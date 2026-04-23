"""Conditional Flow Matching policy (ablation vs ACT)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.encoders import ImageEncoder, FTEncoder, ProprioEncoder

_TIME_EMB_DIM = 64


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for flow time t ∈ [0, 1]."""

    def __init__(self, dim: int = _TIME_EMB_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class FlowNet(nn.Module):
    """Residual MLP with sinusoidal time embedding: (noisy_action, t, cond) -> velocity."""

    def __init__(self, action_dim: int, action_chunk: int, cond_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(_TIME_EMB_DIM)
        self.time_proj = nn.Sequential(nn.Linear(_TIME_EMB_DIM, hidden_dim), nn.SiLU())

        in_dim = action_dim * action_chunk + hidden_dim + cond_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim * action_chunk),
        )
        self.action_dim = action_dim
        self.action_chunk = action_chunk

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        t_emb = self.time_proj(self.time_emb(t))
        inp = torch.cat([x_flat, t_emb, cond], dim=-1)
        h = self.input_proj(inp)
        for block in self.res_blocks:
            h = block(h)
        return self.out_proj(h).reshape(B, self.action_chunk, self.action_dim)


class CFMPolicy(nn.Module):
    """Conditional Flow Matching policy for a single subtask."""

    def __init__(
        self,
        image_feature_dim: int = 512,
        ft_feature_dim: int = 128,
        proprio_feature_dim: int = 64,
        fusion_dim: int = 256,
        flow_hidden_dim: int = 512,
        flow_layers: int = 6,
        action_dim: int = 20,
        action_chunk: int = 32,
    ):
        super().__init__()
        self.action_chunk = action_chunk
        self.action_dim = action_dim

        self.img_enc = ImageEncoder(feature_dim=image_feature_dim)
        self.ft_enc = FTEncoder(feature_dim=ft_feature_dim)
        self.prop_enc = ProprioEncoder(feature_dim=proprio_feature_dim)

        obs_dim = image_feature_dim + ft_feature_dim + proprio_feature_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(obs_dim, fusion_dim),
            nn.SiLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.flow = FlowNet(action_dim, action_chunk, fusion_dim, flow_hidden_dim, flow_layers)

    def _encode(self, images, proprio, ft):
        feat = torch.cat([
            self.img_enc(images),
            self.ft_enc(ft),
            self.prop_enc(proprio),
        ], dim=-1)
        return self.cond_proj(feat)

    def loss(self, images, proprio, ft, actions_gt: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        cond = self._encode(images, proprio, ft)

        x0 = torch.randn_like(actions_gt)
        t = torch.rand(B, device=actions_gt.device)
        t_broadcast = t.view(B, 1, 1)
        x_t = (1 - t_broadcast) * x0 + t_broadcast * actions_gt
        v_target = actions_gt - x0

        v_pred = self.flow(x_t, t, cond)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, images, proprio, ft, n_steps: int = 10) -> torch.Tensor:
        B = images.shape[0]
        cond = self._encode(images, proprio, ft)
        x = torch.randn(B, self.action_chunk, self.action_dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            v = self.flow(x, t, cond)
            x = x + v * dt
        return x
