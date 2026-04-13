"""Conditional Flow Matching policy (ablation vs ACT)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.encoders import ImageEncoder, FTEncoder, ProprioEncoder


class FlowNet(nn.Module):
    """MLP that predicts velocity field: (noisy_action, t, conditioning) -> velocity."""

    def __init__(self, action_dim: int, action_chunk: int, cond_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        in_dim = action_dim * action_chunk + 1 + cond_dim  # noisy_a flat + t + cond
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, action_dim * action_chunk))
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim
        self.action_chunk = action_chunk

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, Tp, action_dim)
        t:    (B,) in [0, 1]
        cond: (B, cond_dim)
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        inp = torch.cat([x_flat, t.unsqueeze(1), cond], dim=-1)
        v = self.net(inp).reshape(B, self.action_chunk, self.action_dim)
        return v


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
        self.cond_proj = nn.Linear(obs_dim, fusion_dim)

        self.flow = FlowNet(action_dim, action_chunk, fusion_dim, flow_hidden_dim, flow_layers)

    def _encode(self, images, proprio, ft):
        feat = torch.cat([
            self.img_enc(images),
            self.ft_enc(ft),
            self.prop_enc(proprio),
        ], dim=-1)
        return self.cond_proj(feat)  # (B, fusion_dim)

    def loss(self, images, proprio, ft, actions_gt: torch.Tensor) -> torch.Tensor:
        """Conditional flow matching loss (MSE on velocity field)."""
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
        """ODE integration from noise to action via Euler steps."""
        B = images.shape[0]
        cond = self._encode(images, proprio, ft)
        x = torch.randn(B, self.action_chunk, self.action_dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            v = self.flow(x, t, cond)
            x = x + v * dt
        return x
