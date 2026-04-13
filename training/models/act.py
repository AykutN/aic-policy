"""ACT (Action Chunking Transformer) policy."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.encoders import ImageEncoder, FTEncoder, ProprioEncoder


class ACTPolicy(nn.Module):
    """Hierarchical ACT policy for a single subtask.

    Architecture:
      1. Encode observations (image + F/T + proprio) -> fused feature vector
      2. CVAE encoder (during training): actions_gt -> latent z
      3. Transformer decoder: (fused_features + z) -> action sequence (Tp, action_dim)

    At inference: z is sampled from N(0, I).
    """

    def __init__(
        self,
        image_feature_dim: int = 512,
        ft_feature_dim: int = 128,
        proprio_feature_dim: int = 64,
        transformer_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 4,
        latent_dim: int = 64,
        action_dim: int = 20,
        action_chunk: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_chunk = action_chunk
        self.action_dim = action_dim

        # Observation encoders
        self.img_enc = ImageEncoder(feature_dim=image_feature_dim)
        self.ft_enc = FTEncoder(feature_dim=ft_feature_dim)
        self.prop_enc = ProprioEncoder(feature_dim=proprio_feature_dim)

        obs_dim = image_feature_dim + ft_feature_dim + proprio_feature_dim

        # Project fused observation to transformer_dim
        self.obs_proj = nn.Linear(obs_dim, transformer_dim)

        # CVAE encoder: maps action sequence -> (mu, log_var)
        self.cvae_enc = nn.Sequential(
            nn.Linear(action_dim * action_chunk + obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2),  # mu + log_var
        )

        # Latent projection to transformer_dim
        self.latent_proj = nn.Linear(latent_dim, transformer_dim)

        # Transformer decoder: takes (obs_token + latent_token) as memory, decodes Tp tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # Action query tokens (learned, one per timestep in chunk)
        self.action_queries = nn.Embedding(action_chunk, transformer_dim)

        # Output head
        self.action_head = nn.Linear(transformer_dim, action_dim)

    def encode_obs(self, images, proprio, ft):
        img_feat = self.img_enc(images)       # (B, img_dim)
        ft_feat = self.ft_enc(ft)             # (B, ft_dim)
        prop_feat = self.prop_enc(proprio)    # (B, prop_dim)
        return torch.cat([img_feat, ft_feat, prop_feat], dim=-1)  # (B, obs_dim)

    def forward(self, images, proprio, ft, actions_gt=None):
        """
        images:     (B, To_img, n_cams, C, H, W)
        proprio:    (B, To_prop, 34)
        ft:         (B, To_prop, 6)
        actions_gt: (B, Tp, 20) or None (inference)

        Returns: (pred_actions, mu, log_var) -- mu/log_var are None at inference
        """
        B = images.shape[0]
        obs_feat = self.encode_obs(images, proprio, ft)  # (B, obs_dim)

        # CVAE: encode latent from ground truth actions (training only)
        if actions_gt is not None:
            cvae_input = torch.cat([obs_feat, actions_gt.reshape(B, -1)], dim=-1)
            stats = self.cvae_enc(cvae_input)
            mu, log_var = stats.chunk(2, dim=-1)  # (B, latent_dim)
            z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        else:
            mu = log_var = None
            z = torch.randn(B, self.latent_dim, device=images.device)

        # Build memory: [obs_token, latent_token] -- shape (B, 2, transformer_dim)
        obs_token = self.obs_proj(obs_feat).unsqueeze(1)       # (B, 1, D)
        latent_token = self.latent_proj(z).unsqueeze(1)        # (B, 1, D)
        memory = torch.cat([obs_token, latent_token], dim=1)   # (B, 2, D)

        # Action query tokens
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Tp, D)

        # Decode
        out = self.transformer(queries, memory)     # (B, Tp, D)
        pred_actions = self.action_head(out)        # (B, Tp, action_dim)

        return pred_actions, mu, log_var


def act_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 0.1,
) -> torch.Tensor:
    """L1 reconstruction + KL divergence."""
    recon_loss = F.l1_loss(pred, target)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_weight * kl_loss
