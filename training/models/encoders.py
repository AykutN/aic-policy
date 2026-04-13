"""Shared encoder modules for ACT and CFM."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ImageEncoder(nn.Module):
    """ResNet-18 per camera, temporal mean pool, then linear projection.

    Input:  (B, To, n_cams, C, H, W)
    Output: (B, feature_dim)
    """

    def __init__(self, feature_dim: int = 512, pretrained: bool = True):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove final FC — keep spatial avg pool output (512-dim)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        # Unfreeze last 2 residual blocks
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "layer3" in name or "layer4" in name
        self.proj = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, To, n_cams, C, H, W = x.shape
        # Flatten batch × time × cameras
        x = x.reshape(B * To * n_cams, C, H, W)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B*To*n_cams, 512)
        feat = feat.reshape(B, To * n_cams, 512).mean(dim=1)  # temporal+camera mean
        return self.proj(feat)  # (B, feature_dim)


class FTEncoder(nn.Module):
    """1D-CNN over F/T temporal window.

    Input:  (B, window, 6)
    Output: (B, feature_dim)
    """

    def __init__(self, input_dim: int = 6, window: int = 16, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window, input_dim) → (B, input_dim, window)
        x = x.permute(0, 2, 1)
        feat = self.net(x).squeeze(-1)  # (B, 128)
        return self.proj(feat)


class ProprioEncoder(nn.Module):
    """Linear projection over proprioceptive temporal window (mean pool over time).

    Input:  (B, window, 34)
    Output: (B, feature_dim)
    """

    def __init__(self, input_dim: int = 34, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean pool over time window first
        x = x.mean(dim=1)  # (B, input_dim)
        return self.net(x)  # (B, feature_dim)
