# IL Training Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train hierarchical ACT + CFM imitation learning policies from 420 HDF5 episodes, deployable as `aic_model.Policy` subclass in Gazebo eval.

**Architecture:** Minimal `training/` directory (no ROS package yet). Two subtask models per architecture: approach (`subtask=0`) and insert (`subtask=1`). Shared encoder stack between ACT and CFM. After IL baseline validates, wrap in ROS package for submission.

**Tech Stack:** PyTorch 2.x, torchvision (ResNet-18), h5py, PyYAML, AWS g4dn.xlarge T4 16GB

---

## Reference: HDF5 Dataset Format

Each episode file `episode_NNNN.hdf5` has:
```
attrs: num_steps, success, plug_name, port_name, image_scale
observations/
    left_image:    (T, H, W, 3)  uint8
    center_image:  (T, H, W, 3)  uint8
    right_image:   (T, H, W, 3)  uint8
    joint_pos:     (T, 7)        float32
    joint_vel:     (T, 7)        float32
    gripper_pos:   (T, 1)        float32
    tcp_pos:       (T, 3)        float32
    tcp_quat:      (T, 4)        float32
    tcp_vel_lin:   (T, 3)        float32
    tcp_vel_ang:   (T, 3)        float32
    tcp_error:     (T, 6)        float32
    wrench_force:  (T, 3)        float32
    wrench_torque: (T, 3)        float32
    subtask:       (T,)          int8    -- 0=APPROACH, 1=INSERT
    plug_type:     (T,)          int8    -- 0=SFP, 1=SC
    port_id:       (T,)          int8
    timestamp:     (T,)          float64
actions/
    action_pos:       (T, 3)   float32
    action_quat:      (T, 4)   float32
    action_stiffness: (T, 6)   float32
    action_damping:   (T, 6)   float32
    action_gripper:   (T, 1)   float32
```

Action vector = concat(action_pos, action_quat, action_stiffness, action_damping, action_gripper) = **20D**

Proprioceptive vector = concat(tcp_pos, tcp_quat, tcp_vel_lin, tcp_vel_ang, tcp_error, joint_pos, joint_vel, gripper_pos) = 3+4+3+3+6+7+7+1 = **34D**

F/T vector = concat(wrench_force, wrench_torque) = **6D**

---

## Task 1: Directory scaffold + configs

**Files:**
- Create: `training/__init__.py`
- Create: `training/models/__init__.py`
- Create: `training/configs/act.yaml`
- Create: `training/configs/cfm.yaml`

**Step 1: Create directories**

```bash
cd ~/ws_aic/src/aic
mkdir -p training/models training/configs
touch training/__init__.py training/models/__init__.py
```

**Step 2: Create `training/configs/act.yaml`**

```yaml
# ACT hyperparameters
model:
  image_feature_dim: 512      # ResNet-18 output per camera
  ft_feature_dim: 128         # 1D-CNN output
  proprio_feature_dim: 64     # Linear projection output
  transformer_dim: 256
  transformer_heads: 4
  transformer_layers: 4
  latent_dim: 32              # CVAE latent
  action_dim: 20              # 3+4+6+6+1
  action_chunk: 32            # Tp

data:
  obs_window_image: 4         # To for RGB (200ms)
  obs_window_proprio: 16      # To for F/T + proprio (800ms)
  train_val_split: 0.9        # episode-level
  image_size: [256, 288]      # H x W after scale (IMAGE_SCALE=0.25)

training:
  batch_size: 32
  lr: 1e-4
  weight_decay: 1e-4
  epochs: 200
  early_stopping_patience: 20
  checkpoint_every_n: 1       # save to EBS every epoch
  s3_sync_every_n: 5          # sync to S3 every N epochs
  s3_bucket: "s3://aic-yusuf/checkpoints"
  checkpoint_dir: "~/checkpoints"
```

**Step 3: Create `training/configs/cfm.yaml`**

```yaml
# CFM hyperparameters (inherits same encoder config as act.yaml)
model:
  image_feature_dim: 512
  ft_feature_dim: 128
  proprio_feature_dim: 64
  fusion_dim: 256
  flow_hidden_dim: 512        # MLP hidden dim for flow network
  flow_layers: 6
  action_dim: 20
  action_chunk: 32

data:
  obs_window_image: 4
  obs_window_proprio: 16
  train_val_split: 0.9
  image_size: [256, 288]

training:
  batch_size: 32
  lr: 1e-4
  weight_decay: 1e-4
  epochs: 200
  early_stopping_patience: 20
  checkpoint_every_n: 1
  s3_sync_every_n: 5
  s3_bucket: "s3://aic-yusuf/checkpoints"
  checkpoint_dir: "~/checkpoints"
  n_flow_steps: 10            # ODE integration steps at inference
```

**Step 4: Commit**

```bash
git add training/
git commit -m "feat: training directory scaffold and configs"
```

---

## Task 2: HDF5 Dataset loader

**Files:**
- Create: `training/dataset.py`
- Create: `training/test_dataset.py`

**Step 1: Write the failing test**

```python
# training/test_dataset.py
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path

def make_fake_episode(path, T=150, H=256, W=288):
    with h5py.File(path, "w") as f:
        f.attrs["num_steps"] = T
        f.attrs["success"] = True
        f.attrs["plug_name"] = "sfp_plug"
        f.attrs["port_name"] = "port_0"
        f.attrs["image_scale"] = 0.25
        obs = f.create_group("observations")
        obs.create_dataset("left_image", data=np.zeros((T, H, W, 3), dtype=np.uint8))
        obs.create_dataset("center_image", data=np.zeros((T, H, W, 3), dtype=np.uint8))
        obs.create_dataset("right_image", data=np.zeros((T, H, W, 3), dtype=np.uint8))
        obs.create_dataset("joint_pos", data=np.zeros((T, 7), dtype=np.float32))
        obs.create_dataset("joint_vel", data=np.zeros((T, 7), dtype=np.float32))
        obs.create_dataset("gripper_pos", data=np.zeros((T, 1), dtype=np.float32))
        obs.create_dataset("tcp_pos", data=np.zeros((T, 3), dtype=np.float32))
        obs.create_dataset("tcp_quat", data=np.zeros((T, 4), dtype=np.float32))
        obs.create_dataset("tcp_vel_lin", data=np.zeros((T, 3), dtype=np.float32))
        obs.create_dataset("tcp_vel_ang", data=np.zeros((T, 3), dtype=np.float32))
        obs.create_dataset("tcp_error", data=np.zeros((T, 6), dtype=np.float32))
        obs.create_dataset("wrench_force", data=np.zeros((T, 3), dtype=np.float32))
        obs.create_dataset("wrench_torque", data=np.zeros((T, 3), dtype=np.float32))
        # subtask: first 100 steps = 0 (APPROACH), rest = 1 (INSERT)
        subtask = np.zeros(T, dtype=np.int8)
        subtask[100:] = 1
        obs.create_dataset("subtask", data=subtask)
        obs.create_dataset("plug_type", data=np.zeros(T, dtype=np.int8))
        obs.create_dataset("port_id", data=np.zeros(T, dtype=np.int8))
        obs.create_dataset("timestamp", data=np.zeros(T, dtype=np.float64))
        act = f.create_group("actions")
        act.create_dataset("action_pos", data=np.zeros((T, 3), dtype=np.float32))
        act.create_dataset("action_quat", data=np.zeros((T, 4), dtype=np.float32))
        act.create_dataset("action_stiffness", data=np.zeros((T, 6), dtype=np.float32))
        act.create_dataset("action_damping", data=np.zeros((T, 6), dtype=np.float32))
        act.create_dataset("action_gripper", data=np.zeros((T, 1), dtype=np.float32))


def test_dataset_returns_correct_keys():
    from training.dataset import AICDataset
    with tempfile.TemporaryDirectory() as d:
        make_fake_episode(os.path.join(d, "episode_0000.hdf5"))
        make_fake_episode(os.path.join(d, "episode_0001.hdf5"))
        ds = AICDataset(d, subtask=0, obs_window_image=4, obs_window_proprio=16, action_chunk=32)
        sample = ds[0]
        assert "images" in sample          # (To_img, 3, C, H, W)
        assert "proprio" in sample         # (To_prop, 34)
        assert "ft" in sample              # (To_prop, 6)
        assert "actions" in sample         # (Tp, 20)
        assert "plug_type" in sample       # scalar int
        assert "port_id" in sample         # scalar int


def test_dataset_subtask_filter():
    from training.dataset import AICDataset
    with tempfile.TemporaryDirectory() as d:
        make_fake_episode(os.path.join(d, "episode_0000.hdf5"), T=150)
        ds0 = AICDataset(d, subtask=0, obs_window_image=4, obs_window_proprio=16, action_chunk=32)
        ds1 = AICDataset(d, subtask=1, obs_window_image=4, obs_window_proprio=16, action_chunk=32)
        # subtask=0 has 100 steps, but need obs_window_proprio steps before first valid index
        # subtask=1 has 50 steps
        assert len(ds0) > 0
        assert len(ds1) > 0
        # No overlap
        assert len(ds0) + len(ds1) <= 150


def test_dataset_action_shape():
    from training.dataset import AICDataset
    with tempfile.TemporaryDirectory() as d:
        make_fake_episode(os.path.join(d, "episode_0000.hdf5"), T=200)
        ds = AICDataset(d, subtask=1, obs_window_image=4, obs_window_proprio=16, action_chunk=32)
        sample = ds[0]
        assert sample["actions"].shape == (32, 20)
        assert sample["images"].shape == (4, 3, 3, 256, 288)   # (To, n_cams, C, H, W)
        assert sample["proprio"].shape == (16, 34)
        assert sample["ft"].shape == (16, 6)


if __name__ == "__main__":
    test_dataset_returns_correct_keys()
    test_dataset_subtask_filter()
    test_dataset_action_shape()
    print("All tests passed.")
```

**Step 2: Run test to verify it fails**

```bash
cd ~/ws_aic/src/aic
pixi run python3 -m pytest training/test_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'training.dataset'`

**Step 3: Implement `training/dataset.py`**

```python
"""HDF5 dataset loader for AIC IL training."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class AICDataset(Dataset):
    """Loads (observation, action) pairs from AIC HDF5 episodes.

    Observation windows:
      images:  last `obs_window_image` steps, 3 cameras → (To_img, 3, C, H, W)
      proprio: last `obs_window_proprio` steps → (To_prop, 34)
      ft:      last `obs_window_proprio` steps → (To_prop, 6)
    Action:    next `action_chunk` steps → (Tp, 20)
    """

    # Proprioceptive fields and their dims
    PROPRIO_FIELDS = [
        ("tcp_pos", 3), ("tcp_quat", 4), ("tcp_vel_lin", 3), ("tcp_vel_ang", 3),
        ("tcp_error", 6), ("joint_pos", 7), ("joint_vel", 7), ("gripper_pos", 1),
    ]  # total 34D
    FT_FIELDS = [("wrench_force", 3), ("wrench_torque", 3)]  # 6D
    ACTION_FIELDS = [
        ("action_pos", 3), ("action_quat", 4),
        ("action_stiffness", 6), ("action_damping", 6), ("action_gripper", 1),
    ]  # 20D

    def __init__(
        self,
        dataset_dir: str,
        subtask: int,
        obs_window_image: int = 4,
        obs_window_proprio: int = 16,
        action_chunk: int = 32,
        augment: bool = False,
        train: bool = True,
        train_val_split: float = 0.9,
    ):
        self.subtask = subtask
        self.To_img = obs_window_image
        self.To_prop = obs_window_proprio
        self.Tp = action_chunk
        self.augment = augment

        # Collect episode files, episode-level train/val split
        all_episodes = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
        n_train = int(len(all_episodes) * train_val_split)
        episodes = all_episodes[:n_train] if train else all_episodes[n_train:]

        # Build index: list of (episode_path, step_idx)
        # step_idx is the CURRENT step (we look back To steps and forward Tp steps)
        self._index: List[Tuple[Path, int]] = []
        for ep_path in episodes:
            with h5py.File(ep_path, "r") as f:
                subtask_arr = f["observations"]["subtask"][:]
                T = len(subtask_arr)
            # Valid indices: need To_prop steps of history AND Tp steps of future
            # AND must belong to the requested subtask
            lookback = max(obs_window_image, obs_window_proprio)
            for t in range(lookback, T - action_chunk):
                if subtask_arr[t] == subtask:
                    self._index.append((ep_path, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_path, t = self._index[idx]
        with h5py.File(ep_path, "r") as f:
            obs = f["observations"]
            act = f["actions"]

            # Images: (To_img, 3, C, H, W)
            img_start = t - self.To_img + 1
            imgs = np.stack([
                obs["left_image"][img_start:t + 1],
                obs["center_image"][img_start:t + 1],
                obs["right_image"][img_start:t + 1],
            ], axis=1)  # (To_img, 3_cams, H, W, C)
            # → (To_img, 3_cams, C, H, W), float32 [0,1]
            imgs = torch.from_numpy(imgs).float() / 255.0
            imgs = imgs.permute(0, 1, 4, 2, 3)

            # Proprioceptive: (To_prop, 34)
            prop_start = t - self.To_prop + 1
            proprio_parts = [
                obs[field][prop_start:t + 1] for field, _ in self.PROPRIO_FIELDS
            ]
            proprio = np.concatenate(proprio_parts, axis=-1).astype(np.float32)

            # F/T: (To_prop, 6)
            ft_parts = [obs[field][prop_start:t + 1] for field, _ in self.FT_FIELDS]
            ft = np.concatenate(ft_parts, axis=-1).astype(np.float32)

            # Actions: (Tp, 20)
            act_parts = [act[field][t:t + self.Tp] for field, _ in self.ACTION_FIELDS]
            actions = np.concatenate(act_parts, axis=-1).astype(np.float32)
            # Pad last episode steps with last action if needed
            if actions.shape[0] < self.Tp:
                pad = np.repeat(actions[-1:], self.Tp - actions.shape[0], axis=0)
                actions = np.concatenate([actions, pad], axis=0)

            plug_type = int(obs["plug_type"][t])
            port_id = int(obs["port_id"][t])

        if self.augment:
            imgs = self._augment(imgs)

        return {
            "images": imgs,
            "proprio": torch.from_numpy(proprio),
            "ft": torch.from_numpy(ft),
            "actions": torch.from_numpy(actions),
            "plug_type": torch.tensor(plug_type, dtype=torch.long),
            "port_id": torch.tensor(port_id, dtype=torch.long),
        }

    def _augment(self, imgs: torch.Tensor) -> torch.Tensor:
        """Random crop + color jitter per timestep."""
        To, n_cams, C, H, W = imgs.shape
        brightness, contrast, saturation, hue = 0.2, 0.2, 0.2, 0.05
        out = []
        for t in range(To):
            frame_cams = []
            for c in range(n_cams):
                img = imgs[t, c]  # (C, H, W)
                img = TF.adjust_brightness(img, 1 + (torch.rand(1).item() - 0.5) * 2 * brightness)
                img = TF.adjust_contrast(img, 1 + (torch.rand(1).item() - 0.5) * 2 * contrast)
                img = TF.adjust_saturation(img, 1 + (torch.rand(1).item() - 0.5) * 2 * saturation)
                img = TF.adjust_hue(img, (torch.rand(1).item() - 0.5) * 2 * hue)
                frame_cams.append(img)
            out.append(torch.stack(frame_cams))
        return torch.stack(out)
```

**Step 4: Run tests**

```bash
pixi run python3 -m pytest training/test_dataset.py -v
```

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add training/dataset.py training/test_dataset.py
git commit -m "feat: HDF5 dataset loader with subtask filtering and multi-rate windows"
```

---

## Task 3: Shared encoder modules

**Files:**
- Create: `training/models/encoders.py`
- Create: `training/test_encoders.py`

**Step 1: Write the failing test**

```python
# training/test_encoders.py
import torch

def test_image_encoder_output_shape():
    from training.models.encoders import ImageEncoder
    enc = ImageEncoder(feature_dim=512)
    # (B, To_img, n_cams, C, H, W)
    x = torch.zeros(2, 4, 3, 3, 256, 288)
    out = enc(x)
    assert out.shape == (2, 512), f"Got {out.shape}"


def test_ft_encoder_output_shape():
    from training.models.encoders import FTEncoder
    enc = FTEncoder(input_dim=6, window=16, feature_dim=128)
    x = torch.zeros(2, 16, 6)
    out = enc(x)
    assert out.shape == (2, 128), f"Got {out.shape}"


def test_proprio_encoder_output_shape():
    from training.models.encoders import ProprioEncoder
    enc = ProprioEncoder(input_dim=34, feature_dim=64)
    x = torch.zeros(2, 16, 34)
    out = enc(x)
    assert out.shape == (2, 64), f"Got {out.shape}"


if __name__ == "__main__":
    test_image_encoder_output_shape()
    test_ft_encoder_output_shape()
    test_proprio_encoder_output_shape()
    print("All encoder tests passed.")
```

**Step 2: Run test to verify it fails**

```bash
pixi run python3 -m pytest training/test_encoders.py -v
```

**Step 3: Implement `training/models/encoders.py`**

```python
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
```

**Step 4: Run tests**

```bash
pixi run python3 -m pytest training/test_encoders.py -v
```

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add training/models/encoders.py training/test_encoders.py
git commit -m "feat: shared image/FT/proprio encoder modules"
```

---

## Task 4: ACT model

**Files:**
- Create: `training/models/act.py`
- Create: `training/test_act.py`

**Step 1: Write the failing test**

```python
# training/test_act.py
import torch

def test_act_forward_shape():
    from training.models.act import ACTPolicy
    model = ACTPolicy(
        image_feature_dim=512,
        ft_feature_dim=128,
        proprio_feature_dim=64,
        transformer_dim=256,
        transformer_heads=4,
        transformer_layers=4,
        latent_dim=32,
        action_dim=20,
        action_chunk=32,
    )
    B = 2
    images = torch.zeros(B, 4, 3, 3, 256, 288)
    proprio = torch.zeros(B, 16, 34)
    ft = torch.zeros(B, 16, 6)
    actions_gt = torch.zeros(B, 32, 20)  # only used during training

    # Training forward (with CVAE encoder)
    pred_actions, mu, log_var = model(images, proprio, ft, actions_gt)
    assert pred_actions.shape == (B, 32, 20), f"Got {pred_actions.shape}"
    assert mu.shape == (B, 32), f"Got {mu.shape}"

    # Inference forward (no actions_gt → sample from prior)
    pred_actions_inf, _, _ = model(images, proprio, ft, actions_gt=None)
    assert pred_actions_inf.shape == (B, 32, 20)


def test_act_loss():
    from training.models.act import act_loss
    pred = torch.zeros(2, 32, 20)
    target = torch.ones(2, 32, 20)
    mu = torch.zeros(2, 32)
    log_var = torch.zeros(2, 32)
    loss = act_loss(pred, target, mu, log_var, kl_weight=0.1)
    assert loss.item() > 0


if __name__ == "__main__":
    test_act_forward_shape()
    test_act_loss()
    print("All ACT tests passed.")
```

**Step 2: Run test to verify it fails**

```bash
pixi run python3 -m pytest training/test_act.py -v
```

**Step 3: Implement `training/models/act.py`**

```python
"""ACT (Action Chunking Transformer) policy."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.encoders import ImageEncoder, FTEncoder, ProprioEncoder


class ACTPolicy(nn.Module):
    """Hierarchical ACT policy for a single subtask.

    Architecture:
      1. Encode observations (image + F/T + proprio) → fused feature vector
      2. CVAE encoder (during training): actions_gt → latent z
      3. Transformer decoder: (fused_features + z) → action sequence (Tp, action_dim)

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
        latent_dim: int = 32,
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

        # CVAE encoder: maps action sequence → (mu, log_var)
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

        Returns: (pred_actions, mu, log_var) — mu/log_var are None at inference
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

        # Build memory: [obs_token, latent_token] — shape (B, 2, transformer_dim)
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
```

**Step 4: Run tests**

```bash
pixi run python3 -m pytest training/test_act.py -v
```

Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add training/models/act.py training/test_act.py
git commit -m "feat: ACT policy model with CVAE encoder and Transformer decoder"
```

---

## Task 5: CFM model

**Files:**
- Create: `training/models/cfm.py`
- Create: `training/test_cfm.py`

**Step 1: Write the failing test**

```python
# training/test_cfm.py
import torch

def test_cfm_forward_shape():
    from training.models.cfm import CFMPolicy
    model = CFMPolicy(
        image_feature_dim=512,
        ft_feature_dim=128,
        proprio_feature_dim=64,
        fusion_dim=256,
        flow_hidden_dim=512,
        flow_layers=6,
        action_dim=20,
        action_chunk=32,
    )
    B = 2
    images = torch.zeros(B, 4, 3, 3, 256, 288)
    proprio = torch.zeros(B, 16, 34)
    ft = torch.zeros(B, 16, 6)
    actions_gt = torch.rand(B, 32, 20)

    loss = model.loss(images, proprio, ft, actions_gt)
    assert loss.item() > 0

    # Inference
    pred = model.sample(images, proprio, ft, n_steps=5)
    assert pred.shape == (B, 32, 20), f"Got {pred.shape}"


if __name__ == "__main__":
    test_cfm_forward_shape()
    print("All CFM tests passed.")
```

**Step 2: Run test to verify it fails**

```bash
pixi run python3 -m pytest training/test_cfm.py -v
```

**Step 3: Implement `training/models/cfm.py`**

```python
"""Conditional Flow Matching policy (ablation vs ACT)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.encoders import ImageEncoder, FTEncoder, ProprioEncoder


class FlowNet(nn.Module):
    """MLP that predicts velocity field: (noisy_action, t, conditioning) → velocity."""

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

        # Sample noise and time
        x0 = torch.randn_like(actions_gt)
        t = torch.rand(B, device=actions_gt.device)
        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        t_broadcast = t.view(B, 1, 1)
        x_t = (1 - t_broadcast) * x0 + t_broadcast * actions_gt
        # Target velocity field
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
```

**Step 4: Run tests**

```bash
pixi run python3 -m pytest training/test_cfm.py -v
```

Expected: 1 test PASS

**Step 5: Commit**

```bash
git add training/models/cfm.py training/test_cfm.py
git commit -m "feat: conditional flow matching policy model"
```

---

## Task 6: Training loop (ACT)

**Files:**
- Create: `training/train_act.py`

No unit test for training loop — verify manually with 2-epoch smoke test.

**Step 1: Implement `training/train_act.py`**

```python
#!/usr/bin/env python3
"""ACT training loop.

Usage:
    python3 training/train_act.py --dataset ~/aic_dataset --subtask 0
    python3 training/train_act.py --dataset ~/aic_dataset --subtask 1
    python3 training/train_act.py --dataset ~/aic_dataset --subtask 1 --resume
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from training.dataset import AICDataset
from training.models.act import ACTPolicy, act_loss


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def s3_sync(local: str, s3: str):
    subprocess.run(["aws", "s3", "sync", local, s3, "--quiet"], check=False)


def save_checkpoint(model, optimizer, epoch, val_loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


def train(args):
    cfg = load_config(args.config)
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["training"]

    checkpoint_dir = Path(tc["checkpoint_dir"]).expanduser() / f"act_subtask{args.subtask}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ACTPolicy(
        image_feature_dim=mc["image_feature_dim"],
        ft_feature_dim=mc["ft_feature_dim"],
        proprio_feature_dim=mc["proprio_feature_dim"],
        transformer_dim=mc["transformer_dim"],
        transformer_heads=mc["transformer_heads"],
        transformer_layers=mc["transformer_layers"],
        latent_dim=mc["latent_dim"],
        action_dim=mc["action_dim"],
        action_chunk=mc["action_chunk"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    train_ds = AICDataset(
        args.dataset, subtask=args.subtask,
        obs_window_image=dc["obs_window_image"],
        obs_window_proprio=dc["obs_window_proprio"],
        action_chunk=mc["action_chunk"],
        augment=True, train=True,
        train_val_split=dc["train_val_split"],
    )
    val_ds = AICDataset(
        args.dataset, subtask=args.subtask,
        obs_window_image=dc["obs_window_image"],
        obs_window_proprio=dc["obs_window_proprio"],
        action_chunk=mc["action_chunk"],
        augment=False, train=False,
        train_val_split=dc["train_val_split"],
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, tc["epochs"]):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["images"].to(device)
            proprio = batch["proprio"].to(device)
            ft = batch["ft"].to(device)
            actions = batch["actions"].to(device)

            optimizer.zero_grad()
            pred, mu, log_var = model(images, proprio, ft, actions)
            loss = act_loss(pred, actions, mu, log_var, kl_weight=0.1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                proprio = batch["proprio"].to(device)
                ft = batch["ft"].to(device)
                actions = batch["actions"].to(device)
                pred, mu, log_var = model(images, proprio, ft, actions)
                val_loss += act_loss(pred, actions, mu, log_var).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # --- Checkpoint ---
        ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
        save_checkpoint(model, optimizer, epoch, val_loss, latest_ckpt)

        # S3 sync every N epochs
        if (epoch + 1) % tc["s3_sync_every_n"] == 0:
            print(f"  S3 sync...")
            s3_sync(str(checkpoint_dir), tc["s3_bucket"] + f"/act_subtask{args.subtask}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Final S3 sync
    s3_sync(str(checkpoint_dir), tc["s3_bucket"] + f"/act_subtask{args.subtask}")
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {checkpoint_dir}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset directory")
    parser.add_argument("--subtask", type=int, required=True, choices=[0, 1])
    parser.add_argument("--config", default="training/configs/act.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
```

**Step 2: Smoke test (2 epochs, tiny dataset)**

```bash
# On AWS, with a few episodes downloaded:
cd ~/ws_aic/src/aic
aws s3 sync s3://aic-yusuf/aic_dataset/ ~/aic_dataset/ --quiet

pixi run python3 training/train_act.py \
  --dataset ~/aic_dataset \
  --subtask 0 \
  --config training/configs/act.yaml
```

Expected: 2 epoch lines printed, checkpoint written to `~/checkpoints/act_subtask0/`.

**Step 3: Commit**

```bash
git add training/train_act.py
git commit -m "feat: ACT training loop with checkpoint and S3 sync"
```

---

## Task 7: Training loop (CFM)

**Files:**
- Create: `training/train_cfm.py`

Same structure as `train_act.py`. Key differences: uses `CFMPolicy.loss()` directly (no CVAE), no KL term.

**Step 1: Implement `training/train_cfm.py`**

```python
#!/usr/bin/env python3
"""CFM training loop (ablation).

Usage:
    python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 0
    python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 1 --resume
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from training.dataset import AICDataset
from training.models.cfm import CFMPolicy


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def s3_sync(local, s3):
    subprocess.run(["aws", "s3", "sync", local, s3, "--quiet"], check=False)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


def train(args):
    cfg = load_config(args.config)
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["training"]

    checkpoint_dir = Path(tc["checkpoint_dir"]).expanduser() / f"cfm_subtask{args.subtask}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CFMPolicy(
        image_feature_dim=mc["image_feature_dim"],
        ft_feature_dim=mc["ft_feature_dim"],
        proprio_feature_dim=mc["proprio_feature_dim"],
        fusion_dim=mc["fusion_dim"],
        flow_hidden_dim=mc["flow_hidden_dim"],
        flow_layers=mc["flow_layers"],
        action_dim=mc["action_dim"],
        action_chunk=mc["action_chunk"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from epoch {start_epoch}")

    train_ds = AICDataset(args.dataset, subtask=args.subtask,
        obs_window_image=dc["obs_window_image"], obs_window_proprio=dc["obs_window_proprio"],
        action_chunk=mc["action_chunk"], augment=True, train=True,
        train_val_split=dc["train_val_split"])
    val_ds = AICDataset(args.dataset, subtask=args.subtask,
        obs_window_image=dc["obs_window_image"], obs_window_proprio=dc["obs_window_proprio"],
        action_chunk=mc["action_chunk"], augment=False, train=False,
        train_val_split=dc["train_val_split"])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["images"].to(device)
            proprio = batch["proprio"].to(device)
            ft = batch["ft"].to(device)
            actions = batch["actions"].to(device)
            optimizer.zero_grad()
            loss = model.loss(images, proprio, ft, actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += model.loss(
                    batch["images"].to(device),
                    batch["proprio"].to(device),
                    batch["ft"].to(device),
                    batch["actions"].to(device),
                ).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir / f"epoch_{epoch:04d}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, latest_ckpt)

        if (epoch + 1) % tc["s3_sync_every_n"] == 0:
            s3_sync(str(checkpoint_dir), tc["s3_bucket"] + f"/cfm_subtask{args.subtask}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    s3_sync(str(checkpoint_dir), tc["s3_bucket"] + f"/cfm_subtask{args.subtask}")
    print(f"Done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subtask", type=int, required=True, choices=[0, 1])
    parser.add_argument("--config", default="training/configs/cfm.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
```

**Step 2: Commit**

```bash
git add training/train_cfm.py
git commit -m "feat: CFM training loop"
```

---

## Task 8: Evaluation policy (Gazebo runner)

**Files:**
- Create: `training/evaluate.py`

**Step 1: Implement `training/evaluate.py`**

```python
#!/usr/bin/env python3
"""Evaluation policy: loads trained ACT checkpoints, runs as aic_model.Policy in Gazebo.

Usage (same as any other policy):
    pixi run ros2 run aic_model aic_model \
      --ros-args -p use_sim_time:=true \
      -p policy:=training.evaluate.TrainedPolicy \
      -p approach_ckpt:=/home/ubuntu/checkpoints/act_subtask0/best.pt \
      -p insert_ckpt:=/home/ubuntu/checkpoints/act_subtask1/best.pt
"""
from __future__ import annotations

import numpy as np
import torch
import cv2
from pathlib import Path

from aic_model.policy import Policy, GetObservationCallback, MoveRobotCallback, SendFeedbackCallback
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

from training.models.act import ACTPolicy
from training.configs import load_yaml

IMAGE_SCALE = 0.25
OBS_WINDOW_IMAGE = 4
OBS_WINDOW_PROPRIO = 16
ACTION_CHUNK = 32
ACTION_DIM = 20

APPROACH_STEPS = 100  # matches DataCollectorPolicy


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def obs_to_tensors(obs_buffer: list, device) -> tuple:
    """Convert rolling observation buffer to model input tensors."""
    # obs_buffer: list of dicts, most recent last
    To_img = OBS_WINDOW_IMAGE
    To_prop = OBS_WINDOW_PROPRIO

    # Images: use last To_img obs
    img_buf = obs_buffer[-To_img:]
    imgs = np.stack([[
        o["left_image"], o["center_image"], o["right_image"]
    ] for o in img_buf])  # (To_img, 3, H, W, C)
    imgs = torch.from_numpy(imgs).float() / 255.0
    imgs = imgs.permute(0, 1, 4, 2, 3).unsqueeze(0).to(device)  # (1, To_img, 3, C, H, W)

    # Proprio: use last To_prop obs
    prop_buf = obs_buffer[-To_prop:]
    proprio = np.stack([np.concatenate([
        o["tcp_pos"], o["tcp_quat"], o["tcp_vel_lin"], o["tcp_vel_ang"],
        o["tcp_error"], o["joint_pos"], o["joint_vel"], [o["gripper_pos"]],
    ]) for o in prop_buf]).astype(np.float32)
    proprio = torch.from_numpy(proprio).unsqueeze(0).to(device)  # (1, To_prop, 34)

    # F/T
    ft = np.stack([np.concatenate([
        o["wrench_force"], o["wrench_torque"]
    ]) for o in prop_buf]).astype(np.float32)
    ft = torch.from_numpy(ft).unsqueeze(0).to(device)  # (1, To_prop, 6)

    return imgs, proprio, ft


def action_to_motion_update(action: np.ndarray, frame_id: str, stamp) -> MotionUpdate:
    """Convert 20D action vector to MotionUpdate message."""
    from geometry_msgs.msg import Pose, Vector3, Wrench
    from aic_control_interfaces.msg import TrajectoryGenerationMode
    pos = action[:3]
    quat = action[3:7]
    stiffness = action[7:13]
    damping = action[13:19]
    # action[19] = gripper (ignored for now — gripper held constant)

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])

    return MotionUpdate(
        header=Header(frame_id=frame_id, stamp=stamp),
        pose=pose,
        target_stiffness=np.diag(stiffness).flatten(),
        target_damping=np.diag(damping).flatten(),
        trajectory_generation_mode=__import__(
            "aic_control_interfaces.msg", fromlist=["TrajectoryGenerationMode"]
        ).TrajectoryGenerationMode(mode=0),
    )


class TrainedPolicy(Policy):
    """Hierarchical trained policy: approach model (subtask=0) → insert model (subtask=1)."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = load_yaml("training/configs/act.yaml")
        mc = cfg["model"]

        def _load_model(ckpt_path: str) -> ACTPolicy:
            m = ACTPolicy(
                image_feature_dim=mc["image_feature_dim"],
                ft_feature_dim=mc["ft_feature_dim"],
                proprio_feature_dim=mc["proprio_feature_dim"],
                transformer_dim=mc["transformer_dim"],
                transformer_heads=mc["transformer_heads"],
                transformer_layers=mc["transformer_layers"],
                latent_dim=mc["latent_dim"],
                action_dim=mc["action_dim"],
                action_chunk=mc["action_chunk"],
            ).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            return m

        # Read checkpoint paths from ROS params (set via --ros-args -p approach_ckpt:=...)
        approach_ckpt = parent_node.declare_parameter("approach_ckpt", "").value
        insert_ckpt = parent_node.declare_parameter("insert_ckpt", "").value

        self.approach_model = _load_model(approach_ckpt)
        self.insert_model = _load_model(insert_ckpt)
        self.get_logger().info(f"Loaded approach: {approach_ckpt}")
        self.get_logger().info(f"Loaded insert: {insert_ckpt}")

    def _obs_to_dict(self, obs) -> dict:
        """Convert aic_model_interfaces.msg.Observation → plain dict for buffer."""
        H = int(round(1024 * IMAGE_SCALE))
        W = int(round(1152 * IMAGE_SCALE))

        def decode_img(data):
            arr = np.frombuffer(bytes(data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros((H, W, 3), dtype=np.uint8)
            return cv2.resize(img, (W, H))

        return {
            "left_image": decode_img(obs.left_image.data),
            "center_image": decode_img(obs.center_image.data),
            "right_image": decode_img(obs.right_image.data),
            "tcp_pos": np.array([obs.tcp_pose.position.x, obs.tcp_pose.position.y, obs.tcp_pose.position.z]),
            "tcp_quat": np.array([obs.tcp_pose.orientation.x, obs.tcp_pose.orientation.y,
                                   obs.tcp_pose.orientation.z, obs.tcp_pose.orientation.w]),
            "tcp_vel_lin": np.array([obs.tcp_velocity.linear.x, obs.tcp_velocity.linear.y, obs.tcp_velocity.linear.z]),
            "tcp_vel_ang": np.array([obs.tcp_velocity.angular.x, obs.tcp_velocity.angular.y, obs.tcp_velocity.angular.z]),
            "tcp_error": np.array(obs.tcp_error),
            "joint_pos": np.array(obs.joint_states.position[:7]),
            "joint_vel": np.array(obs.joint_states.velocity[:7]),
            "gripper_pos": obs.joint_states.position[6],
            "wrench_force": np.array([obs.wrench.force.x, obs.wrench.force.y, obs.wrench.force.z]),
            "wrench_torque": np.array([obs.wrench.torque.x, obs.wrench.torque.y, obs.wrench.torque.z]),
        }

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        lookback = max(OBS_WINDOW_IMAGE, OBS_WINDOW_PROPRIO)
        obs_buffer = []

        # Warm up buffer
        for _ in range(lookback):
            obs_buffer.append(self._obs_to_dict(get_observation()))

        step = 0
        action_queue = []  # remaining actions from last chunk

        while True:
            if not action_queue:
                # Decide subtask based on step count
                model = self.approach_model if step < APPROACH_STEPS else self.insert_model
                imgs, proprio, ft = obs_to_tensors(obs_buffer, self.device)
                with torch.no_grad():
                    pred, _, _ = model(imgs, proprio, ft, actions_gt=None)
                action_queue = pred[0].cpu().numpy().tolist()  # list of (ACTION_DIM,) arrays

            action = np.array(action_queue.pop(0))
            stamp = self._parent_node.get_clock().now().to_msg()
            mu = move_robot(motion_update=action_to_motion_update(action, "base_link", stamp))

            obs_buffer.append(self._obs_to_dict(get_observation()))
            if len(obs_buffer) > lookback:
                obs_buffer.pop(0)

            step += 1

            # Termination: check wrench for insertion success (simple threshold)
            wrench = obs_buffer[-1]["wrench_force"]
            if step > APPROACH_STEPS and np.linalg.norm(wrench) < 0.5 and step > APPROACH_STEPS + 50:
                # Low force after insertion phase → likely seated
                send_feedback("Insertion complete")
                return True

            if step > 2000:
                send_feedback("Max steps reached")
                return False
```

**Step 2: Commit**

```bash
git add training/evaluate.py
git commit -m "feat: evaluation policy subclass for Gazebo deployment"
```

---

## Task 9: AGENT_TRAIN.md — AWS run instructions

**Files:**
- Create: `AGENT_TRAIN.md`

**Step 1: Create `AGENT_TRAIN.md`**

```markdown
# AGENT_TRAIN.md — IL Training on AWS

## Prerequisites
- g4dn.xlarge on-demand (NOT spot — training must not be interrupted)
- Dataset at ~/aic_dataset/ (420+ episodes)
- Code at ~/ws_aic/src/aic/

## Setup (once)

```bash
cd ~/ws_aic/src/aic

# Install PyTorch if not present (Deep Learning AMI has it)
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Install training deps
pip install pyyaml h5py torchvision

# Verify dataset
pixi run python3 aic_data_collector/scripts/inspect_dataset.py ~/aic_dataset
```

## Run All 4 Models

```bash
cd ~/ws_aic/src/aic
mkdir -p ~/checkpoints

# ACT approach (subtask=0) — ~3-4 hours
nohup python3 training/train_act.py \
  --dataset ~/aic_dataset --subtask 0 \
  > ~/checkpoints/act_subtask0_train.log 2>&1 &

# After subtask 0 finishes, run subtask 1:
python3 training/train_act.py \
  --dataset ~/aic_dataset --subtask 1

# CFM models (ablation, run after ACT):
python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 0
python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 1
```

## Resume After Interrupt

```bash
python3 training/train_act.py --dataset ~/aic_dataset --subtask 1 --resume
```

## Monitor

```bash
tail -f ~/checkpoints/act_subtask0/train.log
watch -n 30 'ls -lh ~/checkpoints/act_subtask*/best.pt 2>/dev/null'
```

## Evaluate in Gazebo

```bash
# Terminal 1: eval container
export DBX_CONTAINER_MANAGER=docker
distrobox enter --root aic_eval -- /entrypoint.sh \
  ground_truth:=false start_aic_engine:=true

# Terminal 2: trained policy
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=training.evaluate.TrainedPolicy \
  -p approach_ckpt:=$HOME/checkpoints/act_subtask0/best.pt \
  -p insert_ckpt:=$HOME/checkpoints/act_subtask1/best.pt
```

## Expected Results

| Model | Subtask | Expected val_loss | Expected success rate |
|---|---|---|---|
| ACT | approach (0) | < 0.05 | N/A (open-loop) |
| ACT | insert (1) | < 0.08 | 55-70% |
| CFM | insert (1) | < 0.01 (MSE) | 55-75% |

## Next Step: DPPO Fine-tuning
After IL baseline achieves >50% success rate, proceed to DPPO fine-tuning
(not yet implemented — see docs/plans/ for future design doc).
```

**Step 2: Commit**

```bash
git add AGENT_TRAIN.md
git commit -m "docs: AGENT_TRAIN.md — IL training instructions on AWS"
```

---

## Summary: Run Order on AWS

```bash
# 1. Run tests locally first (Mac or AWS)
pixi run python3 -m pytest training/test_dataset.py training/test_encoders.py training/test_act.py training/test_cfm.py -v

# 2. On AWS (on-demand g4dn.xlarge):
python3 training/train_act.py --dataset ~/aic_dataset --subtask 0  # ~3-4h
python3 training/train_act.py --dataset ~/aic_dataset --subtask 1  # ~3-4h
python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 0  # ~3h
python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 1  # ~3h

# 3. Evaluate best models in Gazebo — see AGENT_TRAIN.md
```
