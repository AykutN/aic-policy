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
        n_train = max(1, int(len(all_episodes) * train_val_split))
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
