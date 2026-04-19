"""HDF5 dataset loader for AIC IL training."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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
        self._file_cache: Dict[str, h5py.File] = {}  # lazy per-worker file handles

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
            for t in range(lookback, T - action_chunk + 1):
                if subtask_arr[t] == subtask:
                    self._index.append((ep_path, t))

    def __len__(self) -> int:
        return len(self._index)

    def _get_file(self, ep_path: Path) -> h5py.File:
        key = str(ep_path)
        if key not in self._file_cache:
            self._file_cache[key] = h5py.File(ep_path, "r", swmr=True)
        return self._file_cache[key]

    def __getitem__(self, idx: int) -> dict:
        ep_path, t = self._index[idx]
        f = self._get_file(ep_path)
        obs = f["observations"]
        act = f["actions"]

        # Images returned as uint8 (To, 3_cams, C, H, W) — float conversion + resize on GPU
        img_start = t - self.To_img + 1
        imgs = np.stack([
            obs["left_image"][img_start:t + 1],
            obs["center_image"][img_start:t + 1],
            obs["right_image"][img_start:t + 1],
        ], axis=1)  # (To, 3, H, W, C) uint8
        imgs = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 1, 4, 2, 3)))  # (To, 3, C, H, W) uint8

        # Proprioceptive: (To_prop, 34)
        prop_start = t - self.To_prop + 1
        proprio = np.concatenate(
            [obs[field][prop_start:t + 1] for field, _ in self.PROPRIO_FIELDS], axis=-1
        ).astype(np.float32)

        # F/T: (To_prop, 6)
        ft = np.concatenate(
            [obs[field][prop_start:t + 1] for field, _ in self.FT_FIELDS], axis=-1
        ).astype(np.float32)

        # Actions: (Tp, 20)
        actions = np.concatenate(
            [act[field][t:t + self.Tp] for field, _ in self.ACTION_FIELDS], axis=-1
        ).astype(np.float32)
        if actions.shape[0] < self.Tp:
            pad = np.repeat(actions[-1:], self.Tp - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        plug_type = int(obs["plug_type"][t])
        port_id = int(obs["port_id"][t])

        return {
            "images": imgs,  # uint8, raw resolution — preprocess_images() in train loop
            "proprio": torch.from_numpy(proprio),
            "ft": torch.from_numpy(ft),
            "actions": torch.from_numpy(actions),
            "plug_type": torch.tensor(plug_type, dtype=torch.long),
            "port_id": torch.tensor(port_id, dtype=torch.long),
            "augment": torch.tensor(self.augment),
        }

