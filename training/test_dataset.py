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
        assert sample["actions"].shape == (32, 20), f"Got {sample['actions'].shape}"
        assert sample["images"].shape == (4, 3, 3, 256, 288), f"Got {sample['images'].shape}"   # (To, n_cams, C, H, W)
        assert sample["proprio"].shape == (16, 34), f"Got {sample['proprio'].shape}"
        assert sample["ft"].shape == (16, 6), f"Got {sample['ft'].shape}"


if __name__ == "__main__":
    test_dataset_returns_correct_keys()
    test_dataset_subtask_filter()
    test_dataset_action_shape()
    print("All tests passed.")
