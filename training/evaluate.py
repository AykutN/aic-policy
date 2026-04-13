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
import yaml
from pathlib import Path

from aic_model.policy import Policy, GetObservationCallback, MoveRobotCallback, SendFeedbackCallback
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Pose, Vector3, Wrench
from std_msgs.msg import Header

from training.models.act import ACTPolicy


IMAGE_SCALE = 0.25
OBS_WINDOW_IMAGE = 4
OBS_WINDOW_PROPRIO = 16
APPROACH_STEPS = 100  # matches DataCollectorPolicy
MAX_STEPS = 2000


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_act_model(ckpt_path: str, cfg: dict, device) -> ACTPolicy:
    mc = cfg["model"]
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
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _obs_msg_to_dict(obs) -> dict:
    """Convert aic_model_interfaces.msg.Observation to plain dict for buffer."""
    H = int(round(1024 * IMAGE_SCALE))
    W = int(round(1152 * IMAGE_SCALE))

    def decode_img(data):
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((H, W, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (W, H))

    return {
        "left_image": decode_img(obs.left_image.data),
        "center_image": decode_img(obs.center_image.data),
        "right_image": decode_img(obs.right_image.data),
        "tcp_pos": np.array([obs.tcp_pose.position.x, obs.tcp_pose.position.y, obs.tcp_pose.position.z], dtype=np.float32),
        "tcp_quat": np.array([obs.tcp_pose.orientation.x, obs.tcp_pose.orientation.y,
                               obs.tcp_pose.orientation.z, obs.tcp_pose.orientation.w], dtype=np.float32),
        "tcp_vel_lin": np.array([obs.tcp_velocity.linear.x, obs.tcp_velocity.linear.y, obs.tcp_velocity.linear.z], dtype=np.float32),
        "tcp_vel_ang": np.array([obs.tcp_velocity.angular.x, obs.tcp_velocity.angular.y, obs.tcp_velocity.angular.z], dtype=np.float32),
        "tcp_error": np.array(obs.tcp_error, dtype=np.float32),
        "joint_pos": np.array(obs.joint_states.position[:7], dtype=np.float32),
        "joint_vel": np.array(obs.joint_states.velocity[:7], dtype=np.float32),
        "gripper_pos": np.float32(obs.joint_states.position[6]),
        "wrench_force": np.array([obs.wrench.force.x, obs.wrench.force.y, obs.wrench.force.z], dtype=np.float32),
        "wrench_torque": np.array([obs.wrench.torque.x, obs.wrench.torque.y, obs.wrench.torque.z], dtype=np.float32),
    }


def _buffer_to_tensors(obs_buffer: list, device):
    """Convert rolling observation buffer to (images, proprio, ft) tensors."""
    To_img = OBS_WINDOW_IMAGE
    To_prop = OBS_WINDOW_PROPRIO

    img_buf = obs_buffer[-To_img:]
    imgs = np.stack([[o["left_image"], o["center_image"], o["right_image"]] for o in img_buf])
    imgs = torch.from_numpy(imgs).float() / 255.0
    imgs = imgs.permute(0, 1, 4, 2, 3).unsqueeze(0).to(device)  # (1, To_img, 3, C, H, W)

    prop_buf = obs_buffer[-To_prop:]
    proprio = np.stack([np.concatenate([
        o["tcp_pos"], o["tcp_quat"], o["tcp_vel_lin"], o["tcp_vel_ang"],
        o["tcp_error"], o["joint_pos"], o["joint_vel"], [o["gripper_pos"]],
    ]) for o in prop_buf]).astype(np.float32)
    proprio = torch.from_numpy(proprio).unsqueeze(0).to(device)  # (1, To_prop, 34)

    ft = np.stack([np.concatenate([o["wrench_force"], o["wrench_torque"]]) for o in prop_buf]).astype(np.float32)
    ft = torch.from_numpy(ft).unsqueeze(0).to(device)  # (1, To_prop, 6)

    return imgs, proprio, ft


def _action_to_motion_update(action: np.ndarray, frame_id: str, stamp) -> MotionUpdate:
    """Convert 20D action vector to MotionUpdate ROS message."""
    pos = action[:3]
    quat = action[3:7]
    stiffness = action[7:13]
    damping = action[13:19]
    # action[19] = gripper — held constant (pre-grasped), not commanded

    pose = Pose()
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])

    return MotionUpdate(
        header=Header(frame_id=frame_id, stamp=stamp),
        pose=pose,
        target_stiffness=np.diag(stiffness.astype(np.float64)).flatten(),
        target_damping=np.diag(damping.astype(np.float64)).flatten(),
        feedforward_wrench_at_tip=Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        ),
        wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        trajectory_generation_mode=TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        ),
    )


class TrainedPolicy(Policy):
    """Hierarchical trained policy: approach model (subtask=0) -> insert model (subtask=1)."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = _load_yaml("training/configs/act.yaml")

        approach_ckpt = parent_node.declare_parameter("approach_ckpt", "").value
        insert_ckpt = parent_node.declare_parameter("insert_ckpt", "").value

        if not approach_ckpt or not insert_ckpt:
            raise ValueError("approach_ckpt and insert_ckpt ROS params must be set")

        self.approach_model = _load_act_model(approach_ckpt, cfg, self.device)
        self.insert_model = _load_act_model(insert_ckpt, cfg, self.device)
        self.get_logger().info(f"Loaded approach: {approach_ckpt}")
        self.get_logger().info(f"Loaded insert: {insert_ckpt}")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        lookback = max(OBS_WINDOW_IMAGE, OBS_WINDOW_PROPRIO)
        obs_buffer = []

        # Warm up buffer with initial observations
        for _ in range(lookback):
            obs_buffer.append(_obs_msg_to_dict(get_observation()))

        step = 0
        action_queue = []  # remaining actions from last chunk

        while step < MAX_STEPS:
            if not action_queue:
                model = self.approach_model if step < APPROACH_STEPS else self.insert_model
                imgs, proprio, ft = _buffer_to_tensors(obs_buffer, self.device)
                with torch.no_grad():
                    pred, _, _ = model(imgs, proprio, ft, actions_gt=None)
                action_queue = pred[0].cpu().numpy().tolist()

            action = np.array(action_queue.pop(0))
            stamp = self._parent_node.get_clock().now().to_msg()
            move_robot(motion_update=_action_to_motion_update(action, "base_link", stamp))

            obs_buffer.append(_obs_msg_to_dict(get_observation()))
            if len(obs_buffer) > lookback:
                obs_buffer.pop(0)

            step += 1

            # Termination: low insertion force after enough insert steps
            if step > APPROACH_STEPS + 50:
                wrench = obs_buffer[-1]["wrench_force"]
                if np.linalg.norm(wrench) < 0.5:
                    send_feedback("Insertion complete")
                    return True

        send_feedback("Max steps reached without insertion")
        return False
