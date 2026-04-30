#!/usr/bin/env python3
"""Evaluation policy: loads trained CFM checkpoints, runs as aic_model.Policy in Gazebo.

Usage (same as any other policy):
    pixi run ros2 run aic_model aic_model \
      --ros-args -p use_sim_time:=true \
      -p policy:=training.evaluate.TrainedPolicy \
      -p approach_ckpt:=/workspace/checkpoints/cfm_subtask0/best.pt \
      -p insert_ckpt:=/workspace/checkpoints/cfm_subtask1/best.pt
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

from training.models.cfm import CFMPolicy


IMAGE_SCALE = 0.25
OBS_WINDOW_IMAGE = 4
OBS_WINDOW_PROPRIO = 16
APPROACH_STEPS = 100  # matches DataCollectorPolicy
CONTROL_HZ = 20
MAX_STEPS = 600  # 30 seconds at 20Hz — stay well under the 60s scoring threshold


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_cfm_model(ckpt_path: str, cfg: dict, device):
    mc = cfg["model"]
    model = CFMPolicy(
        image_feature_dim=mc["image_feature_dim"],
        ft_feature_dim=mc["ft_feature_dim"],
        proprio_feature_dim=mc["proprio_feature_dim"],
        fusion_dim=mc["fusion_dim"],
        flow_hidden_dim=mc["flow_hidden_dim"],
        flow_layers=mc["flow_layers"],
        action_dim=mc["action_dim"],
        action_chunk=mc["action_chunk"],
        n_tasks=mc.get("n_tasks", 2),
        task_emb_dim=mc.get("task_emb_dim", 32),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_key = "ema_state" if "ema_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    act_mean = ckpt["act_mean"].to(device)
    act_std = ckpt["act_std"].to(device)
    return model, act_mean, act_std


def _obs_msg_to_dict(obs) -> dict:
    """Convert aic_model_interfaces.msg.Observation to plain dict for buffer."""
    H = int(round(1024 * IMAGE_SCALE))
    W = int(round(1152 * IMAGE_SCALE))

    def decode_img(ros_img):
        img = np.frombuffer(bytes(ros_img.data), dtype=np.uint8).reshape(
            ros_img.height, ros_img.width, 3
        )
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    cs = obs.controller_state
    return {
        "left_image": decode_img(obs.left_image),
        "center_image": decode_img(obs.center_image),
        "right_image": decode_img(obs.right_image),
        "tcp_pos": np.array([cs.tcp_pose.position.x, cs.tcp_pose.position.y, cs.tcp_pose.position.z], dtype=np.float32),
        "tcp_quat": np.array([cs.tcp_pose.orientation.x, cs.tcp_pose.orientation.y,
                               cs.tcp_pose.orientation.z, cs.tcp_pose.orientation.w], dtype=np.float32),
        "tcp_vel_lin": np.array([cs.tcp_velocity.linear.x, cs.tcp_velocity.linear.y, cs.tcp_velocity.linear.z], dtype=np.float32),
        "tcp_vel_ang": np.array([cs.tcp_velocity.angular.x, cs.tcp_velocity.angular.y, cs.tcp_velocity.angular.z], dtype=np.float32),
        "tcp_error": np.array(list(cs.tcp_error), dtype=np.float32),
        "joint_pos": np.array(obs.joint_states.position[:7], dtype=np.float32),
        "joint_vel": np.array(obs.joint_states.velocity[:7], dtype=np.float32),
        "gripper_pos": np.float32(obs.joint_states.position[6]),
        "wrench_force": np.array([obs.wrist_wrench.wrench.force.x, obs.wrist_wrench.wrench.force.y, obs.wrist_wrench.wrench.force.z], dtype=np.float32),
        "wrench_torque": np.array([obs.wrist_wrench.wrench.torque.x, obs.wrist_wrench.wrench.torque.y, obs.wrist_wrench.wrench.torque.z], dtype=np.float32),
    }


def _buffer_to_tensors(obs_buffer: list, to_img: int, to_prop: int, device):
    """Convert rolling observation buffer to (images, proprio, ft) tensors."""
    To_img = to_img
    To_prop = to_prop

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
    """Hierarchical CFM policy: approach model (subtask=0) -> insert model (subtask=1)."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = _load_yaml(str(Path(__file__).parent / "configs" / "cfm.yaml"))
        self._obs_window_image = cfg["data"]["obs_window_image"]
        self._obs_window_proprio = cfg["data"]["obs_window_proprio"]
        self._n_flow_steps = cfg["inference"]["n_flow_steps"]
        mc = cfg["model"]
        self._execute_steps = cfg["inference"].get("execute_steps", mc["action_chunk"])
        self._n_tasks = mc.get("n_tasks", 2)

        approach_ckpt = parent_node.declare_parameter("approach_ckpt", "").value
        insert_ckpt = parent_node.declare_parameter("insert_ckpt", "").value

        if not approach_ckpt or not insert_ckpt:
            raise ValueError("approach_ckpt and insert_ckpt ROS params must be set")

        self.approach_model, self.approach_act_mean, self.approach_act_std = _load_cfm_model(approach_ckpt, cfg, self.device)
        self.insert_model, self.insert_act_mean, self.insert_act_std = _load_cfm_model(insert_ckpt, cfg, self.device)
        self.get_logger().info(f"Loaded approach: {approach_ckpt}")
        self.get_logger().info(f"Loaded insert: {insert_ckpt}")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        # task_id: 0=SFP, 1=SC
        task_id = torch.tensor(
            [0 if "sfp" in task.plug_name.lower() else 1],
            dtype=torch.long, device=self.device
        )

        max_steps = int(task.time_limit * CONTROL_HZ) if task.time_limit > 0 else MAX_STEPS
        max_steps = min(max_steps, MAX_STEPS)
        self.get_logger().info(
            f"insert_cable: plug={task.plug_name} port={task.port_name} "
            f"module={task.target_module_name} time_limit={task.time_limit}s max_steps={max_steps}"
        )

        lookback = max(self._obs_window_image, self._obs_window_proprio)
        obs_buffer = []

        # Warm up buffer with initial observations
        for _ in range(lookback):
            obs_buffer.append(_obs_msg_to_dict(get_observation()))

        step = 0
        action_queue = []  # remaining actions from last chunk

        while step < max_steps:
            if not action_queue:
                if step < APPROACH_STEPS:
                    model, act_mean, act_std = self.approach_model, self.approach_act_mean, self.approach_act_std
                else:
                    model, act_mean, act_std = self.insert_model, self.insert_act_mean, self.insert_act_std
                imgs, proprio, ft = _buffer_to_tensors(obs_buffer, self._obs_window_image, self._obs_window_proprio, self.device)
                with torch.no_grad():
                    pred = model.sample(imgs, proprio, ft, task_id, n_steps=self._n_flow_steps)
                    pred = pred * act_std + act_mean  # de-normalize
                action_queue = pred[0].cpu().numpy().tolist()[:self._execute_steps]

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

        send_feedback(f"Max steps ({max_steps}) reached without insertion")
        return False
