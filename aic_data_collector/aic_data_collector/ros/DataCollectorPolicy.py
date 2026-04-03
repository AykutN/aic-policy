"""
DataCollectorPolicy
───────────────────
CheatCode'u "otomatik uzman" olarak kullanarak tüm observation-action çiftlerini
HDF5 formatında kaydeden wrapper policy.

Nasıl çalışır:
  1. CheatCode, ground_truth TF frame'lerini kullanarak başarılı insertion yapar.
  2. Bu policy, her move_robot() çağrısını intercept ederek o anki observation
     ve CheatCode'un gönderdiği komutu (action) kaydeder.
  3. Episode sonunda tüm veri bir HDF5 dosyasına yazılır.
  4. Sonuçta elimizde: (kamera görüntüsü + sensör) → action eşleştirme tablosu.

Çalıştırma:
  pixi run ros2 run aic_model aic_model \\
    --ros-args -p use_sim_time:=true \\
    -p policy:=aic_data_collector.ros.DataCollectorPolicy

Gereksinim: eval container ground_truth:=true ile başlatılmalı.
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np

from aic_control_interfaces.msg import MotionUpdate
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


class DataCollectorPolicy(Policy):
    """CheatCode'u çalıştırırken her adımı HDF5'e kaydeden wrapper."""

    # Görüntü ölçekleme — RunACT ile aynı (0.25x = 288×256 px per camera)
    IMAGE_SCALE = 0.25

    def __init__(self, parent_node):
        super().__init__(parent_node)

        # Uzman policy — ground truth TF ile çalışıyor
        self.cheat = CheatCode(parent_node)

        # Kayıt dizini
        self.save_dir = Path("/tmp/aic_dataset")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Mevcut episode sayısını bul (önceki çalıştırmalardan devam et)
        self.episode_count = len(list(self.save_dir.glob("episode_*.hdf5")))

        self.get_logger().info(
            f"DataCollectorPolicy başlatıldı. "
            f"Mevcut episode sayısı: {self.episode_count}. "
            f"Kayıt dizini: {self.save_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Dönüşüm yardımcıları
    # ──────────────────────────────────────────────────────────────────────────

    def _ros_img_to_numpy(self, ros_img) -> np.ndarray:
        """ROS Image mesajını (H, W, 3) uint8 numpy dizisine çevirir (ölçekli)."""
        img = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(
            ros_img.height, ros_img.width, 3
        )
        if self.IMAGE_SCALE != 1.0:
            img = cv2.resize(
                img,
                None,
                fx=self.IMAGE_SCALE,
                fy=self.IMAGE_SCALE,
                interpolation=cv2.INTER_AREA,
            )
        return img

    def _motion_update_to_action(self, mu: MotionUpdate) -> dict:
        """MotionUpdate mesajından action vektörlerini çıkarır."""
        # Stiffness ve damping 6×6 matrisin köşegen elemanları (6D)
        stiffness_matrix = np.array(mu.target_stiffness, dtype=np.float32).reshape(6, 6)
        damping_matrix = np.array(mu.target_damping, dtype=np.float32).reshape(6, 6)

        return {
            "action_pos": np.array(
                [mu.pose.position.x, mu.pose.position.y, mu.pose.position.z],
                dtype=np.float32,
            ),
            "action_quat": np.array(
                [
                    mu.pose.orientation.x,
                    mu.pose.orientation.y,
                    mu.pose.orientation.z,
                    mu.pose.orientation.w,
                ],
                dtype=np.float32,
            ),
            "action_stiffness": stiffness_matrix.diagonal().copy(),
            "action_damping": damping_matrix.diagonal().copy(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Ana policy metodu
    # ──────────────────────────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        self.get_logger().info(
            f"[Episode {self.episode_count}] Başlıyor — "
            f"plug={task.plug_name}, port={task.port_name}"
        )

        # Bu episode için adım tamponu
        buffer: list[dict] = []

        # ── Recording wrapper ─────────────────────────────────────────────────
        def recording_move_robot(
            motion_update: Optional[MotionUpdate] = None,
            joint_motion_update=None,
        ):
            """Orijinal move_robot'u sarmak — her çağrıda observation+action kaydeder."""

            obs = get_observation()

            if obs is not None and motion_update is not None:
                try:
                    step = {
                        # ── Görsel gözlemler ──────────────────────────────────
                        "left_image": self._ros_img_to_numpy(obs.left_image),
                        "center_image": self._ros_img_to_numpy(obs.center_image),
                        "right_image": self._ros_img_to_numpy(obs.right_image),
                        # ── Propriyosepsiyon ──────────────────────────────────
                        "joint_pos": np.array(
                            obs.joint_states.position[:7], dtype=np.float32
                        ),
                        "joint_vel": np.array(
                            obs.joint_states.velocity[:7], dtype=np.float32
                        ),
                        # ── TCP pose & velocity ───────────────────────────────
                        "tcp_pos": np.array(
                            [
                                obs.controller_state.tcp_pose.position.x,
                                obs.controller_state.tcp_pose.position.y,
                                obs.controller_state.tcp_pose.position.z,
                            ],
                            dtype=np.float32,
                        ),
                        "tcp_quat": np.array(
                            [
                                obs.controller_state.tcp_pose.orientation.x,
                                obs.controller_state.tcp_pose.orientation.y,
                                obs.controller_state.tcp_pose.orientation.z,
                                obs.controller_state.tcp_pose.orientation.w,
                            ],
                            dtype=np.float32,
                        ),
                        "tcp_vel_lin": np.array(
                            [
                                obs.controller_state.tcp_velocity.linear.x,
                                obs.controller_state.tcp_velocity.linear.y,
                                obs.controller_state.tcp_velocity.linear.z,
                            ],
                            dtype=np.float32,
                        ),
                        "tcp_vel_ang": np.array(
                            [
                                obs.controller_state.tcp_velocity.angular.x,
                                obs.controller_state.tcp_velocity.angular.y,
                                obs.controller_state.tcp_velocity.angular.z,
                            ],
                            dtype=np.float32,
                        ),
                        # ── Kuvvet-Tork sensörü ───────────────────────────────
                        "wrench_force": np.array(
                            [
                                obs.wrist_wrench.wrench.force.x,
                                obs.wrist_wrench.wrench.force.y,
                                obs.wrist_wrench.wrench.force.z,
                            ],
                            dtype=np.float32,
                        ),
                        "wrench_torque": np.array(
                            [
                                obs.wrist_wrench.wrench.torque.x,
                                obs.wrist_wrench.wrench.torque.y,
                                obs.wrist_wrench.wrench.torque.z,
                            ],
                            dtype=np.float32,
                        ),
                        # ── CheatCode'un komutu (öğreneceğimiz şey) ──────────
                        **self._motion_update_to_action(motion_update),
                        # ── Timestamp ─────────────────────────────────────────
                        "timestamp": np.float64(time.time()),
                    }
                    buffer.append(step)

                except Exception as exc:
                    self.get_logger().warn(f"Adım kaydedilemedi: {exc}")

            # Gerçek robota komutu her zaman ilet
            move_robot(
                motion_update=motion_update,
                joint_motion_update=joint_motion_update,
            )

        # ── CheatCode'u recording wrapper ile çalıştır ────────────────────────
        success = self.cheat.insert_cable(
            task, get_observation, recording_move_robot, send_feedback
        )

        # ── Episode'u kaydet ──────────────────────────────────────────────────
        if buffer:
            self._save_episode(buffer, success, task)
        else:
            self.get_logger().warn(
                f"[Episode {self.episode_count}] Hiç adım kaydedilmedi!"
            )

        self.episode_count += 1
        return success

    # ──────────────────────────────────────────────────────────────────────────
    # HDF5 kayıt
    # ──────────────────────────────────────────────────────────────────────────

    def _save_episode(self, buffer: list[dict], success: bool, task: Task) -> None:
        """Tampondaki tüm adımları tek bir HDF5 dosyasına yazar."""

        T = len(buffer)
        fpath = self.save_dir / f"episode_{self.episode_count:04d}.hdf5"

        self.get_logger().info(
            f"[Episode {self.episode_count}] Kaydediliyor: "
            f"{T} adım, başarı={success} → {fpath}"
        )

        with h5py.File(fpath, "w") as f:

            # ── Metadata ─────────────────────────────────────────────────────
            f.attrs["success"] = bool(success)
            f.attrs["plug_name"] = task.plug_name      # "sfp_module" | "sc_plug"
            f.attrs["port_name"] = task.port_name      # "sfp_port_0" | "sc_port_0" ...
            f.attrs["module_name"] = task.target_module_name
            f.attrs["num_steps"] = T
            f.attrs["image_scale"] = self.IMAGE_SCALE
            f.attrs["recorded_at"] = time.time()

            # ── Gözlemler (observations/) ─────────────────────────────────────
            obs_g = f.create_group("observations")

            # Kamera görüntüleri: (T, H, W, 3) uint8 — gzip sıkıştırmayla
            for cam_key in ("left_image", "center_image", "right_image"):
                imgs = np.stack([step[cam_key] for step in buffer])  # (T, H, W, 3)
                obs_g.create_dataset(
                    cam_key,
                    data=imgs,
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, *imgs.shape[1:]),  # Her frame ayrı chunk → hızlı random access
                )

            # Sayısal state vektörleri
            for key in (
                "joint_pos", "joint_vel",
                "tcp_pos", "tcp_quat",
                "tcp_vel_lin", "tcp_vel_ang",
                "wrench_force", "wrench_torque",
                "timestamp",
            ):
                obs_g.create_dataset(
                    key, data=np.stack([step[key] for step in buffer])
                )

            # ── Eylemler (actions/) ───────────────────────────────────────────
            act_g = f.create_group("actions")
            for key in ("action_pos", "action_quat", "action_stiffness", "action_damping"):
                act_g.create_dataset(
                    key, data=np.stack([step[key] for step in buffer])
                )

        self.get_logger().info(f"[Episode {self.episode_count}] Kaydedildi ✓")
