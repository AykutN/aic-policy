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
  4. Her çalıştırma için ayrı bir .log dosyası ve özet JSON satırı üretilir.

Observation space (RunACT ile birebir aynı — 26D state):
  tcp_pos(3) + tcp_quat(4) + tcp_vel_lin(3) + tcp_vel_ang(3)
  + tcp_error(6) + joint_pos(7) = 26

Çalıştırma:
  pixi run ros2 run aic_model aic_model \\
    --ros-args -p use_sim_time:=true \\
    -p policy:=aic_data_collector.ros.DataCollectorPolicy

Gereksinim: eval container ground_truth:=true ile başlatılmalı.
"""

import json
import logging
import time
from datetime import datetime
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

    # Görüntü ölçekleme — RunACT ile aynı (0.25x → 288×256 px per camera)
    IMAGE_SCALE = 0.25

    def __init__(self, parent_node):
        super().__init__(parent_node)

        # Uzman policy — ground truth TF ile çalışıyor
        self.cheat = CheatCode(parent_node)

        # Kayıt dizini
        self.save_dir = Path.home() / "aic_dataset"
        self.log_dir = self.save_dir / "logs"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Bu çalıştırmaya özgü dosya logger'ı kur
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = run_id
        self._setup_file_logger(run_id)

        # Mevcut episode sayısını bul (önceki çalıştırmalardan devam et)
        self.episode_count = len(list(self.save_dir.glob("episode_*.hdf5")))

        # Bu çalıştırmada toplanan episode sayacı
        self._run_episodes = 0
        self._run_success = 0

        msg = (
            f"DataCollectorPolicy başlatıldı | run_id={run_id} | "
            f"mevcut_episode={self.episode_count} | kayıt_dizini={self.save_dir}"
        )
        self.get_logger().info(msg)
        self._log.info(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Loglama kurulumu
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_file_logger(self, run_id: str) -> None:
        """Her çalıştırma için ayrı bir .log dosyası oluşturur."""
        log_path = self.log_dir / f"run_{run_id}.log"

        self._log = logging.getLogger(f"data_collector.{run_id}")
        self._log.setLevel(logging.DEBUG)

        # Dosyaya detaylı format
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._log.addHandler(fh)
        self._log.info(f"Log dosyası açıldı: {log_path}")

    def _append_summary(self, record: dict) -> None:
        """Her episode'un özetini summary.jsonl dosyasına ekler."""
        summary_path = self.save_dir / "summary.jsonl"
        with open(summary_path, "a") as f:
            f.write(json.dumps(record) + "\n")

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
        # Stiffness ve damping: 6×6 tam matris → sadece köşegen alınır (6D)
        stiffness_diag = (
            np.array(mu.target_stiffness, dtype=np.float32)
            .reshape(6, 6)
            .diagonal()
            .copy()
        )
        damping_diag = (
            np.array(mu.target_damping, dtype=np.float32)
            .reshape(6, 6)
            .diagonal()
            .copy()
        )

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
            "action_stiffness": stiffness_diag,  # (6,)
            "action_damping": damping_diag,        # (6,)
            "action_gripper": np.array([current_gripper_pos], dtype=np.float32),
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

        ep_id = self.episode_count
        t_start = time.time()

        msg = (
            f"[ep={ep_id:04d}] Başlıyor — "
            f"plug={task.plug_name} port={task.port_name} "
            f"module={task.target_module_name}"
        )
        self.get_logger().info(msg)
        self._log.info(msg)

        # Bu episode için adım tamponu
        buffer: list[dict] = []
        step_errors = 0

        # ── Recording wrapper ─────────────────────────────────────────────────
        def recording_move_robot(
            motion_update: Optional[MotionUpdate] = None,
            joint_motion_update=None,
        ) -> None:
            """Orijinal move_robot'u sararak her çağrıda observation+action kaydeder."""
            nonlocal step_errors

            obs = get_observation()

            if obs is not None and motion_update is not None:
                try:
                    cs = obs.controller_state  # kısaltma

                    step = {
                        # ── Görsel (3 kamera) ─────────────────────────────────
                        "left_image":   self._ros_img_to_numpy(obs.left_image),
                        "center_image": self._ros_img_to_numpy(obs.center_image),
                        "right_image":  self._ros_img_to_numpy(obs.right_image),

                        # ── Joint states ──────────────────────────────────────
                        # joint_states.position[:7] = 6 arm joints + 1 gripper
                        "joint_pos": np.array(
                            obs.joint_states.position[:7], dtype=np.float32
                        ),
                        "joint_vel": np.array(
                            obs.joint_states.velocity[:7], dtype=np.float32
                        ),

                        # ── TCP pose ──────────────────────────────────────────
                        "tcp_pos": np.array(
                            [
                                cs.tcp_pose.position.x,
                                cs.tcp_pose.position.y,
                                cs.tcp_pose.position.z,
                            ],
                            dtype=np.float32,
                        ),
                        "tcp_quat": np.array(
                            [
                                cs.tcp_pose.orientation.x,
                                cs.tcp_pose.orientation.y,
                                cs.tcp_pose.orientation.z,
                                cs.tcp_pose.orientation.w,
                            ],
                            dtype=np.float32,
                        ),

                        # ── TCP velocity ──────────────────────────────────────
                        "tcp_vel_lin": np.array(
                            [
                                cs.tcp_velocity.linear.x,
                                cs.tcp_velocity.linear.y,
                                cs.tcp_velocity.linear.z,
                            ],
                            dtype=np.float32,
                        ),
                        "tcp_vel_ang": np.array(
                            [
                                cs.tcp_velocity.angular.x,
                                cs.tcp_velocity.angular.y,
                                cs.tcp_velocity.angular.z,
                            ],
                            dtype=np.float32,
                        ),

                        # ── TCP tracking error (6D) — RunACT'te de var ────────
                        # = istenen pose ile şu anki pose arasındaki fark
                        "tcp_error": np.array(
                            list(cs.tcp_error), dtype=np.float32
                        ),  # shape (6,)

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

                        # ── Action (CheatCode'un gönderdiği komut) ────────────
                        # Gripper bilgisini joint_pos'un 7. elemanından [6] alıyoruz.
                        # Eğer dataset'ten model eğiteceksek gripper durumunu da vermemiz şart.
                        **self._motion_update_to_action(motion_update, current_gripper_pos=obs.joint_states.position[6]),

                        # ── Timestamp ─────────────────────────────────────────
                        "timestamp": np.float64(time.time()),
                    }
                    buffer.append(step)

                except Exception as exc:
                    step_errors += 1
                    self._log.warning(f"[ep={ep_id:04d}] Adım kaydedilemedi: {exc}")
                    self.get_logger().warn(f"[ep={ep_id}] Adım kaydedilemedi: {exc}")

            # Gerçek robota komutu her zaman ilet — recording hatası robotu durdurmamalı
            move_robot(
                motion_update=motion_update,
                joint_motion_update=joint_motion_update,
            )

        # ── CheatCode'u recording wrapper ile çalıştır ────────────────────────
        success = self.cheat.insert_cable(
            task, get_observation, recording_move_robot, send_feedback
        )

        elapsed = time.time() - t_start

        # ── Episode'u kaydet ──────────────────────────────────────────────────
        if buffer:
            self._save_episode(buffer, success, task, elapsed)
        else:
            warn = (
                f"[ep={ep_id:04d}] UYARI: Hiç adım kaydedilmedi! "
                f"CheatCode ground_truth olmadan mı çalıştı?"
            )
            self.get_logger().error(warn)
            self._log.error(warn)

        self._run_episodes += 1
        self._run_success += int(success)
        self.episode_count += 1

        summary_line = (
            f"[ep={ep_id:04d}] Tamamlandı | "
            f"success={success} | steps={len(buffer)} | "
            f"step_errors={step_errors} | elapsed={elapsed:.1f}s | "
            f"run_total={self._run_episodes} run_success={self._run_success}"
        )
        self.get_logger().info(summary_line)
        self._log.info(summary_line)

        return success

    # ──────────────────────────────────────────────────────────────────────────
    # HDF5 kayıt
    # ──────────────────────────────────────────────────────────────────────────

    def _save_episode(
        self,
        buffer: list[dict],
        success: bool,
        task: Task,
        elapsed: float,
    ) -> None:
        """Tampondaki tüm adımları tek bir HDF5 dosyasına yazar."""

        T = len(buffer)
        ep_id = self.episode_count
        fpath = self.save_dir / f"episode_{ep_id:04d}.hdf5"

        self._log.info(f"[ep={ep_id:04d}] HDF5 yazılıyor: {T} adım → {fpath}")

        try:
            with h5py.File(fpath, "w") as f:

                # ── Metadata ─────────────────────────────────────────────────
                f.attrs["episode_id"]   = ep_id
                f.attrs["success"]      = bool(success)
                f.attrs["plug_name"]    = task.plug_name       # "sfp_module" | "sc_plug"
                f.attrs["port_name"]    = task.port_name       # "sfp_port_0" | "sc_port_0" ...
                f.attrs["module_name"]  = task.target_module_name
                f.attrs["num_steps"]    = T
                f.attrs["image_scale"]  = self.IMAGE_SCALE
                f.attrs["elapsed_sec"]  = elapsed
                f.attrs["run_id"]       = self._run_id
                f.attrs["recorded_at"]  = datetime.now().isoformat()

                # ── Gözlemler ─────────────────────────────────────────────────
                obs_g = f.create_group("observations")

                # Kamera görüntüleri: (T, H, W, 3) uint8 — gzip ile sıkıştır
                for cam_key in ("left_image", "center_image", "right_image"):
                    imgs = np.stack([step[cam_key] for step in buffer])  # (T, H, W, 3)
                    obs_g.create_dataset(
                        cam_key,
                        data=imgs,
                        dtype=np.uint8,
                        compression="gzip",
                        compression_opts=4,
                        # Her frame ayrı chunk → eğitimde random access için hızlı
                        chunks=(1, *imgs.shape[1:]),
                    )

                # Sayısal state vektörleri
                for key in (
                    "joint_pos",     # (T, 7)
                    "joint_vel",     # (T, 7)
                    "tcp_pos",       # (T, 3)
                    "tcp_quat",      # (T, 4)
                    "tcp_vel_lin",   # (T, 3)
                    "tcp_vel_ang",   # (T, 3)
                    "tcp_error",     # (T, 6) — RunACT'te de kullanılan
                    "wrench_force",  # (T, 3)
                    "wrench_torque", # (T, 3)
                    "timestamp",     # (T,)
                ):
                    obs_g.create_dataset(
                        key, data=np.stack([step[key] for step in buffer])
                    )

                # ── Eylemler ──────────────────────────────────────────────────
                act_g = f.create_group("actions")
                for key in (
                    "action_pos",        # (T, 3)
                    "action_quat",       # (T, 4)
                    "action_stiffness",  # (T, 6)
                    "action_damping",    # (T, 6)
                    "action_gripper",    # (T, 1)
                ):
                    act_g.create_dataset(
                        key, data=np.stack([step[key] for step in buffer])
                    )

            self.get_logger().info(f"[ep={ep_id:04d}] HDF5 kaydedildi ✓ ({fpath.stat().st_size / 1e6:.1f} MB)")
            self._log.info(f"[ep={ep_id:04d}] HDF5 kaydedildi ✓ ({fpath.stat().st_size / 1e6:.1f} MB)")

        except Exception as exc:
            err = f"[ep={ep_id:04d}] HDF5 KAYIT HATASI: {exc}"
            self.get_logger().error(err)
            self._log.error(err)
            raise  # yukarıya taşı — sessizce geçme

        # ── summary.jsonl'e özet satırı ekle ─────────────────────────────────
        self._append_summary({
            "episode_id":   ep_id,
            "run_id":       self._run_id,
            "success":      bool(success),
            "plug_name":    task.plug_name,
            "port_name":    task.port_name,
            "module_name":  task.target_module_name,
            "num_steps":    T,
            "elapsed_sec":  round(elapsed, 2),
            "file_mb":      round(fpath.stat().st_size / 1e6, 2),
            "recorded_at":  datetime.now().isoformat(),
        })
