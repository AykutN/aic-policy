#!/usr/bin/env python3
"""
Dataset istatistiklerini inceleyen yardımcı script.
Kullanım: python3 scripts/inspect_dataset.py /tmp/aic_dataset
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def human_size(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def inspect(dataset_dir: str):
    d = Path(dataset_dir)
    episodes = sorted(d.glob("episode_*.hdf5"))

    if not episodes:
        print("Hiç episode bulunamadı!")
        return

    print(f"\n{'='*60}")
    print(f"Dataset: {d}")
    print(f"Toplam episode: {len(episodes)}")
    print(f"{'='*60}\n")

    total_bytes = 0
    step_counts = []
    success_count = 0
    sfp_count = 0
    sc_count = 0

    for ep_path in episodes:
        size = ep_path.stat().st_size
        total_bytes += size

        with h5py.File(ep_path, "r") as f:
            T = f.attrs["num_steps"]
            success = f.attrs["success"]
            plug = f.attrs.get("plug_name", "?")
            port = f.attrs.get("port_name", "?")

            step_counts.append(T)
            if success:
                success_count += 1
            if "sfp" in str(plug).lower():
                sfp_count += 1
            else:
                sc_count += 1

    step_counts = np.array(step_counts)

    print(f"Başarı oranı:     {success_count}/{len(episodes)} = {success_count/len(episodes)*100:.1f}%")
    print(f"SFP insertion:    {sfp_count}")
    print(f"SC insertion:     {sc_count}")
    print(f"\nAdım istatistikleri:")
    print(f"  Ortalama:       {step_counts.mean():.1f} adım")
    print(f"  Min/Max:        {step_counts.min()} / {step_counts.max()}")
    print(f"  Toplam:         {step_counts.sum():,} adım")
    print(f"\nDisk kullanımı:  {human_size(total_bytes)}")
    print(f"Ep başına orta.: {human_size(total_bytes // len(episodes))}")
    print(f"Toplam tahmini (200 ep): {human_size(total_bytes // len(episodes) * 200)}\n")

    # İlk episode detayı
    print(f"{'─'*60}")
    print("İlk episode detayı:")
    with h5py.File(episodes[0], "r") as f:
        print(f"  plug_name:     {f.attrs.get('plug_name')}")
        print(f"  port_name:     {f.attrs.get('port_name')}")
        print(f"  success:       {f.attrs.get('success')}")
        print(f"  num_steps:     {f.attrs.get('num_steps')}")
        print(f"  image_scale:   {f.attrs.get('image_scale')}")
        obs = f["observations"]
        print(f"  left_image:    {obs['left_image'].shape} {obs['left_image'].dtype}")
        print(f"  joint_pos:     {obs['joint_pos'].shape}")
        print(f"  wrench_force:  {obs['wrench_force'].shape}")
        act = f["actions"]
        print(f"  action_pos:    {act['action_pos'].shape}")
        print(f"  action_quat:   {act['action_quat'].shape}")
        print(f"  stiffness:     {act['action_stiffness'].shape}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/aic_dataset"
    inspect(path)
