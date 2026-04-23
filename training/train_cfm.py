#!/usr/bin/env python3
"""CFM training loop (ablation).

Usage:
    python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 0
    python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 1 --resume
"""
from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

import kornia.augmentation as Ka
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import AICDataset
from training.models.cfm import CFMPolicy

_GPU_AUG = Ka.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0)


def preprocess_images(imgs: torch.Tensor, image_size: tuple, augment: bool) -> torch.Tensor:
    """uint8 (B, To, n_cams, C, H, W) → float32 resized, per-image augmented on GPU."""
    B, To, n_cams, C, H, W = imgs.shape
    flat = imgs.reshape(B * To * n_cams, C, H, W).float() / 255.0
    flat = F.interpolate(flat, size=image_size, mode="bilinear", align_corners=False)
    if augment:
        flat = _GPU_AUG(flat).clamp(0.0, 1.0)
    return flat.reshape(B, To, n_cams, C, *image_size)


def compute_action_stats(dataset, device):
    """Compute per-dim mean and std over all training actions."""
    print("Computing action normalization stats...")
    loader = DataLoader(dataset, batch_size=512, num_workers=4, shuffle=False)
    all_actions = []
    for batch in tqdm(loader, desc="Scanning actions", leave=False):
        all_actions.append(batch["actions"])
    all_actions = torch.cat(all_actions, dim=0)          # (N, Tp, 20)
    mean = all_actions.mean(dim=(0, 1)).to(device)       # (20,)
    std  = all_actions.std(dim=(0, 1)).clamp(min=1.0).to(device)  # clamp so constant dims don't explode
    print(f"  Action mean: {mean.cpu().numpy().round(2)}")
    print(f"  Action std:  {std.cpu().numpy().round(2)}")
    return mean, std


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rclone_sync_async(local: str, remote: str):
    """Non-blocking rclone copy — GPU keeps running while upload happens in background."""
    subprocess.Popen(
        ["rclone", "copy", local, remote, "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def save_checkpoint(model, optimizer, scheduler, scaler, ema_state,
                    act_mean, act_std,
                    epoch, val_loss, patience_counter, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "ema_state": ema_state,
        "act_mean": act_mean.cpu(),
        "act_std": act_std.cpu(),
        "val_loss": val_loss,
        "patience_counter": patience_counter,
    }, path)


def prune_old_checkpoints(checkpoint_dir: Path, keep_n: int):
    """Keep only the last keep_n epoch_XXXX.pt files to avoid filling disk."""
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    for old in ckpts[:-keep_n]:
        old.unlink(missing_ok=True)


def update_ema(ema_state: dict, model: nn.Module, step: int, decay: float = 0.999):
    """EMA with warmup: starts fast, converges to target decay."""
    adaptive_decay = min(decay, (1.0 + step) / (10.0 + step))
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if ema_state[k].is_floating_point():
                ema_state[k].mul_(adaptive_decay).add_(v, alpha=1.0 - adaptive_decay)
            else:
                ema_state[k].copy_(v)


def train(args):
    cfg = load_config(args.config)
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["training"]

    seed = tc.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    checkpoint_dir = Path(tc["checkpoint_dir"]).expanduser() / f"cfm_subtask{args.subtask}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed={seed}")

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
    scaler = GradScaler(device.type)
    ema_state = copy.deepcopy(model.state_dict())

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    act_mean = act_std = None

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        ema_state = ckpt["ema_state"]
        act_mean = ckpt["act_mean"].to(device)
        act_std  = ckpt["act_std"].to(device)
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resumed epoch {start_epoch} | best_val={best_val_loss:.4f} | patience={patience_counter}")

    img_size = tuple(dc["image_size"]) if dc.get("image_size") else (128, 144)

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
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset for subtask={args.subtask}.")

    if act_mean is None:
        act_mean, act_std = compute_action_stats(train_ds, device)

    nw = tc.get("num_workers", 4)
    pf = tc.get("prefetch_factor", 2)
    pw = nw > 0
    mp_ctx = "spawn" if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pw, persistent_workers=pw,
                              multiprocessing_context=mp_ctx,
                              prefetch_factor=pf if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                            num_workers=nw, pin_memory=pw, persistent_workers=pw,
                            multiprocessing_context=mp_ctx,
                            prefetch_factor=pf if nw > 0 else None)

    rclone_remote = tc.get("rclone_remote", "")
    rclone_every  = tc.get("rclone_sync_every_n", 5)
    keep_n        = tc.get("keep_last_n_checkpoints", 3)

    print(f"Training | epochs={tc['epochs']} | batch={tc['batch_size']} | workers={nw} | prefetch={pf}")
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        train_loss = 0.0
        grad_norm_avg = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)
        for batch in pbar:
            images  = preprocess_images(batch["images"].to(device, non_blocking=True), img_size, augment=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            ft      = batch["ft"].to(device, non_blocking=True)
            actions = (batch["actions"].to(device, non_blocking=True) - act_mean) / act_std

            optimizer.zero_grad()
            with autocast(device.type):
                loss = model.loss(images, proprio, ft, actions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_state, model, global_step)
            global_step += 1
            train_loss += loss.item()
            grad_norm_avg += grad_norm
            pbar.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{grad_norm:.2f}")
        train_loss /= len(train_loader)
        grad_norm_avg /= len(train_loader)
        scheduler.step()

        # Validate with EMA weights
        model.eval()
        original_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema_state)
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False, dynamic_ncols=True):
                with autocast(device.type):
                    actions_norm = (batch["actions"].to(device, non_blocking=True) - act_mean) / act_std
                    val_loss += model.loss(
                        preprocess_images(batch["images"].to(device, non_blocking=True), img_size, augment=False),
                        batch["proprio"].to(device, non_blocking=True),
                        batch["ft"].to(device, non_blocking=True),
                        actions_norm,
                    ).item()
        val_loss /= len(val_loader)
        model.load_state_dict(original_state)

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val(ema)={val_loss:.4f} | gnorm={grad_norm_avg:.3f} | lr={lr_now:.2e}")

        save_checkpoint(model, optimizer, scheduler, scaler, ema_state,
                        act_mean, act_std,
                        epoch, val_loss, patience_counter,
                        checkpoint_dir / f"epoch_{epoch:04d}.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, ema_state,
                        act_mean, act_std,
                        epoch, val_loss, patience_counter, latest_ckpt)
        prune_old_checkpoints(checkpoint_dir, keep_n)

        if rclone_remote and (epoch + 1) % rclone_every == 0:
            print(f"  rclone sync (async) → {rclone_remote}")
            rclone_sync_async(str(checkpoint_dir), f"{rclone_remote}/cfm_subtask{args.subtask}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            path = checkpoint_dir / "best.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": ema_state,
                "act_mean": act_mean.cpu(),
                "act_std": act_std.cpu(),
                "val_loss": val_loss,
            }, path)
            print(f"  ✓ best.pt updated (val={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= tc["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    if rclone_remote:
        rclone_sync_async(str(checkpoint_dir), f"{rclone_remote}/cfm_subtask{args.subtask}")
    print(f"Done. Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {checkpoint_dir}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subtask", type=int, required=True, choices=[0, 1])
    parser.add_argument("--config", default="training/configs/cfm.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
