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
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.dataset import AICDataset
from training.models.act import ACTPolicy, act_loss


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def s3_sync(local: str, s3: str):
    subprocess.run(["aws", "s3", "sync", local, s3, "--quiet"], check=False)


def save_checkpoint(model, optimizer, scaler, epoch, val_loss, patience_counter, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "val_loss": val_loss,
        "patience_counter": patience_counter,
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
    scaler = GradScaler("cuda")

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, patience={patience_counter}")

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

    nw = tc.get("num_workers", 4)
    pw = nw > 0
    pm = nw > 0
    mp_ctx = "spawn" if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pm, persistent_workers=pw,
                              multiprocessing_context=mp_ctx)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                            num_workers=nw, pin_memory=pm, persistent_workers=pw,
                            multiprocessing_context=mp_ctx)

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
            with autocast("cuda"):
                pred, mu, log_var = model(images, proprio, ft, actions)
                loss = act_loss(pred, actions, mu, log_var, kl_weight=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
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
                with autocast("cuda"):
                    pred, mu, log_var = model(images, proprio, ft, actions)
                    val_loss += act_loss(pred, actions, mu, log_var, kl_weight=tc.get("kl_weight", 0.1)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # --- Checkpoint ---
        ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        save_checkpoint(model, optimizer, scaler, epoch, val_loss, patience_counter, ckpt_path)
        save_checkpoint(model, optimizer, scaler, epoch, val_loss, patience_counter, latest_ckpt)

        # S3 sync every N epochs
        if (epoch + 1) % tc["s3_sync_every_n"] == 0:
            print(f"  S3 sync...")
            s3_sync(str(checkpoint_dir), tc["s3_bucket"] + f"/act_subtask{args.subtask}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scaler, epoch, val_loss, patience_counter, checkpoint_dir / "best.pt")
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
