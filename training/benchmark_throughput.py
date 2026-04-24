#!/usr/bin/env python3
"""GPU/CPU throughput benchmark — finds optimal batch_size, num_workers."""
from __future__ import annotations
import gc, time
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from training.dataset import AICDataset
from training.models.cfm import CFMPolicy

DEVICE   = torch.device("cuda")
IMG_SIZE = (128, 144)
VRAM_CAP = torch.cuda.get_device_properties(0).total_memory // 1024**2
N_WARMUP = 10
N_BENCH  = 60


def worker_init_fn(worker_id):
    """Use file-system IPC to avoid /dev/shm limits."""
    import torch as _t
    _t.multiprocessing.set_sharing_strategy("file_system")


def preprocess(imgs):
    B, To, n, C, H, W = imgs.shape
    flat = imgs.reshape(B * To * n, C, H, W).float() / 255.0
    return F.interpolate(flat, size=IMG_SIZE, mode="bilinear", align_corners=False).reshape(B, To, n, C, *IMG_SIZE)


def run(batch_size, num_workers, compile_model=False):
    ds = AICDataset(
        "/workspace/aic_dataset", subtask=1,
        obs_window_image=4, obs_window_proprio=16, action_chunk=32,
        augment=False, train=True, train_val_split=0.9,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    model = CFMPolicy().to(DEVICE)
    if compile_model:
        print(f"  (compiling model...)", flush=True)
        model = torch.compile(model)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda")
    act_mean = torch.zeros(20, device=DEVICE)
    act_std  = torch.ones(20,  device=DEVICE)

    it = iter(loader)

    def step(b):
        imgs = preprocess(b["images"].to(DEVICE))
        prop = b["proprio"].to(DEVICE)
        ft   = b["ft"].to(DEVICE)
        acts = (b["actions"].to(DEVICE) - act_mean) / act_std
        opt.zero_grad()
        with autocast("cuda"):
            loss = model.loss(imgs, prop, ft, acts)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(N_WARMUP):
        step(next(it))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        step(next(it))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps  = N_BENCH * batch_size / elapsed
    vram = torch.cuda.max_memory_allocated() // 1024**2

    del model, opt, scaler, loader, ds
    torch.cuda.empty_cache()
    gc.collect()
    return sps, vram


CONFIGS = [
    (64,  12, False),
    (128, 12, False),
    (256, 12, False),
    (512, 12, False),
    (256, 16, False),
    (128, 12, True),   # compile
    (256, 12, True),   # compile
]

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {VRAM_CAP} MiB")
    print(f"{'batch':>6} {'workers':>8} {'compile':>8} {'samples/s':>11} {'VRAM MiB':>10} {'VRAM%':>7}")
    best = (0, None)
    for bs, nw, comp in CONFIGS:
        label = f"bs={bs} nw={nw}{' +compile' if comp else ''}"
        print(f"  Testing {label}...", flush=True)
        try:
            sps, vram = run(bs, nw, comp)
            pct = vram / VRAM_CAP * 100
            star = " ◀" if sps > best[0] else ""
            print(f"{bs:>6} {nw:>8} {str(comp):>8} {sps:>11.1f} {vram:>10} {pct:>6.1f}%{star}")
            if sps > best[0]:
                best = (sps, {"batch_size": bs, "num_workers": nw, "compile": comp})
        except RuntimeError as e:
            print(f"{bs:>6} {nw:>8} {str(comp):>8}  FAILED: {str(e)[:60]}")
            torch.cuda.empty_cache(); gc.collect()
    print(f"\n✓ Best: {best[1]}  ({best[0]:.0f} samples/s)")
