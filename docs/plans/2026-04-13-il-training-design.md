# IL Training Pipeline Design

**Date:** 2026-04-13  
**Status:** Approved  
**Deadline:** 2026-05-15 (Qualification submission)

---

## Context

- 420+ episodes collected via DataCollectorPolicy (CheatCode oracle)
- HDF5 format, gzip compressed, stored at `~/aic_dataset/` (EBS + S3 backup)
- Each episode has `subtask` marker: `0=APPROACH` (first 100 steps), `1=INSERT` (remaining steps)
- Training on AWS g4dn.xlarge (T4 16GB, on-demand for training to avoid interruption)
- ~$20 AWS credit remaining

---

## Architecture Decision: Approach A (Minimal Pipeline)

**Chosen over:**
- B (Full ROS package from start) — too slow, deadline risk
- C (LeRobot adapter) — adapter work equals building from scratch

**Rationale:** Get a working model first, wrap in ROS later. If model fails, iteration is fast. ROS wrapper (`my_policy/`) added after IL baseline is validated.

---

## Model Architecture

### Hierarchical: 2 subtask models trained separately

```
train_act.py --subtask 0  →  ACT approach policy
train_act.py --subtask 1  →  ACT insert policy
train_cfm.py --subtask 0  →  CFM approach policy (ablation)
train_cfm.py --subtask 1  →  CFM insert policy (ablation)
```

### ACT (primary)

| Component | Detail |
|---|---|
| Visual encoder | ResNet-18 pretrained, last layers unfrozen, 3 cameras separate encoders → concat |
| F/T encoder | 1D-CNN, `To=16` (800ms temporal window) → 128-dim |
| Proprioceptive encoder | Linear projection → 64-dim |
| Fusion | All features concat → Transformer encoder (4 layers, 256-dim, 4 heads) |
| Action head | CVAE decoder → `Tp=32` action chunk |
| Action space | pos(3) + quat(4) + stiffness(6) + damping(6) + gripper(1) = **20D** |
| Loss | L1 reconstruction + KL divergence (CVAE) |

### CFM (ablation)

- Same encoder stack as ACT (shared feature extractor code)
- Decoder: Conditional Flow Matching, conditioning from encoder output
- Loss: MSE on velocity field

---

## Data Pipeline

### Observation windows (multi-rate)

| Stream | Window (To) | Temporal coverage | Reason |
|---|---|---|---|
| RGB images (3×) | `To=4` | 200ms | Motion context, memory-constrained |
| F/T wrench | `To=16` | 800ms | Full insertion force profile/trend |
| Joint pos/vel, TCP, gripper | `To=16` | 800ms | Proprioceptive context |

### Filtering

- `subtask=0` steps → approach model training data
- `subtask=1` steps → insert model training data

### Splits

- Train/val: 90%/10%, **episode-level split** (not step-level — avoids data leakage)
- Augmentation: random crop, color jitter (sim-to-real robustness)

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW, lr=1e-4, cosine annealing |
| Batch size | 32 (safe for T4 16GB with To=4 images) |
| Epochs | 200, early stopping patience=20 |
| Action chunk | `Tp=32` (1.6s at 20Hz) |

**Estimated cost:** ~3-4 hours/model × 4 models = ~12-16 GPU-hours → ~$2-2.5 on-demand

---

## Checkpoint Strategy

```bash
# Every epoch → EBS
~/checkpoints/act_subtask{0,1}_epoch{N}.pt
~/checkpoints/act_subtask{0,1}_latest.pt  # symlink

# Every 5 epochs → S3
aws s3 sync ~/checkpoints/ s3://aic-yusuf/checkpoints/

# Resume after spot interrupt
python train_act.py --subtask 1 --resume-from ~/checkpoints/act_subtask1_latest.pt
```

---

## File Structure

```
aic/
└── training/
    ├── dataset.py          # HDF5 → PyTorch Dataset (shared by ACT + CFM)
    ├── models/
    │   ├── act.py          # ACT: ResNet + 1D-CNN + Transformer + CVAE
    │   └── cfm.py          # CFM: same encoders, flow matching decoder
    ├── train_act.py        # Training loop, checkpointing, S3 sync
    ├── train_cfm.py        # Training loop, checkpointing, S3 sync
    ├── evaluate.py         # aic_model.Policy subclass, subtask dispatcher
    ├── configs/
    │   ├── act.yaml
    │   └── cfm.yaml
    └── checkpoints/        # .pt files, gitignored
```

---

## Evaluation

- `evaluate.py` implements `aic_model.Policy` subclass
- Subtask dispatcher: reads `subtask` state, routes to approach or insert model
- Metrics: success rate (30 trials), avg steps, avg peak force
- Run in Gazebo eval container (same as data collection)

---

## Next Steps (after IL baseline)

1. DPPO fine-tuning on MuJoCo (headless rollouts)
2. `my_policy/` ROS 2 package — wrap trained weights for submission
3. Domain randomization robustness testing
4. Docker submission image

---

## What This Does NOT Cover (deferred)

- ALIGN subtask (implementation plan had 3 subtasks; our data only has 2: APPROACH + INSERT)
- Failure recovery / spiral search (DPPO phase)
- Real hardware (Phase 2, after qualification)
