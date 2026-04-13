# AGENT_TRAIN.md — IL Training on AWS

## Prerequisites
- g4dn.xlarge on-demand (NOT spot — training must not be interrupted)
- Dataset at ~/aic_dataset/ (420+ episodes from data collection phase)
- Repo at ~/ws_aic/src/aic/

## Setup (once per instance)

```bash
cd ~/ws_aic/src/aic

# Verify dataset
pixi run python3 aic_data_collector/scripts/inspect_dataset.py ~/aic_dataset

# Install training deps (if not already in env)
pip install pyyaml torchvision

# Run sanity tests
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 -c "
import sys; sys.path.insert(0, '.')
from training.test_dataset import test_dataset_returns_correct_keys, test_dataset_subtask_filter, test_dataset_action_shape
from training.test_encoders import test_image_encoder_output_shape, test_ft_encoder_output_shape, test_proprio_encoder_output_shape
from training.test_act import test_act_forward_shape, test_act_loss
from training.test_cfm import test_cfm_forward_shape
test_dataset_returns_correct_keys(); test_dataset_subtask_filter(); test_dataset_action_shape()
test_image_encoder_output_shape(); test_ft_encoder_output_shape(); test_proprio_encoder_output_shape()
test_act_forward_shape(); test_act_loss()
test_cfm_forward_shape()
print('All tests passed.')
"

mkdir -p ~/checkpoints
```

## Training (run in order)

### ACT approach policy (subtask=0) — ~3-4 hours

```bash
cd ~/ws_aic/src/aic
nohup python3 training/train_act.py \
  --dataset ~/aic_dataset \
  --subtask 0 \
  > ~/checkpoints/act_subtask0_train.log 2>&1 &
echo "ACT subtask 0 started (PID=$!)"

# Monitor:
tail -f ~/checkpoints/act_subtask0_train.log
```

### ACT insert policy (subtask=1) — ~3-4 hours

```bash
python3 training/train_act.py \
  --dataset ~/aic_dataset \
  --subtask 1 \
  2>&1 | tee ~/checkpoints/act_subtask1_train.log
```

### CFM approach policy (ablation, subtask=0) — ~3 hours

```bash
python3 training/train_cfm.py \
  --dataset ~/aic_dataset \
  --subtask 0 \
  2>&1 | tee ~/checkpoints/cfm_subtask0_train.log
```

### CFM insert policy (ablation, subtask=1) — ~3 hours

```bash
python3 training/train_cfm.py \
  --dataset ~/aic_dataset \
  --subtask 1 \
  2>&1 | tee ~/checkpoints/cfm_subtask1_train.log
```

## Resume After Interrupt

```bash
python3 training/train_act.py --dataset ~/aic_dataset --subtask 1 --resume
python3 training/train_cfm.py --dataset ~/aic_dataset --subtask 1 --resume
```

## Monitor Checkpoints

```bash
# Watch best.pt files appear as training completes
watch -n 60 'ls -lh ~/checkpoints/*/best.pt 2>/dev/null && echo "" && du -sh ~/checkpoints/'

# Check S3 sync worked
aws s3 ls s3://aic-yusuf/checkpoints/ --recursive | tail -10
```

## Evaluate in Gazebo

```bash
# Terminal 1: eval container (no ground_truth — tests real policy)
export DBX_CONTAINER_MANAGER=docker
distrobox enter --root aic_eval -- /entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true

# Terminal 2: trained policy
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=training.evaluate.TrainedPolicy \
  -p approach_ckpt:=$HOME/checkpoints/act_subtask0/best.pt \
  -p insert_ckpt:=$HOME/checkpoints/act_subtask1/best.pt
```

## Expected Results

| Model | Subtask | Expected val_loss | Expected Gazebo success rate |
|---|---|---|---|
| ACT | approach (0) | < 0.05 | N/A (open-loop approach) |
| ACT | insert (1) | < 0.08 | 55–70% |
| CFM | approach (0) | < 0.01 MSE | N/A |
| CFM | insert (1) | < 0.01 MSE | 55–75% |

If Gazebo success rate is > 50%: proceed to DPPO fine-tuning.
If success rate is < 30%: check val_loss curves for overfitting and reduce batch_size or increase augmentation.

## Download Checkpoints to Mac

```bash
# From Mac terminal:
aws s3 sync s3://aic-yusuf/checkpoints/ ~/Desktop/aic_checkpoints/
```

## Next Step: DPPO Fine-tuning

After IL baseline achieves > 50% success rate, write `AGENT_DPPO.md` for RL fine-tuning.
Training code: `training/train_dppo.py` (not yet written).
