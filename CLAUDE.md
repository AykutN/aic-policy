# CLAUDE.md — AIC Project Status & Context

## Project Overview
- **Goal:** Intrinsic AI for Industry Challenge — Robotic cable insertion (SFP/SC) using UR5e + Robotiq Hand-E.
- **Tech Stack:** ROS 2 (Humble), Gazebo Sim, MuJoCo (for fast RL/IL training), Pixi (package manager).
- **Core Strategy:** Hierarchical FSM + Flow Matching / Diffusion Policies + Variable Impedance Control (VIC).

## Recent Work (April 3, 2026)
- **Data Collection Analysis:** Reviewed `aic_data_collector/aic_data_collector/ros/DataCollectorPolicy.py`.
    - Uses `CheatCode` (ground truth expert) to collect successful insertion demonstrations.
    - Records 26D State: TCP pose(7), TCP velocity(6), TCP error(6), Joint positions(7).
    - Records Action: Target pose(7), Stiffness(6), Damping(6).
    - Storage: HDF5 format with gzip compression for 3x RGB camera views (scaled 0.25x).
- **Strategy Alignment:** Confirmed implementation matches `docs/dev_docs/implementation_plan.md`.
    - VIC parameters (stiffness/damping) are correctly captured in actions.
    - Observation space covers all required modalities for IL (Imitation Learning).

## Key Commands
- **Run Data Collector:**
  ```bash
  pixi run ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_data_collector.ros.DataCollectorPolicy
  ```
- **Build Workspace:** `pixi run build` (or `colcon build`)
- **Launch Gazebo:** `pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py`

## Current Priorities
1.  **Validate Data Quality:** Ensure `CheatCode` generates high-quality HDF5 files.
2.  **Gripper State:** Verify if `CheatCode` actions include gripper commands and if they should be explicitly added to the recorded action vector.
3.  **IL Training:** Start training the first baseline policy (Flow Matching/Diffusion) using the collected HDF5 data.

## System/Environment Context
- **OS:** macOS ARM-64 (Low free RAM).
- **Simulator:** Running via Docker (Gazebo eval container) or local MuJoCo.
- **Special Setup:** Using a custom `load` script for Victoria 3 symlinks (unrelated to AIC but present in environment).
- **Launcher Fix:** `launcher-settings.json` modified for `distPlatform: pdx` and `steamAppId: 1158310`.
