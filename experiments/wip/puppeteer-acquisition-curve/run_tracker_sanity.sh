#!/bin/bash
# Sanity check: tracker training for 2000 steps
# Verifies data loading, env, model, save work correctly
set -e

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
MOCAP_DATA="/data/jameskimh/mocapact/small"
LOGFILE="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve/tracker_sanity.log"

cd "$PUPPETEER_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] START: Tracker sanity check (2000 steps)" | tee "$LOGFILE"
echo "HDF5 files in data_dir: $(ls $MOCAP_DATA/*.hdf5 2>/dev/null | wc -l)" | tee -a "$LOGFILE"

python3 train.py \
    "hydra/launcher=basic" \
    task=tracking \
    num_clips=all \
    data_dir="$MOCAP_DATA" \
    steps=2000 \
    save_freq=1000 \
    save_agent=true \
    eval_freq=1000 \
    eval_episodes=3 \
    seed=1 \
    exp_name=tracker_sanity \
    use_wandb=false \
    save_csv=true \
    save_video=false \
    2>&1 | tee -a "$LOGFILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: sanity check" | tee -a "$LOGFILE"
echo "Check logs/tracking/1/tracker_sanity/models/ for checkpoint" | tee -a "$LOGFILE"
