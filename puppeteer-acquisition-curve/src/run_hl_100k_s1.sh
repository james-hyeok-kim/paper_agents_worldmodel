#!/bin/bash
# High-level corridor training with tracker@100k, seed=1 — GPU 1
set -e
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
TRACKER_PT="/data/jameskimh/worldmodel/tracker_checkpoints/tracker_100000.pt"
LOGFILE="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve/ablation_100000_s1.log"

cd "$PUPPETEER_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] START: tracker@100k seed=1 GPU=1" | tee "$LOGFILE"

python3 train.py "hydra/launcher=basic" \
    task=corridor \
    low_level_fp="$TRACKER_PT" \
    steps=200000 \
    seed=1 \
    exp_name=ablation_100000 \
    use_wandb=false \
    save_csv=true \
    save_video=false \
    save_agent=false \
    eval_freq=20000 \
    eval_episodes=5 \
    2>&1 | tee -a "$LOGFILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: tracker@100k seed=1" | tee -a "$LOGFILE"
