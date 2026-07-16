#!/bin/bash
# Condition A (trained tracker), seed=1
set -e
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
TRACKING_PT="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/checkpoints/tracking.pt"
OUTDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"

cd "$PUPPETEER_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] START: Condition A seed=1"
python3 train.py \
    "hydra/launcher=basic" \
    task=corridor \
    low_level_fp="$TRACKING_PT" \
    steps=200000 \
    seed=1 \
    eval_freq=20000 \
    eval_episodes=5 \
    save_csv=true \
    save_video=false \
    save_agent=false \
    use_wandb=false \
    exp_name=condA_s1 \
    2>&1 | tee "$OUTDIR/condA_s1.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: Condition A seed=1"
