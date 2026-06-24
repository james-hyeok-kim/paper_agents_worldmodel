#!/bin/bash
# Condition B (random tracker), seed=2
set -e
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
RANDOM_PT="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve/random_tracker.pt"
OUTDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"

cd "$PUPPETEER_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] START: Condition B seed=2"
python3 train.py \
    "hydra/launcher=basic" \
    task=corridor \
    low_level_fp="$RANDOM_PT" \
    steps=200000 \
    seed=2 \
    eval_freq=20000 \
    eval_episodes=5 \
    save_csv=true \
    save_video=false \
    save_agent=false \
    use_wandb=false \
    exp_name=condB_s2 \
    2>&1 | tee "$OUTDIR/condB_s2.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: Condition B seed=2"
