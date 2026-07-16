#!/bin/bash
# High-level corridor training with tracker@500k, seed=2 — GPU 1
set -e
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
TRACKER_PT="/data/jameskimh/worldmodel/tracker_checkpoints/tracker_500000.pt"
LOGFILE="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve/ablation_500000_s2.log"

MAX_WAIT=14400
WAITED=0
while [ ! -f "$TRACKER_PT" ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Waiting for $TRACKER_PT ..."
    sleep 60
    WAITED=$((WAITED + 60))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] TIMEOUT waiting for 500k checkpoint. Aborting."
        exit 1
    fi
done

cd "$PUPPETEER_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] START: tracker@500k seed=2 GPU=1" | tee "$LOGFILE"

python3 train.py "hydra/launcher=basic" \
    task=corridor \
    low_level_fp="$TRACKER_PT" \
    steps=200000 \
    seed=2 \
    exp_name=ablation_500000 \
    use_wandb=false \
    save_csv=true \
    save_video=false \
    save_agent=false \
    eval_freq=20000 \
    eval_episodes=5 \
    2>&1 | tee -a "$LOGFILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: tracker@500k seed=2" | tee -a "$LOGFILE"
