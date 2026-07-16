#!/bin/bash
# Tracker quality ablation — train tracker from scratch with checkpoints
# Checkpoints saved at: 100k, 200k, ..., 500k, ..., 1M, ..., 3M steps
# data_dir: /data/jameskimh/mocapact/small/  (MoCap Act small dataset)
set -e

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
MOCAP_DATA="/data/jameskimh/mocapact/small"
CKPT_SAVE_DIR="/data/jameskimh/worldmodel/tracker_checkpoints"
LOGFILE="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve/tracker_training.log"

mkdir -p "$CKPT_SAVE_DIR"

cd "$PUPPETEER_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] START: Tracker ablation training (3M steps)" | tee -a "$LOGFILE"
echo "data_dir: $MOCAP_DATA" | tee -a "$LOGFILE"
echo "save checkpoints to: logs/tracking/1/tracker_ablation/models/" | tee -a "$LOGFILE"

python3 train.py \
    "hydra/launcher=basic" \
    task=tracking \
    num_clips=all \
    data_dir="$MOCAP_DATA" \
    steps=3000000 \
    save_freq=100000 \
    save_agent=true \
    eval_freq=500000 \
    eval_episodes=20 \
    seed=1 \
    exp_name=tracker_ablation \
    use_wandb=false \
    save_csv=true \
    save_video=false \
    2>&1 | tee -a "$LOGFILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: Tracker training complete" | tee -a "$LOGFILE"

# Copy key checkpoints to dedicated directory
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Copying checkpoints..." | tee -a "$LOGFILE"
MODELS_DIR="$PUPPETEER_DIR/logs/tracking/1/tracker_ablation/models"
for STEP in 100000 500000 1000000 3000000; do
    SRC="$MODELS_DIR/${STEP}.pt"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$CKPT_SAVE_DIR/tracker_${STEP}.pt"
        echo "  Copied tracker_${STEP}.pt" | tee -a "$LOGFILE"
    else
        echo "  WARNING: $SRC not found" | tee -a "$LOGFILE"
    fi
done
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Checkpoints copied to $CKPT_SAVE_DIR" | tee -a "$LOGFILE"
