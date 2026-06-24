#!/bin/bash
# Run high-level corridor training for each tracker quality level
# Usage: called after tracker training completes with all checkpoints
# Runs: tracker_100k, tracker_500k, tracker_1m, tracker_3m × 2 seeds each
set -e

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
CKPT_DIR="/data/jameskimh/worldmodel/tracker_checkpoints"
LOGDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"

run_highlevel() {
    local TRACKER_STEPS=$1
    local SEED=$2
    local GPU=$3
    local TRACKER_PT="${CKPT_DIR}/tracker_${TRACKER_STEPS}.pt"
    local EXP_NAME="ablation_${TRACKER_STEPS}"
    local LOGFILE="${LOGDIR}/ablation_${TRACKER_STEPS}_s${SEED}.log"

    if [ ! -f "$TRACKER_PT" ]; then
        echo "ERROR: $TRACKER_PT not found, skipping" | tee -a "$LOGFILE"
        return 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] START: tracker@${TRACKER_STEPS} seed=${SEED} GPU=${GPU}" | tee "$LOGFILE"
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py \
        "hydra/launcher=basic" \
        task=corridor \
        low_level_fp="$TRACKER_PT" \
        steps=200000 \
        seed=$SEED \
        exp_name="$EXP_NAME" \
        use_wandb=false \
        save_csv=true \
        save_video=false \
        eval_freq=20000 \
        eval_episodes=5 \
        2>&1 | tee -a "$LOGFILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: tracker@${TRACKER_STEPS} seed=${SEED}" | tee -a "$LOGFILE"
}

# Sequential pairs using 2 GPUs
# Pair 1: tracker_100k (GPU0 s1, GPU1 s2) → in parallel
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Phase 1: tracker@100k"
run_highlevel 100000 1 0 &
run_highlevel 100000 2 1 &
wait

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Phase 2: tracker@500k"
run_highlevel 500000 1 0 &
run_highlevel 500000 2 1 &
wait

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Phase 3: tracker@1M"
run_highlevel 1000000 1 0 &
run_highlevel 1000000 2 1 &
wait

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Phase 4: tracker@3M"
run_highlevel 3000000 1 0 &
run_highlevel 3000000 2 1 &
wait

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] ALL DONE: All ablation runs complete"
echo "Results in: ${PUPPETEER_DIR}/logs/corridor/"
