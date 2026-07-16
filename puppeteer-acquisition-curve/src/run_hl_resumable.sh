#!/bin/bash
# Generic auto-resuming high-level corridor trainer.
# Usage: run_hl_resumable.sh <tracker_step> <seed> <gpu_id>
# On crash (process dies without "Training completed successfully"), automatically
# finds the latest saved checkpoint and relaunches with checkpoint=<path> to resume.
set -u

TRACKER_STEP=$1
SEED=$2
GPU=$3

EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/results"
PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
TRACKER_PT="/data/jameskimh/worldmodel/tracker_checkpoints/tracker_${TRACKER_STEP}.pt"
EXP_NAME="ablation_${TRACKER_STEP}"
MODELS_DIR="$PUPPETEER_DIR/logs/corridor/${SEED}/${EXP_NAME}/models"
LOGFILE="$EXPDIR/ablation_${TRACKER_STEP}_s${SEED}.log"
SUPERVISOR_LOG="$EXPDIR/ablation_${TRACKER_STEP}_s${SEED}_supervisor.log"

MAX_RETRIES=20

slog() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$SUPERVISOR_LOG"; }

slog "=== Resumable run started: tracker@${TRACKER_STEP} seed=${SEED} GPU=${GPU} ==="

ATTEMPT=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))

    # Find latest checkpoint if any (numeric filenames only, e.g. 20000.pt)
    CKPT=""
    if [ -d "$MODELS_DIR" ]; then
        CKPT=$(ls "$MODELS_DIR" 2>/dev/null | grep -E '^[0-9]+\.pt$' | sort -t. -k1 -n | tail -1)
        if [ -n "$CKPT" ]; then
            CKPT="$MODELS_DIR/$CKPT"
        fi
    fi

    CKPT_ARG=""
    if [ -n "$CKPT" ]; then
        slog "Attempt $ATTEMPT: resuming from $CKPT"
        CKPT_ARG="checkpoint=$CKPT"
    else
        slog "Attempt $ATTEMPT: fresh start (no checkpoint found)"
    fi

    cd "$PUPPETEER_DIR"
    export MUJOCO_GL=egl
    export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=$GPU
    # Reduce CUDA caching-allocator fragmentation risk over long single-process runs
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python3 train.py "hydra/launcher=basic" \
        task=corridor \
        low_level_fp="$TRACKER_PT" \
        steps=200000 \
        seed=$SEED \
        exp_name="$EXP_NAME" \
        $CKPT_ARG \
        use_wandb=false \
        save_csv=true \
        save_video=false \
        save_agent=true \
        save_freq=20000 \
        eval_freq=20000 \
        eval_episodes=5 \
        2>&1 | tee -a "$LOGFILE"

    if grep -q "Training completed successfully" "$LOGFILE"; then
        slog "SUCCESS: tracker@${TRACKER_STEP} seed=${SEED} completed on attempt $ATTEMPT"
        echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: tracker@${TRACKER_STEP} seed=${SEED}" | tee -a "$LOGFILE"
        exit 0
    fi

    slog "CRASHED: tracker@${TRACKER_STEP} seed=${SEED} attempt $ATTEMPT did not complete. Retrying..."
    sleep 10
done

slog "FAILED: exceeded $MAX_RETRIES retries for tracker@${TRACKER_STEP} seed=${SEED}"
exit 1
