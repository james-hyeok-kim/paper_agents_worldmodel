#!/bin/bash
# Generic auto-resuming high-level trainer -- parameterized task + tracker path + exp_name,
# so it works for both the corridor tracker-quality ablation and cross-task generalization runs.
# Usage: run_generic_resumable.sh <task> <tracker_pt_path> <seed> <gpu> <exp_name>
set -u

TASK=$1
TRACKER_PT=$2
SEED=$3
GPU=$4
EXP_NAME=$5

EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/results"
PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
MODELS_DIR="$PUPPETEER_DIR/logs/${TASK}/${SEED}/${EXP_NAME}/models"
LOGFILE="$EXPDIR/${EXP_NAME}_s${SEED}.log"
SUPERVISOR_LOG="$EXPDIR/${EXP_NAME}_s${SEED}_supervisor.log"

MAX_RETRIES=20

slog() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$SUPERVISOR_LOG"; }

slog "=== Resumable run started: task=${TASK} tracker=${TRACKER_PT} seed=${SEED} GPU=${GPU} exp=${EXP_NAME} ==="

ATTEMPT=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))

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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python3 train.py "hydra/launcher=basic" \
        task=$TASK \
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
        slog "SUCCESS: ${EXP_NAME} seed=${SEED} completed on attempt $ATTEMPT"
        echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] DONE: ${EXP_NAME} seed=${SEED}" | tee -a "$LOGFILE"
        exit 0
    fi

    slog "CRASHED: ${EXP_NAME} seed=${SEED} attempt $ATTEMPT did not complete. Retrying..."
    sleep 10
done

slog "FAILED: exceeded $MAX_RETRIES retries for ${EXP_NAME} seed=${SEED}"
exit 1
