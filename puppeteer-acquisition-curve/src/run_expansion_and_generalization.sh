#!/bin/bash
# Dispatcher: runs a fixed job list with a concurrency cap, cycling GPU0/GPU1.
# Part A: 500k/1M/3M seed 6-11 (18 runs) -- extend n=5 -> n=11 for statistical power.
# Part B: gaps-corridor / walls-corridor x {0-step, 500k} x seed 1-3 (12 runs) -- generalization.
# Concurrency cap chosen to stay well under the 16-core cgroup CPU quota alongside the
# already-running 10M seed-expansion queue (~1.3 CPU cores/job observed).
set -u
EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/results"
SRCDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/src"
TRACKER_DIR="/data/jameskimh/worldmodel/tracker_checkpoints"
LOG="$EXPDIR/expansion_generalization_queue.log"
MAX_CONCURRENT=6

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$LOG"; }

# Build job list as lines: "task|tracker_pt|seed|exp_name"
JOBS=()
for TRACKER in 500000 1000000 3000000; do
    for SEED in 6 7 8 9 10 11; do
        JOBS+=("corridor|${TRACKER_DIR}/tracker_${TRACKER}.pt|${SEED}|ablation_${TRACKER}")
    done
done
for TASK in gaps-corridor walls-corridor; do
    TSHORT=$(echo "$TASK" | cut -d- -f1)
    for SEED in 1 2 3; do
        JOBS+=("${TASK}|${EXPDIR}/random_tracker.pt|${SEED}|${TSHORT}_0step")
        JOBS+=("${TASK}|${TRACKER_DIR}/tracker_500000.pt|${SEED}|${TSHORT}_500k")
    done
done

log "Expansion+generalization queue started: ${#JOBS[@]} jobs total, max ${MAX_CONCURRENT} concurrent."

GPU_TOGGLE=0
RUNNING=0
IDX=0

launch_next() {
    if [ $IDX -ge ${#JOBS[@]} ]; then
        return 1
    fi
    IFS='|' read -r TASK TRACKER_PT SEED EXP_NAME <<< "${JOBS[$IDX]}"
    GPU=$GPU_TOGGLE
    GPU_TOGGLE=$(( (GPU_TOGGLE + 1) % 2 ))
    log "LAUNCH [$((IDX+1))/${#JOBS[@]}]: task=${TASK} tracker=$(basename "$TRACKER_PT") seed=${SEED} exp=${EXP_NAME} GPU=${GPU}"
    bash "$SRCDIR/run_generic_resumable.sh" "$TASK" "$TRACKER_PT" "$SEED" "$GPU" "$EXP_NAME" \
        > "$EXPDIR/${EXP_NAME}_s${SEED}_stdout.log" 2>&1 &
    IDX=$((IDX + 1))
    RUNNING=$((RUNNING + 1))
    return 0
}

while [ $IDX -lt ${#JOBS[@]} ] && [ $RUNNING -lt $MAX_CONCURRENT ]; do
    launch_next
done

while [ $RUNNING -gt 0 ]; do
    wait -n
    RUNNING=$((RUNNING - 1))
    if [ $IDX -lt ${#JOBS[@]} ]; then
        launch_next
    fi
done

log "Expansion+generalization queue COMPLETE (${#JOBS[@]} jobs)."
