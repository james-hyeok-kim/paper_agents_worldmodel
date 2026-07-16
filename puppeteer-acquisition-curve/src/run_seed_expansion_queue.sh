#!/bin/bash
# Queue: 500k/1M/3M x seed 3,4,5 (9 runs), sequential on GPU 1, each auto-resuming on crash.
set -e
EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"
LOG="$EXPDIR/seed_expansion_queue.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$LOG"; }

log "Seed expansion queue started (500k/1M/3M x seed 3,4,5, GPU 1)."

for TRACKER in 500000 1000000 3000000; do
    for SEED in 3 4 5; do
        log "--- HL@${TRACKER} seed=${SEED} (GPU 1, auto-resume) ---"
        bash "$EXPDIR/run_hl_resumable.sh" "$TRACKER" "$SEED" 1
        log "--- HL@${TRACKER} seed=${SEED} DONE ---"
    done
done

log "Seed expansion queue COMPLETE."
