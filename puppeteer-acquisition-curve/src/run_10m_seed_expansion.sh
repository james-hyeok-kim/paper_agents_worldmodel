#!/bin/bash
# Queue: 10M-tracker x seed 3,4,5 (3 runs), parallel across GPU0/GPU1 where possible.
# seed=3 (GPU0) and seed=4 (GPU1) run concurrently, then seed=5 (GPU0) runs after.
# Each run auto-resumes on crash via run_hl_resumable.sh.
set -u
EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/results"
SRCDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/src"
LOG="$EXPDIR/10m_seed_expansion_queue.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$LOG"; }

log "10M seed expansion queue started (tracker=10000000 x seed 3,4,5, GPU0+GPU1 parallel)."

log "--- HL@10000000 seed=3 (GPU0) + seed=4 (GPU1) launching in parallel ---"
bash "$SRCDIR/run_hl_resumable.sh" 10000000 3 0 > "$EXPDIR/10m_seed3_stdout.log" 2>&1 &
PID3=$!
bash "$SRCDIR/run_hl_resumable.sh" 10000000 4 1 > "$EXPDIR/10m_seed4_stdout.log" 2>&1 &
PID4=$!

wait $PID3
log "--- HL@10000000 seed=3 DONE ---"
wait $PID4
log "--- HL@10000000 seed=4 DONE ---"

log "--- HL@10000000 seed=5 (GPU0) ---"
bash "$SRCDIR/run_hl_resumable.sh" 10000000 5 0 > "$EXPDIR/10m_seed5_stdout.log" 2>&1
log "--- HL@10000000 seed=5 DONE ---"

log "10M seed expansion queue COMPLETE."
