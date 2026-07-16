#!/bin/bash
# Sequential HL@3M seed=1 then seed=2 on GPU 1, each auto-resuming on crash.
set -e
EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"
LOG="$EXPDIR/hl_3m_sequential_resumable.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$LOG"; }

log "Resumable sequential 3M pipeline started."

log "--- HL@3M seed=1 (GPU 1, auto-resume) ---"
bash "$EXPDIR/run_hl_resumable.sh" 3000000 1 1
log "--- HL@3M seed=1 DONE ---"

log "--- HL@3M seed=2 (GPU 1, auto-resume) ---"
bash "$EXPDIR/run_hl_resumable.sh" 3000000 2 1
log "--- HL@3M seed=2 DONE ---"

log "Resumable sequential 3M pipeline COMPLETE."
