#!/bin/bash
# Watch HL@100k s1 log. When "DONE" appears, launch the remaining pipeline on GPU 1.
EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"
S1_LOG="$EXPDIR/ablation_100000_s1.log"
WLOG="$EXPDIR/watcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$WLOG"; }

log "Watcher started. Monitoring: $S1_LOG"

while true; do
    if grep -q "DONE: tracker@100k seed=1" "$S1_LOG" 2>/dev/null; then
        log "HL@100k s1 DONE detected. Launching remaining pipeline..."
        nohup bash "$EXPDIR/run_pipeline_remaining.sh" > "$EXPDIR/pipeline_stdout.log" 2>&1 &
        PIPELINE_PID=$!
        log "Pipeline launched with PID $PIPELINE_PID. Watcher exiting."
        break
    fi
    sleep 60
done
