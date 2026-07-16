#!/bin/bash
# Pipeline: chain all remaining high-level ablation runs on GPU 1
# Assumes HL@100k s1 is already running. Runs s2 first, then 500k s1/s2, 1M s1/s2, 3M s1/s2 in sequence.
set -e

EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"
PLOG="$EXPDIR/pipeline.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$PLOG"; }

log "Pipeline started."

# Step 1: HL@100k s2
log "--- Launching HL@100k seed=2 ---"
bash "$EXPDIR/run_hl_100k_s2.sh"
log "--- HL@100k seed=2 DONE ---"

# Step 2: HL@500k s1 (includes internal wait for checkpoint)
log "--- Launching HL@500k seed=1 ---"
bash "$EXPDIR/run_hl_500k_s1.sh"
log "--- HL@500k seed=1 DONE ---"

# Step 3: HL@500k s2
log "--- Launching HL@500k seed=2 ---"
bash "$EXPDIR/run_hl_500k_s2.sh"
log "--- HL@500k seed=2 DONE ---"

# Step 4: HL@1M s1 (includes internal wait for checkpoint)
log "--- Launching HL@1M seed=1 ---"
bash "$EXPDIR/run_hl_1m_s1.sh"
log "--- HL@1M seed=1 DONE ---"

# Step 5: HL@1M s2
log "--- Launching HL@1M seed=2 ---"
bash "$EXPDIR/run_hl_1m_s2.sh"
log "--- HL@1M seed=2 DONE ---"

# Step 6: HL@3M s1 (includes internal wait for checkpoint)
log "--- Launching HL@3M seed=1 ---"
bash "$EXPDIR/run_hl_3m_s1.sh"
log "--- HL@3M seed=1 DONE ---"

# Step 7: HL@3M s2
log "--- Launching HL@3M seed=2 ---"
bash "$EXPDIR/run_hl_3m_s2.sh"
log "--- HL@3M seed=2 DONE ---"

log "ALL PIPELINE RUNS COMPLETE."
