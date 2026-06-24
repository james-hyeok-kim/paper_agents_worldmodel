#!/bin/bash
# Daemon: auto-copy target tracker checkpoints to /data, delete non-targets from workspace
# Target checkpoints: 500000, 1000000, 3000000 (100k already done manually)
# Runs in background, checks every 60 seconds

MODELS_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer/logs/tracking/1/tracker_ablation/models"
CKPT_DIR="/data/jameskimh/worldmodel/tracker_checkpoints"
LOGFILE="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve/ckpt_daemon.log"

TARGETS="500000 1000000 3000000"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $1" | tee -a "$LOGFILE"; }

log "Daemon started. Watching: $MODELS_DIR"

while true; do
    for PT in "$MODELS_DIR"/*.pt; do
        [ -f "$PT" ] || continue
        BASENAME=$(basename "$PT" .pt)

        # Check if it's a target checkpoint
        IS_TARGET=0
        for T in $TARGETS; do
            [ "$BASENAME" = "$T" ] && IS_TARGET=1 && break
        done

        if [ "$IS_TARGET" = "1" ]; then
            DEST="$CKPT_DIR/tracker_${BASENAME}.pt"
            if [ ! -f "$DEST" ]; then
                cp "$PT" "$DEST"
                log "COPIED target checkpoint: tracker_${BASENAME}.pt → $CKPT_DIR"
            fi
            # Delete from workspace after confirmed copy
            if [ -f "$DEST" ]; then
                rm -f "$PT"
                log "DELETED from workspace: $BASENAME.pt (kept in /data)"
            fi
        else
            # Non-target: delete immediately
            rm -f "$PT"
            log "DELETED non-target: $BASENAME.pt"
        fi
    done

    # Check if 3M training is done (final.pt appears)
    if [ -f "$MODELS_DIR/final.pt" ]; then
        # final.pt is the 3M model (same as 3000000.pt)
        DEST="$CKPT_DIR/tracker_final.pt"
        if [ ! -f "$DEST" ]; then
            cp "$MODELS_DIR/final.pt" "$DEST"
            log "COPIED final.pt → tracker_final.pt in /data"
        fi
        rm -f "$MODELS_DIR/final.pt"
        log "Training complete (final.pt handled). Daemon exiting."
        break
    fi

    sleep 60
done
