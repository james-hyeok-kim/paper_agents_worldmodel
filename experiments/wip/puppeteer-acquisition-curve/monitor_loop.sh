#!/bin/bash
# 30분마다 진행 상황 스냅샷 기록
OUTDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"
LOGS_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer/logs/corridor"
PID_FILE="$OUTDIR/nohup.pid"
MAIN_PID=$(cat "$PID_FILE" 2>/dev/null)

while kill -0 "$MAIN_PID" 2>/dev/null; do
    TS="$(date '+%Y-%m-%d %H:%M:%S KST')"
    echo "=== $TS ===" >> "$OUTDIR/progress_snapshots.log"
    for tag in "1/condA_s1" "2/condA_s2" "1/condB_s1" "2/condB_s2"; do
        csv="$LOGS_DIR/$tag/eval.csv"
        if [ -f "$csv" ]; then
            last=$(tail -1 "$csv")
            echo "  $tag: $last" >> "$OUTDIR/progress_snapshots.log"
        else
            echo "  $tag: (no eval yet)" >> "$OUTDIR/progress_snapshots.log"
        fi
    done
    sleep 1800
done

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Main process ($MAIN_PID) finished." >> "$OUTDIR/progress_snapshots.log"
