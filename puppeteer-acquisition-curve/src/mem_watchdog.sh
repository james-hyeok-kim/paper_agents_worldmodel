#!/bin/bash
# Memory watchdog v2: track the ACTUAL cgroup limit (not host free -h, which is blind to
# the container's memory.max) plus this job's own train.py RSS, every 30s.
EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/results"
LOGFILE="$EXPDIR/mem_watchdog.log"
CGROUP_MAX="/sys/fs/cgroup/memory.max"
CGROUP_CUR="/sys/fs/cgroup/memory.current"
CGROUP_EVENTS="/sys/fs/cgroup/memory.events"

MAXVAL=$(cat "$CGROUP_MAX" 2>/dev/null || echo "unknown")
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Watchdog v2 started. cgroup memory.max=$MAXVAL" >> "$LOGFILE"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S KST')
    CUR=$(cat "$CGROUP_CUR" 2>/dev/null || echo "?")
    OOM_EVENTS=$(grep -E "^oom" "$CGROUP_EVENTS" 2>/dev/null | tr '\n' ' ')
    TRAINPY_RSS=$(ps -C python3 -o pid,rss,cmd --no-headers 2>/dev/null | grep "train.py" | awk '{sum+=$2} END {print sum/1024/1024 " GiB"}')
    echo "[$TS] cgroup.current=$((CUR/1024/1024/1024))GiB train.py_rss=${TRAINPY_RSS:-0GiB} $OOM_EVENTS" >> "$LOGFILE"
    sleep 30
done
