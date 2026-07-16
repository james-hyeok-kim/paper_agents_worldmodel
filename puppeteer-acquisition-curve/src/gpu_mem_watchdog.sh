#!/bin/bash
# GPU memory watchdog: track train.py CUDA memory usage on GPU0+GPU1 over time,
# to catch allocator fragmentation / gradual growth before it causes a crash.
EXPDIR="/home/jovyan/workspace/paper_agents_worldmodel/puppeteer-acquisition-curve/results"
LOGFILE="$EXPDIR/gpu_mem_watchdog.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] GPU watchdog started (GPU0+GPU1)." >> "$LOGFILE"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S KST')
    GPU_USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
    PROC_MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
    echo "[$TS] gpu.used(idx,MiB)=[${GPU_USED}] procs=[${PROC_MEM}]" >> "$LOGFILE"
    sleep 60
done
