#!/bin/bash
# 전체 실험 실행: CondA(s1,s2) 병렬 → 완료 후 CondB(s1,s2) 병렬
# nohup으로 세션 독립 실행용

set -e
OUTDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"
PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
TRACKING_PT="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/checkpoints/tracking.pt"
RANDOM_PT="$OUTDIR/random_tracker.pt"

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] === RUN ALL STARTED ===" | tee -a "$OUTDIR/run_all.log"

# --- Condition A ---
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Starting CondA s1 (GPU 0)" | tee -a "$OUTDIR/run_all.log"
CUDA_VISIBLE_DEVICES=0 python3 "$PUPPETEER_DIR/train.py" \
    "hydra/launcher=basic" task=corridor low_level_fp="$TRACKING_PT" \
    steps=200000 seed=1 eval_freq=20000 eval_episodes=5 \
    save_csv=true save_video=false save_agent=false use_wandb=false \
    exp_name=condA_s1 >> "$OUTDIR/condA_s1.log" 2>&1 &
PID_A1=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Starting CondA s2 (GPU 1)" | tee -a "$OUTDIR/run_all.log"
CUDA_VISIBLE_DEVICES=1 python3 "$PUPPETEER_DIR/train.py" \
    "hydra/launcher=basic" task=corridor low_level_fp="$TRACKING_PT" \
    steps=200000 seed=2 eval_freq=20000 eval_episodes=5 \
    save_csv=true save_video=false save_agent=false use_wandb=false \
    exp_name=condA_s2 >> "$OUTDIR/condA_s2.log" 2>&1 &
PID_A2=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Waiting for CondA (PIDs: $PID_A1 $PID_A2)" | tee -a "$OUTDIR/run_all.log"
wait $PID_A1 && echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] CondA s1 done" | tee -a "$OUTDIR/run_all.log"
wait $PID_A2 && echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] CondA s2 done" | tee -a "$OUTDIR/run_all.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] === CondA COMPLETE ===" | tee -a "$OUTDIR/run_all.log"
cat "$PUPPETEER_DIR/logs/corridor/1/condA_s1/eval.csv" | tee -a "$OUTDIR/run_all.log"
cat "$PUPPETEER_DIR/logs/corridor/2/condA_s2/eval.csv" | tee -a "$OUTDIR/run_all.log"

# --- Condition B ---
echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Starting CondB s1 (GPU 0)" | tee -a "$OUTDIR/run_all.log"
CUDA_VISIBLE_DEVICES=0 python3 "$PUPPETEER_DIR/train.py" \
    "hydra/launcher=basic" task=corridor low_level_fp="$RANDOM_PT" \
    steps=200000 seed=1 eval_freq=20000 eval_episodes=5 \
    save_csv=true save_video=false save_agent=false use_wandb=false \
    exp_name=condB_s1 >> "$OUTDIR/condB_s1.log" 2>&1 &
PID_B1=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Starting CondB s2 (GPU 1)" | tee -a "$OUTDIR/run_all.log"
CUDA_VISIBLE_DEVICES=1 python3 "$PUPPETEER_DIR/train.py" \
    "hydra/launcher=basic" task=corridor low_level_fp="$RANDOM_PT" \
    steps=200000 seed=2 eval_freq=20000 eval_episodes=5 \
    save_csv=true save_video=false save_agent=false use_wandb=false \
    exp_name=condB_s2 >> "$OUTDIR/condB_s2.log" 2>&1 &
PID_B2=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] Waiting for CondB (PIDs: $PID_B1 $PID_B2)" | tee -a "$OUTDIR/run_all.log"
wait $PID_B1 && echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] CondB s1 done" | tee -a "$OUTDIR/run_all.log"
wait $PID_B2 && echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] CondB s2 done" | tee -a "$OUTDIR/run_all.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] === ALL DONE ===" | tee -a "$OUTDIR/run_all.log"
cat "$PUPPETEER_DIR/logs/corridor/1/condB_s1/eval.csv" | tee -a "$OUTDIR/run_all.log"
cat "$PUPPETEER_DIR/logs/corridor/2/condB_s2/eval.csv" | tee -a "$OUTDIR/run_all.log"
