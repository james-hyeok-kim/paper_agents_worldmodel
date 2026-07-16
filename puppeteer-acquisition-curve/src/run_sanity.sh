#!/bin/bash
# Sanity check: Condition A (trained tracker), seed=1, 3100 steps
# seed_steps = max(1000, 5*500) = 2500 → 3100 steps로 update 버스트 + online update 검증

set -e
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:$LD_LIBRARY_PATH

PUPPETEER_DIR="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer"
TRACKING_PT="/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/checkpoints/tracking.pt"
OUTDIR="/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve"

cd "$PUPPETEER_DIR"

echo "=== SANITY CHECK: Condition A (trained tracker), 3100 steps ==="
python3 train.py \
    "hydra/launcher=basic" \
    task=corridor \
    low_level_fp="$TRACKING_PT" \
    steps=3100 \
    seed=1 \
    eval_freq=2000 \
    eval_episodes=1 \
    save_csv=true \
    save_video=false \
    save_agent=false \
    use_wandb=false \
    exp_name=sanity_condA \
    2>&1 | tee "$OUTDIR/sanity_condA.log"

echo ""
echo "=== Sanity check eval.csv ==="
cat "$PUPPETEER_DIR/logs/corridor/1/sanity_condA/eval.csv" 2>/dev/null || echo "(eval.csv not found at expected path)"

echo ""
echo "=== SANITY DONE ==="
