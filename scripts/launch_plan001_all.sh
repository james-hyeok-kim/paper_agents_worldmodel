#!/usr/bin/env bash
# Plan 001 + confound-fix runs (13 total)
# Run from dreamerv3-torch directory.
# GPU 0: Batch A (multi-seed, 4 runs) + 2 size-matched baselines
# GPU 1: Batch B K-sweep (3 runs) + Batch C multi-env (4 runs)

set -e
LOGBASE=/data/jameskimh/worldmodel/dual-rate-paper
DRDIR=/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/dual-rate-world-model/dreamerv3-torch

export LD_LIBRARY_PATH=/home/jovyan/egl_libs
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "$DRDIR"

echo "[$(date +%H:%M:%S)] Launching 13 runs..."

# ── GPU 0: Batch A + size-matched baselines ──────────────────────────
CUDA_VISIBLE_DEVICES=0 python dreamer.py --configs dmc_vision \
  --task dmc_walker_walk --logdir $LOGBASE/baseline_seed1 \
  --steps 100000 --seed 1 >> $LOGBASE/baseline_seed1.log 2>&1 &
echo "  [GPU0] baseline_seed1 (PID $!)"

CUDA_VISIBLE_DEVICES=0 python dreamer.py --configs dmc_vision \
  --task dmc_walker_walk --logdir $LOGBASE/baseline_seed2 \
  --steps 100000 --seed 2 >> $LOGBASE/baseline_seed2.log 2>&1 &
echo "  [GPU0] baseline_seed2 (PID $!)"

CUDA_VISIBLE_DEVICES=0 python dreamer.py --configs dmc_vision dual_rate \
  --task dmc_walker_walk --logdir $LOGBASE/dualrate_K3_seed1 \
  --steps 100000 --seed 1 >> $LOGBASE/dualrate_K3_seed1.log 2>&1 &
echo "  [GPU0] dualrate_K3_seed1 (PID $!)"

CUDA_VISIBLE_DEVICES=0 python dreamer.py --configs dmc_vision dual_rate \
  --task dmc_walker_walk --logdir $LOGBASE/dualrate_K3_seed2 \
  --steps 100000 --seed 2 >> $LOGBASE/dualrate_K3_seed2.log 2>&1 &
echo "  [GPU0] dualrate_K3_seed2 (PID $!)"

# Size-matched baselines (confound fix: deter=384 = same as dual_rate total)
CUDA_VISIBLE_DEVICES=0 python dreamer.py --configs dmc_vision \
  --task dmc_walker_walk --logdir $LOGBASE/baseline_deter384_seed0 \
  --steps 100000 --seed 0 --dyn_deter 384 >> $LOGBASE/baseline_deter384_seed0.log 2>&1 &
echo "  [GPU0] baseline_deter384_seed0 (PID $!)"

CUDA_VISIBLE_DEVICES=0 python dreamer.py --configs dmc_vision \
  --task dmc_walker_walk --logdir $LOGBASE/baseline_deter384_seed1 \
  --steps 100000 --seed 1 --dyn_deter 384 >> $LOGBASE/baseline_deter384_seed1.log 2>&1 &
echo "  [GPU0] baseline_deter384_seed1 (PID $!)"

# ── GPU 1: Batch B (K-sweep) + Batch C (multi-env) ───────────────────
CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision dual_rate \
  --task dmc_walker_walk --logdir $LOGBASE/dualrate_K2_seed0 \
  --steps 100000 --seed 0 --slow_K 2 >> $LOGBASE/dualrate_K2_seed0.log 2>&1 &
echo "  [GPU1] dualrate_K2_seed0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision dual_rate \
  --task dmc_walker_walk --logdir $LOGBASE/dualrate_K4_seed0 \
  --steps 100000 --seed 0 --slow_K 4 >> $LOGBASE/dualrate_K4_seed0.log 2>&1 &
echo "  [GPU1] dualrate_K4_seed0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision dual_rate \
  --task dmc_walker_walk --logdir $LOGBASE/dualrate_K6_seed0 \
  --steps 100000 --seed 0 --slow_K 6 >> $LOGBASE/dualrate_K6_seed0.log 2>&1 &
echo "  [GPU1] dualrate_K6_seed0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision \
  --task dmc_cheetah_run --logdir $LOGBASE/baseline_cheetah_seed0 \
  --steps 100000 --seed 0 >> $LOGBASE/baseline_cheetah_seed0.log 2>&1 &
echo "  [GPU1] baseline_cheetah_seed0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision dual_rate \
  --task dmc_cheetah_run --logdir $LOGBASE/dualrate_cheetah_seed0 \
  --steps 100000 --seed 0 >> $LOGBASE/dualrate_cheetah_seed0.log 2>&1 &
echo "  [GPU1] dualrate_cheetah_seed0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision \
  --task dmc_hopper_hop --logdir $LOGBASE/baseline_hopper_seed0 \
  --steps 100000 --seed 0 >> $LOGBASE/baseline_hopper_seed0.log 2>&1 &
echo "  [GPU1] baseline_hopper_seed0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 python dreamer.py --configs dmc_vision dual_rate \
  --task dmc_hopper_hop --logdir $LOGBASE/dualrate_hopper_seed0 \
  --steps 100000 --seed 0 >> $LOGBASE/dualrate_hopper_seed0.log 2>&1 &
echo "  [GPU1] dualrate_hopper_seed0 (PID $!)"

echo "[$(date +%H:%M:%S)] All 13 runs launched. Monitor with:"
echo "  watch -n 30 'tail -1 $LOGBASE/*/metrics.jsonl 2>/dev/null | grep step'"
echo ""
echo "Note: Existing partial logs (~6-8k steps) will be appended."
echo "All runs start from step 0 (no checkpoints saved from previous stalled runs)."
