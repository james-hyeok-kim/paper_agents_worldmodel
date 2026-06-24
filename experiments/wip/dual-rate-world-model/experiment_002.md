# Experiment 002 — Dual-Rate World Model 논문 확정 실험

## 메타데이터
- 시작: 2026-06-12 21:30 KST
- 상태: RUNNING (11개 학습 실행 중)
- 기반: experiment_001 (seed=0, Walker-Walk) → GO 판정

## 실험 설계

### 목적
seed=0 단일 결과를 다중 seed, K sweep, 다중 환경으로 확장하여 논문 수준 근거 확보.

### 실험 구성

| Run ID | Task | Config | Seed | K | GPU | Status |
|---|---|---|---|---|---|---|
| baseline_seed1 | walker_walk | dmc_vision | 1 | — | 0 | RUNNING |
| baseline_seed2 | walker_walk | dmc_vision | 2 | — | 0 | RUNNING |
| dualrate_K3_seed1 | walker_walk | dual_rate | 1 | 3 | 0 | RUNNING |
| dualrate_K3_seed2 | walker_walk | dual_rate | 2 | 3 | 0 | RUNNING |
| dualrate_K2_seed0 | walker_walk | dual_rate | 0 | 2 | 0 | RUNNING |
| dualrate_K4_seed0 | walker_walk | dual_rate | 0 | 4 | 0 | RUNNING |
| dualrate_K6_seed0 | walker_walk | dual_rate | 0 | 6 | 1 | RUNNING |
| baseline_cheetah | cheetah_run | dmc_vision | 0 | — | 1 | RUNNING |
| dualrate_cheetah | cheetah_run | dual_rate | 0 | 3 | 1 | RUNNING |
| baseline_hopper | hopper_hop | dmc_vision | 0 | — | 1 | RUNNING |
| dualrate_hopper | hopper_hop | dual_rate | 0 | 3 | 1 | RUNNING |

### 기존 결과 (seed=0)
- baseline_100k: `/data/jameskimh/worldmodel/dual-rate-world-model/baseline_100k/`
- dualrate_K3_100k: `/data/jameskimh/worldmodel/dual-rate-world-model/dualrate_K3_100k/`

### 새 결과
- `/data/jameskimh/worldmodel/dual-rate-paper/*/metrics.jsonl`

## 측정 지표
1. **eval_return** (primary): late-stage mean (75k-105k), 3 seed mean ± std
2. **fps**: 학습 효율 (baseline 대비 비율)
3. **separation_score**: fast/slow 분리 안정성 (>0.5)
4. **FLOPs reduction**: analytical (K 함수)

## 성공 기준
- 다중 seed: 3 seed late-stage mean, dual-rate ≥ baseline × 0.85
- K sweep: K=3이 quality 기준 Pareto-optimal
- 다중 환경: 3개 중 2개 이상 competitive (≥0.85×)

## 시각화 파일
- `figures/fig1_eval_return.pdf/png` — multi-seed eval curve
- `figures/fig2_separation.pdf/png` — separation score trajectory
- `figures/fig3_k_sweep.pdf/png` — K Pareto
- `figures/fig4_multi_env.pdf/png` — multi-env bar chart

스크립트: `visualize_results.py`

## 예상 완료
2026-06-13 09:00-12:00 KST (학습 ~10-12시간)
