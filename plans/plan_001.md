# Plan 001 — Dual-Rate World Model 논문 확정 실험

## 목표
seed=0 + Walker-Walk의 단일 결과를 논문 수준으로 확장.
다중 seed, K sweep, 다중 환경, 시각화를 포함한 full paper result 확보.

## 가설
- Dual-Rate RSSM (K=3)이 baseline 대비 late-stage 품질 < 15% 하락으로 1.78× FLOPs 감소
- K sweep에서 Pareto trade-off 존재 (K 증가 → FLOPs↑, quality↓)
- 최소 2/3 환경에서 dual-rate ≥ baseline × 0.85 (15% 이내)
- 학습 전반에 걸쳐 separation_score > 0.5 유지

## 실험 목록

### Batch A: 다중 seed (Walker-Walk, 100k steps)
| Run ID | Config | seed | logdir |
|---|---|---|---|
| baseline_s1 | dmc_vision | 1 | baseline_seed1 |
| baseline_s2 | dmc_vision | 2 | baseline_seed2 |
| dualrate_K3_s1 | dmc_vision dual_rate | 1 | dualrate_K3_seed1 |
| dualrate_K3_s2 | dmc_vision dual_rate | 2 | dualrate_K3_seed2 |

### Batch B: K sweep (Walker-Walk, seed=0, 100k steps)
| Run ID | slow_K | logdir |
|---|---|---|
| dualrate_K2 | 2 | dualrate_K2_seed0 |
| dualrate_K4 | 4 | dualrate_K4_seed0 |
| dualrate_K6 | 6 | dualrate_K6_seed0 |

### Batch C: 다중 환경 (seed=0, K=3, 100k steps)
| Run ID | Task | logdir |
|---|---|---|
| baseline_cheetah | dmc_cheetah_run | baseline_cheetah_seed0 |
| dualrate_cheetah | dmc_cheetah_run | dualrate_cheetah_seed0 |
| baseline_hopper | dmc_hopper_hop | baseline_hopper_seed0 |
| dualrate_hopper | dmc_hopper_hop | dualrate_hopper_seed0 |

## GPU 할당
- **GPU 0** (33% util, ~62GB free): baseline_s1, baseline_s2, dualrate_K3_s1, dualrate_K3_s2, dualrate_K2, dualrate_K4
- **GPU 1** (63% util, ~59GB free): dualrate_K6, baseline_cheetah, dualrate_cheetah, baseline_hopper, dualrate_hopper

## 성공 기준 (metric × threshold)
| Metric | 기준 | 판정 |
|---|---|---|
| Late-stage eval_return (seed mean) | baseline ± 15% | GO |
| K=3 at best K | Pareto-optimal or tied | GO |
| 2/3 환경 competitive | ≥0.85× baseline | GO |
| separation_score | >0.5 throughout | 확인 (이미 0.70) |

## 실패 시 fallback
- 극단 seed outlier → 제외 후 보고 (2/3 기준)
- 환경 미달 → 추가 환경 실험 (reacher_hard, walker_run)
- K sweep 단조 아닌 경우 → 그대로 Pareto 곡선으로 보고

## 예상 시간
- 병렬 학습: ~10-12시간 (각 run ~5.5시간, 5-6 병렬)
- 시각화: 학습 완료 후 30분

## 시각화 계획
1. eval_return curves: baseline vs dualrate (3 seeds, shaded ±std)
2. FPS comparison: bar chart per condition
3. Separation score trajectory (dualrate only)
4. K sweep Pareto: FLOPs reduction vs late-stage quality

## 파일 관리
- 학습 로그: `/data/jameskimh/worldmodel/dual-rate-paper/*/metrics.jsonl`
- 시각화 스크립트: `experiments/wip/dual-rate-world-model/visualize_results.py`
- 실험 파일: `experiments/wip/dual-rate-world-model/experiment_002.md`
- 결과 파일: `experiments/wip/dual-rate-world-model/result_002.md`

## 변경 이력
- 2026-06-12 KST: 초기 작성, 11개 run 계획
