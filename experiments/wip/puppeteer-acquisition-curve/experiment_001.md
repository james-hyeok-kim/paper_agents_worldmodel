# Experiment 001 — Puppeteer Acquisition Curve (R8)

## 가설
고정된 유능한 low-level tracker(tracking.pt) 위에서 high-level policy가 random-init tracker 대비 corridor task를 더 빠르게 획득한다.

## 조건

| 조건 | 설명 | low_level_fp |
|------|------|--------------|
| **A** (competent) | 훈련된 tracking.pt (frozen) | `checkpoints/tracking.pt` |
| **B** (random) | 동일 아키텍처 random-init tracker (frozen) | `experiments/wip/puppeteer-acquisition-curve/random_tracker.pt` |

- Task: `corridor` (RunThroughCorridor, target_velocity=6.0)
- Steps: 200,000 per run
- Seeds: 1, 2
- Eval: every 20,000 steps (eval_episodes=5)
- High-level action space: 15-dim (appendage delta)
- Episode length: 500 steps / seed_steps: 2,500

## 성공 기준 (Gate)
- **PASS (비자명)**: A AUC_100k ≥ 2× B AUC_100k
- **FAIL (자명)**: A ≈ B → tracker quality 무관 → 실험 의미 없음

## 측정 metric
- `eval_return` vs steps (20k interval) — eval.csv에서 추출
- `AUC_100k`: 0~100k step 구간 return 적분 (trapezoid rule)
- `T_50`: Condition A final return의 50% 도달 step

## Ceiling 참조
- corridor-1.pt, corridor-2.pt, corridor-3.pt: 10M step 훈련 완료 모델

## 인프라
- GPU: NVIDIA B200 × 2 (CUDA 13.0, PyTorch 2.11.0+cu130)
- 코드: `/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer/`
- NumPy 2.0 호환 패치: `dm_control_wrapper.py` np.infty → np.inf
- 패키지 추가 설치: hydra-core, omegaconf, torchrl, stable-baselines3

## 실행 스케줄
1. Sanity check (3100 steps) — 완료 2026-06-23 KST (eval.csv 확인됨)
2. Condition A seed=1,2 병렬 (GPU 0, 1) — 시작: 2026-06-23 05:00 KST
3. Condition B seed=1,2 병렬 (GPU 0, 1) — A 완료 후 즉시 시작
4. 분석 및 result_001.md 작성

## 파일 목록
- `experiment_001.md` — 이 파일
- `result_001.md` — 실험 완료 후 작성
- `make_random_tracker.py` — Condition B random tracker 생성
- `random_tracker.pt` — Condition B low-level policy
- `run_sanity.sh`, `sanity_condA.log`
- `run_condA_s1.sh`, `condA_s1.log`
- `run_condA_s2.sh`, `condA_s2.log`
- `run_condB_s1.sh`, `condB_s1.log`
- `run_condB_s2.sh`, `condB_s2.log`
