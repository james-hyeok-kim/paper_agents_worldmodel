# Experiment 003 — Dual-Rate Ratio/K 재조정 (사전등록 kill-test)

## 메타데이터
- 시작: 2026-06-21 KST
- 상태: RUNNING
- 기반: experiment_002 결과 = Phase 1 FAIL (size-matched 대조 시 dual-rate 49.6%)

## 배경 (experiment_002 진단)
- 원래 seed=0 +12.4%는 **confound** (dual-rate deter=384 vs baseline deter=512).
- size-matched baseline_deter384 추가 후: dual-rate 375 vs baseline_deter384 755 (85k eval).
- dual-rate 최고 seed(465) < deter384 최저 seed(729), 구간 겹치지 않음 → robust FAIL.
- **penalty 진단**: ib_penalty/sm_penalty ~0.0003 (거의 0). lambda 조정 무의미.
- **원인**: deter_slow=256(전체의 67%)가 K=3에서 2/3 시간 freeze → dynamics 추적 손상.

## 가설
ratio를 바꿔 frozen mass = deter_slow×(1−1/K)를 줄이면 dynamics 손상이 완화되어
deter384 수준 품질을 회복할 수 있다.

## 사전등록 기준 (pre-registered, 변경 금지)
- **통과**: 최고 config가 85k eval_return에서 **deter384(~755) 도달/초과** (2-seed mean)
- **무승부 = 실패**: wall-clock 이미 0.926×(느림) → quality에서 *이겨야* 함
- **kill**: 최고 config가 deter384 못 넘으면 → dual-rate 아이디어 死, MWM/STORM 무의미

## 실험 구성
| Config | deter_slow | deter_fast | K | frozen mass | seeds | logdir |
|---|---|---|---|---|---|---|
| A | 128 | 256 | 3 | 85 | 0, 1 | dualrate_A_s0/s1 |
| B | 128 | 256 | 2 | 64 | 0, 1 | dualrate_B_s0/s1 |

- task: dmc_walker_walk, 100k steps
- 대조군: baseline_deter384 (seed0=781, seed1=729, mean=755 @ 85k) — experiment_002에서 확보
- nice=15, CPU 부하 절반 이하 유지

## 측정 지표
- eval_return @ 85k (primary, 2-seed mean) — deter384와 비교
- separation_score (collapse 확인)

## 비교 기준값 (experiment_002, 85k eval_return)
| 조건 | seed별 | mean |
|---|---|---|
| baseline_deter384 (deter=384, vanilla) | 781.7 / 729.7 | **755.7** ← 넘어야 할 바 |
| dualrate_K3 (256/128, K=3) | 284.5 / 465.6 | 375.1 (기존 FAIL) |
| baseline_512 (deter=512, vanilla) | 251.2 / 653.8 | 452.5 (noise, 무시) |

## 예상 완료
~1.5-2시간 (4 run 병렬, niced)
