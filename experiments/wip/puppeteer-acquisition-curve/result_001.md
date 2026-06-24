# Result 001 — Puppeteer Acquisition Curve (R8 bet)

## 판정: PASS ✓ — 비자명(non-obvious) 결과 (2026-06-24 KST)

## 수치 결과

### Learning Curve (eval_return vs steps)

| steps | CondA s1 | CondA s2 | CondB s1 | CondB s2 |
|-------|----------|----------|----------|----------|
| 0 | 5.38 | 0.08 | 2.50 | 1.55 |
| 20k | 9.30 | 7.36 | 2.11 | 2.83 |
| 40k | 10.35 | 9.80 | 2.34 | 1.25 |
| 60k | 12.55 | 8.05 | 2.24 | 1.93 |
| 80k | 11.63 | 11.33 | 3.17 | 1.47 |
| 100k | 12.29 | 12.12 | 1.55 | 2.50 |
| 120k | 13.97 | 12.93 | 1.99 | 2.72 |
| 140k | 11.72 | 10.15 | 2.70 | 1.74 |
| 160k | 18.64 | 16.86 | 2.82 | 1.47 |
| 180k | 15.53 | 14.73 | 2.40 | 1.78 |

### Gate Metrics

| Metric | CondA (trained tracker) | CondB (random tracker) | 비율 |
|--------|------------------------|------------------------|------|
| **AUC_100k** | 953,554 | 213,920 | **4.46×** |
| **Final return (180k)** | 15.13 | 2.09 | **7.25×** |
| **T_50** | ~20–40k steps | >180k steps | **>4.5× 빠름** |

### Gate 판정
- **기준**: AUC_100k(A) ≥ 2× AUC_100k(B)
- **결과**: **4.46× ≥ 2× → PASS ✓**

## 해석

**발견: tracker competence가 high-level 학습에 결정적**

- Condition A (훈련된 tracker): 20k~40k steps 이내에 return ≥ 7.5 도달 (T_50 = 20–40k)
- Condition B (random tracker): 180k steps 내내 return 1.5–3.2 범위에서 정체 (T_50 > 180k)
- **Condition B는 전혀 학습되지 않음**: 180k steps 전 구간에서 return이 초기값(~2.0)과 동일, 학습 곡선이 flat

**이 결과의 비자명성**:
1. "frozen tracker만 있으면 high-level이 쉽게 학습된다"는 게 자명하지 않음 — random tracker로도 같은 action space(15-dim appendage delta)를 제공하지만 전혀 학습 불가
2. Tracker의 역할이 단순한 "action abstraction"이 아니라 "물리적으로 의미있는 명령 해석"임을 시사
3. 고정된 유능한 low-level이 주어져야만 hierarchy가 sample efficiency를 실제로 제공

**한계 요인 (what limits acquisition)**:
- CondA에서도 return이 noisy하고 volatile (140k에서 drop, 180k에서 다시 하락)
- 최고점(160k)에서 plateau 안 됨 → 200k를 훨씬 넘기거나, MPC planning이 한계일 수 있음
- Tracker quality가 NECESSARY 조건이지만 sufficient 조건은 아님

## 실험 설정

| 항목 | 값 |
|------|-----|
| Base | Puppeteer (TD-MPC2 hierarchical) |
| Task | corridor (RunThroughCorridor, target_velocity=6.0) |
| Steps | 200k (eval 20k마다) |
| Seeds | 2 (seed=1, seed=2) |
| CondA tracker | tracking.pt (10M steps 훈련) |
| CondB tracker | random init (동일 아키텍처, 미훈련) |
| GPU | NVIDIA B200 × 2 (병렬) |
| 실행 시간 | CondA: ~08:00h, CondB: ~07:30h |

## 다음 액션

1. **validator 통과 여부**: Gate PASS → worldmodel-idea-validator에 정식 제출
2. **확장 실험**: 더 어려운 task (gaps-corridor, hurdles-corridor)에서도 동일 효과 확인
3. **ceiling 비교**: corridor-1/2/3.pt (10M steps)와의 gap → 200k만으로 얼마나 도달?
4. **mechanistic 분석**: tracker quality가 왜 필수인가? action distribution, tracking error 분석

## 파일
- `experiment_001.md` — 실험 설계
- `result_001.md` — 이 파일
- `analyze_results.py` — 분석 스크립트
- Eval data: `puppeteer/logs/corridor/{seed}/{cond}/eval.csv`
