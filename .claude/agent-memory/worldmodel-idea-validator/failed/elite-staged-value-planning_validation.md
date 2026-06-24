---
slug: elite-staged-value-planning
verdict: FAIL
date: 2026-06-19 KST
validated-by: team-lead (직접, advisor-ordered microbenchmark die-point)
base: TD-MPC2 · gate: planning wall-clock microbenchmark
---

# PoC Validation — elite-staged-value-planning → FAIL (microbenchmark die-point)

## 요약
advisor 지시로 **가장 싼 die-point(wall-clock 마이크로벤치)를 맨 먼저** 실행 — 2-stage 구현/elite-recall/학습 전에. 결과 FAIL이라 후속 작업 전부 절약.

## 결과 (실물 TD-MPC2, walker-walk, 512 samples/6 iter/H3/64 elites)
proposed = cheap-rank(distilled scalar MLP, **best case**) on 512 → full qnet on 64 elites. elite는 timing용 random stub. baseline = qnet on all 512.

| device | baseline ms/step | 2-stage ms/step | speedup |
|---|---|---|---|
| GPU (B200) | 16.94 | 18.02 | **0.94× (느려짐)** |
| CPU | 4686.7 | 4636.5 | **1.011× (flat)** |

gate(≥1.2×) **양 device 모두 미달** → FAIL. edge/CPU pivot도 불가(CPU도 flat).

## 왜 FAIL (메커니즘 교훈)
- **terminal Q-ensemble은 planning wall-clock bottleneck이 아니다.** TD-MPC2 Q는 vmap vectorized 단일 op(`qnet(z)`가 5-head 한 번에) → 512→64 sample 축소가 kernel-launch-bound라 GPU에서 무이득, 추가한 cheap-rank launch로 오히려 0.94×.
- gen-r4의 "terminal ~50%"는 **forward-count(FLOPs) 기준엔 맞지만 wall-clock엔 틀림**. wall-clock은 dynamics term(horizon-sequential, 512)이 지배. CPU(FLOP-bound)서도 terminal 축소가 1.1%뿐 → terminal이 작음.
- 코드 추가 발견: planning Q value는 `return_type='avg'` = 5-head를 vmap으로 계산 후 **2개만 subsample 평균**. value엔 2-head만 쓰임(compute는 vectorized 5-head).

## 교훈 (재사용)
"vectorized ensemble op의 sample 수 축소"는 GPU wall-clock 이득 없음(dual-rate 0.926× 패턴 재확인). planning efficiency는 **iterations 또는 dynamics rollout(horizon-sequential)** 을 쳐야 wall-clock 이득 가능 → [[adaptive-mppi-budget]]이 맞는 bottleneck(단 novelty thin).

artifacts: `experiments/wip/elite-staged-value-planning/{microbench.py, microbench_results.json, microbench.log}`
