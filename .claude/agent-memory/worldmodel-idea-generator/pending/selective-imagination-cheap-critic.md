---
slug: selective-imagination-cheap-critic
status: literature-checked
verdict: INCREMENTAL
created: 2026-06-11 KST
category: C
venue-fit: [NeurIPS, ICLR]
blacklist-delta:
  - "BL-02 (fixed early-stopping): step 수 임계가 아니라, 매 branch 시작점에서 latent-only critic이 value-variance(정보 이득 proxy)를 추정해 '어느 trajectory를 full로 imagine할지'를 공간적으로 선택. 시간축 truncation이 아니라 trajectory-space 선택."
  - "BL-08 (per-step learned stopping controller): BL-08은 매 step continue/stop 이진 결정 → reward-cliff에서 실패. 본 아이디어는 stop 결정을 하지 않는다. 모든 선택된 trajectory는 full horizon을 끝까지 imagine하므로 truncation drift/cliff 문제가 구조적으로 없음. 대신 '몇 개를' imagine할지를 batch 차원에서 조절."
  - "BL-07 (branch KL merge): merge로 forward를 합치지 않음(발산 환경에서 실패). 발산하는 branch는 그대로 두되, 시작점에서 low-information branch 자체를 launch하지 않음 → 발산 여부와 무관."
---

# Selective Imagination via a Cheap Latent-Only Information Critic

## 핵심 가설
DreamerV3 imagination의 비용은 batch×horizon RSSM forward에 비례하지만, policy gradient에 기여하는 정보의 대부분은 소수의 high-value-variance 시작 state에서 나온다. 매 시작 state에 대해 latent-only로 value-variance를 추정하는 초경량 critic(RSSM forward의 ~5% 비용)으로 imagination budget을 상위 정보 trajectory에 집중 배분하면, imagination forward를 40~55% 줄여 model-based update를 1.5~1.9× 가속하면서 policy return drop을 4% 이내로 유지한다.

## 동기 (Why Now)
DreamerV3는 replay에서 뽑은 모든 state를 동일하게 H-step imagine한다. 그러나 학습 후반에는 다수 state가 이미 수렴된 영역(value 평탄, advantage≈0)이라 imagination이 거의 정보를 주지 않는다. BL-08의 교훈: 시간축에서 자르면 reward-cliff에 당한다. 따라서 자르지 말고, 처음부터 정보가 적은 trajectory를 시작하지 않는 trajectory-space sparsity로 전환한다. 이는 world-model imagination의 batch 차원 redundancy를 직접 겨냥한다.

## 제안 방법
- 시작 state 집합 {s_i} (replay batch B개)에 대해, 경량 critic `g(s_i) -> v̂_var`가 latent-only로 value-variance를 추정. g는 value head ensemble의 disagreement 또는 stochastic prior에서 K개 z 샘플의 value 분산을 1-step만 펼쳐 추정(full H-step rollout 불필요).
- budget M < B를 v̂_var 상위로 배분(top-M + 탐색용 소량 random). 선택된 state만 full H-step imagine.
- 미선택 state는 imagine 안 하고, 그 자리의 actor-critic gradient는 (a) skip 또는 (b) 직전 step의 1-step bootstrap value로 저비용 대체.
- critic g는 실제 imagined return과의 회귀로 online 학습(self-supervised, 추가 label 불필요).

```
for batch s_1..s_B:
    score_i = g(s_i)            # latent-only, K z-samples, 1-step value spread (~5% of full rollout)
select M = top-k(score) ∪ ε-random
for i in select:
    imagine full H-step          # 비용 집중
update actor-critic on imagined(select) (+ 1-step bootstrap for the rest)
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: imagination forward = DreamerV3 model-based update의 hot-path 대부분(actor-critic step은 H×B RSSM prior + heads). budget을 B→M=0.45B로 줄이면 imagination 비용 ~0.45×.
- critic 오버헤드: 1-step×K(=3~5) z-spread ≈ full H(=15)-step의 5~8%.
- 추정 speedup: 1/(0.45 + 0.07) ≈ 1.9× (imagination-dominated 가정). 보수적으로 actor-critic 외 고정비 포함 시 1.5~1.7×. → gate 1.5× 도달 가능 구간.

## Novelty 포인트 (최소 3개)
1. (vs world model) imagination을 시간축이 아니라 trajectory(batch)축에서 sparsify. DreamerV3/TD-MPC2는 모든 시작 state를 균일 imagine.
2. (vs active learning / prioritized replay) prioritized replay는 어떤 transition을 학습할지 고르지만, 본 아이디어는 어떤 latent state를 imagine할지를 latent-only proxy로 고름 — imagination 연산 자체를 절감(replay는 안 함).
3. (vs BL-08) stop 결정이 없어 cliff-free. 선택된 trajectory는 항상 full horizon → value bootstrap 보장. imbalance label 문제도 없음(critic은 회귀, 이진 분류 아님).

## 선행 연구 위험 요소
- Prioritized Experience Replay, prioritized model rollouts (MBPO 변형)
- Value-equivalence / value-aware sampling
- Active inference, expected information gain in planning
- Uncertainty-driven exploration (ensemble disagreement)
- "Which experiences to imagine" 류 model-based RL 논문

## 예상 실험 Skeleton
- Base model: DreamerV3 (imagination 루프 fork)
- Benchmark: DMControl proprio + visual, Atari 100k
- 측정: imagination forward 호출 수, critic 오버헤드, model update steps/sec, episode return, critic 추정-실제 return 상관
- 예상 결과: imagination 40~55%↓, update 1.5~1.9× 가속, return drop < 4%

## 빠른 PoC 가능 여부
가능(1~2일). synthetic: toy MDP에서 value-variance가 큰 state만 imagine해도 policy gradient norm/return이 유지되는지, latent-only 1-step proxy가 full H-step return-variance와 상관(>0.7)되는지 측정. critic 오버헤드 대비 절감 곡선까지 2일 내.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **도달 가능**. **precondition:** imagination이 model-based update의 hot-path 대부분이라는 가정(≥70%)을 PoC가 먼저 측정. 1.5~1.9×는 imagination-dominated 환경에서. actor-critic 외 고정비가 크면 ceiling 하락.
- quality_delta < 0.05: **핵심은 per-update 품질이 아니라 sample efficiency**. trajectory 55%를 drop하면 그 actor-critic gradient도 사라지므로, validator가 잡을 quality metric은 **고정 env-step에서의 return(=sample efficiency)**이지 단순 reconstruction이 아님. 1-step bootstrap fallback이 gradient 손실을 보완하는지가 go/no-go. cliff-free 구조 + 회귀 critic이 BL-08 실패 원인을 회피하지만, 진짜 검증점은 (a) latent-only proxy 상관 > 0.7, (b) 동일 env-step에서 return drop < 5%.

## Venue Fit 이유
imagination compute allocation은 sample/compute efficiency 핵심 주제 → NeurIPS. proxy-critic + 학습 곡선 분석은 ICLR 적합.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| latent-only proxy가 full return-variance를 못 잡음 | 중 | K z-samples ensemble + value-head disagreement 병용, 상관 모니터 |
| 미선택 state gradient 누락이 exploration 저해 | 중 | ε-random 선택 + 1-step bootstrap fallback |
| imagination이 hot-path가 아닌 환경에서 ceiling 하락 | 중 | imagination-dominated 환경(visual DreamerV3)에 적용 한정, FLOP 분해 보고 |
