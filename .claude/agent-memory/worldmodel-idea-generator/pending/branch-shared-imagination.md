---
slug: branch-shared-imagination
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: D
venue-fit: [ICLR, NeurIPS, CoRL]
blacklist-delta:
  - "BL-01 (단순 KV cache 재사용): BL-01은 '시간축 내' KV 재사용(이전 토큰 캐싱)이다. 본 아이디어는 '병렬 imagination 분기 간' 공유다 — 같은 시점에서 여러 action으로 갈라지는 분기들이 공유하는 latent prefix와 reward/value 연산을 묶어 1회만 계산한다. 축이 시간(temporal)이 아니라 분기(branch/batch)."
  - "BL-01과 달리 stochastic transition을 다룬다: 분기들이 같은 stochastic z 샘플을 공유할지 갈라질지를 KL 기반으로 결정하는 selective branching이 핵심. 동일 z prefix는 1회 계산, divergence 임계 초과 시에만 분기 복제."
  - "MPPI/MCTS의 trajectory 집합에서 공통 prefix subtree를 latent-level로 dedup해 world model forward 횟수를 줄이는 점이 reward-conditioned로 동작 (같은 reward 궤적이면 공유 유지)."
---

# Branch-Shared Imagination: KL 기반 동적 Merge Tree로 병렬 분기 연산 공유

## 핵심 가설
TD-MPC2의 MPPI planning과 DreamerV3의 multi-trajectory imagination은 같은 초기 state에서 갈라지는 수백 개 분기를 독립적으로 forward한다. 분기 간 latent state divergence(KL)가 작은 prefix 구간을 식별해 world model forward를 1회로 공유하면, planning/imagination forward 횟수를 2~3× 줄여 planning latency를 1.8~2.5× 단축하면서 control 성능 drop을 3% 이내로 유지한다.

## 동기 (Why Now)
TD-MPC2는 매 결정마다 수백 개 candidate trajectory를 latent에서 rollout한다(MPPI). 이 분기들은 초기 몇 step 동안 거의 동일한 latent를 지나다가 나중에 갈라진다(action 차이가 latent에 누적되기까지 시간 지연). 그러나 현재는 모든 분기가 매 step 독립적으로 RSSM/latent dynamics forward를 호출 → 동일 연산의 대량 중복. 이 중복은 stochastic이라 단순 dedup이 안 되고, "언제 갈라지는가"를 latent divergence로 판단해야 한다 — world-model-specific 문제다.

## 제안 방법
- planning/imagination 분기를 트리로 관리. 노드 = (h, z, 누적 action prefix).
- 매 step 후 분기 쌍의 latent divergence `KL(z_i || z_j)` (또는 h L2)를 측정, 임계 이하 분기들을 **하나의 대표 노드로 merge** → 다음 forward를 1회만.
- divergence가 임계 초과하면 분기 split(연산 분리). reward/value head도 merge된 노드에선 1회 평가 후 broadcast.
- merge/split을 union-find로 관리, GPU에서는 동적 batching으로 구현(merge된 분기는 batch에서 제거).
- 정확도 보존: merge는 근사이므로, planning 종료 시점 가까운 step(영향 큰 구간)에서는 merge 임계를 낮춰 split 우선.

```
nodes = [root]*N
for t in horizon:
    unique = dedup_by_divergence(nodes, tau_t)   # tau_t는 t에 따라 감소
    latents = wm.forward(unique, actions)         # forward = |unique| << N
    nodes = broadcast(latents, mapping)
    if diverged(nodes): split()
```

## Novelty 포인트 (최소 3개)
1. (vs world model) imagination/planning을 trajectory 집합이 아닌 동적 merge tree로 보고, world model forward 횟수를 분기 수가 아닌 unique latent 수로 만든다.
2. (vs KV cache/BL-01) 시간축이 아니라 분기축 공유. stochastic transition에서 "언제 분기가 실제로 갈라지는가"를 KL로 판단하는 selective branching이 핵심 — 결정론 모델엔 trivial하지만 stochastic WM에선 비자명.
3. (vs MCTS dedup) reward/value head 평가까지 merge 노드에서 1회 공유 → reward-conditioned dedup. step별 가변 merge 임계로 정확도-속도 trade-off를 명시적 제어.

## 선행 연구 위험 요소
- MCTS transposition tables, EfficientZero의 reanalyze
- Beam search state merging, hypothesis recombination (NLP)
- TD-MPC2 MPPI 구현, Sampled MuZero
- Trajectory clustering for planning

## 예상 실험 Skeleton
- Base model: TD-MPC2 (planning latency 명확), 보조로 DreamerV3 imagination batch
- Benchmark: DMControl (humanoid, dog — 고차원, 분기 많음), MetaWorld
- 측정: planning latency (ms/decision), world model forward 호출 수, merge 비율, episode return, success rate
- 예상 결과: forward 2~3× 감소, planning 1.8~2.5× 단축, return/success drop < 3%

## 예상 Contribution
- planning/imagination을 dynamic merge tree로 재구성하는 일반 프레임워크 (TD-MPC2, DreamerV3 양쪽 적용)
- stochastic latent에서의 selective branching 기준(KL 임계 스케줄) 제시

## 빠른 PoC 가능 여부
가능. synthetic: 같은 state에서 갈라지는 toy 분기들의 latent divergence가 실제로 지연되어 발생하는지(=공유 가능 prefix 길이) 측정 1~2일. Full: TD-MPC2 fork에 merge tree 삽입, 단일 DMControl task 4주 이내.

## Venue Fit 이유
planning latency는 robotics 실시간성과 직결 → CoRL 매력적. 동적 merge tree라는 알고리즘 novelty + latency/quality 곡선은 ICLR/NeurIPS에도 적합.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| dynamic batching 구현이 GPU에서 비효율(불규칙 텐서) | 높 | 고정 bucket 기반 merge(분기를 K개 클러스터로 quantize), padding 최소화 |
| merge 근사가 planning 품질 해침 | 중 | terminal 근처 step은 split 우선(가변 임계), ablation으로 임계 민감도 |
| 분기가 빨리 갈라지는 환경(chaotic)에서 이득 없음 | 중 | 적용 가능 조건(분기 지연이 큰 환경) 분석을 contribution으로 |
