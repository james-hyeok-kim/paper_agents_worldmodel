---
slug: subtree-reuse-muzero
status: literature-checked
verdict: INCREMENTAL
literature-checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: D
venue-fit: [NeurIPS, ICLR, AAAI]
blacklist-delta:
  - "BL-07 (MPPI 분기 KL merge): BL-07은 TD-MPC2 MPPI(연속 action, iid 샘플)에서 분기를 KL로 merge → high-action regime에서 9.9% merge로 1.11× 실패. 본 아이디어는 (1) merge하지 않고 inter-step(환경 step 간) search tree를 carry-over해 재활용, (2) 연속 MPPI가 아니라 discrete-action MCTS(MuZero/EfficientZero)에 적용. BL-07 실패 원인(연속 action 발산으로 merge 불가)을 베이스 시스템 교체 + 재활용 축 전환(분기 간 → step 간)으로 회피."
  - "BL-01 (KV cache): transformer KV가 아니라 환경 step 사이에 살아남는 MCTS subtree의 node(latent state, visit, value, policy prior)를 carry-over해 그 노드들의 dynamics forward 재계산을 방지. learned-model search graph의 inter-step redundancy 제거."
  - "BL-08 (adaptive horizon): search depth/budget을 자르지 않음. 동일 simulation budget을 유지하되, 직전 search에서 이미 평가한(=carry-over된) 노드만큼 신규 dynamics forward를 절약."
---

# Inter-Step Subtree-Reuse MCTS: Learned-Model 탐색의 환경 Step 간 연산 재활용

## 핵심 가설
MuZero/EfficientZero는 환경 step마다 MCTS를 root부터 새로 구축하지만, 선택된 action 아래 subtree의 대부분은 다음 step에서도 그대로 유효하다(보통 폐기됨). 직전 search에서 평가된 subtree(latent state, visit count, value, policy prior)를 다음 search의 초기 트리로 carry-over하고, learned-model 특유의 신뢰도 보정(value-staleness 가중 + stochastic outcome-conditioned 키)을 적용하면, 환경 step당 신규 `recurrent_inference`(dynamics forward) 호출을 25~40% 줄여 planning을 1.4~1.7× 가속하면서 action 선택 품질(return) drop을 3% 이내로 유지한다.

## 동기 (Why Now)
**중요한 설계 정정(이전 버전 수정):** canonical MuZero는 simulation당 `recurrent_inference`를 leaf expansion에서 *한 번만* 호출하고 hidden state를 노드에 저장한다. selection은 저장된 노드를 pUCT로 재방문할 뿐 dynamics를 다시 부르지 않는다. 따라서 "intra-search (root, action-prefix) memoization"은 트리가 이미 하는 일이라 절감이 없다. 실재하는 redundancy는 **inter-step**에 있다 — 환경이 한 step 전진하면 직전 트리의 root는 선택된 자식으로 이동하고, 그 subtree(이미 dynamics forward로 평가된 수십~수백 노드)는 보통 버려진다. AlphaZero 일부 구현이 부분 reuse를 하지만, learned model에서는 (a) 시간이 지나며 노드의 value가 stale해지고, (b) stochastic transition에서 실제 outcome이 carry-over한 subtree와 어긋날 수 있어 naive reuse가 위험하다. 이 두 learned-model 특유 문제를 푸는 것이 본 아이디어다.

## 제안 방법
- **subtree carry-over:** 환경 step 후 root ← root.child[selected_action]. 그 subtree의 모든 노드(latent state h, N(visit), Q(value), P(prior))를 다음 search의 초기 트리로 보존 → 이미 평가된 노드의 `recurrent_inference` 재호출 방지. 신규 simulation은 이 트리를 확장(expand)만.
- **value-staleness 보정(learned-model 특유):** carry-over된 Q는 시간이 지나 stale → carry-over 노드의 Q에 staleness 가중 λ^(age)를 곱해 신뢰도 하향, 신규 backup이 빠르게 갱신하도록. visit count도 일부 decay(N ← γ·N)해 over-commitment 방지.
- **stochastic outcome reconciliation:** Stochastic MuZero의 chance node에서 실제 관측된 outcome이 carry-over subtree의 가정과 다르면, 어긋난 outcome 가지만 폐기하고 일치 가지만 reuse(outcome-conditioned 키). deterministic(Atari/board)에서는 전체 reuse 무손실.
- **안전:** carry-over 트리가 policy drift로 더 이상 유효하지 않으면(현재 policy prior와 저장된 P의 KL 초과) 해당 subtree 폐기 후 재구축.

```
# 환경 step 후
root = root.child[selected_action]          # subtree carry-over (재평가 없이)
for node in root.subtree:
    node.Q *= λ ** node.age                  # value-staleness 보정
    node.N  = int(γ * node.N)                # visit decay
# Stochastic: 관측 outcome과 어긋난 chance 가지 폐기
# 이후 신규 simulation은 보존된 트리를 확장 → recurrent_inference 절약
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: `recurrent_inference`(dynamics network) forward = MCTS planning hot-path의 대부분(환경 step당 N simulation × 평균 expansion depth).
- carry-over로 절약되는 forward = 선택된 자식 subtree에 보존된 노드 수 / search당 신규 expand 노드 수. N=50 simulation, 작은 branching에서 선택된 자식이 보유한 visit 비율은 흔히 25~40%(고-visit action일수록 큼).
- carry-over/decay/reconcile 오버헤드: forward의 ~1%.
- 추정 speedup: 1/(1 - 0.33) ≈ 1.5×. 작은 action space + 깊은 search(N↑)일수록 상향, board game류는 더 높음. → gate 1.5× 경계~도달. **단, 선택 자식의 retained visit fraction은 환경 의존이므로 PoC가 먼저 측정.**

## Novelty 포인트 (최소 3개)
1. (vs BL-07) 분기 간 merge(손실, 연속 action 발산으로 실패)가 아니라 step 간 무손실 subtree carry-over. 재활용 축을 분기→시간으로 전환 + discrete-action 베이스로 매칭.
2. (vs AlphaZero partial tree reuse) classical reuse는 ground-truth state 기반이라 value가 정확하지만, learned-model에서는 value-staleness 보정 + stochastic outcome reconciliation이 필요 — 이 두 learned-model 특유 메커니즘이 핵심 기여.
3. (vs transposition table) TT는 board-state 해시지만, 본 아이디어는 learned latent에 대한 step 간 subtree 보존 + visit/value transfer. Stochastic MuZero로의 outcome-conditioned 확장 포함.

## 선행 연구 위험 요소
- **AlphaZero/Leela tree reuse between moves** — 가장 가까운 위험. 차별점: learned-model value-staleness 보정 + stochastic outcome reconciliation(ground-truth state가 없는 latent model 특유).
- MuZero, EfficientZero, Stochastic MuZero, Gumbel MuZero
- Transposition tables / DAG-MCTS in classical search
- Memoization in planning, search graph reuse
- Sampled MuZero (continuous action) — 적용 경계(prefix 불일치) 명시 필요

## 예상 실험 Skeleton
- Base model: EfficientZero (Atari 100k 표준 공개구현) 또는 MuZero
- Benchmark: Atari 100k, 보조로 작은 action space board-game(재활용률↑)
- 측정: **(선행) recurrent_inference가 planning hot-path 비중**, 환경 step당 신규 dynamics forward 수, 선택 자식 retained-visit fraction, value-staleness 보정 전후 return, planning wall-clock/sec, episode return, action 선택 일치율(vs full rebuild)
- 예상 결과: 신규 dynamics forward 25~40%↓, planning 1.4~1.7×, return drop < 3%(deterministic carry-over는 staleness 보정만 정확하면 거의 무손실)

## 빠른 PoC 가능 여부
가능(2일). synthetic: 작은 discrete MDP + random dynamics net에서 N-simulation MCTS를 환경 step마다 돌리고, 선택 자식 subtree의 retained-visit fraction(=절약률)과 value-staleness 보정 유무에 따른 action 일치율을 측정. deterministic에서 carry-over 무손실성, stochastic에서 outcome reconciliation의 폐기율을 정량화. **핵심:** intra-search가 아니라 inter-step 절약이므로, 환경 step 간 트리 재활용률이 25%+인지 직접 측정. 2일.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **경계~도달 가능**. inter-step retained-visit fraction이 핵심 레버(intra-search는 절감 없음으로 정정). 작은 action space + N↑에서 30~40% 재활용이면 1.5×. 작으면 미달 위험 → PoC가 retained fraction을 먼저 측정해 go/no-go 판정.
- quality_delta < 0.05: **유망**. deterministic carry-over는 value-staleness 보정만 정확하면 거의 무손실(BL-07의 손실 merge와 대조). 위험은 staleness 보정 부정확 또는 stochastic reconciliation 폐기율↑. 보정 λ/γ 튜닝과 policy-drift 폐기로 완화.

## Venue Fit 이유
MCTS/planning 효율 + learned-model 탐색은 model-based RL 핵심 + 게임 AI 연결 → NeurIPS/AAAI. learned-model search graph 재활용 + staleness 이론은 ICLR 적합.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| 선택 자식 retained-visit fraction이 낮아 절감 미달 | 중 | PoC 선측정, 작은 action space·깊은 search 환경 우선, 적용 조건 정의 |
| value-staleness 보정 부정확으로 stale Q가 action 왜곡 | 중 | λ^age decay + 신규 backup 우선, KL-drift 폐기 |
| stochastic 모델에서 outcome 어긋나 reuse 폐기율↑ | 중 | deterministic(Atari/board)에 핵심 결과 집중, stochastic은 outcome-conditioned로 별도 |
| 메모리 오버헤드(subtree 보존) | 낮 | 선택 자식 subtree만 보존(나머지 폐기) + LRU |
