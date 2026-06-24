---
slug: action-quantized-planning-cache
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: D
venue-fit: [NeurIPS, ICLR, CoRL]
blacklist-delta:
  - "BL-07 (MPPI 분기 KL merge): BL-07은 발산하는 분기를 사후 merge하려다 실패 — high-action regime에서 분기가 빠르게 발산해 merge rate 9.9%. 본 아이디어는 merge가 아니다. MPPI의 N개 sampled action sequence가 latent×action 공간에서 충돌(near-duplicate (s,a) 쌍)할 때 transition forward를 memoize/재사용 — 사후 분기 통합이 아니라 중복 (s,a) 입력의 연산 dedup. action을 quantize해 충돌을 능동 유도."
  - "BL-12 (2-tier model routing): 모델 tier 선택이 아니다. 단일 모델을 쓰되, MPPI iteration 간 그리고 sample 간 (s_quant, a_quant) 키로 transition 결과를 cache. 모델을 바꾸는 게 아니라 같은 모델의 중복 호출을 제거."
  - "subtree-reuse-muzero(자체 pending)와의 차별: 그건 discrete-action MCTS의 step간 subtree carry-over. 본 아이디어는 continuous-action MPPI(TD-MPC2)에서 iteration 내/간 (s,a) 양자화 충돌을 cache. discrete tree가 아니라 continuous latent의 LSH/grid 양자화 기반 memoization."
---

# Action-Quantized Planning Cache: MPPI Iteration의 중복 (s,a) Transition Memoization

## 핵심 가설
TD-MPC2의 MPPI planning은 매 iteration N개 action sequence를 sample하고 각각을 latent에서 H-step rollout한다. iteration이 수렴할수록 sample들이 elite 근방으로 좁혀져 (latent, action) 입력이 서로 가까워진다. (s,a)를 양자화해 충돌하는 transition forward를 memoize/재사용하면, planning의 transition call을 1.6~2.4× 줄이면서 planning 품질(선택 action의 value)을 5% 이내로 유지한다.

## 동기 (Why Now)
TD-MPC2 planning 비용은 (MPPI iteration 수 × N samples × H horizon)번의 latent transition forward에 지배된다. 핵심 관찰: MPPI는 iteration마다 분포를 elite 쪽으로 좁히므로, 후반 iteration에서는 많은 sampled trajectory가 거의 같은 latent를 거의 같은 action으로 통과한다 — 즉 (s,a) 입력이 중복된다. BL-07은 이 중복을 "사후 분기 merge"로 잡으려다 발산 때문에 실패했다. 그러나 merge(다른 미래를 하나로 합침)가 아니라 **같은 (s,a) 입력의 forward 결과를 재사용**(deterministic transition이면 정확히 같은 출력)하면 발산과 무관하다 — 입력이 같으면 출력도 같으니까. action을 양자화하면 이 충돌을 능동적으로 유도해 cache hit를 높인다.

## 제안 방법
- **(s,a) 양자화 키:** latent s를 LSH 또는 learned coarse grid로, action a를 적응적 bin으로 양자화 → 키 `k = (hash(s), bin(a))`. transition `s' = f(s,a)`를 `cache[k]`에 저장.
- **iteration 내/간 재사용:** MPPI의 N samples를 rollout할 때, 같은 step에서 (s,a)가 같은 bin이면 transition forward 1회만 수행하고 나머지는 cache에서 읽음. 다음 iteration에서도 cache 유지(분포가 좁혀지므로 hit↑).
- **quantization-error 보정(능동, drift 안전):** cache hit가 정확한 (s,a)가 아니라 근사 bin이므로 오차 발생. (1) bin 중심과 실제 (s,a)의 거리가 τ 초과면 cache miss 처리(full forward), (2) elite trajectory(value 상위)는 항상 full forward로 정밀 평가 → 최종 선택 action의 정확도 보장. cache는 "어느 영역이 promising한가"의 coarse screening에만 사용.
- **value-aware bin 해상도:** value gradient가 큰 (s,a) 영역은 fine bin(정밀), 평탄 영역은 coarse bin(aggressive cache). 중요한 곳은 정확, 안 중요한 곳은 절감.

```
cache = {}
for iter in MPPI_iters:
   samples = sample_actions(dist)              # N action sequences
   for each sample, each step t:
      k = (lsh(s), bin(a, resolution(s)))
      if k in cache and dist(k_center,(s,a))<τ and not is_elite:
         s = cache[k]                           # 재사용, forward skip
      else:
         s = f(s,a); cache[k] = s               # full forward
   dist = refit(elite_samples)                  # elite는 항상 full forward로 평가
# 절감 = cache hit한 transition forward call
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: MPPI planning의 latent transition forward call. **TD-MPC2에서 planning이 전체 inference 비용의 압도적 부분**(N×H×iters forward) — DreamerV3 imagination(BL-11의 37%)과 달리 planning은 TD-MPC2의 핵심 hot-path. PoC가 planning vs encoder/value-head 비중 선측정.
- cache hit rate가 절감을 직접 결정. 후반 iteration에서 분포가 좁혀져 hit 40~60% 가정 → forward 0.4~0.6× → 1.6~2.4×.
- **GPU 주의(BL-10/13 교훈):** cache hit를 batch에서 indexing으로 처리해 작은 개별 forward를 피해야 함. unique (s,a) bin만 모아 single batched forward → 중복 제거가 batch 크기를 줄임(작은 call 분리가 아니라 batch 축소). 이것이 GPU에서 실질 speedup의 핵심 설계.
- **핵심:** 절감은 중복 (s,a)의 forward를 batch에서 제거(unique만 forward)하는 것 — sub-batch 분리(BL-11 0.49×)가 아니라 batch dedup(batch가 작아짐).

## Novelty 포인트 (최소 3개)
1. (vs BL-07) 발산하는 분기를 merge하는 게 아니라, MPPI sample 간 중복 (s,a) 입력의 forward를 memoize. 입력이 같으면 출력도 같으므로 발산과 무관 — BL-07 실패 원인(분기 발산)을 구조적으로 우회.
2. (vs world model planning) continuous-action MPPI에서 latent×action 양자화 기반 transition cache를 도입하는 첫 시도. TD-MPC2는 모든 sample을 독립 full forward. MPPI 수렴 특성(elite 집중)을 cache hit로 직접 활용.
3. (vs MCTS transposition table / subtree-reuse) discrete tree의 정확한 state 일치가 아니라, continuous latent의 LSH 양자화 + value-aware 해상도 + elite full-forward 보장. quantization-error를 elite 정밀평가로 bound하는 메커니즘이 novelty.

## 선행 연구 위험 요소
- MCTS transposition tables (discrete state 일치) — continuous LSH 양자화로 차별
- subtree-reuse-muzero (자체 pending) — discrete MCTS step간 vs continuous MPPI iteration간
- BL-07 (자체) — merge vs memoization 명시
- Locality-sensitive hashing for RL, episodic memory / model-based caching
- MPPI/CEM planning 효율화, amortized planning
- Speculative/approximate computation reuse

## 예상 실험 Skeleton
- Base model: TD-MPC2 (MPPI planning loop에 (s,a) cache)
- Benchmark: DMControl, MetaWorld (TD-MPC2 표준, continuous control)
- 측정: **(선행) planning vs 비-planning 비용 분해**, cache hit rate(iteration별), unique (s,a) bin 수, planning latency, 선택 action value(full 대비), episode return
- 예상 결과: transition call 40~60%↓, planning 1.6~2.4×, return drop < 5%, 후반 iteration hit↑ 관찰

## 빠른 PoC 가능 여부
가능(2~3일). 핵심 go/no-go: (1) **MPPI iteration에서 실제로 (s,a) 충돌이 양자화 후 충분히 발생하는지** — TD-MPC2 checkpoint로 후반 iteration의 (latent,action) 분포 좁힘 측정, bin 해상도 vs hit-rate vs quantization-error 곡선. (2) cache hit를 batch dedup으로 처리할 때 GPU에서 실제 speedup이 나는지(unique forward batch 축소 측정). (3) elite full-forward로 quantization-error가 최종 action 품질에 영향 없는지. 2~3일.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **cache hit rate × batch-dedup 효율이 관건**. MPPI 수렴이 충분히 (s,a)를 집중시키고 batch dedup이 GPU에서 batch를 줄이면 도달. high-action 환경에서 충돌 부족하면(BL-07과 같은 영역) hit↓ 위험 — PoC가 hit-rate 선검증.
- quality_delta < 0.05: **elite full-forward가 안전장치**. cache는 coarse screening, 최종 선택은 정밀 평가 → quantization-error가 선택 품질로 전파 안 됨. value-aware 해상도가 중요 영역 정밀 유지.

## Venue Fit 이유
latent planning의 computation reuse는 model-based control 효율의 핵심 → NeurIPS. continuous control/로봇 planning 각도 → CoRL. LSH/근사 reuse 이론 → ICLR.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| high-action 환경에서 (s,a) 충돌 부족(BL-07 영역) | 중 | value-aware 해상도, 충돌 적은 환경은 coarse bin 강화, hit-rate 선측정 |
| batch dedup이 GPU에서 indexing overhead로 이득 상쇄 | 중 | unique (s,a)만 batched forward, gather/scatter 최적화, PoC GPU 측정 |
| quantization-error가 elite 평가까지 오염 | 낮 | elite는 항상 full forward, bin 거리 τ 게이트 |
