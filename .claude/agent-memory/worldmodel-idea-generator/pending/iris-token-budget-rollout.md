---
slug: iris-token-budget-rollout
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: C
venue-fit: [NeurIPS, ICLR]
blacklist-delta:
  - "BL-09 (encoder amortize, decoder가 진짜 bottleneck): BL-09 실패 분석이 정확히 지목한 '진짜 bottleneck인 transformer forward를 직접 절감'을 IRIS에서 수행. DreamerV3 imagination은 latent-only라 적용 불가지만, IRIS는 imagination 중 token을 autoregressive transformer로 생성 → 여기서 layer depth를 token별로 조절. 베이스 시스템을 transformer가 hot-path인 IRIS로 정확히 매칭."
  - "BL-01 (KV cache 재사용): KV cache를 단순 재사용하는 게 아니라, token별로 transformer layer depth를 early-exit/router로 가변화. 쉬운 token은 얕은 layer에서 조기 종료, reward-sensitive token만 full depth. depth-level computational sparsity."
  - "BL-03 (픽셀 codec): 픽셀 압축이 아님. discrete token sequence에서 token별로 몇 개 transformer layer를 통과시킬지의 sequence-level depth budgeting."
---

# IRIS Token-Depth Rollout: Autoregressive World Model에서 Token별 Layer-Depth 예산화

## 핵심 가설
IRIS류 transformer world model의 imagination 비용은 (token 수 × transformer layer depth)에 지배되며, 한 프레임의 token 중 상당수는 얕은 layer만으로 정확히 예측된다. **expensive layer를 통과시키기 전에** 입력 단(context embedding)에서 동작하는 경량 router가 token별 필요 depth를 결정해 쉬운 token을 조기 종료(early-exit)시키면, 평균 transformer FLOPs를 45~60% 줄여 imagination을 1.6~2.1× 가속하면서 reward/return 예측 drop을 5% 이내로 유지한다.

## 동기 (Why Now)
BL-09의 핵심 교훈은 "진짜 bottleneck을 쳐라"였다. DreamerV3 imagination은 latent-only로 transformer/decoder를 거의 호출하지 않아 절감 기회가 없다. IRIS는 다르다 — imagination이 token-by-token autoregressive 생성이고, 한 step마다 K개 token 각각이 L개 transformer layer를 전부 통과한다. 그래서 절감 기회가 실재하는 곳은 IRIS다. **중요한 설계 제약(이전 버전 수정):** token의 예측 entropy는 full forward의 *출력*이므로 "entropy 보고 skip"은 이미 비용을 다 치른 뒤라 절감이 없다. 따라서 결정은 반드시 expensive 연산 *이전*에 내려야 한다 → 입력단 router 또는 얕은 layer 기반 early-exit으로 설계한다.

## 제안 방법 (Mixture-of-Depths / early-exit, 결정이 expensive 연산을 선행)
- **입력단 router** `ρ(context_embed, pos) -> depth ∈ {m_low, m_mid, L}`: token이 expensive layer를 통과하기 *전에*, context embedding과 position만으로 필요 depth를 예측. router는 작은 MLP(full layer의 ~2%).
- **early-exit 보강:** router가 mid를 선택한 token은 layer m까지 실행 후 provisional head로 확신도 확인, 충분하면 m+1..L skip. (router 오분류에 대한 2차 안전망)
- **reward-sensitivity 마스크:** reward head gradient saliency가 큰 token은 router와 무관하게 항상 full depth L. high-stakes token 품질 보장.
- **학습:** router는 "이 token을 depth d로 예측했을 때 full-depth 대비 token/reward 오차가 임계 이하인가"의 라벨로 학습(full-depth teacher의 token logits와 비교, self-supervised). depth budget loss로 평균 depth↓ 압력.
- **drift 안전:** frame 경계마다 reward head residual 확인, 누적 오차 크면 다음 frame budget 상향.

```
for t in rollout:
  for token k in frame:
     d_k = router(context_embed, pos_k)        # expensive layer 이전, ~2% 비용
     if k ∈ reward-sensitive: d_k = L
     h = run_layers(1..d_k)                     # 결정된 depth만 실행 (절감 = skip된 layer)
     if d_k < L and confident(provisional_head(h)): pass   # early-exit 확정
     tok_k = head(h)
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: transformer layer 연산 = IRIS imagination의 압도적 hot-path(token 수 × L layer). **이 비율(≥70% 가정)은 PoC가 먼저 측정해야 하는 전제** — IRIS checkpoint FLOP 분해로 tokenizer/head 대비 transformer 비중 선검증.
- router가 token의 평균 depth를 L → ~0.45L로 낮춤(쉬운 token이 다수): layer 연산 ~0.45×.
- router 오버헤드: full forward의 ~2%.
- 추정 speedup: 1/(0.45 + 0.02) ≈ 2.1× on transformer, hot-path 70% 가정 시 전체 1/(0.3 + 0.7×0.47) ≈ 1.5~1.7×. → gate 1.5× 도달. 평균 depth 0.4L까지 낮추면 여유 확보.
- **핵심:** 절감은 router 결정 *이후* 실제로 skip하는 layer에서 나온다(entropy 사후판단 아님) → Amdahl 수치가 실재.

## Novelty 포인트 (최소 3개)
1. (vs world model) autoregressive WM에서 token별 layer-depth를 reward-sensitivity로 가이드하는 input-side router — IRIS/TWM은 모든 token을 full depth로 생성.
2. (vs LLM early-exit / Mixture-of-Depths) 일반 MoD는 정확도/perplexity 보존이 목표지만, 본 아이디어는 reward/dynamics 관련성 기준으로 depth를 배분하고 reward-sensitive token에 full depth를 강제 — world-model objective(return 예측)에 종속된 depth budgeting. RL imagination 환경의 token redundancy를 직접 활용.
3. (vs BL-09) bottleneck을 정확히 매칭(transformer가 hot-path인 IRIS) + 결정이 expensive 연산을 선행(proxy 비용 함정 회피) + reward-sensitive 마스크로 품질 안전장치 내장.

## 선행 연구 위험 요소
- **Mixture-of-Depths (Raposo 2024), early-exit transformers (CALM, DeeBERT), Adaptive Computation Time** — 가장 가까운 위험. 차별점: depth 배분이 reward-sensitivity/return-objective에 종속, RL imagination 환경 한정.
- Speculative decoding (정확도 무손실 목표라 다름)
- IRIS, TWM, STORM token-based world models
- Token pruning / merging in ViT (ToMe), patch-importance / saliency
- Layer-skip in autoregressive video/world models

## 예상 실험 Skeleton
- Base model: IRIS (transformer WM에 input-side router + early-exit head 추가), 비교용 TWM
- Benchmark: Atari 100k (IRIS 표준)
- 측정: **(선행) transformer 비중 FLOP 분해**, token당 평균 depth, transformer FLOPs, imagination steps/sec, reward 예측 정확도, episode return, FVD of imagined frames
- 예상 결과: layer 연산 45~60%↓, imagination 1.6~2.1× 가속, return drop < 5%

## 빠른 PoC 가능 여부
가능(2~3일). synthetic: 정적 배경+움직이는 패치 toy token grid에 작은 L-layer transformer를 학습시키고, input-side router가 쉬운 token을 얕은 depth로 보내도 reward-relevant region 예측이 보존되는지, 평균 depth vs 예측오차 곡선. **핵심 PoC 측정:** (1) IRIS checkpoint에서 transformer가 hot-path 비중 ≥ 가정인지 FLOP 분해, (2) router 결정이 forward 이전에 가능해 실제 layer skip이 일어나는지(entropy 사후판단 아님 검증). 2~3일.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **도달 가능하나 hot-path 비율이 전제(PoC 선측정)**. 평균 depth 0.45L 달성 + transformer 비중 70%면 1.5~1.7×. IRIS는 transformer가 압도적이라 BL-09(encoder가 작은 부분)와 달리 ceiling 높음. **이전 버전의 entropy-after-forward 함정을 input-side router로 해소해 절감이 실재.**
- quality_delta < 0.05: **중간 위험**. reward-sensitivity 마스킹 + early-exit 확신도 체크 + frame-level correction이 관건. router 오분류 시 drift 가능 → 2차 안전망으로 완화.

## Venue Fit 이유
autoregressive WM compute는 token-WM 효율의 최전선 → NeurIPS. transformer adaptive computation 각도는 ICLR 적합.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| 얕은 depth token의 누적 drift로 long rollout 붕괴 | 중 | early-exit 확신도 체크 + frame-level reward self-correction, budget 동적 상향 |
| router가 reward-중요 token을 얕은 depth로 오분류 | 중 | reward-head gradient saliency로 항상-full 마스크 |
| hot-path 비율이 가정보다 낮아 ceiling 하락 | 중 | IRIS(transformer-dominated)에 한정, FLOP 분해 선검증(PoC 1단계) |
