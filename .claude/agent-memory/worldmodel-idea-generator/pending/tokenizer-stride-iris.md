---
slug: tokenizer-stride-iris
status: literature-checked
verdict: NO-GO
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: G
venue-fit: [NeurIPS, ICLR]
blacklist-delta:
  - "BL-13 (transformer layer depth budgeting): BL-13 실패 분석이 정확히 지목 — IRIS의 진짜 bottleneck은 transformer가 아니라 tokenizer(VQVAE)가 76%. 본 아이디어는 그 76%를 직접 친다. transformer depth가 아니라 tokenizer call 빈도를 절감. BL-13이 redirect로 명시한 'tokenizer 자체를 절감'을 정면 수행."
  - "BL-09 (encoder amortize, 에피소드당 1회): episode당 1회가 아니라 K-frame stride로 VQVAE encoder를 호출하고, 사이 frame은 transformer가 예측한 latent token을 decode-skip한 채 직접 사용. BL-09는 static/dynamic 분리로 encoder만 절감(decoder가 진짜 bottleneck이라 희석)했지만, IRIS에서는 encoder+decoder(tokenizer)가 둘 다 76%의 큰 덩어리라 stride skip이 둘 다 절감 → 희석 없음."
  - "BL-03 (픽셀 codec): 픽셀 압축이 아니라, VQVAE encoding/decoding call 자체를 frame stride로 skip. imagination 중에는 token-space에서만 rollout하고 픽셀 재구성을 K frame마다만 수행."
---

# Tokenizer-Stride IRIS: VQVAE Call을 Frame-Stride로 Skip해 진짜 Bottleneck 절감

## 핵심 가설
IRIS 비용의 76%는 transformer가 아니라 VQVAE tokenizer(CNN encoder/decoder)에 있다(BL-13 측정). IRIS imagination은 이미 token-space autoregressive라 encoder는 1회뿐이므로 절감 레버는 **decoder**다 — actor/reward/value가 매 frame 소비하는 decoder call을 K frame마다 1회로 줄이고 actor/reward/value를 token-space head로 옮기면, end-to-end imagination을 1.4~1.7× 가속하면서 return drop을 5% 이내로 유지한다. (decoder가 imagination 비용 ≥45%일 때 gate 1.5× 도달.)

## 동기 (Why Now)
BL-13의 교훈은 명확했다: IRIS에서 transformer를 최적화해봐야 Amdahl headroom 24%뿐, tokenizer가 76%. 그런데 IRIS imagination loop를 보면 tokenizer가 매 frame 호출될 필요가 없다 — imagination은 본질적으로 token-space의 autoregressive 예측이고, transformer가 다음 frame의 token을 직접 생성한다. 픽셀로 decode했다가 다시 encode하는 round-trip은 (a) reward head가 픽셀을 보거나 (b) 사람이 시각화할 때만 필요. 순수 token rollout이라면 tokenizer를 frame마다 부를 이유가 없다. 이 round-trip을 frame-stride로 줄이면 76%의 hot-path를 직접 절감한다.

## 제안 방법
- **imagination 중 token-space rollout:** transformer가 next-frame token을 직접 예측 → 다음 step의 transformer 입력으로 그 token을 그대로 사용(픽셀 decode-encode round-trip 생략). 원래 IRIS도 이 부분은 token-space일 수 있으나, **reward/value head가 픽셀 또는 decode된 feature를 입력으로 받는 구현에서는 매 step decode가 발생** — 이를 token-space head로 우회.
- **token-space reward/value head:** reward/value를 픽셀이 아닌 token embedding에서 직접 예측하는 head를 distillation으로 학습(픽셀 head를 teacher로). imagination 중 decoder call 제거.
- **K-frame stride decode(검증/품질 anchor용):** K frame마다 1회만 decoder로 픽셀 재구성 → token-space rollout의 drift를 anchor. encoder는 imagination 시작 시 1회(실제 obs)만, 이후는 token-space 유지.
- **drift 안전(능동):** K-frame anchor 시점에 token-space rollout으로 예측한 latent를 decode→재encode한 token과 비교, 불일치 크면 re-anchor(실제 token으로 reset)하고 stride K 하향. anchor가 drift를 능동 보정.

```
tok = VQVAE_encode(obs_0)               # encoder 1회 (원래 IRIS도 1회 — 추가 절감 없음)
for t in 1..H:
   tok = transformer_next(tok, a_t)     # token-space autoregressive (encoder 미호출)
   a_t = actor_tokenspace(tok)           # actor도 token-space (원래는 decoded frame 소비)
   r = reward_head_tokenspace(tok)       # decode 없이 reward/value
   if t % K == 0:
      px = VQVAE_decode(tok)             # K frame마다 1회 (drift anchor)
      tok_check = VQVAE_encode(px)
      if drift(tok, tok_check) > τ: tok = tok_check; K = max(1, K-1)  # re-anchor
# 절감 = skip된 (H - H/K)번의 decoder call (encoder는 원래 1회라 절감 대상 아님)
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: tokenizer **decoder** call(아래 encoder 정정 참조). **BL-13이 측정한 76%가 전제** — 단 그 76%가 "imagination 중 매 frame 호출되는 부분"인지 PoC가 재확인 필요(표현학습 phase의 tokenizer와 imagination phase의 tokenizer 구분).
- **encoder 정정(double-count 제거):** IRIS imagination은 token-space autoregressive다 — transformer가 예측한 token이 다음 step 입력으로 직접 들어가고, **encoder는 초기 실제 obs 1회만** 호출된다(imagined frame을 re-encode하지 않음). 따라서 "(H-1) encoder call skip"은 존재하지 않는다. **유일한 절감 레버는 decoder-skip이다.**
- **decoder 소비처(중요):** IRIS에서 decoder를 imagination 중 부르는 주체는 reward/value head만이 아니라 **actor(policy)도 decoded frame에서 행동**한다. 따라서 decoder를 skip하려면 reward/value head뿐 아니라 **actor도 token-space 입력으로 옮겨야** 한다 — 이것이 추가 quality 리스크(위험 요소 참조).
- imagination 중 decoder call: 원래 매 frame(actor+reward+value가 소비) → K frame마다 1회(anchor). K=4면 decoder 75% skip.
- decoder가 imagination 비용의 40%라 가정 시(encoder 제외 후 보수적): 평균 ≈ 0.6 + 0.4×0.25 = 0.7× → 1.4×. decoder 비중 50%면 0.5 + 0.5×0.25 = 0.625× → 1.6×. **ceiling은 이전 추정(1.8~2.5×)보다 낮다** — decoder-only이고 encoder 절감이 없으므로. gate 1.5×는 decoder가 imagination 비용 ≥45%일 때 도달.
- **핵심:** 절감은 skip된 decoder CALL에서 나옴(GPU 선호, 큰 CNN call을 통째로 제거). BL-13처럼 sub-batch 분리(0.66× 역효과)가 아니라 frame축 call skip.

## Novelty 포인트 (최소 3개)
1. (vs IRIS/TWM/STORM) imagination을 순수 token-space로 유지하고 픽셀 round-trip을 frame-stride로 제거 + token-space reward/value head. token-WM에서 tokenizer를 imagination hot-path에서 빼는 첫 명시적 설계.
2. (vs BL-13) transformer가 아니라 측정된 진짜 bottleneck(tokenizer 76%)을 타격. BL-13이 redirect로 지정한 정확한 방향. encoder+decoder를 함께 skip해 BL-09의 희석 문제(decoder가 큰데 encoder만 절감) 회피.
3. (vs latent rollout 일반) drift를 K-frame anchor의 decode→re-encode 일관성으로 능동 보정하고 stride를 adaptive하게 조정 — token-space drift를 픽셀-anchor로 bound하는 메커니즘이 novelty.

## 선행 연구 위험 요소
- IRIS, TWM, STORM, DreamerV3(decoder) — 기존 token-space 정도 확인 필수(이미 token-space면 novelty 약화 → reward head decode 의존성이 실재하는지가 관건)
- Latent video prediction (decode 생략) — 일반론, RL imagination 효율 목표 아님
- VQVAE/VQGAN 효율화 (encoder 경량화) — call skip이 아니라 call 경량화라 다름
- Frame skipping / temporal subsampling in video models
- MaskGIT/parallel token decoding

## 예상 실험 Skeleton
- Base model: IRIS (imagination loop의 tokenizer call 분석 + token-space head)
- Benchmark: Atari 100k (IRIS 표준)
- 측정: **(선행) imagination 중 tokenizer call 횟수/FLOP 분해**, encoder/decoder skip 비율, token-space drift(decode anchor 대비), imagination steps/sec, episode return, FVD
- 예상 결과: tokenizer call 60~80%↓(imagination 동안), end-to-end imagination 1.8~2.5×, return drop < 5%

## 빠른 PoC 가능 여부
가능(2~3일). 핵심 go/no-go: (1) **IRIS imagination 코드에서 tokenizer가 실제로 매 frame 호출되는지** 프로파일링(이미 token-space면 reward head decode 의존성만 절감 가능 → 이득 재추정). (2) token-space reward head가 픽셀 head 대비 정확도 보존하는지(distillation). (3) K-frame token-space rollout의 drift가 anchor로 bound되는지. synthetic 불필요, IRIS checkpoint 직접 프로파일.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **decoder가 imagination hot-path인지가 전제(encoder는 1회라 제외)**. decoder가 imagination 비용 ≥45%면 K=4에서 1.5~1.7× 도달. encoder 절감이 없으므로 ceiling은 1.4~1.7×(이전 1.8~2.5× 과대추정 정정). decoder 비중이 작으면 gate 미달 위험 → PoC가 decoder FLOP 비중 선측정.
- quality_delta < 0.05: **token-space head 정확도 + drift anchor가 관건**. distillation으로 reward head 보존, K-frame anchor로 drift bound. token-space rollout이 IRIS 원본 대비 충실하면 PASS.

## Venue Fit 이유
token-WM의 진짜 bottleneck(tokenizer) 절감은 token-based world model 효율의 핵심 미해결 문제 → NeurIPS. tokenizer round-trip 제거라는 분명한 메시지 → ICLR.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| encoder가 원래 1회라 절감 레버가 decoder뿐 → ceiling 하락 | 높 | decoder-only로 Amdahl 재추정(1.4~1.7×), decoder가 imagination 비용 ≥45%인 환경/구현 확인 |
| actor를 token-space로 옮기면 policy 품질 저하(actor가 원래 decoded frame 소비) | 중 | actor token-space head를 픽셀-actor teacher로 distill, K-frame마다 decoded-frame로 actor 보정 |
| token-space rollout drift가 long horizon에서 누적 | 중 | K-frame decode anchor + adaptive stride, re-anchor |
| token-space reward/value head 정확도 저하 | 중 | 픽셀 head teacher distillation, reward-sensitive frame은 full decode |
