---
slug: static-dynamic-decoupled-wm
status: literature-checked
verdict: INCREMENTAL
created: 2026-06-11 KST
category: G
venue-fit: [ICLR, NeurIPS, CoRL]
blacklist-delta:
  - "BL-03 (픽셀 codec 압축): 픽셀이 아니라 latent를 static(에피소드 내 불변: 배경/조명/scene id)과 dynamic(매 step 변화: agent/object 운동)으로 분해. static latent를 에피소드 시작 시 1회만 인코딩해 캐시, imagination 전체에서 재사용 → encoder forward 절감. 손실 압축 아님."
  - "BL-04 (distillation)와 무관: 모델 압축이 아니라 입력 표현의 정보 분해. encoder를 static/dynamic 두 경로로 분리하고 static 경로를 에피소드당 1회만 호출."
  - "BL-01 (KV cache)와 다름: latent transition의 캐싱이 아니라 encoder/decoder의 static 성분 캐싱. dynamic 경로만 매 step 갱신하고 reward head는 두 성분 결합 사용 — invariant feature를 imagination rollout 전체에 broadcast."
---

# Static/Dynamic Decoupled World Model: 불변 Latent의 에피소드 단위 Amortization

## 핵심 가설
시각 RL 환경에서 관측의 상당 부분(배경, 조명, 카메라, scene 정체성)은 에피소드 내내 불변이다. encoder를 static/dynamic 두 경로로 분리해 static latent를 에피소드당 1회만 인코딩하고 imagination 전체에서 broadcast하면, encoder/decoder forward 연산을 25~45% 줄여 model-based update를 1.4~1.7× 가속하면서 reconstruction/return drop을 4% 이내로 유지한다.

## 동기 (Why Now)
DreamerV3/IRIS의 encoder는 매 step 전체 이미지를 인코딩한다. 그러나 대부분의 정보(배경 텍스처, 정적 구조물)는 에피소드 내내 동일하고, 의사결정에 중요한 변화는 작은 영역에 국한된다. static 성분을 한 번만 계산해 캐시하면 encoder 비용을 크게 줄일 수 있다 — 특히 imagination 중 decoder를 호출하는 경우(시각 reward, reconstruction loss)에 절감이 크다. 이는 perception과 dynamics를 분리하는 world-model-specific 구조 기회다.

## 제안 방법
- encoder를 분리: `e_static(o_0) -> c` (에피소드 시작 관측에서 불변 코드 c 추출, 1회), `e_dyn(o_t) -> u_t` (매 step, 경량).
- latent state = `[c, u_t, s_dyn_t]`. dynamics transition은 `s_dyn`만 갱신, c는 상수 carry.
- 분해 유도: c는 에피소드 내 시간 불변(temporal-invariance contrastive loss: 같은 에피소드 다른 step의 c는 가까이, 다른 에피소드는 멀리), u는 변화 성분만 담도록 c→u 정보 누수 차단(reconstruction은 c+u 둘 다 필요하게).
- decoder: `decode(c, u_t, s_dyn_t)` — c는 broadcast, dynamic만 step별. imagination reward에도 c 재사용.
- 안전장치: scene이 급변하면(c invariance 깨짐) refresh — c와 현재 관측 잔차가 임계 초과 시 e_static 재호출.

```
c = e_static(o_0)                 # 에피소드당 1회
for t in rollout:
    u_t   = e_dyn(o_t)            # 경량
    s_dyn = f([s_dyn, u_t, a])    # dynamic만
    out   = decode(c, u_t, s_dyn) # c broadcast
```

## Novelty 포인트 (최소 3개)
1. (vs world model) encoder/decoder를 static/dynamic으로 명시 분해하고 static을 에피소드당 1회로 amortize — perception 연산의 temporal redundancy를 직접 제거. DreamerV3/IRIS encoder는 매 step full.
2. (vs disentanglement 논문) 목적이 해석가능성이 아니라 efficiency. temporal-invariance contrastive로 static을 식별하고 invariance-break refresh로 동적 scene에 robust.
3. (vs object-centric WM) 객체 분할 없이 "시간 불변 vs 가변" 단일 축으로 분해 → 경량, imagination reward/decoder까지 절감 적용.

## 선행 연구 위험 요소
- Object-centric world models (SLATE, SAVi), Slot Attention
- Content/motion disentanglement (video prediction: MoCoGAN, DDPAE)
- DreamerV3 encoder, IRIS tokenizer
- Invariant representation learning, contrastive temporal coherence
- "Denoised MDPs", task-relevant feature 분리

## 예상 실험 Skeleton
- Base model: DreamerV3 (encoder/decoder 분리), 보조 IRIS tokenizer
- Benchmark: DMControl (정적 배경), Atari 100k, Distracting Control Suite(배경 변화 robustness 검증)
- 측정: encoder/decoder FLOPs 절감, static refresh 빈도, model update steps/sec, reconstruction PSNR, episode return
- 예상 결과: encode/decode 25~45% 절감, update 1.4~1.7× 가속, PSNR/return drop < 4%

## 예상 Contribution
- perception 단계의 temporal redundancy를 제거하는 static/dynamic encoder 분해 + invariance-break refresh
- imagination reward/decoder까지 절감을 확장하는 broadcast 메커니즘

## 빠른 PoC 가능 여부
가능. synthetic: 정적 배경 + 움직이는 객체 toy video에서 c가 시간 불변으로 학습되는지, c skip 후 reconstruction 오차, refresh가 배경 전환을 잡는지 2~3일. Full: DreamerV3 encoder 분리 fork 3~4일.

## Venue Fit 이유
perception efficiency + robustness(distracting bg)는 robotics 실환경과 연결 → CoRL 매력적. 표현 분해 novelty + efficiency 곡선은 ICLR/NeurIPS 적합.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| static/dynamic 분해 collapse (c가 비거나 모든 정보 독점) | 높 | temporal-invariance contrastive + reconstruction이 c,u 둘 다 요구하도록 구조적 강제 |
| 배경이 자주 변하는 환경에서 refresh 빈발 → 이득 소멸 | 중 | refresh 빈도를 보고, 적용 가능 조건(정적 배경 비율) 분석 |
| static 가정이 깨지는 task에서 품질 저하 | 중 | invariance-break 임계 + fallback(full encode) 안전장치 |
