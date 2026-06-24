---
slug: spectral-multilevel-rssm
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: E
venue-fit: [NeurIPS, ICLR]
blacklist-delta:
  - "dual-rate(성공/INCREMENTAL)와의 차별: dual-rate는 slow/fast 2-tier를 수작업 정의하고 information bottleneck으로 collapse를 막는다. 본 아이디어는 latent 차원을 update-frequency로 자동 할당 — 각 latent 차원의 시간 자기상관(temporal autocorrelation)을 측정해 spectral clustering으로 L개 frequency band에 배정. 수작업 2-tier가 아니라 데이터 기반 multi-band 분해. novelty가 'frequency를 학습으로 결정'에 있음."
  - "BL-08 (per-step entropy adaptive horizon): horizon을 자르지 않음. update frequency를 차원별로 할당하되 모든 차원이 끝까지 살아있음. cliff 없음. 그리고 frequency는 per-step 학습 게이트가 아니라 학습 후 고정(per-dimension band assignment)이라 BL-08의 label-imbalance/cliff 함정 회피."
  - "BL-10 (cheaper-per-step GRU): cheaper update가 아니라 high-frequency band 차원만 매 step 갱신하고 low-frequency band는 1/K_band 빈도로 call skip. 절감은 skipped CALL에서 나옴(GPU 선호). 차원을 band별로 묶어 큰 sub-GRU 1개를 sparse 호출 → kernel overhead amortize."
---

# Spectral Multi-Level RSSM: 차원별 시간 자기상관으로 Update-Frequency 자동 할당

## 핵심 가설
RSSM latent의 각 차원은 서로 다른 고유 시간 스케일을 가진다(일부는 천천히, 일부는 빠르게 변함). 차원별 temporal autocorrelation을 측정해 L개 frequency band로 spectral clustering하고, band b의 차원은 K_b step마다만 갱신(나머지는 call skip)하면, dynamics call을 40~60% 줄여 rollout을 1.7~2.3× 가속하면서 return drop을 5% 이내로 유지한다.

## 동기 (Why Now) — 중심 thesis: L>2가 2-tier가 구조적으로 못 닿는 Pareto point다
dual-rate(1.93× CONDITIONAL-GO이나 문헌상 INCREMENTAL)의 한계는 단순한 튜닝 문제가 아니라 **구조적**이다. 2-tier는 각 차원을 fast(K=1) 또는 slow(K=K_s) 둘 중 하나에만 배정할 수 있다. 그런데 실제 latent 차원의 특성 시간 스케일은 연속 스펙트럼이고, 중간 스케일(예: K≈3) 차원이 다수면 2-tier는 반드시 손해를 본다: 그 차원을 fast로 보내면 절감 손실, slow로 보내면 K_s가 너무 길어 drift/quality 손실. **즉 중간-스케일 질량이 큰 latent에서 2-tier의 절감-품질 곡선에는 도달 불가능한 영역(Pareto gap)이 존재한다.** L-band(L>2)는 그 차원들에 맞는 중간 period를 부여해 이 gap을 채운다.

본 아이디어의 load-bearing 주장(중심 결과로 검증할 것): **"latent 자기상관 스펙트럼이 multi-modal(2-tier로 환원 불가)이고, L-band가 dual-rate(L=2)의 어떤 (K_fast, K_slow) 조합으로도 닿지 못하는 speedup-quality Pareto point를 실현한다."** 이게 성립하지 않으면(스펙트럼이 사실상 bimodal) 본 아이디어는 dual-rate로 환원되며 novelty가 없다 — 따라서 PoC의 1순위는 자기상관 스펙트럼의 multi-modality 측정이다. 부수적으로 band assignment를 데이터로 결정하면 환경별 수작업 튜닝도 사라진다.

## 제안 방법
- **band assignment(학습 후 고정):** warmup 동안 각 latent 차원 i의 시간 자기상관 ρ_i(τ)를 추정 → 특성 시간 스케일 τ_i. spectral clustering으로 차원들을 L개 band {B_1,...,B_L}에 배정, band b에 update period K_b 부여(K_1=1 fast ... K_L=large slow).
- **band별 sparse update:** band b의 차원 부분공간은 K_b step마다 1회 sub-GRU로 갱신. 사이 step은 **그 band의 차원만** 이전 값 유지(carry)하되, 매 step 갱신되는 fast band가 decoder/reward로 들어가 보정 → 전체 latent는 매 step 신선(BL-06의 통째 carry-forward와 다름: 느린 차원만 carry, 빠른 차원이 매 step 보정).
- **drift 안전(능동):** 각 slow band 갱신 시점에 직전 carry된 값과 full-update 값의 residual을 모니터, residual이 τ 초과하면 그 band의 K_b를 하향(adaptive period). carry되는 차원은 정의상 자기상관 높은(=느린) 차원이라 drift 본질적으로 작음 — band assignment가 drift bound를 데이터로 보장.
- **stochastic z:** prior는 항상 매 step 전체 latent로 샘플(diversity), slow 차원의 prior 입력은 carry된 값 사용.

```
# warmup: estimate τ_i per dim, cluster into L bands with periods K_b
for t in horizon:
   for band b in 1..L:
      if t % K_b == 0:
         h[B_b] = subGRU_b(h[B_b], context)   # band b 차원만 갱신, K_b 빈도
      # else: h[B_b] carry (느린 차원만)
   z = sample(prior(h))                        # 매 step 전체 (fast 차원이 보정)
   out = decode(h)
# 절감 = slow band의 skipped subGRU calls
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: RSSM transition의 GRU 연산(imagination hot-path). BL-10/residual-corrected가 명시한 precondition 동일: **prior MLP가 transition의 59%를 차지**하므로 GRU만 절감하면 ceiling이 낮음. 따라서 본 아이디어의 sub-GRU band 분할은 **GRU 부분(41%)** 내에서 절감 → 전체 transition 기준 효과 제한적일 수 있음. **PoC가 먼저 측정:** prior MLP를 band-skip에 포함시킬 수 있는가? slow band 차원의 prior head도 K_b마다만 갱신하면 prior MLP도 절감 대상에 포함 → ceiling 상향.
- band-skip로 prior+GRU 둘 다 slow 차원에서 1/K_b 호출 시: 평균 transition 비용 ≈ Σ_b (|B_b|/d)·(1/K_b). 예: 절반 차원이 K=4면 평균 ≈ 0.5 + 0.5×0.25 = 0.625× → transition 1.6×.
- **핵심:** slow band의 sub-GRU+sub-prior call을 통째로 skip(GPU 선호). band를 큰 묶음으로 유지해 sub-call kernel overhead amortize.

## Novelty 포인트 (최소 3개)
1. (vs dual-rate / Clockwork RNN) 고정 2-tier나 수작업 clock이 아니라, 차원별 temporal autocorrelation 측정 → spectral clustering으로 L-band를 데이터 기반 자동 할당. frequency 분해가 학습됨.
2. (vs world model) RSSM 차원을 시간 스펙트럼으로 분해해 차원별 update period를 차등화하는 첫 시도. slow 차원만 carry(=느린 차원이라 drift 작음)라 BL-06 함정 회피 — band assignment 자체가 drift bound를 데이터로 보장.
3. (vs multi-scale SSM / hierarchical) abstraction을 만드는 게 아니라 기존 flat latent를 frequency로 재배치 — 아키텍처 추가 없이 update schedule만 변경. prior+GRU를 함께 band-skip해 BL-10의 ceiling 한계(GRU만으론 부족)를 정면 해결.

## 선행 연구 위험 요소
- Clockwork RNN (Koutnik 2014), Phased LSTM — 고정 clock이라 다름
- dual-rate-world-model (자체 INCREMENTAL) — 2-tier vs 데이터기반 L-band 차별 필수
- Slow Feature Analysis, spectral methods for dynamics
- Multi-timescale RL, hierarchical SSM (S4 hierarchy)
- HiPPO/관련 frequency 분해 — representation이지 efficiency 아님

## 예상 실험 Skeleton
- Base model: DreamerV3 (RSSM 차원 band 분할 + sub-GRU/sub-prior)
- Benchmark: DMControl (proprio+pixel, 시간 스케일 다양), Atari 100k
- 측정: **transition FLOPs(prior+GRU 분해)**, band assignment 분포, slow band drift residual, rollout steps/sec, return, FVD
- 예상 결과: dynamics call 40~60%↓, rollout 1.7~2.3×, return drop < 5%, dual-rate 대비 Pareto 개선

## 빠른 PoC 가능 여부
가능(2~3일). **1순위 go/no-go(중심 thesis 검증): 학습된 RSSM의 차원별 자기상관 스펙트럼이 multi-modal(≥3 mode)인가, 아니면 bimodal(=dual-rate로 환원)인가.** bimodal이면 이 아이디어는 즉시 폐기(dual-rate와 구별 불가). multi-modal이면 진행. synthetic: 명시적 3-scale toy(상수 배경 + 느린 drift + 빠른 점) video로 RSSM 학습 후, (1) 자기상관 스펙트럼 mode 수 측정, (2) L-band(L=3)의 speedup-quality 점이 dual-rate(L=2)의 모든 (K_fast,K_slow) 격자 위 어떤 점보다 Pareto-우월한지 직접 비교(이게 핵심 novelty 증거), (3) prior MLP를 band-skip에 포함 가능한지(ceiling).

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **prior+GRU 동시 band-skip이 관건**. GRU만이면 BL-10처럼 ceiling 부족(<1.5× 위험). prior head도 slow 차원에서 skip 가능하면 1.7×+ 도달.
- quality_delta < 0.05: **유망**. carry되는 차원이 정의상 느린(자기상관 높은) 차원이라 drift 본질적으로 작음 — BL-06과의 결정적 차이. band assignment가 drift를 데이터로 bound.

## Venue Fit 이유
latent dynamics의 spectral 효율화 + drift 이론은 NeurIPS. frequency 기반 representation 분해는 ICLR.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| prior MLP를 band-skip 못해 ceiling이 GRU(41%)로 제한 | 중 | slow 차원 prior head도 함께 skip하는 설계, PoC 선검증 |
| 자기상관 스펙트럼이 bimodal이라 dual-rate로 환원(novelty 소멸) | 중 | **이게 kill criterion** — PoC 1순위로 multi-modality 선측정, bimodal이면 아이디어 폐기(L=2 fallback은 novelty 없으므로 진행 불가) |
| band 경계가 환경마다 불안정 | 낮 | warmup autocorrelation 추정, episode 단위 재추정 |
