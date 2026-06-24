---
slug: residual-corrected-sparse-rssm
status: literature-checked
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
created: 2026-06-11 KST
category: B
venue-fit: [NeurIPS, ICLR]
blacklist-delta:
  - "BL-06 (carry-forward delta-skip): BL-06은 GRU h를 통째로 carry-forward → 누적 drift quality 1.9(gate 38× 초과)로 실패. 본 아이디어는 carry-forward를 하지 않는다. 대신 full GRU 대신 저차원 latent 부분공간에서 동작하는 lightweight residual corrector가 매 step h의 변화량(delta)을 명시적으로 추정·적용 → drift를 매 step 능동 보정. skip이 아니라 cheap-update."
  - "BL-09 (decoder bottleneck): RSSM dynamics(GRU transition)를 직접 절감. DreamerV3 imagination은 latent-only이므로 GRU transition이 hot-path. encoder/decoder 절감(BL-09 실패)이 아니라 transition 자체를 저차원 residual로 근사."
  - "BL-08 (adaptive horizon): horizon을 자르지 않음. 모든 step을 끝까지 rollout하되 각 step의 transition 비용을 낮춤. truncation 없음 → cliff 문제 없음."
---

# Residual-Corrected Sparse RSSM: Drift 없는 저차원 Transition 근사

## 핵심 가설
BL-06의 carry-forward가 실패한 이유는 transition을 통째로 생략해 drift가 누적됐기 때문이다. transition을 생략하는 대신, full GRU update를 매 step 실행하지 않고 저차원 부분공간(rank-r projection)에서 동작하는 경량 residual corrector가 h의 step별 변화량을 추정해 적용하면, drift를 능동 보정하면서 transition FLOPs를 35~50% 줄여 imagination을 1.5~1.8× 가속하고 quality drop을 5% 이내로 유지한다.

## 동기 (Why Now)
DreamerV3 imagination은 매 step full GRU+prior MLP를 돈다(latent-only hot-path). BL-06 실패 분석: deterministic h를 그냥 carry하면 drift가 decoder/reward를 망친다. 그러나 h의 step별 변화 Δh_t는 대개 저차원 manifold에 놓인다(연속 dynamics의 국소 선형성). full GRU 대신 Δh를 rank-r corrector로 추정하면, 정보를 버리지 않고(=carry-forward 아님) 연산만 줄인다. 핵심은 "skip이 아니라 cheaper-but-correcting update".

## 제안 방법
- full transition `h_t = GRU(h_{t-1}, [z,a])`를 매 step 대신, K step마다 1회만 full GRU(anchor), 사이 step은 저차원 corrector로 보간/외삽.
- corrector: `Δĥ_t = U · r(P·h_{t-1}, z_t, a_t)` — P∈R^{r×d} (r≪d) 하향 projection, r은 작은 MLP, U∈R^{d×r} 상향. h_t = h_{t-1} + Δĥ_t. (carry-forward면 Δ=0이지만 여기선 Δ를 추정)
- corrector는 anchor full-GRU 출력과의 residual로 학습(`Δĥ ≈ h_full - h_prev`). teacher가 full GRU라 추가 라벨 불필요.
- drift 누적 방지: K step마다 full GRU로 re-anchor(corrector 오차 reset). K는 corrector residual norm으로 적응(오차 크면 K↓).
- stochastic z는 항상 prior에서 정상 샘플(diversity 유지) — BL-06과 달리 z를 frozen h에서 뽑지 않음.

```
h = GRU_full(h, z, a)            # anchor (every K steps)
for k in 1..K-1:
   Δ = U·r(P·h, z, a)            # rank-r corrector, cheap
   h = h + Δ                     # cheaper update, NOT carry-forward (Δ≠0)
   z = sample(prior(h))          # 정상 stochastic
   if ||residual|| > τ: break    # re-anchor early
```

## Amdahl 체크 (gate 도달 가능성)
- 타깃: RSSM transition의 **GRU(d×d) 부분**. **중요 precondition:** stochastic z를 매 step prior에서 정상 샘플하므로(diversity 유지), prior MLP는 corrector step에서도 그대로 실행된다 → corrector가 절약하는 것은 GRU 연산뿐. 따라서 1.7× 추정은 **GRU가 prior MLP를 압도(GRU ≳ 2× prior)할 때만** 성립. PoC가 GRU:prior FLOP 비를 먼저 측정해야 함.
- corrector 비용: rank-r (r/d ≈ 1/8~1/4) → full GRU의 ~15~25%.
- K=4 (1 full GRU + 3 corrector): 평균 GRU 비용 ≈ (1 + 3×0.2)/4 = 0.4× → GRU 60%↓.
- 추정 speedup: GRU가 transition의 ~70%(prior MLP ~30%)면 transition 평균 비용 ≈ 0.7×0.4 + 0.3 = 0.58× → transition 1.7×. transition이 imagination hot-path 70%면 전체 ≈ 1/(0.3 + 0.7×0.58) ≈ 1.45~1.6×. → gate 1.5× 경계. GRU 비중이 클수록 상회.

## Novelty 포인트 (최소 3개)
1. (vs BL-06) carry-forward(Δ=0)가 아니라 Δ를 저차원으로 능동 추정·보정 → drift 누적을 구조적으로 차단. re-anchor로 오차 reset.
2. (vs world model) RSSM transition을 anchor+residual로 분해해 step별 비용 차등화 — DreamerV3는 모든 step full GRU.
3. (vs low-rank / LoRA) 파라미터 압축이 아니라 시간축 연산 sparsity. corrector는 full-GRU teacher의 residual을 online으로 근사하는 distillation-in-time(BL-04의 정적 teacher-student와 무관, 같은 모델 내 시간 amortization).

## 선행 연구 위험 요소
- Low-rank RNN / linear RNN (S4, Mamba 류) — 구조 압축이라 다름 강조 필요
- Multistep/coarse Euler integration in latent ODE (Neural ODE, latent dynamics)
- Hypernetwork residual prediction
- BL-06 자체 (carry-forward) — 차별점 명시 필수
- Mixture-of-Depths / adaptive computation

## 예상 실험 Skeleton
- Base model: DreamerV3 (RSSM transition fork)
- Benchmark: DMControl proprio (연속 dynamics, 국소 선형성 가정 검증에 적합), Atari 100k
- 측정: transition FLOPs, corrector residual norm 분포, re-anchor 빈도, imagination steps/sec, h-drift(vs full GRU), reward/return drop
- 예상 결과: transition 35~50%↓, imagination 1.5~1.8×, return drop < 5%, drift bounded(re-anchor로)

## 빠른 PoC 가능 여부
가능(2일). synthetic: 학습된 toy RSSM(또는 random GRU)에서 Δh가 실제로 저차원인지(SVD energy), rank-r corrector가 K-step 사이 h를 full GRU 대비 얼마나 정확히 재현하는지, re-anchor 주기 vs drift 곡선. BL-06과 직접 비교(carry-forward Δ=0 vs corrector Δ≠0)로 drift 차이 정량화 2일.

## Validator Gate 달성 평가
- rollout_speedup > 1.5×: **도달 가능**. K와 r로 조절. transition이 hot-path인 한 1.5~1.7×.
- quality_delta < 0.05: **핵심 검증점이자 BL-06 회피의 본질**. corrector가 drift를 0.05 이내로 잡는지가 전부. re-anchor + 저차원 Δ추정이 BL-06의 1.9를 0.05 이하로 끌어내릴 수 있는지가 PoC의 go/no-go. Δh 저차원성이 성립하면 유망, 아니면 FAIL.

## Venue Fit 이유
latent dynamics 연산 효율 + drift 분석은 model-based RL/efficiency 핵심 → NeurIPS. 저차원 residual 근사 이론은 ICLR 적합.

## 위험 요소
| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| Δh가 저차원이 아니어서 corrector 부정확(=BL-06 재현) | 중 | PoC에서 Δh SVD energy 선검증, rank-r 부족 시 K↓로 fallback |
| re-anchor 빈발로 이득 소멸 | 중 | residual-norm 적응 K, 적용 환경(smooth dynamics) 한정 |
| corrector 학습이 full-GRO와 경쟁해 불안정 | 낮 | teacher residual 회귀로 분리 학습, stop-gradient |
