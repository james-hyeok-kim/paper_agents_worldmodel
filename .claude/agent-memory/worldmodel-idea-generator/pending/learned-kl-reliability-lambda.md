---
slug: learned-kl-reliability-lambda
status: pending
created: 2026-06-18 KST
category: quality/sample-eff (DreamerV3 RSSM)
venue-fit: [NeurIPS, ICLR, ICML]
axis: quality/sample-efficiency (NOT raw speed)
base: DreamerV3-torch (experiments/wip/dual-rate-world-model/dreamerv3-torch)
blacklist-delta:
  - "BL-08 (RSSM prior-entropy 기반 adaptive horizon, learned controller): BL-08은 prior entropy/reward-var로 imagination을 '조기 종료(truncate)'하려다 reward-cliff를 사전에 못 봐서 실패(speedup 1.0×). 본 아이디어는 (1) truncation이 아니라 λ-return의 step별 신뢰 weighting(모든 step 살아있음, cliff 무관), (2) raw prior entropy(aleatoric, 단일 step)가 아니라 *학습된 보정 신호*(또는 imagination-내 critic-spread)로 step별 신뢰를 추정 — 단 이 신호가 실제 multi-step drift를 잡는지는 검증 대상(BL-08 함정 재유입 위험을 본문에 명시, PoC #1이 die point), (3) 효율이 아니라 sample-eff/return이 목표(잘라서 빠르게가 아니라 나쁜 target을 덜 믿어서 정확하게)."
  - "BL-02 (고정 임계값 early stopping): 임계값도 stopping도 없음. 연속 weight ∈ [0,1]를 학습된 reliability로 부여."
  - "vs '효율' BL-06~13 전체: 본 아이디어는 forward call을 하나도 줄이지 않는다 — imagination cost 동일. 오직 policy/critic target의 *품질*을 바꾼다. GPU kernel overhead / Amdahl 문제와 무관(지난 라운드 13개 실패 원인을 구조적으로 회피)."
---

# Learned Reliability-Weighted λ-Return: Imagination 신뢰도를 추정해 정책 target을 보정 (ensemble-free, RSSM-internal)

## 핵심 메커니즘 (한 문장)
DreamerV3 imagination은 **prior-only rollout**(관측 없이 `img_step`만 반복, 코드 확인: `ImagBehavior._imagine` L412-418)이라 imagined step에는 posterior-prior KL이 존재하지 않는다. 그래서 **경량 reliability head `ρ(feat)`** 를 둬서 "이 prior step에서 만약 관측이 있었다면 posterior-prior KL이 얼마였을지"를 예측하도록 **world-model 학습 시(실제 replay 시퀀스에서 KL이 진짜 존재)** supervise하고, imagination 시 prior-only로 `ρ`를 호출해 λ-return의 각 horizon step 기여를 `w_t = exp(-β·ρ_t)`로 가중한다. 모델이 신뢰 어려운(KL이 클 것으로 예측되는) imagined 구간을 정책/critic target에서 부드럽게 덜 믿게 만든다.

## 왜 quality/sample-eff를 개선하는가
- DreamerV3 actor/critic은 **전적으로 imagined trajectory로 학습**된다(`ImagBehavior._train`). imagination이 길어질수록 prior-only rollout이 실제 dynamics에서 벗어나고(compounding error), 그 위에서 계산된 λ-return(`_compute_target` L427-445)은 far-horizon에서 신뢰가 낮다. 현재 DreamerV3는 이 신뢰를 **고정 discount/λ로만** 반영 — step별 모델 신뢰도 차이를 무시한다.
- reliability weighting은 모델이 틀린 구간의 bootstrap target 기여를 줄여 **value overestimation/optimistic hallucination**(검색 근거: "noisy value function biased the agent towards optimistic hallucinations")을 억제 → critic이 더 정확해지고, actor가 환각된 고가치 영역으로 끌려가지 않음 → **동일 env step에서 더 높은 return = sample-efficiency 개선**.
- 핵심: forward 연산을 줄이지 않으므로(지난 라운드 13개 efficiency 실패의 GPU/Amdahl 함정 회피) 순수하게 **target 품질**만 바꾼다.

## 왜 신호가 실제로 존재하는가 (Gate B / 아키텍처 호환 — 검증 완료)
- **문제**: imagination은 prior-only라 posterior가 없음 → posterior-prior KL을 직접 읽을 수 없음(코드 확인). prior entropy(`state_ent = dynamics.get_dist(imag_state).entropy()`, L361)는 이미 계산돼 있으나 **aleatoric**이고 Dreamer가 과대추정하는 신호(BL-08 함정).
- **해결 (후보 신호 2개, ablation으로 결정)**:
  - **(a) 학습된 KL-predictor**: KL은 *world-model 학습 시 실제 replay 시퀀스에서는 진짜 존재*한다(`WorldModel._train`의 `kl_loss`, L160-163, shape = (batch, length)). 이를 라벨로 `ρ(prior_feat) → KL_predicted`를 회귀 학습, imagination 시 prior-only 호출. `ρ`는 작은 MLP(2-layer), cost 무시 가능.
  - **(b) critic-distribution spread (label-shift-free, 권장 우선순위 높음)**: DreamerV3 critic은 symexp two-hot **분포**를 출력(`self.value(imag_feat)`, L433은 distribution). 그 분포의 spread/엔트로피는 **imagination 안에서 직접 계산**되며 train/test label-shift가 없다. 미학습/OOD imagined 영역에서 critic이 value에 불확실 → 신뢰 낮음. (a)의 약점(아래)을 구조적으로 회피.
- **(a)의 핵심 위험 — 정직하게 명시**: 라벨 `kl_value` = KL(posterior‖prior)는 **실제 replay에서 posterior로 re-anchor된 trajectory 상의 one-step 관측 surprise**다. in-distribution 잘 학습된 모델에선 이는 **aleatoric 우세**(상태가 본질적으로 stochastic해서 큰 KL이지, 모델이 틀려서가 아님)다. 반면 imagination 불신뢰의 정체는 **prior-only rollout을 t step 누적한 compounding epistemic drift**(unbounded, 누적)로 *다른 양*이다. 따라서 ρ가 완벽히 학습돼도 "자신만만하게 틀린"(low aleatoric, high actual drift) step을 못 잡고 high weight를 줄 수 있다 — 이것이 BL-08 함정이 head를 통해 재유입되는 경로. **"ρ가 epistemic 모델오차를 잡는다"는 가정이 아니라 검증해야 할 중심 가설**이며, 이게 본 아이디어의 die point다(아래 PoC #1 gate). dual-rate와 공유하는 건 "RSSM 내부 신호 + 학습 보정"의 *형태*이지, 신호가 옳다는 보장이 아니다.

## 의사코드 (DreamerV3 hook 지점 명시)
```
# WorldModel._train (실제 replay, KL 존재):
kl_loss, kl_value, _, _ = dynamics.kl_loss(post, prior, ...)   # 기존, L160
rho_pred = reliability_head(get_feat(prior).detach())          # 추가
rho_loss = mse(rho_pred, kl_value.detach())                    # KL을 라벨로 회귀
model_loss = ... + rho_scale * rho_loss                        # 기존 loss에 추가

# ImagBehavior._compute_target (prior-only imagination):
rho_t = reliability_head(imag_feat.detach())                   # prior-only proxy
w_rel = torch.exp(-beta * rho_t)                               # ∈(0,1], step별 신뢰
target = lambda_return_reweighted(reward, value, discount, w_rel, lambda_)
# λ-return 재귀에서 각 k-step 기여를 w_rel로 가중(또는 effective discount = discount * w_rel)
```
구현 옵션 2가지: (a) effective discount `d_t ← d_t · w_rel_t` (가장 적은 코드 변경, 신뢰 낮은 step 이후를 자동 단축), (b) λ-return의 n-step 혼합 가중치를 직접 reweight. ablation으로 비교.

## PoC에서 측정할 metric + 예상 이득
- Base: DreamerV3-torch, DMControl 2-3개 task(walker-walk, cheetah-run, cartpole-swingup; 짧은 budget 100k-500k env steps).
- **PoC #1 gate (signal validity — 이걸 먼저, 가장 싸게)**: 학습 중 prior-only rollout을 생성하고 각 step의 imagined state를 **그 step의 실제 posterior state(ground truth)와 비교한 multi-step drift**를 측정 → 신호(ρ 또는 critic-spread)가 그 *drift*와 상관하는가. **주의: ρ를 자신이 학습한 라벨(one-step KL)과 비교하는 건 memorization이라 trivially pass → false pass. 반드시 multi-step drift와 상관**을 본다. 상관이 낮으면(BL-11 0.085 전례) 해당 신호 기각, 다른 신호로 교체. (a) KL-predictor가 fail하면 (b) critic-spread로.
- **Primary (sample-eff)**: 고정 env-step(예: 100k, 250k)에서의 eval_return — baseline 대비 향상. 게이트는 **"fixed-step return ≥ baseline, with upside"**. (dual-rate의 robust 결과는 -25% params에서 late-stage -1.3%, best-checkpoint에서 +12.4% — +12%는 단일 best point라 기대치로 인용하면 cherry-pick. 보수적 목표는 "drop 없이 같은 step에서 더 빨리".)
- **Quality**: critic value calibration — imagined value vs 실제 Monte-Carlo return의 오차(value overestimation 감소 여부).
- **Final return**: 수렴 후 return은 baseline 이상(drop 없어야 함 — gate).
- Gate: PoC #1(신호-drift 상관) 통과 AND fixed-step return이 baseline 대비 유의하게 높음(seed 3-5) AND final return drop 없음.

## 핵심 ablation
1. **ρ를 학습 KL-predictor vs 단순 prior-entropy(`state_ent`)로 교체** — BL-08 함정(entropy는 aleatoric) 대비 학습된 epistemic proxy의 우위를 직접 입증. 이게 메인 ablation.
2. **weighting vs truncation** — 동일 ρ로 step을 자르면(MACURA/M2AC 스타일) vs 부드럽게 가중하면. cliff-free 가설 검증.
3. β sweep (0 = baseline DreamerV3) — β=0에서 정확히 baseline 복원 확인 + 이득이 β에 monotone인지.
4. ρ supervision 제거(reliability head를 imagination-time self-consistency로만 학습) — KL-label supervision의 기여 분리.

## 선행 연구 위험 요소 (freshness self-check 완료)
- **"Acting upon Imagination: When to Trust Imagined Trajectories" (arXiv:2105.05716)** — 가장 가까움. **차별점 3개(fetch로 확인)**: (1) ensemble of probabilistic NN 사용 → 본 방법은 ensemble-free 단일 RSSM 내부 학습 head, (2) truncation/replanning → 본 방법은 λ-return target weighting, (3) random-shooting MPC 대상 → 본 방법은 Dreamer/RSSM imagination 대상.
- **DreamerV3-XP (arXiv:2510.21418)** — adaptive λ를 *성능 추세로 전역 스칼라* 조정. 본 방법은 *per-horizon-step* 신뢰를 RSSM 내부 모델오차로 가중. global scalar vs per-step + 신호 출처가 다름.
- **MACURA (ICML 2024), M2AC (NeurIPS 2020)** — per-state truncation/masking, ensemble 필요, fixed threshold. 본 방법은 weighting + ensemble-free + 학습된 연속 신뢰. (landscape/rollout_efficiency.md에 등재)
- **STEVE / Uncertainty-based Value Expansion (arXiv:1912.05328), RAVE** — value target을 ensemble variance로 가중. 본 방법은 ensemble 없이 학습된 KL-predictor. → literature-checker가 STEVE 대비 ensemble-free novelty 정밀 확인 필요(주요 리스크).
- COPlanner (ICLR 2024) — ensemble uncertainty penalty, action 회피 기반. ensemble 필요.

## venue
NeurIPS/ICLR 본트랙. "imagination의 prior-only 한계를 학습된 KL-proxy로 메워 정책 target을 보정" 메시지가 분명하고, DreamerV3에 drop-in. efficiency가 아닌 quality/sample-eff 기여.

## 위험 요소
| 위험 | 가능성 | 완화 |
|---|---|---|
| 신호(ρ 또는 critic-spread)가 실제 multi-step imagination drift와 상관이 낮음(BL-11의 0.085 재발 / BL-08 함정 head 재유입) | **높음 (메인 die point)** | PoC #1 go/no-go: 신호 vs *multi-step drift*(imagined state vs 실제 posterior) 상관 선측정 — ρ를 자기 학습 라벨과 비교하는 false-pass 금지. (a) KL-predictor fail 시 (b) critic-distribution spread(label-shift 없음)로 교체 |
| STEVE류 ensemble value-weighting과 novelty 충돌 | 중 | literature-checker가 ensemble-free + KL-label supervision 조합의 novelty 확정 |
| reweighting이 학습 불안정(bias 도입) | 중 | β=0 baseline 복원 확인, β warmup, effective-discount 방식(이론적으로 valid한 sub-discounting) |
| 이득이 짧은 PoC에서 안 보임 | 중 | actor/critic target 직접 변경이라 비교적 조기 발현(Gate B 근거). 안 보이면 horizon 긴 task로 |
