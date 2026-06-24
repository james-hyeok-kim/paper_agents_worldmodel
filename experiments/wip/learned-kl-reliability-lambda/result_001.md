# Result 001 — learned-kl-reliability-lambda PoC #1 (signal validity + feasibility)

slug: learned-kl-reliability-lambda · stage: validator PoC gate · 2026-06-18 KST · synthetic (GL-free)
실행자: team-lead (validator agent spawn 3회 무응답 → 직접 수행)

## Run 1 (effective-discount reweighting) — 신호 validity PASS, feasibility 무효

**Signal validity (idea의 die point #1) — 강하게 통과:**
| 신호 | t-controlled Spearman (headline) | partial(ctrl t) | pooled | vs latent-L2 |
|---|---|---|---|---|
| critic_spread | **0.535** | 0.549 | 0.282 | 0.116 |
| learned_rho (KL-predictor) | 0.373 | 0.436 | 0.360 | 0.050 |
| prior_entropy (BL-08 baseline) | **-0.185** | -0.211 | 0.017 | — |

- 신호가 **decision-relevant drift**(=|reward err|+|value err|, 같은 head 통과)와 강하게 상관, 그러나 **latent-L2와는 약함(0.116)** → advisor #2의 BL-11 함정(latent-L2가 value harm 못 잡음)의 정반대, 정확히 value-target harm을 잡음.
- t-controlled(0.535) > pooled(0.282), partial-ctrl-t(0.549)≈within-step → **state-dependent 신호가 진짜**, t-추세 artifact 아님. BL-11의 0.085 대비 압도.
- prior_entropy(BL-08)는 음의 상관 → aleatoric 신호는 실패(예측대로). critic_spread가 prior-entropy 실패한 곳에서 성공 = 순수 aleatoric 아님(일부 epistemic 포착).

**Feasibility (reweighting) — 결과 무효 (confound):**
- effective-discount `d=γ·w (w<1)` reweighting: 두 신호 모두 best_beta=0, β>0에서 단조 악화(4.15→7.8).
- **원인(advisor 수학 확인)**: env 보상이 전부 음수 → w<1이 음수 target을 0쪽으로 축소 → 큰 음수 true return에서 멀어짐 → 신호 품질과 무관하게 β=0 선호. **테스트가 rigged. 무효.**

## Run 2 사전등록 (PRE-REGISTERED, 실행 전 commit) — 2026-06-18 KST

**Sign-unbiased formulation = selective bootstrap:**
- anchor(=post[0])에서 imagined rollout. 신호 > τ인 첫 step m에서 critic value로 bootstrap:
  `T_imag(m) = Σ_{k=1..m} γ^{k-1} r_imag[k] + γ^m · v_imag[m]`. flag 없으면 m=H(full).
- truth: `G = Σ_{k=1..H} γ^{k-1} r_real[k] + γ^H · v_post[H]` (real env reward + ground-truth-state(posterior) critic bootstrap).
- 오차 metric: mean_n |T_imag(m) − G|.
- **sanity**: τ=∞ → m=H 전부 → T_imag(H)=full baseline 정확 복원(±1e-6). 안 되면 formulation 버그.

**통제 (advisor 3개):**
1. **Oracle**: 신호=true drift로 selective bootstrap.
2. **State-agnostic baseline**: 신호 없이 균일 m(고정 bootstrap horizon) 중 best.
3. τ(신호)·m(agnostic) 선택은 **tune set(eval traj 전반부)**에서, 오차는 **held-out test set(후반부)**에서 보고 → cherry-pick 방지, 모든 방법 동일 자유도.

**판정 기준 (이 기준으로 다음 run 결과를 그대로 commit):**
- CONDITIONAL-GO ⟺ **(i)** oracle_test ≤ 0.95 × min(full_test, agnostic_test) **AND (ii)** critic_spread_test ≤ 0.95 × min(full_test, agnostic_test). (≥5% 상대 오차 감소, full·agnostic 둘 다 대비)
- oracle 통과+learned 실패 → FAIL (신호 품질). oracle 실패 → robust FAIL (메커니즘 자체). 둘 다 통과 → CONDITIONAL-GO.
- 1회 실행 후 결과 그대로 채택(추가 튜닝 금지).

**한계(synthetic, 다음 DMControl 실험이 분리해야):** env가 high-noise=high-drift 영역을 구조적으로 결합 → critic_spread 성공의 일부가 aleatoric 탐지일 수 있음(prior-entropy 실패가 반증이지만 완전 분리는 실물 실험에서).

## Run 2 결과 (selective bootstrap, 사전등록 기준) — robust FAIL

**판정: FAIL (robust — 메커니즘 자체)**. sanity_full_recovered=true (τ=∞가 full 정확 복원 → formulation 무편향).

| 방법 | held-out test 오차 (|T − G|) |
|---|---|
| full-rollout (개입 없음) | 4.846 |
| **state-agnostic best uniform m=1** | **3.918** |
| oracle (true drift 신호) | 3.942 |
| critic_spread (학습 신호) | 3.918 |
| learned_rho | 3.919 |
| **bar = 0.95 × min(full, agnostic)** | **3.722** |

- agnostic best m = **1**: "무조건 step 1에서 critic value로 bootstrap"이 최선의 균일 전략. full rollout(4.846) 대비 3.918로 개선 — imagined rollout을 더 굴릴수록 drift만 누적.
- **oracle(완벽 신호)=3.942 < agnostic 못 이김(>3.722 bar)**. step별 신뢰로 bootstrap 위치를 골라도 trivial 균일 early-bootstrap을 못 이김. learned 신호도 동일(3.918≈agnostic).
- 결론: **신호 validity는 통과(0.535)했지만, 그 신호를 λ-target reweighting에 쓰는 메커니즘은 이득 0** — 완벽 신호로도 state-agnostic baseline 초과 불가. advisor 사전등록 분기 "oracle 실패 → robust FAIL, 추가 iteration 없음"에 해당.

**왜 (메커니즘 교훈):** 이 regime에서 λ-target 정확도는 critic bootstrap이 지배하고, 가장 이른 신뢰 지점에서 bootstrap하는 것이 이미 최적. imagination rollout을 "더 잘 가중"하는 것은 "더 일찍 bootstrap"이 주는 이득을 넘지 못함. 즉 state-dependent reliability는 actionable advantage가 없음.

**한계(정직)**: synthetic이고 critic이 real-MC-return으로 잘 학습돼 bootstrap이 강함 → 실제 DreamerV3(긴 effective horizon, mid-rollout 정보가 중요할 수 있음)와 다를 여지. 단 (1) 기준을 사전등록하고 1회로 commit했고, (2) oracle 통제가 신호품질과 무관하게 decisive하게 실패했으므로, 이 PoC 게이트에서는 FAIL이 정직한 판정.

**signal validity가 흥미로운 양성 결과(critic_spread가 BL-08 entropy 실패 지점에서 drift 예측)**라는 점은 별도 가치 — 단, 본 아이디어의 *사용처*(λ reweighting)는 기각.

## Run 3 — REAL DreamerV3 (DMControl) — synthetic caveat 해소, FAIL 확정

GL infra 수리(`LD_LIBRARY_PATH=/home/jovyan/egl_libs`, 설치 불필요) 후, 실제 vanilla DreamerV3를 dmc_proprio walker_walk에 학습(50k step 설정, train_return ~621/1000, 의미있게 학습됨). `experiment_real.py`로 **동일 사전등록 분석**을 실제 RSSM/critic으로 재현 (N=512, H=15).

| 신호 | t-controlled Spearman | (synthetic 대비) |
|---|---|---|
| critic_spread | **-0.058** | synthetic 0.535 → 실물 음수 |
| learned_rho | 0.114 | 0.373 → 0.114 |
| prior_entropy (BL-08) | 0.131 | -0.185 → 0.131 |

feasibility (selective bootstrap, held-out test): **full = agnostic(m=15) = oracle = critic_spread = 5.096** (전부 동일). sanity_full_recovered=true.

**판정: FAIL (signal validity gate failed on real model).**

**해석 (synthetic과 다른 메커니즘, 같은 결론):**
- 실물 DreamerV3에선 **신호가 drift를 예측 못함**(critic_spread -0.058) — synthetic의 강한 신호(0.535)가 실물에선 사라짐. (실물 critic 분포 spread가 state별 변별력이 낮거나, H=15 prior-only imagination이 충분히 정확해 drift 자체가 작고 균일.)
- **agnostic_best_m=15 = full rollout이 최적** → "신뢰 낮은 far step"이 존재하지 않음. oracle(완벽 신호)조차 full을 못 이김(개입 = 손해). reweighting/selective-bootstrap이 고칠 **대상 자체가 없음.**
- synthetic("신호 valid하나 critic이 지배 → m=1")과 실물("imagination 정확 → m=15, 신호 무효")은 정반대 regime이지만 **둘 다 아이디어 전제를 반박.** synthetic caveat("실물은 다를 수 있다")는 해소: 실물은 더 깨끗하게 FAIL.

**한계(정직)**: 단일 task(walker-walk), 단일 seed, proprio, 50k step(중간 수준 학습). 더 긴 학습/어려운 task/긴 effective horizon에서 imagination drift가 커지면 신호가 살아날 여지는 이론상 남으나, 표준 DreamerV3(H=15) regime에서 신호가 음수~0.13이고 full rollout이 최적이라 메커니즘 동작 징후가 없음. → funnel 종결 FAIL로 충분.

**artifacts**: `experiment_real.py`, `experiment_real_results.json`, `real_train.log`, ckpt `/data/jameskimh/worldmodel/kl-reliability-poc/walker_proprio_real_s0/`

