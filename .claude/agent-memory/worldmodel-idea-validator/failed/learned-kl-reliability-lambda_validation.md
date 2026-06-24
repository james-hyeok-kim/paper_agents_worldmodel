---
slug: learned-kl-reliability-lambda
verdict: FAIL
date: 2026-06-18 KST
validated-by: team-lead (validator agent spawn 무응답 → 직접 수행, advisor-hardened)
gate: signal-validity PASS, feasibility (selective bootstrap) robust FAIL
---

# PoC Validation — learned-reliability-weighted-λ-return → FAIL

## 요약
quality/sample-eff 아이디어라 efficiency 게이트(speedup>1.5×) 대신 **(1) 신호 validity + (2) reweighting feasibility**로 게이트. synthetic, GL-free (DMControl은 EGL/libglvnd 부재로 불가 → 다음 단계 과제). 기준 **사전등록 후 1회 실행 commit** (result_001.md).

## 결과
- **Signal validity: PASS.** critic_spread가 decision-relevant drift(|reward err|+|value err|, 같은 head)와 **t-controlled Spearman 0.535** (partial-ctrl-t 0.549, pooled 0.282; BL-11 전례 0.085 압도). learned KL-predictor 0.373. prior-entropy(BL-08) -0.185(실패, 예측대로). 신호는 latent-L2와는 약함(0.116) → value-target harm을 정확히 포착(BL-11 함정 회피).
- **Feasibility: robust FAIL.** sign-unbiased selective bootstrap(flagged step에서 critic value bootstrap), held-out test 오차 |T_imag − G_truth|:
  - full-rollout 4.846 / state-agnostic best uniform m=1 **3.918** / **oracle(true drift) 3.942** / critic_spread 3.918.
  - bar(0.95×min)=3.722. **oracle조차 trivial 균일 early-bootstrap(m=1)을 못 이김** → 신호 품질과 무관하게 메커니즘 무이득.
  - sanity: τ=∞가 full 정확 복원(formulation 무편향 확인).

## 왜 FAIL (메커니즘 교훈)
λ-target 정확도는 critic bootstrap이 지배 — 가장 이른 신뢰 지점에서 bootstrap이 이미 최적이고, imagination rollout을 "더 잘 가중"하는 것은 "더 일찍 bootstrap"의 이득을 못 넘음. **state-dependent reliability가 actionable advantage를 주지 못함.** (초기 effective-discount reweighting은 음수-보상 env에서 |return| 축소 confound로 무효였고, advisor 검증 후 sign-unbiased selective bootstrap + oracle/agnostic 통제 + tune/test split으로 재설계해 robust 판정.)

## 한계
synthetic + critic이 real-MC-return으로 잘 학습돼 bootstrap이 강함. 실제 DreamerV3(긴 horizon, mid-rollout 정보 중요 가능)와 다를 여지 있으나, oracle 통제가 decisive하게 실패했으므로 PoC 게이트에서는 FAIL.

## 부산물 (재활용 가치)
critic-distribution spread가 prior-only imagination의 reward/value drift를 state별로 예측한다는 **signal validity 양성 결과**는 별개 가치(uncertainty-aware imagination 진단/조기 종료 등). 단 본 아이디어의 사용처(λ-return reweighting)는 기각.

artifacts: `experiments/wip/learned-kl-reliability-lambda/{poc.py, poc_results.json, result_001.md, poc_run2.log}`

## REAL DreamerV3 후속 검증 (2026-06-19 KST) — FAIL 재확정
synthetic의 단 하나 caveat("synthetic critic이 너무 좋아 bootstrap 지배 → 실물은 다를 수 있다")를 직접 검증. GL infra 수리(LD_LIBRARY_PATH=/home/jovyan/egl_libs, 설치 불필요) 후 실제 vanilla DreamerV3를 dmc_proprio walker_walk에 학습(train_return~621), 동일 사전등록 분석(N=512, H=15) 재현:
- signal validity **실패**: critic_spread t-ctrl Spearman **-0.058**(synthetic 0.535에서 붕괴), learned_rho 0.114, BL08 0.131 (전부 <0.3).
- feasibility: full=agnostic(**m=15**)=oracle=critic_spread=5.096 (전부 동일). 실물에선 H=15 imagination이 충분히 정확해 full rollout이 최적 → reweighting할 대상 없음. oracle도 무이득.
- 결론: synthetic("신호 valid하나 reweight 무이득, m=1")과 실물("신호 무효 + imagination 정확, m=15")은 정반대 regime이나 **둘 다 FAIL**. caveat 해소, 종결.
artifacts(real): `experiment_real.py`, `experiment_real_results.json`, ckpt `/data/jameskimh/worldmodel/kl-reliability-poc/walker_proprio_real_s0/`
