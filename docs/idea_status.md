# Idea Status Dashboard

World Model 연구 아이디어 전체 상태 현황.

## 파이프라인

```
생성(generator) → 문헌검증(literature-checker) → PoC(validator) → 실험계획(planner) → 실험실행(runner)
```

## 아이디어 현황

| Slug | 카테고리 | 문헌검증 | PoC | 실험 | 상태 |
|---|---|---|---|---|---|
| predictive-horizon-controller | A | INCREMENTAL | FAIL (speedup 1.0×) | — | 기각 |
| latent-delta-rollout | B | INCREMENTAL | FAIL (quality ~1.9) | — | 기각 |
| branch-shared-imagination | D | INCREMENTAL | FAIL (speedup 1.11×) | — | 기각 |
| **dual-rate-world-model** | E | INCREMENTAL | **CONDITIONAL-GO** (1.93×) | **GO** | **논문 가능** |
| static-dynamic-decoupled-wm | G | INCREMENTAL | FAIL (speedup 1.2×) | — | 기각 |

## Round 2 아이디어

| Slug | 카테고리 | 문헌검증 | PoC | 실험 | 상태 |
|---|---|---|---|---|---|
| residual-corrected-sparse-rssm | B | INCREMENTAL | FAIL (speedup 0.72×, quality 0.25) | — | 기각 |
| selective-imagination-cheap-critic | C | INCREMENTAL | FAIL (speedup 0.49×, proxy corr 0.085) | — | 기각 |
| policy-gated-progressive-wm | F | INCREMENTAL | FAIL (speedup 0.49×, cheap 71% of exp) | — | 기각 |
| subtree-reuse-muzero | D | INCREMENTAL (weak) | — | — | 보류 (re-grounding 문제) |
| iris-token-budget-rollout | C | INCREMENTAL | FAIL (speedup 0.66×, tokenizer 76%) | — | 기각 |

## Round 2 실패 패턴 요약
- **GPU kernel overhead**: 소형 모델(D=128 vs 512) 교체 speedup 이론치 16× → 실제 1.41×. call을 완전히 skip해야만 실질 speedup 가능
- **Amdahl precondition**: 가정된 bottleneck이 실제로는 hot-path 아닌 경우 다수 (prior MLP > GRU, encoder > imagination, tokenizer > transformer)
- **Proxy quality**: 1-step latent proxy가 H-step return variance를 예측 못함 (Pearson 0.085)
- **성공 패턴**: dual-rate처럼 K-1 GRU call을 완전히 생략하는 구조만 실질 speedup 달성

## Round 3 아이디어

| Slug | 카테고리 | 문헌검증 | PoC | 실험 | 상태 |
|---|---|---|---|---|---|
| residual-corrected-sparse-rssm | B | INCREMENTAL | FAIL (speedup 0.72×, drift 0.25) | — | 기각 |
| selective-imagination-cheap-critic | C | INCREMENTAL | FAIL (proxy corr 0.085) | — | 기각 |
| policy-gated-progressive-wm | F | INCREMENTAL | FAIL (speedup 0.49×) | — | 기각 |
| amortized-rollout-operator | D | INCREMENTAL | FAIL (GPU 0.99×, CPU 6.1×) | — | 부분 결과 |
| spectral-multilevel-rssm | E | INCREMENTAL | 실행 중 | — | pending |
| value-bootstrapped-depth-scheduler | A | INCREMENTAL | FAIL (delta 1.7, speedup 0.73×) | — | 기각 |
| action-quantized-planning-cache | C | INCREMENTAL | FAIL (speedup 0.84×) | — | 기각 |
| tokenizer-stride-iris | C | NO-GO | — | — | 기각 |

## Round 4 (2026-06-18 KST) — family pivot 시도 → 포화 확인

요청: transformer/diffusion/JEPA 등 **새 WM family**에서 efficiency 아이디어 5개.
결과: **7/7 probe cell 포화** (2025-26 문헌에 선점). clean survivor 0~1개. → 5개 생성 대신 saturation map 반환.

| Probe | 차단 |
|---|---|
| STORM transformer-WM | hot-path 아님 (transformer cheap, 학습 9.3 V100h) — 탈락 |
| DINO-WM (JEPA eff) | rollout hot-path cite 불가 — 탈락 |
| diffusion adaptive NFE | DISK (arXiv:2602.00440) 선점, DIAMOND n=3 floor |
| JEPA planning-aware quality | arXiv:2602.18639 + C-JEPA 선점 |
| diffusion quality-at-fixed-NFE | DIAMOND monolithic U-Net → spatial-selective 불가(unsound) |
| prediction-error prioritized replay | MaPER/Curious Replay/UPER 선점 |
| diffusion memory / drift | EDELINE / Self-Forcing 등 선점 |

**Open lane (generator 권고): DreamerV3 quality/sample-efficiency 축** — BL-06~13(전부 call-reduction efficiency)에 안 막혀 있고, 유일 성공작 dual-rate의 실제 성과(+12% sample-eff)와 일치.
상세: `.claude/agent-memory/worldmodel-idea-generator/pending/round4-pivot-saturation-map.md`

## Round 4b (2026-06-18 KST) — DreamerV3 quality/sample-eff 축 (방향 재확정 후)

요청: DreamerV3 RSSM에서 return/sample-eff 개선 아이디어 (raw speed 아님). literature-fresh 먼저, collision manufacturing 금지.

| Slug | 카테고리 | 문헌검증 | PoC | 상태 |
|---|---|---|---|---|
| learned-kl-reliability-lambda | quality/sample-eff | INCREMENTAL | FAIL (synth: signal 0.535 but reweight 무이득) | 기각 (실물 재확인) |

**실물 DreamerV3 후속 검증 (2026-06-19 KST):** GL infra 수리 후 실제 DreamerV3(dmc_proprio walker_walk, train_return~621)에서 동일 분석(N=512). signal validity **붕괴**(critic_spread t-ctrl -0.058, learned_rho 0.114, 전부<0.3), feasibility는 full=agnostic(m=15)=oracle=5.096 전부 동일(imagination이 H=15에서 정확→reweight 대상 없음). synthetic과 정반대 regime이지만 **FAIL 재확정**. caveat 해소. artifacts: experiment_real.py / experiment_real_results.json.

**PoC 판정 (2026-06-18 KST, 사전등록·advisor-hardened, synthetic):** signal validity는 PASS(critic_spread가 decision-relevant drift와 t-controlled Spearman 0.535, BL-08 prior-entropy는 -0.185), 그러나 feasibility robust FAIL — sign-unbiased selective bootstrap에서 **oracle(완벽 신호)조차 trivial state-agnostic early-bootstrap(m=1)을 못 이김**(oracle 3.942 vs agnostic 3.918, bar 3.722). λ-target은 critic bootstrap이 지배 → reliability reweighting 무이득. → BL-14 등재. (DMControl 실물은 EGL/libglvnd 부재로 불가, synthetic으로 게이트.)

생성 1개 (clean). 나머지 후보는 literature-blocked로 미생성 (collision manufacturing 회피):
- 멀티샘플 posterior-expectation critic target → **Probabilistic Dreaming (arXiv:2603.04715)** 정확 선점 (multiple latent particles, variance reduction of critic target)
- reward-relevant capacity allocation / reward-head 강화 → **value-equivalence 클러스터**(Grimm 2020, VAML, Value-Consistent RL 2206.12542) + **MuDreamer/R2-Dreamer/Dreamer-CDP** 선점
- prediction-error/uncertainty prioritized WM replay → **MaPER/Curious Replay/UPER** 선점 (team-lead가 사전 차단 명시)
- JEPA planning-aware representation → **arXiv:2602.18639/C-JEPA** 선점 (team-lead 사전 차단)

핵심 아키텍처 발견(코드 확인): DreamerV3 imagination은 **prior-only rollout**(`ImagBehavior._imagine`의 `img_step`만, posterior 없음). imagined step에 posterior-prior KL이 존재하지 않음 → "이미 계산된 KL을 읽어 가중"식 아이디어는 불가. anchor는 KL을 *학습된 predictor head*로 prior-only proxy화해 우회(BL-08 entropy 함정도 회피).
결론: quality/sample-eff 축은 family-pivot보다 덜 포화됐으나 여전히 상당히 worked. clean survivor 1개.

## Round 5 (2026-06-19 KST) — robustness/OOD/eval-diagnostic pivot

요청: efficiency·quality 벗어나 미탐색 axis(robustness/OOD/eval-diagnostic). 실물 DMControl 가능. literature-fresh 먼저, BL-14 재유입 금지.

| Slug | axis | 문헌검증 | PoC | 상태 |
|---|---|---|---|---|
| decision-fidelity-atlas | eval-diagnostic | INCREMENTAL (thin) | — | **drop (전략적 deprioritize, 2026-06-19)** |

**Novelty (2026-06-19 KST, team-lead 직접):** INCREMENTAL이나 **margin 얇음**. 핵심 개념("state-wrong이어도 decision-right")을 value-equivalence(Grimm 2020/Proper VE 2106.10316, VAML/IterVAML, **Model-Advantage 2106.14080**)가 선점. 차별점은 training objective가 아닌 diagnostic atlas + DF_dyn/DF_rew 분해뿐. deliverable이 method 아닌 benchmark/diagnostic.

생성 1개 (clean, eval-diagnostic 축). 나머지 후보 미생성(documented):
- **OOD/dynamics generalization (contextual-WM cell)** → **cRSSM, DALI, Dreaming of Many Worlds(2403.10967), Augmented WM, Prototypical context-aware** 선점. (cell 한정 포화 — axis 전체 단정 아님)
- **action-noise/policy-perturbation robustness** → **PR-MDP/NR-MDP**(action-robust MDP 정립 이론) + 깨끗한 WM-specific 메커니즘 부재("apply domain randomization to DreamerV3" 수준이라 미생성)
- **sensor/observation-noise robustness** → **Surprise Recognition(2512.01119)** 선점 (Bayesian-surprise rejection sampling, DreamerV3+Cosmos)
- **RSSM multimodal-collapse / "product laws not joint" 진단** → 이미 *문서화된 limitation*(gap 아님) + **Probabilistic Dreaming(2603.04715)** fix 충돌
- **imagination calibration 진단** → 부분 fresh이나 실패모드(RSSM aleatoric 과대팽창)가 이미 문서화(non-obvious 게이트 약함) + decision-fidelity-atlas와 collapse 위험 → 미생성

핵심 근거: 2025 WM 서베이(2510.16732)가 imagination 실패-taxonomy/reliability-diagnostic/imagined-vs-real divergence 메트릭 **없음**을 명시("genuine gap in evaluation infrastructure", fetch 확인) → eval-diagnostic이 셋 중 가장 fresh. anchor는 BL-14를 *증명된 사실*이 아니라 *해소할 열린 질문*으로 reframe(state-fidelity가 decision-fidelity를 예측하는가)하고, NULL을 사전등록(진단이 실패할 수 있게). reliability-reweighting과 hard firewall(BL-14 재유입 금지).
결론: eval-diagnostic 축 clean survivor 1개. robustness/OOD는 열어본 cell이 대부분 선점.

## Round 6 (2026-06-19 KST) — TD-MPC2/Puppeteer (새 base 2종, planning hot-path)

요청: 새 base(TD-MPC2 MPPI planning / Puppeteer hierarchical)에서 idea. MPPI planning이 TD-MPC2 진짜 hot-path(BL-11/Amdahl 함정의 정반대). 실물 측정 가능.

| Slug | base | axis | 문헌검증 | PoC | 상태 |
|---|---|---|---|---|---|
| elite-staged-value-planning | TD-MPC2 | efficiency(planning) | INCREMENTAL | **FAIL (microbench GPU 0.94×/CPU 1.011×)** | 기각 (BL-15) |
| adaptive-mppi-budget | TD-MPC2 | efficiency(planning) | INCREMENTAL (thin) | — | 보류 (novelty thin, 선점) |

**Round 6 PoC 결과 (2026-06-19 KST, advisor-ordered microbench die-point):**
- **elite-staged-value-planning = FAIL.** 실물 TD-MPC2 wall-clock 마이크로벤치(best-case cheap-rank=distilled scalar): GPU **0.94×(느려짐)**, CPU **1.011×(flat)** — 양 device 모두 gate 1.2× 미달, edge pivot도 불가. 원인: terminal Q는 vmap **vectorized 단일 op**(512→64 축소가 kernel-launch-bound, BL-10/12 재확인) → gen-r4의 "terminal ~50%"는 FLOPs엔 맞지만 wall-clock엔 틀림(dynamics rollout이 horizon-sequential로 wall-clock 지배). → BL-15. 교훈: planning-eff는 wall-clock microbench를 GPU+CPU로 맨 먼저.
- **adaptive-mppi-budget = 보류.** novelty INCREMENTAL-thin — per-state iteration 조기종료가 Adaptive Online Planning(1912.01188)·"Learning to Plan, Planning to Learn"(2512.17091)에 선점. 단 microbench가 보여준 *맞는* bottleneck(iterations/dynamics)을 침 → 실행하면 실 speedup 가능성은 있으나 단독 paper 가치 낮음.
- **Round 6 통과 = 0 / push = 0.** (anchor FAIL, 보완 thin.)

## Round 7 (2026-06-19 KST) — TD-MPC2 non-planning 축 — **0 clean (전부 선점)**

요청: planning-efficiency 밖(representation/value-learning/policy/robustness/multitask). 결과: **0 clean.** 도달한 모든 cell이 2024-2026 named paper로 선점. reskin 거부.

| Cell | 차단 (named paper) |
|---|---|
| decoder-free latent robustness | RePo/TIA/Denoised-MDP(2206.15477)/DBC + **distractor harness 부재(이중차단)** |
| consistency/self-predictive representation | **iQRL(2406.02696)** + SPR/Tang(ICLR24, TD-MPC=self-predictive member) + SimNorm=Simplicial Embeddings |
| Q-ensemble value-learning | REDQ(baseline)·UPER(R4b)·Plan2Explore(R5)·EDAC·DEA·TD-M(PC)² — fifth mechanism 부재 |
| pi-planning gap | TD-M(PC)²(2502.03550, ICLR25) |
| multitask interference | One Model for All Tasks(2509.07945) + PCGrad/CAGrad field |

**Cross-round 합성 (R4~R7): net survivable = 0.** generation은 됐으나(R4b 1, R5 1, R6 2) 전부 死(BL-14/INCREMENTAL-drop/BL-15). 유일 asset dual-rate는 이 구간 *이전* + efficiency 아닌 **quality/sample-eff**로 이김. **bottleneck = survival**(wall-clock vmap/kernel/Amdahl 함정 + literature saturation), generation 아님. **"fresh base 추가"는 R6에서 테스트되어 불충분 판명**(gen 1→2, survival 0 유지).

**team-lead 결정지점 (cross-round 데이터 기반):** (1) efficiency-framing 중단 또는 venue가 FLOPs-as-gate 수용, (2) dual-rate(유일 survivor)에 compute 집중, (3) 덜 worked된 새 problem-setting/domain 정의(유일 미테스트 lever — 같은 base/axis는 saturation 지속).
상세: `.claude/agent-memory/worldmodel-idea-generator/pending/round7-nonplanning-saturation-map.md`

## Round 8 (2026-06-19 KST) — least-worked corner (사용자 "계속 생성" 선택) — **0 clean + 1 open-space pointer**

요청: 사용자가 "계속 생성" 명시 선택. efficiency 금지 + 포화 cell 금지 + option3(가장 덜 worked된 새 problem-setting). 결과: **0 clean mechanism-idea.**

**구조적 reframe (R8 headline):** freshness-check 방법론은 *genuinely-new problem-setting을 certify 불가* — novel setting은 대조 literature가 없음. 모든 후보가 둘로 갈림: clean check 통과⟹mapped⟹saturation/wall-clock 死 / genuinely-new⟹literature 없음⟹search가 de-risk 불가. option3은 후자를 요구하는데 freshness-check는 전자용 도구 = 구조적 bind. 따라서 valid open-space 판별은 "literature-clean"이 아니라 **3-part test**: (1)harness 존재, (2)face-value 가치, (3)mechanism 아님.

**기각한 후보(rationalization 회피):** WM-of-frozen-sub-policy/OOD-command 진단(diagnostic-shaped+harness 미검증+OHIO 인접) / Puppeteer transfer·reuse(Puppeteer ICLR25 자체 contribution+THICK Dreamer/OHIO/MetaWorld-skill-comp) / skill composition(MetaWorld 2601.17507).

**THE ONE BET (de-risk 불가, validator에서 산다/죽는다):** "frozen 유능 low-level 하 high-level whole-body task 획득의 sample-efficiency frontier와 한계요인" (Puppeteer transfer setting). harness 존재(TransferWrapper+RunThroughCorridor/Walk/Run, 코드 검증). Puppeteer가 최종 성능만 보고하고 acquisition curve/floor는 미측정. 단 diagnostic-shaped라 "non-obvious AND 결정변경" gate에서 산다/죽는다 — 가치는 human research taste, 내가 literature로 보증 불가.

**EV 순위 R7과 불변:** dual-rate 1편(검증된 survivor) > 죽을 가능성 높은 새 idea. 진짜 미테스트 lever는 *human이 정의하는 새 problem-setting*뿐.

## Round 9 (2026-06-20 KST) — cross-paradigm lever (3-part test 적용) — **0 clean (lever 닫힘)**

요청: team-lead가 3 작동 base 대비(RSSM imagination vs MPPI planning vs hierarchical)를 새 lever로 제시, 3-part test로 평가.

**lead 후보**: cross-paradigm per-state competence complementarity(Dreamer vs TD-MPC2가 같은/다른 state에서 실패하나). freshness는 fresh slice 존재(per-state cross-substrate 비교 미발견; 단 "state-dependent value of planning"은 TD-M(PC)²/2512.17091/2510.04280이 TD-MPC2 내부서 선점).

**4-bar 자가판정(validator 전) → FAIL at bar 4(distinguishable, decisive):** Dreamer vs TD-MPC2 차이를 *paradigm*에 귀속 불가 — 두 agent는 codebase·network·budget·구현 수십 개가 다름(paradigm/implementation 교란). bars 2,3(surprising/decision-changing)도 weak(가장 그럴듯한 결과=둘 다 같은 hard state 실패=unsurprising; router 비현실).

**R9 finding: bar-4 confound가 cross-paradigm lever *전체*를 닫는다.** comparison-shaped→confound(두 codebase) / combine-paradigms→mechanism(mapped/BL, RL-MPC·TD-M(PC)² 선점) / 유일 low-confound인 Puppeteer-vs-flat-TD-MPC2→Puppeteer ICLR25가 이미 보임(선점). 세 경로 다 닫힘 = R8 bind의 cross-paradigm 인스턴스.

**방법론이 win**: 4-bar가 validator *전에* 후보를 죽임(설계대로, track record: decision-fidelity-atlas·R8 bet·BL-14/15 예측). 9라운드 = 9 실패가 아니라 고신뢰 negative map + 미완 자산 1개.

**forward action(R10 생성보다 우선): R8 Puppeteer acquisition-curve bet을 validator로** — pointer만 냈지 4-bar pre-registration 미통과. 9번째 생성 전 8번째 테스트가 맞다. EV 불변(dual-rate > 새 idea).
상세: `.claude/agent-memory/worldmodel-idea-generator/pending/round9-crossparadigm-confound.md`

## Round 10 (2026-06-20 KST) — **첫 human-defined lever**: DreamerV4+Puppeteer inference efficiency — **0 clean (두 arm 닫힘)**

사용자 직접 지시(9라운드 만에 첫 human lever). arm 분리 평가.

**Arm A — Puppeteer 2-level planning efficiency → CLOSED (event-triggered MPC).** 코드확인: Puppeteer는 진짜 2 MPPI를 매 step 구동(high `agent.act` + low `low_level_policy.act`=별도 full TD-MPC2, transfer.py:86,130). high-level adaptive skip은 wall-clock 절감 실재(call-skip, BL-15 함정 아님). **그러나 novelty 차단**(R6에서 막힌 건 Amdahl이 아니라 novelty; code finding은 Amdahl만 확립). 제어이론 freshness(RL 아닌 control 키워드): fixed skip=multi-rate/hierarchical MPC, adaptive skip=**event-triggered/self-triggered MPC**(연산절감으로 *정의된* 서브필드). learned-value-of-replanning trigger 변형조차 선점(Ren 2026 HRL-event-triggered, 2208.10302, MPC+learned-value 2025). → R6 finding을 code+named-field로 확정.

**Arm B — DreamerV4 inference efficiency → CLOSED (저자 자기선점+코드부재).** DreamerV4(2509.24527) 헤드라인이 inference efficiency(shortcut forcing 16×+efficient transformer) — 저자 홈그라운드. 공식코드 없음→analytic-only→wall-clock gate 불가(BL-15 전례). fresh angle이어도 validator 통과 불가.

**통합 강제 안 함**(bottleneck 다름=통합 함정).

**R10 finding(human 재조준용): human lever 형식(human-defined)은 맞았으나 *efficiency* framing이 Puppeteer를 graveyard에 떨어뜨림.** dual-rate(유일 survivor)는 efficiency 아닌 **quality/sample-eff**로 이김 — 패턴: efficiency framing 死, quality/sample-eff 生. arm을 몰래 quality로 안 바꿈(human이 efficiency 명시). human 재조준 표면화. 재조준 시 Puppeteer capability/sample-eff = R8 acquisition-curve bet 공간(validator 미통과).

**forward 불변: R8 bet을 validator 4-bar로(R11 생성보다 우선).**
상세: `.claude/agent-memory/worldmodel-idea-generator/pending/round10-human-lever-efficiency.md`
상세: `.claude/agent-memory/worldmodel-idea-generator/pending/round8-openspace-pointer.md`

## Round 11 (2026-06-20 KST) — DreamerV3+Puppeteer inference efficiency (둘 다 runnable) — **0 by construction**

사용자가 R10 lever 수정(DreamerV4→DreamerV3). 결과 0 — search 없이 코드+문서로 4벽 확정.

**핵심 kill: DreamerV3 arm은 saturation이 아니라 construction으로 0.** 코드확인(`dreamer.py:94 _policy`): DreamerV3 inference=encoder+RSSM+actor 1 forward/step, **decoder 없음·planning 없음** → 이미 싸다. 비용은 전부 *training*(imagination/WM update/decoder reconstruction). 즉 DreamerV3는 inference-efficiency 대상이 *구조적으로* 아님(한쪽 base에 inference hot-path가 거의 없음).

**4-wall**: (1) DreamerV3 inference=near-empty surface; 존재 비용은 training=BL-06~13+reconstruction-free(R7). (2) Puppeteer=event-triggered MPC 선점(R10); non-ET 각도(low-level MPPI→policy-prior ablate)=state-dependent-value-of-planning worked(R9/TD-M(PC)²). (3) 통합=substrate 없음(DreamerV3에 공유할 inference hot-path 자체가 없음)→collision trap. (4) wall-clock microbench=살아남은 후보 0이라 미도달.

**drift 차단**: "decoder training cost" 각도는 off-lever(training≠inference)+pre-empted(reconstruction-free R7). 양 해석(inference/training) 모두 0.

**R11 binding constraint = validator wedge(escalation)**: R9·R10에서 권고한 "R8 bet→validator" 미실행(validator wedge, team-lead 수동). binding은 이제 "라운드가 0인가"가 아니라 **유일 testable asset(R8 bet)이 gate에 도달 못 함**. → validator unblock이 우선, 아니면 loop는 construction상 0만 반환.
상세: `.claude/agent-memory/worldmodel-idea-generator/pending/round11-dreamerv3-puppeteer-efficiency.md`

**핵심: Amdahl을 코드로 검증(6라운드 중 최강 precondition).** `_plan`(tdmpc2.py L140-207)이 매 env step마다 iterations=6 × `_estimate_value`(horizon=3 × num_samples=512 × [next+reward] + terminal pi+5Q). horizon=3에서 **terminal Q-ensemble term ≈ per-sample 비용 50%**, dynamics term ≈ 50%. 두 idea가 *각각 다른 절반*을 침 → orthogonal(collision 아님):
- **elite-staged-value-planning (anchor)**: terminal term. cheap-rank(1~2 head/distilled)로 512 순위→full 5-head는 elite 64개만. chicken-and-egg(elite 찾으려면 512 value 필요) 수정 반영. BL-07 merge 아님(발산 무관), REDQ/DC-MPC critic-subsample은 *learning bias*용이라 차별.
- **adaptive-mppi-budget (보완)**: dynamics term. 수렴 신호(elite-value var/action std/mean Δ, MPPI 내부서 이미 계산=관측가능→BL-08 함정 회피)로 iteration/sample per-state 조기종료. 단 "후반 iteration 잉여" 전제는 EMPIRICAL-ONLY(die-point #1, 실행해야 확인). anchor가 Amdahl-verified라 deliverable이 이 전제에 비의존.

미생성(documented, collision 회피):
- **MPPI 후반 iteration trajectory reuse/merge** → 코드에 이미 warm-start(L168 `_prev_mean`)+policy-prior seed(L155-161 num_pi_trajs) 존재(repo 자체가 선점) + BL-07 continuous-action 발산(merge rate ~10%)으로 메커니즘 공허.
- **Puppeteer hierarchical: high-level 저빈도 replan / command persistence** → options/timed-subgoals/command-commitment heavily worked. "plan less often at high level"은 novel 아님. high/low *interface*(frozen tracker, command repr, credit assignment)에 crisp 메커니즘 없어 미생성(manufacturing 회피).
- **Q-ensemble disagreement를 planning에 quality로** → ensemble-pessimism/UPER 선점(Round 5 발견).
- **decoder-free latent OOD/robustness** → TD-MPC2-specific 메커니즘 없으면 "apply X to control".

결론: TD-MPC2 planning hot-path는 Amdahl-verified라 BL-07 graveyard와 다름. clean 2개(orthogonal, 같은 cost equation의 다른 절반). Puppeteer는 interface 메커니즘 부재로 보류. team-lead가 fresh base 추가한 것이 throughput 레버였음을 입증(R5의 1개 → R6의 2개).

## Round 12-14 (2026-06-22 KST) — MT framing 전환 + 신규 아이디어 2종 — **0 (BL-19, BL-20)**

R12에서 framing을 efficiency에서 multitask/generalization으로 전환. R13, R14 두 아이디어 모두 PoC FAIL.

**R13 — embedding-only task onboarding (BL-19, FAIL)**:
- 가설: frozen TD-MPC2 backbone + 96-dim embedding row만 최적화로 new task 적응.
- PoC 결과: collapse-proof eval에서 embed-only가 full FT보다 3~100× 높은 eval loss. in-manifold(mass=1.0)에서도 gap=66% → NULL 조건 적용. dynamics function 변화에 embedding-only 무력.

**R14 — task-conditioning locus (BL-20, FAIL)**:
- 가설: same-embodiment family에서 dynamics는 wrong embedding에 robust, reward는 fragile → 분리.
- PoC 결과: (1) Tautology — walker-stand/walk/run은 same physics XML 공유, dynamics identical → robustness는 trivial. (2) reward arm OOD — velocity sweep states가 훈련 분포 외부, model이 잘못 예측(v=1에서 pred=0.39 vs actual=0.83). Q1 mean ratio = -0.48, FAIL.
- 핵심 교훈: same-embodiment = same physics → dynamics robustness는 발견이 아닌 설계 결과.

**R15b CLOSED (2026-06-22 KST)** — quality/sample-eff pivot도 0 clean.

- **R15**: efficiency b1/b2/b3 literature closure (PaMoRL, Looped World Models 6일 전 선점)
- **R15b**: quality/sample-eff pivot도 0 clean — binding gate = **un-absorbability** (잘 튜닝된 WM+planner가 같은 데이터로 구조를 흡수). 두 후보 死:
  - controllability-whitening → WM이 command→consequence 매핑 전체 흡수 + WPT/XTRA 선점
  - low-level internals as privileged signal → Puppeteer `high_level_obs ⊇ low_level_obs` (code 반증), privileged 아닌 computable feature

**5번째 escalation. 추가 결정 필요.**

**R15c (2026-06-23 KST)**: b1/b2 mechanism triage → 0. efficiency는 framing-dead 확정.
- b1 sub-directions: dim-reduction(launch-bound=BL-10), factored(BL-18), quantized(baseline already), IB(quality가 아닌 efficiency 불가)
- b2 sub-directions: sampling/elite(BL-15/16), algorithm change(sequential rollout Amdahl dominant), MuZero(infra 없음 + INCREMENTAL already)
- 6번 연속 closure: 모든 WM-novel lever가 {call-skip / batch-reduction / small-op / adaptive-stop / gradient-skip / sequential-depth} 중 하나로 환원
- **남은 경로**: (a) non-launch-bound new base(MuZero infra 없음) 또는 (b) quality/sample-eff의 what-data-enters-buffer 방향(crowded but open)

**R15d (2026-06-23 KST)**: what-data-enters-buffer → 0. data-collection ceiling closure 확정.
- Plan2Explore가 dense-reward DMC에서 oracle과 거의 동등(ceiling) → headroom 없음
- 5각도 전부 死: disagreement=Plan2Explore본체, task-conditioned=dominated, novelty=RND/count선점, pred-error=ICM선점, offline→online=infra부재
- **3개 카테고리 모두 소진**: efficiency(launch-bound) + quality(un-absorbable) + data-collection(ceiling-capped)
- **남은 경로**: (1) sparse-reward/hard-exploration 새 base(infra 신규 구축 필요) 또는 (2) human-defined problem-setting(R8 Puppeteer acquisition-curve bet, runnable infra 존재)

**R8 Puppeteer acquisition-curve bet — PASS ✓ (2026-06-24 KST)**
- CondA(trained tracker) vs CondB(random tracker), corridor task, 200k steps, seed×2
- AUC_100k ratio = **4.46×** (gate ≥2× → PASS ✓)
- Final return: CondA 15.13 vs CondB 2.09 (**7.25×**)
- T_50: CondA 20–40k steps vs CondB >180k (never learns)
- 핵심: tracker competence가 high-level 학습의 필요조건. random tracker로는 전혀 학습 불가.
- 결과 파일: `experiments/wip/puppeteer-acquisition-curve/result_001.md`
- 다음: worldmodel-idea-validator 정식 제출

## 실험 최종 결과 (2026-06-12 KST)

**dual-rate-world-model (DMControl Walker-Walk, seed=0, 115k steps)**:
- FLOPs 감소: **1.783× (분석, K=3)**
- 파라미터: **-25%** (D=384 vs baseline D=512)
- Late-stage quality (75k-105k): baseline 740.0 vs dualrate 730.2 (**-1.3%, 목표 <-15% 달성**)
- Best checkpoint (95k): dualrate **806.8 vs baseline 717.8 (+12.4%)** ← sample efficiency
- Wall-clock fps: 0.926× (GPU kernel launch overhead, 작은 모델의 알려진 한계)
- Training 안정성: separation_score=0.717, collapse 없음

## Round 2 실패 패턴 요약
- **GPU kernel overhead**: 소형 모델(D=128 vs 512) 교체 speedup 이론치 16× → 실제 1.41×. call을 완전히 skip해야만 실질 speedup 가능
- **Amdahl precondition**: 가정된 bottleneck이 실제로는 hot-path 아닌 경우 다수 (prior MLP > GRU, encoder > imagination, tokenizer > transformer)
- **Proxy quality**: 1-step latent proxy가 H-step return variance를 예측 못함 (Pearson 0.085)
- **성공 패턴**: dual-rate처럼 K-1 GRU call을 완전히 생략하는 구조만 실질 speedup 달성
- **amortized-rollout**: CPU 6.1× 가속, GPU 0.55× (대형 batch에서 transformer overhead > parallelism)

## 범례
- 카테고리: A=Adaptive Rollout, B=Latent Compression, C=Selective Imagination, D=Shared Computation, E=Hierarchical, F=Distillation, G=Decoupled
- 문헌검증: NOVEL / INCREMENTAL / NO-GO / pending
- PoC: CONDITIONAL-GO / FAIL / pending
- 실험: GO / WEAK-GO / NO-GO / wip / pending
