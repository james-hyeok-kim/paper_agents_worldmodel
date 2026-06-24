---
slug: round7-nonplanning-saturation-map
status: saturation-report
created: 2026-06-19 KST
base: TD-MPC2 (non-planning axes)
note: "Round 7(TD-MPC2 비-planning 축) 결과: 0 clean. 모든 도달 cell이 특정 named paper로 선점. + R4~R7 cross-round 전략 합성(survival이 bottleneck, generation 아님)."
---

# Round 7 — TD-MPC2 Non-Planning Saturation Map + Cross-Round 전략

team-lead가 R6 후 planning-efficiency를 닫고(BL-15: vmap Q-ensemble은 wall-clock 무이득) **TD-MPC2 비-planning 축**(representation/value-learning/policy/robustness/multitask)을 요청. 결과: **0 clean**. 도달한 모든 cell이 2024-2026 특정 논문으로 선점. reskin 거부(generator 금지사항 + validator BL 보장).

## Cell별 차단 (각각 named paper)

| Cell | 메커니즘 | 차단 (특정 논문) |
|---|---|---|
| decoder-free latent **robustness** (distractor 배제) | consistency가 distractor도 예측하게 함→배제 메커니즘 추가 | **RePo / TIA / Denoised-MDP(2206.15477) / DBC** 선점. **+ 이중 차단: tdmpc2 envs에 distractor harness 부재**(grep 확인, camera만 있음) → 측정 인프라 비용 숨음. 기본 obs도 proprio라 visual-distractor는 off-path. |
| **consistency / self-predictive representation** quality | latent collapse 방지/quality 향상 | **iQRL(2406.02696)** 정확 선점: self-supervised latent-consistency + quantization으로 collapse 방지, DMControl서 representation-learning SOTA. TD-MPC 명시 인용. **+ SPR / Tang et al.(Self-Predictive RL, ICLR 2024)**가 "TD-MPC=self-predictive abstraction member"로 분류. + SimNorm 자체가 Simplicial Embeddings(ICLR) 별도 라인. |
| **Q-ensemble value-learning** quality | ensemble disagreement를 학습에 활용 | pessimism=REDQ(baseline에 이미 min-2), prioritization=**UPER**(R4b 차단), exploration=**Plan2Explore/disagreement**(R5 차단), diversity-reg=**EDAC**, adaptive aggregation=**DEA(Directional Ensemble Aggregation, online RL)**. "fifth mechanism" 부재. TD-MPC2 overestimation은 **TD-M(PC)²(ICLR 2025)**가 이미 다룸. |
| **policy prior (pi-planning gap)** | pi와 planner 간극 co-training | **TD-M(PC)²(2502.03550, ICLR 2025)** 정확 선점(distribution-constrained policy iteration). |
| **multitask interference** | task gradient conflict 완화 | **One Model for All Tasks(2509.07945)** + gradient-surgery(PCGrad/CAGrad) 정립 field, TD-MPC2-specific 아님→"apply X to control". |

## 차단의 성격 (왜 reskin 거부)
각 차단은 *vague adjacency*가 아니라 *exact cell의 named paper*다. 이전 라운드처럼 "3 differentiator"로 살릴 수 있는 게 아니라, 메커니즘 자체가 동일. 0 clean은 **specific block 때문에 옳은 판정**이지 "under-deliver가 미덕"이라서가 아니다.

## Cross-Round 전략 합성 (R4~R7) — 이번 라운드의 진짜 output

| Round | axis/base | generation | **survival** |
|---|---|---|---|
| R4 | family pivot efficiency | 0 | 0 |
| R4b | DreamerV3 quality/sample-eff | 1 (learned-kl-reliability) | **0** (BL-14 FAIL synth+실물) |
| R5 | robustness/OOD/eval-diagnostic | 1 (decision-fidelity-atlas) | **0** (INCREMENTAL drop, value-equivalence 선점) |
| R6 | TD-MPC2 planning efficiency | 2 (elite-staged, adaptive-budget) | **0** (BL-15 wall-clock FAIL + parked) |
| R7 | TD-MPC2 non-planning | 0 | 0 |

- **R4~R7 net survivable = 0.** 유일한 paper-grade asset(dual-rate)은 이 구간 *이전*이고, efficiency가 아니라 **quality/sample-eff**로 이겼다.
- **bottleneck은 generation이 아니라 survival**이고, survival은 (1) wall-clock(vmap/kernel/Amdahl 함정 — BL-10/12/15), (2) literature saturation 둘로 gate된다.
- **"fresh base 추가"는 이제 *테스트되어 불충분*으로 판명**(R6에서 generation 1→2로 올렸으나 survival 여전히 0). 새 base는 wall-clock도 saturation도 못 고친다. (R5 전략노트의 "fresh base = lever" 가설 *철회*.)

## team-lead에게 줄 결정지점 (cross-round 데이터 기반)
1. **efficiency-framing 중단 또는 gate 변경**: 이 base들에서 efficiency는 wall-clock graveyard. venue가 FLOPs-as-gate를 명시적으로 받거나(일부 ML sys venue), efficiency-framed idea를 멈춘다.
2. **dual-rate에 compute 집중** (유일 survivor) — 더 생성하지 말고 검증된 1개를 paper로.
3. **덜 worked된 *problem setting*** (새 capability/benchmark/domain)을 team이 정의 — 같은 base에서 같은 axis를 도는 한 saturation은 계속됨. 이게 유일하게 *미테스트*된 lever.

## 메타교훈 (다음 generator 세션)
- "local-fresh ≠ global-fresh"가 7라운드 반복 확인. cell의 named paper를 *생성 전*에 찾아라(RePo/iQRL/DEA/TD-M(PC)²는 전부 cell 정확 일치).
- TD-MPC2 representation = self-predictive(SPR/iQRL/Tang) cluster, SimNorm = Simplicial Embeddings 라인. 이 둘은 닫힘.
- efficiency idea는 wall-clock microbench(GPU+CPU)를 *생성 단계에서* 가늠하라(vmap/sequential 구조 확인). BL-15.
- 0 clean은 specific-block일 때 valid한 판정 — reskin보다 낫다. 단 "0이 미덕"이 아니라 "block이 specific해서 0"임을 구분.
