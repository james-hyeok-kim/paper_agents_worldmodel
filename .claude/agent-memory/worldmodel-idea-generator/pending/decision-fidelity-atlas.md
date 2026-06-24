---
slug: decision-fidelity-atlas
status: pending
created: 2026-06-19 KST
axis: evaluation-diagnostic
venue-fit: [NeurIPS Datasets&Benchmarks, ICLR, NeurIPS]
base: DreamerV3-torch (experiments/wip/dual-rate-world-model/dreamerv3-torch), 실물 DMControl
blacklist-delta:
  - "BL-14 (imagination reliability로 target reweighting): 본 아이디어는 reweighting을 *하지 않는다*. reliability 신호로 target/policy를 고치는 어떤 메커니즘도 포함 안 함(BL-14 재시도 금지 준수, hard firewall). 순수 *측정/진단* — imagination이 결정(action-ordering)에 충실한 regime을 map. BL-14는 본 진단이 답할 *열린 질문*(state-fidelity가 decision-fidelity를 예측하는가)에 모순된 힌트를 줬을 뿐, 결론이 아니다."
  - "BL-08 (prior-entropy adaptive horizon): horizon을 조정하지 않음. 고정된 표준 DreamerV3를 측정 대상으로 둠."
  - "vs efficiency BL-01~13 전체: 어떤 연산도 바꾸지 않음. 표준 DreamerV3를 그대로 두고 imagination의 결정-충실도를 진단하는 평가 인프라."
---

# Decision-Fidelity Atlas: DreamerV3 Imagination이 "결정"에 충실한 regime의 체계적 진단

## axis & 한 줄
**evaluation-diagnostic.** 표준 DreamerV3 imagination이 *상태 예측 오차*가 아니라 **정책이 의존하는 action-ordering(advantage 순서)을 보존하는가**를 horizon × 관측양식(proprio/vision) × 분포(in-dist/OOD)의 격자에서 측정하고, 그 충실도를 **dynamics 기여 vs reward-head 기여**로 분해하는 진단 프레임워크 + 메트릭.

## 동기 — 왜 지금, 왜 이게 열린 질문인가
- 2025 WM 서베이(arXiv:2510.16732)는 error accumulation을 challenge로 *지목만* 하고, imagination 실패모드 taxonomy / reliability diagnostic / imagined-vs-real divergence 메트릭을 **제공하지 않는다**("a genuine gap in the field's evaluation infrastructure" — fetch 확인). 즉 진단 인프라 자체가 비어 있다.
- 핵심 열린 질문: **state-space에서 imagination이 틀려도 decision-space(어느 action이 더 좋은가)에서는 맞을 수 있는가?** 기존 평가는 거의 전부 pixel/latent reconstruction error나 return-alignment로 *상태/스칼라* 충실도만 본다. 정책 학습이 실제로 의존하는 건 *상대적 advantage 순서*인데 이걸 직접 진단한 표준 도구가 없다.
- **내부 선행 결과(BL-14)는 이 질문에 모순된 힌트만 줬다 (정직하게)**: synthetic에서는 "완벽한 신호(oracle)조차 state-agnostic 균일 전략을 못 이김"(reweighting 무이득) — decision이 state-fidelity에 둔감할 수 있음을 *약하게* 시사. 그러나 실물 DreamerV3(dmc_proprio walker_walk, H=15)에서는 정반대로 imagination이 *충분히 state-정확*해서 full rollout이 최적이었다(premise-collapse). 이 둘은 같은 결론이 아니다 — 하나는 "state-wrong이어도 decision-right", 다른 하나는 "그냥 state-right". **본 atlas는 이 모순을 체계적으로 해소한다**: state-fidelity와 decision-fidelity의 관계가 regime마다 어떻게 갈리는지 map. (이 점이 "BL-14가 증명했다"가 아니라 "BL-14가 던진 질문을 푼다"인 이유.)

## 제안 메트릭 (핵심 기여)
표준 DreamerV3에서, 실제 환경 상태 `s_t`(또는 posterior로 encode된 실제 trajectory)에서 시작해:
1. **Imagined rollout** vs **real rollout**을 *같은 action sequence 후보군*에 대해 비교.
2. **Decision-Fidelity (DF)**: 각 시작 state에서 K개 후보 action(또는 short action-sequence)에 대해 imagined가 매긴 advantage 순서와 real(환경 rollout + 실제 reward)이 매긴 순서의 **rank correlation**(Kendall τ). state-MSE가 아니라 *순서 보존*을 측정.
   - DF=1: imagination이 어느 action이 더 좋은지 완벽히 보존(state가 틀려도 정책엔 무해).
   - DF 낮음: 정책이 imagination에서 잘못된 action을 선호하게 됨 → 학습 손상.
3. **분해 (sub-result, 1·2 framing을 구분 가능하게 만드는 핵심)**:
   - **DF_dyn**: reward를 *real* reward head로 고정하고 dynamics만 imagined → dynamics counterfactual이 순서를 얼마나 보존하는가.
   - **DF_rew**: dynamics를 *real* next-state로 고정하고 reward head만 imagined → reward head가 순서를 얼마나 보존하는가.
   - DF ≈ f(DF_dyn, DF_rew) 분해로 "어디서 깨지는가"를 dynamics vs reward로 귀속.
4. **State-vs-Decision gap**: 같은 격자에서 state-MSE(또는 latent divergence)와 DF를 둘 다 측정 → 둘의 상관/괴리를 plot. 괴리가 크면 "기존 state-fidelity 메트릭이 정책 영향을 오도한다"는 actionable finding.

## 격자(grid) — breakage가 *그럴듯한* 곳을 겨냥 (expected-fine 셀에 atlas를 두지 않음)
- **Horizon**: H ∈ {5, 15, 50, 100} — 표준 15는 BL-14에서 *expected-fine*이므로 중심이 아니라 anchor 한 점으로만. 진짜 질문은 H≫15에서 DF가 언제 무너지는가.
- **관측양식**: proprio vs vision (vision은 encoder 오차가 추가돼 breakage 가능성↑).
- **분포**: in-dist vs OOD(mass/friction 변형, 배경 perturbation) — OOD에서 DF 붕괴가 가장 그럴듯.
- 각 셀에서 DF, DF_dyn, DF_rew, state-MSE 동시 측정.

## 사전 명시한 NULL (진단이 *실패할 수 있어야* 한다)
- **Null A (do-nothing)**: DF가 모든 격자에서 균일하게 높음(≈1) → "imagination은 결정에 항상 충실, 할 일 없음" = weak/negative 결과. 이 경우 정직하게 negative로 보고.
- **Null B (trivial)**: DF가 state-MSE와 완벽히 단조 상관 → state-fidelity 메트릭만으로 충분, decision 메트릭 불필요 = 기여 약화.
- **Decision-flipping 결과(가설)**: DF가 *특정 셀에서만* 붕괴(예: OOD×vision×H=100은 DF 급락하지만 in-dist×proprio는 유지)하고, **state-MSE와 DF가 괴리**(state 많이 틀려도 DF 높은 셀이 존재) → (1) 어느 regime이 개입 필요/불필요인지 map, (2) reliability-reweighting류 개입이 *왜* 무익했는지(BL-14) 설명, (3) state-fidelity 메트릭의 오도를 입증. 이게 paper-shaped 결과.

## 왜 효과 / 무엇을 바꾸는가 (decision it flips)
- WM 연구자에게 "어느 regime에서 imagination을 신뢰할 수 있나"의 표준 측정 도구 제공(현재 없음).
- reliability-aware 개입(BL-14 포함 한 부류)이 표준 regime에서 왜 futile한지 진단적으로 설명 → 헛된 방향 차단.
- dynamics vs reward-head 귀속으로 *어느 컴포넌트*를 고쳐야 하는지 지목 → 후속 연구 방향 결정.

## 실험에서 측정할 metric + 예상
- Base: 표준 DreamerV3-torch, DMControl(walker, cheetah, cartpole 등) proprio+vision, 실물 학습된 checkpoint.
- 측정: DF(Kendall τ), DF_dyn, DF_rew, state-MSE/latent-div — 격자 전체. seed 3-5.
- 예상: in-dist proprio H≤15 DF≈높음(BL-14 일치, anchor), OOD/vision/H≫15에서 DF 점진 붕괴 + state-MSE와 DF 괴리 셀 존재. (정확한 임계는 측정 대상 — 사전 단정 안 함.)

## 측정 전제조건 (검증 완료) 및 proxy 타당성 게이트
- **Env-state resettability (Gap 1, 검증됨)**: DF는 *같은 arbitrary start state*에서 K개 action 후보를 real로 rollout해야 한다 → 환경을 저장된 임의 상태로 reset 필요. vendored harness(`envs/dmc.py`)는 `dm_control suite.load` Environment를 쓰고 `self._env.physics`를 노출 → dm_control Physics의 `get_state()/set_state()/reset_context()`로 임의 상태 저장/복원 가능(현재 wrapper엔 메서드 미노출, thin wrapper 추가만 필요). **만약 resettable이 아니었다면 on-policy 방문 state에서만 DF 측정 가능 → OOD/arbitrary-state 셀(격자의 핵심)이 불가**했을 것. 확인됨.
- **Proxy 타당성 게이트 (Gap 2, validator가 1순위로 칠 것)**: DF(K개 discrete action의 Kendall τ)는 정책이 실제 의존하는 양의 *proxy*다. 연속제어 DreamerV3는 imagined rollout을 통한 **dynamics gradient backprop**(`imag_gradient: dynamics`)으로 학습하지 discrete action ranking으로 학습하지 않는다. 따라서 **low-DF 셀이 실제로 *학습 return 저하 / 정책-gradient 손상*과 대응함을 반드시 검증**해야 한다 — 단지 state-error가 큰 것과 구분(BL-11 0.085/BL-14 valid-but-useless 함정의 diagnostic 버전). 대응 안 하면 atlas가 *정책-무관한 양*을 재는 것. **더 직접적인 fallback 메트릭**: imagined dynamics 통한 policy gradient vs real dynamics 통한 policy gradient의 cosine alignment(ranking proxy가 impact를 못 잡으면 이걸로 전환).

## 핵심 ablation + baseline
1. **DF vs 기존 메트릭**: 같은 격자에서 state-MSE/latent-divergence/return-alignment를 baseline 메트릭으로 두고 DF가 *추가 정보*를 주는지(괴리 셀 존재 여부)가 핵심 ablation. 괴리 없으면 Null B.
2. **DF_dyn vs DF_rew 분해의 타당성**: real-reward/real-dynamics 치환이 실제로 기여를 분리하는지(합성 sanity: 의도적으로 reward head만 망가뜨린 모델에서 DF_rew만 떨어지는가).
3. **Action 후보군 정의 민감도**: K, action-sequence 길이, 후보 샘플링(정책 근방 vs uniform)에 DF가 robust한지.
4. **OOD 강도 sweep**: mass/friction 변형 크기에 따른 DF 붕괴 곡선.

## 선행 연구 위험 (literature self-check)
- **2510.16732 (WM 서베이, 2025)**: imagination 실패 taxonomy/diagnostic **없음**(fetch 확인) → gap 공인.
- **2501.00195 ("Unraveling Generalization in WM")**: SDE 이론 + Jacobian regularization으로 robustness *개선하는 method*(fetch 확인). 진단 아님, DF 메트릭 없음. → 본 진단과 목적 다름(단, "robustness via regularization" 사ub-cell은 회피).
- **2602.08236 ("When/How Much to Imagine")**: imagination Helpful/Misleading/Unnecessary taxonomy가 있으나 **VLM visual spatial reasoning** 대상, RSSM control 아님. → 도메인·메트릭 다름.
- return-alignment(imagined return vs real return) 평가: 스칼라 정렬만, action-ordering 분해 없음.
- simulation lemma(Kearns&Singh 2002): 이론적 value-gap bound, 경험적 RSSM 격자 진단 아님.
- **메인 리스크**: literature-checker가 "imagined action-ranking fidelity" 정확 일치 선행을 찾는가 + DF_dyn/DF_rew 분해의 novelty. (action-conditional counterfactual fidelity는 diffusion WM 쪽에 있으나 RSSM control + advantage-ordering 분해는 미발견.)

## venue
NeurIPS Datasets & Benchmarks(평가 인프라) 최적합. 진단+분해가 분명한 메시지 → ICLR 본트랙도 가능.

## 위험 요소
| 위험 | 가능성 | 완화 |
|---|---|---|
| DF가 격자 전체 균일(Null A) → negative 결과 | 중 | OOD/vision/H≫15로 breakage 그럴듯한 셀 겨냥. negative라도 "표준 regime에선 imagination 결정-충실" 자체가 보고 가치(reliability 개입 무익 설명) |
| DF가 state-MSE와 단조(Null B) → 기여 약화 | 중 | 괴리 셀 탐색이 1순위 측정. real-DreamerV3 OOD에서 state 붕괴해도 DF 유지 셀 가설 |
| DF_dyn/DF_rew 분해가 깔끔히 안 갈림(상호작용) | 중 | 합성 sanity로 분해 타당성 선검증, 안 되면 joint DF만 보고 |
| reliability-reweighting으로 미끄러짐(BL-14 재유입) | 낮(firewall 명시) | 측정만, 어떤 개입도 안 함을 명문화 |
