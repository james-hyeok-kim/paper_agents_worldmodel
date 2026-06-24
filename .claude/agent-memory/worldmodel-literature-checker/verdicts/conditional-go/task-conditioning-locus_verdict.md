---
slug: task-conditioning-locus
verdict: INCREMENTAL
checked-date: 2026-06-22 KST
papers-reviewed: 12
---

## 판정: INCREMENTAL

## 검색 요약

| 검색어 | 결과 수 | 관련 논문 |
|---|---|---|
| task invariant dynamics reward task specific multitask RL arxiv 2022-2025 | 8 | Vid2Act(2306.03360), Decoupling Dynamics&Reward(1804.10689), Task-Agnostic Dynamics Priors(1905.04819) |
| which module encodes task information world model intervention causal attribution | 8 | Causal WM papers (LLM 중심, RL WM 적용 없음) |
| dynamics task agnostic reward task specific transfer model-based RL multitask | 10 | Vid2Act, 1804.10689, Modular Networks(2111.08010) |
| task embedding ablation TD-MPC2 world model reward dynamics per-module | 8 | EZ-M(2603.01452), TD-MPC2(2310.16828), PWM(2407.02466) |
| causal intervention task conditioning model-based RL activation patching embedding swap | 9 | Activation patching 문헌(LLM 중심) — RL/WM에 직접 적용 사례 없음 |
| EZ-M task embedding ablation dynamics reward indispensable | 5 | EZ-M(2603.01452) 확인 |
| task-agnostic dynamics task-specific reward multitask MBRL separation arxiv 2023-2024 | 9 | 1905.04819, 1804.10689, 2111.08010 |
| contextual MDP task representation locus probing dynamics reward | 8 | DALI(2508.20294), CARL, meta-RL context papers |
| Mixture-of-World Models modular latent dynamics multitask | 5 | MoW(2602.01270) |
| Learning Massively Multitask World Models TD-MPC2 | 5 | Newt(2511.19584) |
| representation locus task specific multitask RL 2024-2025 | 8 | 개념적 관련 논문들 — per-module locus 없음 |
| mechanistic interpretability RL embedding swap activation patching task conditioning | 9 | activation patching(LLM), RL 적용 미확인 |

## 관련 논문 목록

1. **Decoupling Dynamics and Reward for Transfer Learning** (Fu et al., 2018, arXiv:1804.10689) — 관련성: dynamics와 reward를 명시적으로 분리해 transfer하는 최초 체계적 프레임워크. "dynamics는 task-agnostic, reward는 task-specific"이라는 개념의 직접 선행. 단, training-based 분리이고 per-module intervention(swap) 없음.

2. **Task-Agnostic Dynamics Priors for Deep RL** (Pong et al., 2019, arXiv:1905.04819) — 관련성: universal task-agnostic dynamics prior를 학습해 transfer. "dynamics가 task에 무관하다"는 가설을 공리로 사용.

3. **EZ-M: Scaling Tasks, Not Samples** (2026, arXiv:2603.01452) — 관련성: [가장 중요한 위험] EfficientZero 기반 multitask MBRL. Ablation에서 "Task Embedding(TE) is indispensable for Dynamics and Reward"라고 결론. **그러나**: (a) training-time removal(TE를 학습 때 제거) 방식, (b) Dynamics+Reward+VP를 개별 분리 없이 동시 제거(-Dyn-Rew-VP_TE 조건), (c) same-embodiment vs cross-embodiment 분리 없음, (d) H1 humanoid 단일 플랫폼. 이 아이디어의 inference-time per-module swap + same-family restriction과 다른 실험 설계.

4. **TD-MPC2: Scalable, Robust World Models** (Hansen et al., 2023, arXiv:2310.16828) — 관련성: 5개 모듈 전부 task embedding을 concat. t-SNE 시각화에서 "embedding similarity가 objective(walk/run)보다 dynamics(embodiment)에 더 align"됨을 관찰. Per-module ablation 없음 — "어느 모듈이 실제로 task label을 쓰는가"는 미검증.

5. **Model-Based Transfer RL with Task-Agnostic Offline Pretraining (Vid2Act)** (Pan et al., 2023, arXiv:2306.03360, OpenReview:RD7Fo7RezT) — 관련성: task-agnostic dynamics knowledge를 offline에서 transfer + task-specific reward 학습 분리. 정량적으로 dynamics-agnostic 성질을 활용. Per-module locus 측정 없음.

6. **Modular Networks Prevent Catastrophic Interference in Model-Based Multi-Task RL** (2021, arXiv:2111.08010) — 관련성: task별로 다른 latent dynamics 모듈을 routing. Dynamics의 task-specificity를 *가정*해 설계했지만 측정하지 않음.

7. **PWM: Policy Learning with Multi-Task World Models** (Georgiev et al., 2024, arXiv:2407.02466) — 관련성: TD-MPC2 기반 multitask WM에서 policy 최적화. 80-task. task conditioning 설계를 가져왔지만 per-module ablation 없음.

8. **Mixture-of-World Models (MoW)** (2026, arXiv:2602.01270) — 관련성: task-specific expert transformer + shared backbone으로 multitask WM. gradient-based clustering으로 task grouping 수행. Dynamics가 task-specific하다고 가정해 expert routing. Per-module locus 측정 없음.

9. **Learning Massively Multitask World Models (Newt)** (2025, arXiv:2511.19584) — 관련성: TD-MPC2 계열의 massive multitask WM. Language conditioning 사용. Ablation: model/batch size scaling + language instruction 유무. Per-module task embedding ablation 없음. Inference-time swap 없음. Same-embodiment 분리 없음.

10. **DALI: Dynamics-Aligned Latent Imagination in Contextual WM** (2025, arXiv:2508.20294) — 관련성: contextual MDP에서 context encoding + WM conditioning. Dynamics-aligned encoder. Locus 측정 없음.

11. **EZ-M Website (yewr.github.io/ez_m/)** — "TE is indispensable for Dynamics and Reward" 확인. 수치 미공개(그래프만).

12. **Multi-Task RL with Context-based Representations** (Sodhani et al., 2021, ICML 2021) — 관련성: context encoder → policy conditioning. Task-specific reward/dynamics 분리 접근. Per-module swap 없음.

## Novelty 분석

### 제안 방법과 유사한 점

- **개념적 전제**: "dynamics는 task-agnostic, reward는 task-specific"이라는 가설은 2018년(1804.10689)부터 MBRL transfer 문헌에서 공리처럼 사용되어 왔음. 이 방향의 *방향성*은 선점됨.
- **EZ-M의 ablation 방향**: "task embedding이 dynamics와 reward에 필수"라는 결론이 training-time ablation으로 이미 존재함. 제안 아이디어의 core prediction(R(reward) 낮음)과 일치. 단, R(dynamics) 높음에 대해서는 EZ-M이 반박 근거(TE 없애면 dynamics 성능 하락).
- **TD-MPC2 t-SNE 관찰**: task embedding이 dynamics(embodiment)에 더 align됨 → dynamics가 task 구분보다 embodiment 구분에 치중할 수 있다는 *정성적* 암시.
- **transfer 문헌 전반**: dynamics 재사용 + reward 적응이라는 2-stage transfer 프레임워크는 Vid2Act 등에서 quantitatively 확인됨.

### 명확히 다른 점 (차별점)

1. **Inference-time frozen intervention vs training-time removal**: EZ-M 포함 모든 선행 연구는 training-time ablation(학습 때 TE 제거). 이 아이디어는 frozen 모델에 inference-time embedding *swap*(다른 task의 embedding으로 교체, zero가 아님)을 사용해 OOD confound 없이 per-module causal 기여를 측정. 이 실험 설계가 기존에 없음.

2. **Same-embodiment family 분리 + cross-embodiment 제외**: 기존 연구는 same-embodiment(physics 동일, reward만 다름) vs cross-embodiment(physics 다름)를 분리하지 않음. 이 아이디어는 *오직 same-family*(walker-{stand,walk,run,run-backwards} 등)에서만 "dynamics가 task label을 무시한다"고 주장 — cross-embodiment는 trivially redundant라 claim 제외. 이 framing 자체가 없음.

3. **Onboarding decision rule**: "새 task가 reward-only variation이면 dynamics를 unfreeze할 필요 없다"는 실무적 처방을 locus measurement로부터 도출. + BL-19(embedding-only onboarding) 실패에 대한 mechanistic 설명. 이 두 가지를 연결한 논문 없음.

## 판정 근거

**INCREMENTAL** 판정. NOVEL이 아닌 이유: "dynamics task-agnostic, reward task-specific"이라는 개념적 방향은 2018년 Decoupling paper부터 MBRL transfer 문헌에서 반복적으로 가정·활용됨. EZ-M(2603.01452)의 training-ablation도 이 방향에 있음. 따라서 핵심 *방향*이 선점됨.

NO-GO가 아닌 이유: (a) inference-time per-module swap(vs training removal), (b) same-embodiment family restriction(vs whole-dataset ablation), (c) onboarding decision rule + BL-19 mechanistic explanation을 조합한 논문이 없음. 이 3가지 차별점이 명확히 다른 각도를 제공.

**EZ-M의 반박 위험(validator에게 전달)**: EZ-M은 training-time 제거 실험에서 "dynamics에도 TE 필수"라고 주장. 이것이 사실이라면 이 아이디어의 핵심 surprise(R(dynamics) 높음)가 나오지 않을 수 있음. EZ-M의 실험과 이 아이디어의 실험 설계가 다르지만(training vs inference-time, joint removal vs per-module), feasibility PoC에서 this tension을 반드시 테스트해야 함.

**판정의 얇음**: 차별점 3가지가 있지만 개념 선점이 두터움. decision-fidelity-atlas와 유사한 "INCREMENTAL (thin)" 성격.

## 권고 사항

- **다음 단계**: conditional-go로 worldmodel-idea-validator 진행. 단, **1순위 실험이 premise check**: all-swapped 붕괴 + same-family에서 R(dynamics) vs R(reward) 분리. EZ-M의 반박(dynamics도 TE 필수)이 inference-time frozen swap에서도 재현되면 핵심 surprise 없음 → validator FAIL 예상.
- **아이디어 파일 status 업데이트**: `status: literature-checked`, `verdict: INCREMENTAL`.
- **claim 강화 포인트**: EZ-M을 "training-removal이라 confound 있다(training distribution 변화)"고 명시적으로 반박하는 framing이 필요. swap-mean-family arm이 EZ-M 반박의 핵심 교차확인.
- **BLACKLIST 추가 권고 없음**: INCREMENTAL이므로 그대로 validator로 진행.
