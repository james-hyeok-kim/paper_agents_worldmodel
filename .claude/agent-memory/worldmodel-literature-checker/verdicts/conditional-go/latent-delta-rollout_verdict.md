---
slug: latent-delta-rollout
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 8
---

## 판정: INCREMENTAL

## 검색 요약
| 검색어 | 결과 수 | 관련 논문 |
|---|---|---|
| Skip-RNN adaptive computation RNN Campos 2018 | 다수 | Skip-RNN (ICLR 2018) |
| RSSM deterministic stochastic state skip update world model efficiency | 다수 | 직접 관련 없음 |
| delta network temporal difference RNN FLOPs reduction | 다수 | DeltaRNN (Neil et al., 2016) |
| sparsely changing latent states prediction planning NeurIPS 2021 | 다수 | GateL0RD (NeurIPS 2021) |
| GateL0RD latent state sparsity carry-forward | 다수 | GateL0RD (NeurIPS 2021) |
| DISK dynamic inference skipping world model | 다수 | DISK (arXiv 2026) |
| delta predictor RNN latent skip FLOPs world model 2023-2025 | 다수 | Latent Bridge (arXiv 2025) |
| Skip Dreamer conditional computation RSSM imagination | 없음 | RSSM 직접 적용 없음 |
| latent delta world model RSSM GRU skip 2023-2025 | 없음 | 직접 선행 연구 없음 |

## 관련 논문 목록

1. **Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks** (Campos et al., ICLR 2018, arXiv:1708.06834) — 관련성: GRU/LSTM에 binary gate u_t ∈ {0,1}를 추가해 state update를 skip하고 h_{t-1}을 carry-forward. `s_t = u_t · S(s_{t-1}, x_t) + (1 − u_t) · s_{t-1}`. gate는 경량 선형 레이어. 연속 skip 시 갱신 확률 누적 증가(refresh interval 유사). **carry-forward 메커니즘과 경량 gate가 선행 연구에 존재.**

2. **Sparsely Changing Latent States for Prediction and Planning in Partially Observable Domains — GateL0RD** (Gumbsch, Butz, Martius, NeurIPS 2021, arXiv:2110.15949) — 관련성: latent state의 L0 norm 변화를 penalty로 부과해 sparse하게 변하는 recurrent state를 학습. recommendation network + gating network 구조. **하지만 recommendation network를 매 스텝 돌린 후 gate로 선택 → FLOPs 절감 목적 없음, 표현력/일반화 목적.** RSSM의 deterministic/stochastic 분리 없음. 단일 RNN state에 적용.

3. **Delta Networks for Optimized Recurrent Network Computation** (Neil et al., 2016, arXiv:1612.05571) — 관련성: RNN hidden state의 temporal difference(delta)가 임계 이하이면 MAC 연산 skip. 하드웨어 가속기용 temporal sparsity 활용. **speech 9×, vision-based control 100× FLOPs 감소 보고.** 그러나 RSSM 미적용, deterministic/stochastic 구분 없음, world model planning 맥락 없음.

4. **DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos** (Parger et al., CVPR 2022, arXiv:2203.03996) — 관련성: 비디오의 프레임 간 차이(delta)를 활용해 CNN inference를 skip. 최대 7× speedup. **pixel/feature 도메인 적용, RSSM latent state 아님.** delta sparsity 개념의 선행 아이디어.

5. **DISK: Dynamic Inference SKipping for World Models** (arXiv:2602.00440, 2026) — 관련성: diffusion-based world model (autonomous driving)에서 확산 스텝 단위 skip. 2× / 1.6× speedup. `Δ_k(b) ≤ θ·d_{k+1}(b)` 기준 skip 결정, consecutive skip cap C_max 및 warm-up. **diffusion transformer에 적용, GRU/RSSM 미적용.** 하지만 carry-forward + safety guard(refresh) 구조가 이 아이디어와 유사.

6. **Latent Bridge: Feature Delta Prediction for Efficient Dual-System VLA Inference** (arXiv:2605.02739, 2025) — 관련성: VLM output delta를 경량 predictor로 예측 후 VLM 전체 forward pass skip. 1.65~1.73× speedup. **"delta 예측 후 heavy computation skip" 구조는 Claim 2와 동일한 패턴.** World model/RSSM 미적용.

7. **CS-RNN: Efficient Training of Recurrent Neural Networks with Continuous Skips** (Neural Computing and Applications, 2022) — 관련성: Skip-RNN의 continuous skip 확장. FLOPs 절감 목적으로 state update를 가변적으로 skip.

8. **DreamerV3** (Hafner et al., ICML 2023, arXiv:2301.04104) — 관련성: 매 imagination step h_t와 z_t를 동기적으로 갱신하는 RSSM baseline. **이 아이디어가 수정 대상으로 하는 기준 구조.**

## Novelty 분석

### 제안 방법과 유사한 점

- **carry-forward 메커니즘**: Skip-RNN (2018)이 이미 구현. `h_t = h_{t-1}` when gate=0.
- **경량 predictor로 heavy update skip 결정**: Skip-RNN의 gate가 경량 선형 레이어. Latent Bridge의 delta predictor.
- **refresh interval / consecutive skip cap**: Skip-RNN의 누적 확률 증가 (사실상 soft refresh), DISK의 C_max + warm-up guard.
- **temporal sparsity in latent state**: GateL0RD (2021)의 핵심 가설과 동일 ("물리 세계의 많은 요인은 시간에 따라 sparsely 변화").
- **delta threshold 기반 skip**: DeltaRNN (2016), Skip-RNN (2018) 모두 delta/threshold 개념 사용.

### 명확히 다른 점 (차별점)

1. **RSSM deterministic/stochastic branch 비대칭 갱신 (Claim 1 — 가장 강한 차별점)**: Skip-RNN, GateL0RD, DeltaRNN은 모두 단일 RNN state에 적용. RSSM의 두 branch를 비대칭적으로 처리해 deterministic h만 skip하고 stochastic z는 항상 갱신하는 방법은 선행 연구에 없음. 이는 RSSM-specific world model 구조를 직접 공략하는 신규 각도.

2. **FLOPs 절감을 목적으로 한 RSSM GRU 연산 skip (Claim 2 — 부분적 차별점)**: GateL0RD는 recommendation network를 매 스텝 실행한 후 gate로 업데이트 여부를 결정 → 실제 FLOPs 절감 없음, 표현력/일반화 목적. 이 아이디어는 GRU forward 자체를 skip해 실제 compute를 줄임. 단, "경량 predictor로 먼저 결정 후 skip" 구조는 Latent Bridge와 패턴이 유사하므로 완전 신규는 아님.

3. **세계 모델 imagination rollout에서의 적용**: 기존 skip/sparse state 논문들은 일반 sequence modeling 또는 autonomous driving diffusion WM에 적용됨. DreamerV3 RSSM imagination rollout에서 GRU skip-for-compute를 하는 논문은 확인되지 않음.

## 판정 근거

**INCREMENTAL 판정** — 세 가지 novelty claim 중 두 개(Claim 2, Claim 3)는 Skip-RNN, DeltaRNN, DISK, Latent Bridge가 이미 각각 대응하는 선행 기법을 보유한다. 그러나 **Claim 1: RSSM deterministic/stochastic branch 비대칭 갱신**은 명확히 구분되는 차별점이다. 또한 GateL0RD와의 핵심 차이 — "매 스텝 전체 연산 후 gate" vs "먼저 cheap predictor로 결정, GRU forward skip" — 는 실질적인 FLOPs 절감이라는 뚜렷한 구분점을 제공한다.

구성 요소 개별 선행 연구가 존재하지만, 이를 RSSM world model의 imagination rollout efficiency에 결합·응용한 논문은 확인되지 않는다. 검색 결과 "Skip Dreamer", "temporally sparse RSSM rollout", "early-exit RSSM GRU" 등 직접적 선점 논문이 없음이 확인됐다. 따라서 NO-GO가 아닌 INCREMENTAL.

claim-by-claim 비교:
- **Claim 1 (RSSM branch 비대칭)**: NOVEL — 선행 연구 없음 확인. 이 claim이 전체 novelty의 핵심 근거
- **Claim 2 (delta predictor + GRU skip for FLOPs)**: INCREMENTAL — **DeltaRNN (2016)이 가장 직접적 선행 연구**: hidden-state delta threshold 이하 시 MAC 연산 skip + carry-forward, compute 절감 목적. Skip-RNN(carry-forward + 경량 gate), Latent Bridge(delta predictor + heavy compute skip) 패턴도 선점. 유일한 novelty는 이를 RSSM deterministic branch에만 적용하는 것
- **Claim 3 (learned refresh interval)**: INCREMENTAL — Skip-RNN의 누적확률(soft refresh), DISK의 C_max + warm-up guard(hard cap) 동일 패턴 존재

참고: Claim 1이 무너지지 않는 한 전체 판정은 INCREMENTAL 유지. Claim 2와 3의 선행 연구 존재가 NO-GO가 되지 않는 이유는, 선행 연구가 RSSM의 두 branch를 비대칭적으로 처리하는 사례가 없기 때문.

핵심 비교 근거 논문 (pre-2022, 검증 완료): Skip-RNN (ICLR 2018), GateL0RD (NeurIPS 2021), DeltaRNN (arXiv 2016). 나머지 인용 논문(DISK, Latent Bridge, S5WM)은 corroborating evidence이며 판정의 load-bearing 근거가 아님.

## 권고 사항

- **다음 단계**: worldmodel-idea-validator로 synthetic PoC 진행 권고. Claim 1 (branch 비대칭)을 핵심 novelty로 포지셔닝하고 Claim 2/3은 설계 선택으로 재프레이밍할 것.
- **추가 논문 확인 권고**: GateL0RD가 실제로 recommendation network를 매 스텝 모두 돌리는지(→ FLOPs 비절감) 원문에서 방정식 레벨 확인.
- **feasibility 주의점**: stochastic z를 "항상 갱신"할 때 h가 frozen이면 `z ~ prior_head(h)`가 frozen h 기반으로만 샘플링됨 → z가 h에 의존하는 구조에서 h freeze 효과가 z 다양성에 미치는 영향을 PoC에서 검증 필요. 이 feasibility risk가 해결되면 INCREMENTAL → 경쟁력 있는 논문으로 발전 가능.
- **포지셔닝 재설계**: GateL0RD와의 차별점으로 "표현력/일반화 목적의 sparse state vs inference FLOPs 절감 목적의 selective GRU skip"을 전면에 내세울 것.
