---
slug: residual-corrected-sparse-rssm
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 11
---

## 판정: INCREMENTAL

핵심 메커니즘 구성요소들이 각각 선행 연구에 존재하지만, "periodic GRU anchor + learned rank-r active Δh corrector + residual-norm adaptive re-anchor, RSSM imagination rollout FLOPs cheapening"이라는 조합은 문헌에서 발견되지 않음. 차별점 4개가 명확히 문서화되며 Validator 단계 진행 권고.

---

## 검색 요약

| 검색어 | 관련 논문 |
|---|---|
| GateL0RD sparsely changing latent states world model NeurIPS 2021 | GateL0RD (Gumbsch et al., NeurIPS 2021) |
| Clockwork VAE multi-timescale latent dynamics world model 2021 | Clockwork VAE (Saxena et al., NeurIPS 2021) |
| Skip-RNN learning to skip state updates ICLR 2018 | Skip-RNN (Campos et al., ICLR 2018) |
| Delta networks efficient RNN state change sparse update | Delta Networks (Neil et al., ICML 2017) |
| Variational Sparse Gating world model recurrent latent state NeurIPS 2022 | VSG (Morningstar et al., NeurIPS 2022) |
| THICK hierarchical world model ICLR 2024 sparse update gating | THICK (Gumbsch et al., ICLR 2024) |
| anchor corrector latent state world model imagination rollout | ReDRAW (2025) — sim-to-real residual, 다른 목표 |
| Sparse Imagination efficient visual world model planning | Sparse Imagination (2506.01392, 2026) — token dropout, 다른 메커니즘 |
| Koopman latent linear dynamics MBRL low-rank 2022 2023 | Koopman MBRL (Chen et al., ICLR 2024) — 선형화, 다른 방향 |
| E2C Embed-to-Control locally linear latent dynamics | E2C (Watter et al., NeurIPS 2015) — locally linear, 다른 방향 |
| DR-RNN deep residual recurrent model reduction | DR-RNN (Kani & Elsheikh, 2017) — PDE model reduction, MBRL 무관 |

---

## 관련 논문 목록 (11개 확인)

1. **GateL0RD: Sparsely Changing Latent States for Prediction and Planning** (Gumbsch et al., NeurIPS 2021, arXiv:2110.15949)
   - 관련성: L0-regularized gating으로 sparsely changing latent states 유지. world model에 적용. THICK의 기반.
   - 차이: gating = binary carry-forward (h_t = h_{t-1}) when gate closed. Δ=0 skip. rank-r active Δ estimation 없음. anchor + corrector 구조 없음. FLOPs cheapening이 주 목표 아님 (robustness/interpretability).
   - **주의**: 내부 수식은 abstract 수준에서만 확인됨. full-text PDF 확인 실패. "novel internal gating function + L0 penalty on latent change"가 carry-forward임은 다중 snippet에서 수렴적으로 지지됨.

2. **Clockwork VAE** (Saxena, Ba, Hafner, NeurIPS 2021, arXiv:2102.09532)
   - 관련성: 계층별 다른 clock speed로 latent 업데이트. 느린 level은 ticks 사이에 고정.
   - 차이: 느린 level의 "고정"이 carry-forward (Δ=0). active Δh prediction 없음. rank-r corrector 없음. residual-norm adaptive re-anchor 없음. video prediction 목표, RSSM imagination rollout FLOPs cheapening 아님.

3. **Skip-RNN: Learning to Skip State Updates** (Campos et al., ICLR 2018, arXiv:1708.06834)
   - 관련성: RNN state update를 binary gate로 skip하여 computation 절감.
   - 차이: skip = carry-forward (Δ=0). active Δh estimation 없음. WM/RSSM imagination에 적용 없음. rank-r projection 없음.

4. **Delta Networks for Optimized Recurrent Network Computation** (Neil et al., ICML 2017)
   - 관련성: threshold 기반으로 Δ≈0인 update를 skip, compute 절감.
   - 차이: threshold-based skip (Δ<ε면 update 건너뜀). learned rank-r active corrector 아님. K step anchor + corrector 구조 없음. RSSM/MBRL imagination에 적용 없음.

5. **Variational Sparse Gating (VSG)** (Morningstar et al., NeurIPS 2022, arXiv:2210.11698)
   - 관련성: stochastic binary gates로 latent feature dimensions sparse 업데이트. DreamerV2 기반.
   - 차이: gate = feature dimension-level binary masking. carry-forward (gate=0이면 해당 dimension이 h_{t-1} 유지). rank-r active Δ estimation 없음. anchor + corrector 구조 없음. robustness가 목표, FLOPs speedup 부차적.

6. **THICK: Learning Hierarchical World Models with Adaptive Temporal Abstractions** (Gumbsch et al., ICLR 2024)
   - 관련성: GateL0RD 기반으로 저레벨 context code가 sparsely 변함. hierarchical WM.
   - 차이: sparse update = GateL0RD carry-forward (Δ=0). low-rank corrector 없음. K step anchor + active residual 없음. 계층 구조 목표, transition FLOPs cheapening 직접 아님.

7. **ReDRAW: Adapting World Models with Latent-State Dynamics Residuals** (2025, arXiv:2504.02252)
   - 관련성: latent state dynamics에 residual correction 적용.
   - 차이: sim-to-real adaptation 목표. 사전학습 WM을 target domain에 보정. FLOPs/inference speed 절감이 목표가 아님. periodic anchor + cheap corrector between steps 구조 아님.

8. **Sparse Imagination for Efficient Visual World Model Planning** (2026, arXiv:2506.01392)
   - 관련성: WM imagination rollout efficiency 목표.
   - 차이: token-level dropout (visual patch). GRU/RSSM transition 자체를 싸게 만들지 않음. learned rank-r Δh corrector 없음. transformer WM 대상.

9. **Koopman Theory for Efficient Dynamics Modeling** (Chen et al., ICLR 2024)
   - 관련성: nonlinear dynamics를 linear latent space에서 근사하여 speedup.
   - 차이: 전체 dynamics를 linear Koopman operator로 대체 (GRU 제거). rank-r corrector 없음. 선형화가 본질, corrector-as-approximation이 아님. parallel convolution으로 2× speedup.

10. **Embed to Control (E2C)** (Watter et al., NeurIPS 2015, arXiv:1506.07365)
    - 관련성: locally linear latent dynamics. low-rank approximate transition.
    - 차이: 전체 transition을 locally linear A(z_t), B(z_t)로 대체. periodic anchor + cheap corrector 사이 구조 없음. 2015년 논문, DreamerV3/RSSM 이전.

11. **DR-RNN: Deep Residual RNN for Model Reduction** (Kani & Elsheikh, 2017, arXiv:1709.00939)
    - 관련성: residual + recurrent. 이름이 유사.
    - 차이: PDE 수치해석 model reduction 목표. MBRL/world model/RSSM과 무관. residual = line search의 residual minimizer (physics), RSSM Δh corrector 아님.

---

## Novelty 분석

### 제안 방법과 유사한 점

**Temporal sparsity 개념** (GateL0RD, Skip-RNN, Clockwork VAE, Delta Nets, VSG, THICK):
- 모두 "매 step full GRU를 안 돌아도 된다"는 아이디어를 공유
- GateL0RD/VSG는 world model RSSM 맥락에 실제 적용

**Anchor + between-steps 구조** (Clockwork VAE):
- 느린 level이 fast level에게 일종의 "고정 base"를 제공
- K step 주기로 특정 level을 hold하는 개념

**Residual correction in latent space** (ReDRAW):
- latent dynamics에 residual correction을 적용한다는 개념

**FLOPs/inference speedup 목표** (Koopman MBRL, Sparse Imagination):
- imagination rollout efficiency가 명시적 목표

### 명확히 다른 점 (차별점 4개)

**차별점 1 — Δ≠0 active estimation vs Δ=0 carry-forward (핵심)**
GateL0RD, Skip-RNN, Clockwork VAE, VSG, THICK, Delta Nets 모두 "update를 건너뛸 때 h_t = h_{t-1} (carry-forward)"이다. 제안 방법은 K step 사이에도 Δh를 능동적으로 추정하는 rank-r corrector를 실행 — skip이 아니라 cheaper-but-nonzero update. BL-06 실패(carry-forward → drift 1.9)가 이 차이를 직접 동기부여.

**차별점 2 — Adaptive residual-norm re-anchor (새로운 drift 제어 메커니즘)**
Clockwork VAE의 clock은 고정 주기(fixed K). 제안 방법은 corrector residual norm ‖Δh_corrector - Δh_full‖가 threshold τ를 초과하면 조기 re-anchor. dynamics가 거칠어지면 자동으로 K를 낮춤. 이 adaptive re-anchor는 선행 연구에서 발견되지 않음.

**차별점 3 — RSSM GRU transition FLOPs cheapening as primary target**
VSG/GateL0RD는 robustness/interpretability 목표. Sparse Imagination은 token dropout. ReDRAW는 sim-to-real adaptation. Koopman은 전체 transition을 linearize. 제안 방법은 DreamerV3 imagination hot-path인 GRU transition의 FLOPs 자체를 35~50% 줄이는 것이 유일한 목표. RSSM 구조(full GRU + stochastic prior MLP)를 유지하면서 GRU만 rank-r로 근사.

**차별점 4 — Teacher residual distillation-in-time (training scheme)**
corrector는 같은 모델 내 full GRU output과의 residual (Δĥ ≈ h_full - h_prev)을 teacher signal로 학습. 별도 teacher network 없이, 학습 시 full GRU를 함께 돌면서 corrector를 "시간 축 내에서" distill. 선행 연구에서 이 training scheme은 발견되지 않음.

---

## 판정 근거

**INCREMENTAL 판정 이유:**

"carry-forward가 아닌 active Δh estimation"은 새로운 angle이지만, 그 기반 기술들 (temporal sparsity in WM: GateL0RD, VSG; periodic clock: Clockwork VAE; residual correction in latent space: ReDRAW; cheap dynamics: Koopman, E2C)이 각각 존재함. 아이디어는 이들의 새로운 *조합*이자 *WM rollout FLOPs cheapening이라는 명확한 응용 각도*를 가짐.

"4점 체크" 결과:
- periodic full GRU anchor: Clockwork VAE가 유사하나 fixed-clock + carry-forward (0/4)
- cheap active Δh estimation between: **어떤 논문에도 없음** (핵심 차별점)
- residual-norm adaptive re-anchor: **어떤 논문에도 없음**
- RSSM imagination rollout FLOPs cheapening target: Sparse Imagination이 WM 효율화이나 다른 메커니즘 (token dropout)

→ 4점 기준으로 3개 이상 겹치는 논문 없음. NOVEL에 가까운 INCREMENTAL.

**NOVEL 아닌 INCREMENTAL인 이유:**
- GateL0RD + Clockwork VAE 조합이 "K step마다 full update, 사이에 cheap update"라는 구조적 틀을 RSSM WM 맥락에서 이미 암시함
- rank-r RNN approximation은 lrRNN 계열에 선행 연구 있음 (파라미터 압축이지만)
- 메커니즘 검증이 full-text가 아닌 abstract/snippet 수준이므로 conservative 판정 권장

**고위험 미검증 구역:**
- GateL0RD 내부 수식의 정확한 gating mechanism (abstract에서 carry-forward로 수렴하나 full-text 미확인)
- VSG의 stochastic gate가 정확히 어떻게 h_t를 계산하는지 (binary gate × Δh_candidate인지 아닌지)

---

## 권고 사항

1. **Validator 단계 진행**: 차별점이 충분히 문서화됨. PoC에서 BL-06 (Δ=0) vs 제안 방법 (Δ≠0 rank-r) drift 비교가 핵심 실험.
2. **논문 작성 시 positioning**: vs GateL0RD (carry-forward → active estimation), vs Clockwork VAE (fixed clock → adaptive re-anchor), vs ReDRAW (adaptation → FLOPs cheapening) 세 축으로 관련 연구 섹션 구성.
3. **추가 검색 권고**: GateL0RD 및 VSG의 full-text 수식 확인 (h_t 계산이 정확히 gate × (h_candidate - h_prev) + h_prev 형태인지 확인). 만약 VSG가 이미 이 형태를 쓴다면 차별점 1이 약화됨.
