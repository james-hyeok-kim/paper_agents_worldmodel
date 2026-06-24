---
slug: round4-pivot-saturation-map
status: saturation-report
created: 2026-06-18 KST
category: META
note: "Round 4 pivot 요청(IRIS/STORM/diffusion/JEPA로 family 전환)에 대한 결과. 5개 생성 대신, pivot 공간이 2025-26 문헌에서 포화되었음을 7개 probe로 확인. 각 cell의 pre-emption 논문을 기록. 향후 이 공간 재탐색 시 반드시 먼저 읽을 것."
---

# Round 4 Pivot — Saturation Map (transformer / diffusion / JEPA WM)

team-lead가 Round 4에서 DreamerV3+call-reduction 사각지대(BL-06~13)를 벗어나 **bottleneck이 근본적으로 다른 WM family**(IRIS/STORM transformer, diffusion DIAMOND, JEPA/non-generative)에서 효율 또는 품질 아이디어 5개를 요청. 결과: **7개 probe 중 7개 cell이 포화**. pivot 공간 자체가 2025-26에 heavily worked. 5개 생성은 self-collision/pre-empted 아이디어 manufacturing이 되어 generator 금지사항("pre-empted/incremental 생성 금지") 및 team-lead precondition(hot-path cite + 3 차별점 + non-pre-empted) 위반. 따라서 saturation map을 deliverable로 반환.

## Hot-path 검증 결과 (Amdahl precondition)

team-lead precondition #2(가정 bottleneck이 실제 hot-path임을 cite로 증명). 검증으로 2개 family가 즉시 탈락:

- **STORM (transformer-WM)**: transformer가 hot-path가 **아님**. STORM은 latent+action을 **단일 token으로 fuse** → transformer가 극도로 cheap. 학습 9.3 V100h (IRIS 168h, DreamerV3 12h보다도 적음). BL-13의 "IRIS에서 transformer는 24%뿐"과 같은 결론의 더 극단적 버전. transformer 최적화 headroom 없음. (arXiv:2310.09615)
- **DINO-WM (JEPA efficiency)**: rollout이 hot-path임을 **cite 불가**. decoder-free지만 frozen DINOv2 ViT encoder(대형) + CEM many-sample이 비용을 지배할 수 있음. predictor rollout이 dominant cost라는 문장 미발견. + 내 pending planning family(action-quantized-planning-cache, branch-shared-imagination[INCREMENTAL], subtree-reuse-muzero)와 충돌. (arXiv:2411.04983)

- **DIAMOND (diffusion-WM)**: denoising forward pass가 hot-path임이 **cite 가능**(유일하게 깨끗). "number of denoising steps is directly related to inference cost." dynamics n=3 NFE/frame (IRIS 16 NFE 대비). 단 n=3은 이미 floor 근처라 효율 headroom 협소. (arXiv:2405.12399, NeurIPS 2024)

## 7개 cell pre-emption 기록

| # | 시도한 아이디어 | 차단 논문 | 차단 이유 |
|---|---|---|---|
| 1 | diffusion adaptive NFE (efficiency) | **DISK** (arXiv:2602.00440) | dual-branch controller가 per-step denoising skip을 step difficulty/local curvature로 결정. 2×/1.6×. 같은 family·같은 목표. + DIAMOND n=3이라 headroom floor 근처. |
| 2 | JEPA planning-aware objective (quality, anchor) | **2602.18639** "Learning Invariant Visual Representations for Planning with JEPA" | 거의 동일: "separate nearby states that matter for planning", "preserve action/reward-relevant info", "directly addressing why accurate world models fail for planning". |
| 3 | JEPA nearby-state discriminability (quality) | **2602.18639** + **C-JEPA** (arXiv:2410.19560) | 2602.18639가 nearby-state separation 직접 다룸. C-JEPA는 VICReg variance/covariance regularizer(=discriminability regularizer) 이미 제공. idea 2와도 self-collide(같은 논문 "What Drives Success" 출처, 같은 메커니즘). |
| 4 | diffusion quality-at-fixed-NFE (spatial budget redistribution) | **아키텍처적으로 unsound** | DIAMOND denoiser는 monolithic full-frame U-Net 2D(image history channel-wise stack + action via AdaGN). region별 NFE 차등 = easy region freeze 후 계속 denoise해도 매 step **full U-Net 실행**(conv receptive field가 전체 frame 필요) → compute 절감 0, "matched average NFE"는 허구. temporal 버전(frame별 whole-frame NFE 차등)은 DISK로 붕괴. (arXiv:2405.12399 §architecture) |
| 5 | prediction-error prioritized WM-training replay (sample-eff) | **MaPER, Curious Replay, UPER, Prioritized Generative Replay** | MaPER=model error+TD error 우선순위 curriculum. Curious Replay=high model prediction error 우선순위. UPER=epistemic uncertainty 우선(aleatoric로 modulate)— 내가 쓰려던 split까지 선점. |
| 6 | diffusion WM memory / long-horizon consistency | **EDELINE** (arXiv:2502.00466) | linear-time sequence modeling으로 DIAMOND의 short-context memory 한계 해결. |
| 7 | diffusion WM long-horizon drift / exposure bias | **Self-Forcing, LIVE, Epona, HorizonDrive, Persistent Robot WM** (arXiv:2602.03747, 2506.24113, 2605.11596, 2603.25685) | closed-loop gap / compounding error를 RL post-training, self-forcing, chain-of-forward로 해결. 매우 활발. |
| (probe) | diffusion action-conditioning / counterfactual fidelity | **Vid2World, CWMDT, Decoupled CFG, WorldPlanner** (arXiv:2511.17481, 2506.14399, 2511.03077) | action-conditioned diffusion WM + counterfactual/CFG 활발. |

## 결론 및 권고

- **pivot 공간(transformer/diffusion/JEPA WM)은 2025-26 문헌에서 포화.** random-family probing base rate ≈ 0 (7/7 차단). 추가 probe도 또 다른 crowded corner를 찾을 가능성이 높음.
- **clean survivor 0~1개.** diffusion만이 hot-path를 cite할 수 있는 유일 family였으나, 효율은 DISK가 선점 + n=3 floor, 품질-at-fixed-NFE는 U-Net 아키텍처상 unsound. 깨끗한 survivor 없음.
- **권고 (team-lead 결정 필요):**
  1. **sub-problem 좁히기**: DIAMOND의 정확히 한 미해결 지점(예: action-conditioning이 약한 환경에서 RL imagination 품질 — Vid2World의 CFG와 차별화한 *agent-training-objective* 각도)을 깊게 파고들되, 반드시 literature-checker 선행.
  2. **다른 axis로 전환**: 효율/품질이 아닌 robustness / OOD generalization / evaluation-diagnostic 각도. 또는 WM이 아닌 인접 문제.
  3. **DreamerV3로 복귀하되 call-reduction이 아닌 축**: dual-rate(유일 pass, +12% sample-eff)가 성공한 이유는 "효율"이 아니라 "sample-efficiency/품질"이었음. DreamerV3에서 품질/sample-eff 축은 BL에 막혀있지 않음 (BL-06~13은 전부 call-reduction 효율).

## 이번 라운드 메타 교훈 (다음 generator 세션 필수)

- **"pending set에서 fresh"와 "literature에서 fresh"는 다르다.** JEPA는 내 pending엔 없었지만 문헌엔 6개+(Value-guided, C-JEPA, Causal-JEPA, DINO-WM, 2602.18639, What-Drives-Success). family 전환 시 pending 충돌뿐 아니라 **literature 충돌을 먼저** 확인.
- **hot-path는 intuition으로 단정 금지.** "decoder-free니까 rollout이 hot-path"(DINO-WM), "transformer-WM이니까 imagination이 hot-path"(STORM) 둘 다 틀림. cite 가능한 문장이 없으면 그 아이디어는 BL-13 재발.
- **아키텍처 제약을 메커니즘 설계 전에 확인.** monolithic U-Net에서 spatial-selective compute는 불가능. 메커니즘이 base model 아키텍처와 호환되는지 먼저.
- **5개 채우려 collision manufacturing 금지.** under-deliver(documented)가 pre-empted 아이디어 5개보다 valuable.
