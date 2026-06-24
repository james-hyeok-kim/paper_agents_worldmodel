---
slug: elite-staged-value-planning
status: pending
created: 2026-06-19 KST
base: TD-MPC2 (baselines/tdmpc2)
axis: efficiency (planning, test-time)
venue-fit: [CoRL, ICLR, NeurIPS]
amdahl: SATISFIED (코드 검증 — 아래)
blacklist-delta:
  - "BL-07 (MPPI branch KL merge): merge/dedup이 아니다. branch를 합치지 않음. 모든 512 sample을 평가하되 *평가 비용을 2단계로* 나눔(cheap-rank → full-elite). BL-07은 continuous action 발산으로 merge rate 9.9%라 실패했는데, 본 방법은 '겹치는 trajectory를 찾는' 것에 의존하지 않음 → 발산해도 무관. (advisor가 지적한 BL-07의 *메커니즘* 실패를 구조적으로 회피.)"
  - "BL-10/BL-12 (소형 모델 교체 GPU overhead): D=128 소형 GRU 교체가 아님. batch 512 × latent 512의 FLOPs-bound regime이라 kernel-overhead 지배 영역(D=128)이 아님. 단, TD-MPC2가 Q-ensemble을 vectorize해 wall-time을 단축했으므로 head 절감의 wall-clock 이득이 FLOPs보다 작을 수 있음 → wall-clock으로 gate(아래)."
  - "vs REDQ/DC-MPC critic subsampling: 기존 critic subsample(2개)은 *overestimation bias 감소/learning*이 목적. 본 방법은 *planning 비용 절감*이 목적이고, cheap-rank가 elite 선택을 보존하도록 full-ensemble을 elite에만 적용하는 2-stage 구조(bias가 아니라 ranking-fidelity 보존이 설계 기준)."
---

# Elite-Staged Value Planning: MPPI terminal value를 cheap-rank → full-elite 2단계로 평가

## base & axis
**TD-MPC2, efficiency(test-time planning).** MPPI의 per-sample terminal-value 평가를 2단계로 분할해 planning FLOPs/wall-clock을 절감.

## Amdahl 검증 (코드, SATISFIED — 6라운드 중 가장 강한 precondition)
TD-MPC2 `_plan`(tdmpc2.py L140-207)은 매 env step마다 `iterations=6`회 `_estimate_value`(L124-137)를 호출. config: horizon=3, num_samples=512, num_q=5, num_elites=64.
- `_estimate_value` per sample: dynamics term = horizon(3) × [`next` + `reward`] = 6 forward; terminal term = [`pi` + `Q`(5 heads avg)] = 6 forward (단 **1회**, horizon 곱 없음).
- 즉 horizon=3에서 **terminal-value term(pi+5Q)이 per-sample 비용의 ~50%**(forward count 기준; head/dynamics MLP 크기 차이는 validator가 정밀 측정). 그리고 이 terminal Q-ensemble은 **매 iteration 512 sample 전부에** 돌아간다(L137 → topk은 L186에서 그 value로).
- **planning은 TD-MPC2의 검증된 test-time hot-path**(DreamerV3 imagination이 hot-path 아니었던 BL-11/Amdahl 함정의 정반대). 따라서 terminal-value term을 줄이는 것은 진짜 bottleneck의 ~절반을 직접 침.

## 제안 메커니즘 (chicken-and-egg 수정 반영)
- **순진한 오류(회피)**: "non-elite는 Q 안 함" → 불가. elite를 *찾으려면* 512개 전부의 value가 필요(topk). non-elite Q를 그냥 skip하면 깨짐.
- **유효 메커니즘 (2-stage)**:
  1. **Cheap-rank**: 512 sample 전부를 *값싼* value로 순위 매김 — Q-ensemble의 부분집합(예: 1~2 head) 또는 단일 경량 value head. 목적은 정확한 value가 아니라 *어느 64개가 elite인가*의 ranking.
  2. **Full-elite**: 선택된 num_elites=64개에만 full 5-head Q를 적용해 softmax weighting(L190-194)에 쓰는 정밀 value 계산.
  3. cheap-rank가 elite 집합을 충분히 보존하면(아래 gate), planning당 full-Q 호출이 512→64로 ~8× 감소(terminal term 한정), dynamics term은 그대로.
- 선택지: cheap-rank를 (a) Q-head 부분집합 vs (b) cheap-rank용 distilled scalar value head(full Q를 teacher로). ablation으로 결정.

```
# _estimate_value 대체 (terminal만):
G = dynamics_rollout(z, actions)          # 기존 (dynamics term, 미변경)
v_cheap = Q_subset(z_term, a_term)        # 512개, 1~2 head (cheap rank)
elite_idxs = topk(G + v_cheap, num_elites)
v_full[elite] = Q_full5(z_term[elite], a_term[elite])  # 64개만 full ensemble
# softmax/weighting은 elite의 v_full로 (기존 L190-194)
```

## 왜 효과 (무엇을, 얼마나)
- terminal Q-ensemble 호출 512→64 (elite만 full). terminal term이 per-sample ~50%이고 그 안에서 5-head가 지배적이면, planning 비용을 의미있게 절감(예상 1.2~1.5× planning speedup, validator가 정밀 측정).
- return 보존이 관건: cheap-rank가 잘못된 elite를 고르면 planning 품질 저하 → episode return 하락. **cheap-rank의 elite-recall이 높으면**(full-rank 대비 동일 elite 다수 포함) return 보존.

## 측정 metric + 예상 (실물 TD-MPC2)
- Base: TD-MPC2 (baselines/tdmpc2), DMControl/locomotion 몇 task, 학습된 checkpoint로 eval(또는 from-scratch 학습).
- 측정: **(1) planning wall-clock/step** (FLOPs 아니라 wall-clock — vectorization 때문), (2) **episode return**(baseline 대비 drop), (3) cheap-rank의 elite-recall@64(full-rank 대비), (4) per-component FLOPs 분해(terminal vs dynamics 실측 — Amdahl 확정).
- 예상: planning 1.2~1.5× (terminal term이 ~50%, 그 중 5-head Q가 elite-only로 8×↓), return drop < 3%. (gate: wall-clock 유의 단축 AND return 비유의 차이.)

## 핵심 ablation + baseline
1. **cheap-rank 방식**: Q-head 1개 vs 2개 vs distilled scalar head — elite-recall vs 비용.
2. **baseline: 무조건 num_q 줄이기**(전 stage에서 5→2) vs 2-stage(전 cheap, elite full). 2-stage가 같은 비용에서 return 더 보존하는가(핵심 — 단순 head 절감과의 차별).
3. **elite-recall → return 인과**: cheap-rank elite-recall을 인위적으로 낮춰(랜덤 섞기) return이 그에 따라 떨어지는지(메커니즘 타당성).
4. wall-clock vs FLOPs 괴리 측정(vectorization 영향 정량).

## 선행 연구 위험 (literature self-check 완료)
- **REDQ / DC-MPC critic subsampling**: 2개 critic subsample이 있으나 *overestimation/learning* 목적. planning 비용을 위한 2-stage elite-preservation 아님. (차별점 frontmatter)
- **AERO-MPPI / Multi-Agent PI (two-stage MPPI)**: 2-stage가 *collision-cost 조기 폐기/guiding trajectory* 목적, learned latent WM value-ensemble 아님. 도메인·목적 다름.
- **TD-MPC2 Q-ensemble vectorization**: wall-time 단축을 *vectorization*으로 이미 함 → 본 방법의 wall-clock 이득이 FLOPs보다 작을 수 있음(위험, wall-clock gate로 대응).
- 메인 리스크: literature-checker가 "cheap-rank then full-elite for sampling-based planning value" 정확 선행 확인 + distilled cheap value head novelty.

## venue
CoRL(continuous control planning 효율) 최적합. 분명한 메커니즘 + 실물 측정 → ICLR/NeurIPS도.

## 위험 요소
| 위험 | 가능성 | 완화 |
|---|---|---|
| terminal term이 dynamics term보다 작아 절감 ceiling 낮음 | 중 | validator가 per-component FLOPs/wall-clock 선측정(horizon=3에서 ~50% 추정이나 MLP 크기차 미반영). 작으면 [[adaptive-mppi-budget]](dynamics term 치는 idea)로 보완 |
| cheap-rank elite-recall 낮아 return 하락 | 중 | distilled cheap value head, elite 수 약간 여유(64→80 후 full 후 64), recall 모니터 |
| vectorization으로 wall-clock 이득 < FLOPs | 중-높 | wall-clock으로 gate, batched cheap/full 구현 최적화 |
