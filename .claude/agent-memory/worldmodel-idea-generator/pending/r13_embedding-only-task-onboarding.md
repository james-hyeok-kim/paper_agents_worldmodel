---
slug: embedding-only-task-onboarding
status: pending
created: 2026-06-22 KST
direction: b1/b2 hybrid (capability + sample-efficiency, new problem-setting framing)
base: TD-MPC2 (cwd; multitask mt30/mt80 released checkpoints + datasets)
axis: sample-efficiency / capability frontier (NOT efficiency-compute, NOT diagnostic)
venue-fit: [NeurIPS, ICLR, CoRL]
blacklist-delta:
  - "BL-04 (WM distillation): teacher→student 압축 아님. 단일 frozen checkpoint에 새 embedding row만 추가, 모델 크기/구조 불변."
  - "BL-09/11/14 (efficiency/diagnostic 死): compute 절감 아님. forward 횟수 불변. quality/sample-eff lever."
  - "전 efficiency BL(06~18): wall-clock/GPU-launch 무관. training-time sample-eff 측정."
collision-cleared:
  - "TD-MPC2 (ICLR2024): held-out task adaptation을 *full-model fine-tune*으로만 보고. embedding-only frozen-backbone 미수행. 저자 명시: 'finetuning world models to new tasks is very much an open research problem.' → CLEAR"
  - "CaDM(ICML2020)/HiP-MDP(NeurIPS2017)/PEARL(ICML2019): shared-dynamics+task-latent 개념 선점하나 전부 jointly-train small-scale. frozen large-scale checkpoint + embedding-only + breakdown-boundary 미연구. PEARL은 policy conditioner(dynamics 아님). → PARTIAL, 차별 framing 필수"
---

## 핵심 가설 (claim-led, falsifiable)

**Pretraining manifold 안의 held-out task에서, 96-dim task embedding *하나만* 학습하고 backbone 전체를 freeze하면 full-model fine-tune의 sample-efficiency와 return을 (거의) 따라잡는다 — 수백만 파라미터의 적응이 96차원으로 붕괴한다 — 그리고 pretrained latent dynamics로부터 계산한 dynamics-similarity 지표가 embedding-only가 full fine-tune 아래로 plateau하는 *경계*를 예측한다.**

핵심은 latent **dynamics model의 표현 용량** 질문이다: 학습된 frozen latent-dynamics가 새 task를 embedding 하나로 지원하는 임계가 어디인가 — policy conditioning(PEARL)이 아니라 dynamics capacity의 문제.

## 동기 (Why Now)

TD-MPC2 저자가 ICLR2024 논문에서 직접 명시: *"finetuning world models to new tasks is very much an open research problem."* 그들은 70-task pretrained 19M agent를 held-out 10 task에 **full model**로 naive fine-tune하고 (20k step에서 2× 개선), task embedding은 단지 *init 선택*(similar-task emb vs random)으로만 쓴다. **embedding-only adaptation(backbone frozen)과 그것이 충분한 task-region의 경계는 측정되지 않았다.** 그리고 이걸 *오늘* 칠 수 있는 유일 이유: TD-MPC2가 12개 multitask 체크포인트(mt30/mt80, 1~317M)와 30/80-task 데이터셋을 공개 → 80-task 재학습 없이 frozen checkpoint를 onboard 가능(mt80-48M.pt는 8GB GPU).

## 제안 방법

세팅: pretrained multitask `WorldModel` (mt30 or mt80). `_task_emb` 테이블에 새 task용 row 1개 추가(96-dim). 새 task에 대해 online RL(또는 offline), **gradient는 그 embedding row에만** 흐름(나머지 encoder/dynamics/reward/pi/Q 전부 frozen, `requires_grad=False`).

```
# pretrained mt model, freeze all
for p in model.parameters(): p.requires_grad_(False)
model._task_emb.weight[new_id].requires_grad_(True)   # 단 96-dim만 학습
# (action_mask는 new task action_dim에 맞춰 등록)
```

비교 arm (3개, baseline이 핵심):
- **A. embedding-only** (위, frozen backbone): 96 params 학습.
- **B. full fine-tune** (TD-MPC2 보고 방식, baseline): 전 backbone 학습.
- **C. scratch** (no pretrain): sample-eff 하한 anchor.

측정: 동일 return milestone 도달 env-step (sample-eff), asymptotic return, 그리고 각 held-out task에 대한 **dynamics-similarity 지표** s(τ). 가설: A가 B를 따라잡는 region과 A가 B 아래로 plateau하는 region이 s(τ)로 분리된다 → similarity-gated onboarding rule(s>θ면 embed-only, 아니면 dynamics unfreeze).

**Dynamics-similarity 지표 후보 (이종 embodiment에서 정의 가능해야 함 — #1 die-point):**
이종 action/obs dim 때문에 raw param/state-space 비교 불가. latent-space에서 정의:
- **s1 (latent-rollout residual):** new task의 (s,a,s') 전이를 *frozen* dynamics로 예측한 1-step latent residual ‖f(z,a;e*)−z'‖의 분포 — best-fit embedding e* 하에서. residual이 작으면 frozen dynamics가 이미 그 transition family를 커버.
- **s2 (embedding reachability):** pretrained task embedding들의 convex hull/Mahalanobis 거리 — best-fit e*가 hull 내부면 in-manifold.
- s1이 더 mechanistic(dynamics 직접 측정), s2는 cheap proxy. PoC에서 둘 다 s(τ)와 A−B gap의 상관 측정.

## Novelty 포인트 (최소 3개)

1. **TD-MPC2 대비**: 그들은 full fine-tune만 보고+embedding은 init용. embedding-only(frozen backbone)와 그 충분성 경계는 그들이 *open problem*으로 남긴 정확한 슬라이스.
2. **CaDM/HiP-MDP 대비**: 그들은 shared-dynamics+task-latent를 *jointly small-scale*로 학습(마찰/질량 변형 범위). 우리는 *frozen large-scale 80-task checkpoint*에서 OOD task onboarding과 *breakdown boundary*를 측정. "frozen checkpoint + embedding-only + similarity-predicted boundary"는 미연구.
3. **PEARL/context-meta-RL 대비 (WM-specific firewall)**: PEARL은 policy를 task-context로 conditioning. 본 idea의 질문은 **frozen latent-dynamics model의 표현 용량** — "학습된 dynamics가 embedding 하나로 새 task를 지원하는가, 언제 dynamics 자체를 풀어야 하는가". dynamics-capacity 질문은 policy-conditioning meta-RL에 없음.
4. **결정-변경(bar3)**: similarity-gated onboarding은 실무 결정을 바꿈 — 새 task가 들어오면 s(τ) 계산 후 embed-only(96 param, 빠름) vs dynamics-unfreeze를 *사전에* 라우팅.

## 선행 연구 위험 요소 (literature checker가 확인할 항목)

- **CaDM (ICML2020) / HiP-MDP (NeurIPS2017)**: shared-dynamics+task-latent 선점. → Intro에서 "jointly-trained small-scale variation" vs "frozen large-scale OOD onboarding + breakdown boundary"로 명시 차별 필수. 이 한 문장이 reviewer의 CaDM 공격을 막음.
- **PEARL (ICML2019)** + context-based meta-RL 일반: policy conditioner이지 dynamics conditioner 아님 — firewall 문장으로 분리.
- **Parameter-efficient fine-tuning / prompt-tuning (LoRA/prefix)**: "frozen backbone + tiny adapter"가 일반 PEFT처럼 보일 위험. 차별: PEFT는 *capacity가 충분하면 항상 된다*가 전제, 본 idea의 contribution은 *언제 깨지는가(boundary)* + dynamics-capacity 기전. 단순 PEFT-port로 안 보이게 boundary/기전을 headline으로.
- TD-MPC2 후속/citing 논문 중 multitask transfer를 다룬 것(예: 2024~2026 TD-MPC2 extension) 재확인 권장.

## 예상 실험 Skeleton

- **Base model**: TD-MPC2 multitask checkpoint (mt30-48M 또는 mt80-48M, 공개). cwd에서 `evaluate.py task=mt30 model_size=48 checkpoint=...` 로 로드 검증됨.
- **Held-out tasks**: mt 학습셋에서 빠진 DMControl/Meta-World task (TD-MPC2가 쓴 70→10 held-out split 재현 또는 leave-one-out).
- **Benchmark**: DMControl + Meta-World (state obs; 8GB GPU에서 돌아감).
- **측정**: (1) sample-eff = milestone 도달 env-step, A vs B vs C. (2) asymptotic return gap A−B. (3) s(τ) (s1 latent-residual, s2 embedding-reachability) vs (A−B) gap 상관. (4) similarity-gated rule의 onboarding cost 절감(embed-only가 충분한 task에서 full FT 회피).
- **예상 결과**: in-manifold held-out task에서 A≈B (gap < ~10% return, sample-eff 동등 또는 더 빠름 — 96 param이라 overfit 적음). OOD task에서 A가 B 아래 plateau, 그 분리를 s(τ)가 예측(Spearman 목표 > 0.5).
- **NULL 사전등록**: A가 *어떤* held-out task에서도 B를 못 따라잡으면 → "frozen worse, as expected" → idea 死. s(τ)가 gap을 예측 못 하면(상관 ~0) → boundary 주장 死(가장 큰 die-point).

## Venue Fit 이유

NeurIPS/ICLR: model-based RL의 transfer/representation-capacity 질문, surprising empirical claim(96-dim≈backbone) + 예측 가능한 boundary. CoRL: 이종 embodiment onboarding(humanoid/manipulation)이 로보틱스 transfer 관심사와 직결.

## 위험 요소

| 위험 | 가능성 | 완화 |
|---|---|---|
| **dynamics-similarity 지표 s(τ)가 gap을 예측 못 함 (#1 die-point)** | MED-HIGH | s1(latent-residual)은 frozen dynamics를 직접 측정 → 가장 mechanistic. PoC에서 s1/s2 둘 다 상관 측정, 둘 다 실패 시 boundary 주장 drop하고 "embed-only sufficiency 자체"만 보고(축소 fallback). |
| embedding-only가 *항상* full FT에 크게 뒤짐 (NULL) | MED | TD-MPC2 init-similarity 효과가 큰 점은 frozen emb로도 transfer 신호 있음을 시사. in-manifold task부터 측정(가장 유리한 region). |
| CaDM/PEFT로 incremental 공격 | MED | Intro firewall 문장(jointly-small vs frozen-large-OOD-boundary; policy-cond vs dynamics-capacity). headline을 boundary/기전으로, adaptation 자체로 안 함. |
| held-out split이 mt 데이터셋에 누출 | LOW-MED | TD-MPC2 공식 70/10 split 재사용 또는 leave-one-out으로 깨끗이 분리. |
| 4-bar self-kill 잔존(bar2 surprise가 약하면) | MED | surprise는 "96-dim≈backbone 붕괴"의 정량(얼마나 작은 adaptation이 얼마나 큰 backbone을 대체하나)에 있음 — gap이 작을수록 더 surprising. validator가 bar2를 친다. |

## 내 4-bar self-judgment (validator 전, 정직)

| bar | 판정 | 근거 |
|---|---|---|
| plausible | PASS | TD-MPC2 init-similarity 효과 = frozen emb 신호 존재. 96-dim이 task를 구분하기 충분하다는 건 mt80 성공이 증거. |
| surprising | **CONDITIONAL** | A≈B면 surprising(96≈backbone). A≪B면 unsurprising. → PoC가 결정. NULL 사전등록으로 양방향 가능. |
| decision-changing | PASS | similarity-gated onboarding rule(embed-only vs unfreeze 라우팅)이 실무 결정. |
| distinguishable | **PASS (R9 confound 회피)** | 같은 codebase, 한 축(embedding row)만 변경, 나머지 frozen → attribution 깨끗(A vs B는 frozen 여부만 차이). R9에서 lever 닫은 cross-codebase confound 없음. |

→ **bar2(surprise)가 유일 risk이고 그건 PoC로만 풀림(NULL 사전등록됨).** bar4를 통과하는 게 R9 이후 핵심 — controlled within-codebase 변형이라 가능. validator로 보내 (1) embed-only가 in-manifold에서 full FT를 따라잡나, (2) s(τ)가 boundary를 예측하나 — 둘을 싸게 칠 것.
