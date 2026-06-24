---
slug: iris-token-budget-rollout
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 14
---

# 판정: INCREMENTAL

## 검색 요약

| 검색어 | 결과 수 | 관련 논문 |
|---|---|---|
| Mixture-of-Depths transformer token routing 2024 | 8 | MoD (Raposo 2024), Mixture-of-Recursions (2025) |
| IRIS world model autoregressive token efficiency | 10 | Delta-IRIS (2024), POP (2024), DART (2024) |
| early exit transformer world model RL | 7 | LayerSkip, S-GRPO, STORM |
| adaptive computation depth token autoregressive video | 6 | SCOPE (2026), TokenTrim, EVATok |
| STORM efficient transformer world model token | 8 | STORM (2023), Improving Transformer WM (2025) |
| D-LLM token adaptive NeurIPS 2024 | 9 | D-LLM (NeurIPS 2024) |
| input router token importance depth world model | 6 | Sparse Imagination (2506.01392), DDP-WM |
| TIDE per-token early exit LLM | 5 | TIDE (2603.21365) |
| token budget adaptive depth IRIS Atari | 7 | Delta-IRIS, POP, DART |
| token-based world model depth skip budget | 7 | Sparse Imagination, POP, ITC |
| reward sensitivity token depth allocation WM | 8 | 직접 매칭 없음 |
| ACM DL adaptive computation world model imagination | 9 | Delta-IRIS (ICML 2024 ACM) |

---

## 관련 논문 목록 (14개 확인)

1. **Mixture-of-Depths: Dynamically allocating compute in transformer-based language models** (Raposo et al., 2024, DeepMind) — 관련성: per-token top-k routing으로 계층별 FLOPs 동적 배분. input-side 라우터로 expensive 연산 전에 결정. 제안과 메커니즘 수준에서 가장 유사한 논문. 단, LLM 언어 모델 한정, WM/RL 적용 없음, reward-sensitivity 개념 없음.

2. **D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models** (Jyk-122 et al., NeurIPS 2024) — 관련성: 각 transformer layer 앞에 decision module을 두어 token별로 layer 실행 여부를 결정(input-side, decide-before-compute). KV-cache eviction 포함. 50% 연산 절감 달성. 메커니즘 수준에서 제안과 거의 일치. 단, NLP 도메인 한정, WM/RL 없음, reward-sensitive 마스킹 없음.

3. **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding** (Elhoushi et al., ACL 2024) — 관련성: layer dropout 훈련 + 조기 종료 추론. early-exit 보강 개념과 유사. 단, LLM 단일 종료 레이어 방식으로 per-token depth가 아닌 시퀀스 전체 동일 종료.

4. **TIDE: Token-Informed Depth Execution for Per-Token Early Exit in LLM Inference** (2603.21365, 2026) — 관련성: checkpoint layer에 학습된 router 부착, 숨겨진 상태 수렴 감지로 per-token early exit. 수학 문제 해결 LLM 도메인. WM/RL 없음, task-specific depth budgeting 없음.

5. **Adaptive Computation Depth via Learned Token Routing (TSA)** (2605.05222, 2025) — 관련성: per-token gate(경량 2층 MLP)로 연속 확률을 생성해 layer skip 여부 결정. 1.7% 파라미터 오버헤드. character-level language modeling만 평가, WM/RL 없음.

6. **Sparse Imagination for Efficient Visual World Model Planning** (2506.01392, 2025/2026) — 관련성: transformer 기반 visual WM에서 random token subset만으로 MPC 롤아웃 수행. WM + 계획 효율화라는 목적에서 가장 직접적으로 겹치는 논문. **핵심 반증**: importance-based token selection(saliency 기반)이 실패하고 random selection이 우위를 보였음 — 본 아이디어의 reward-sensitive 마스킹의 실효성을 의문에 부침. 단, token 수 감소 방식이지 layer depth 가변화가 아님. DINO-ViT + MPC 도메인, IRIS imagination train loop가 아님.

7. **Transformers are Sample-Efficient World Models (IRIS)** (Micheli et al., ICLR 2023) — 관련성: 제안의 base system. 모든 token이 동일한 L=10 layer를 통과. 본 아이디어가 수정 대상으로 삼는 시스템.

8. **Efficient World Models with Context-Aware Tokenization (Δ-IRIS)** (Micheli et al., ICML 2024) — 관련성: delta 인코딩으로 frame-to-frame 중복 token 제거. 훈련 속도 10× 개선. layer depth 가변화는 없으나, IRIS 효율화 방향 선점. 반증 가능성: IRIS 위에 delta 방식으로 이미 큰 폭의 절감이 달성됨.

9. **Improving Token-Based World Models with Parallel Observation Prediction** (2402.05643, 2024) — 관련성: RetNet 기반 POP으로 sequential token generation을 병렬화, imagination 15.4× 가속. depth 가변화 없음.

10. **STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning** (Zhang et al., NeurIPS 2023) — 관련성: transformer WM에 확률적 성분 추가, Atari 100k SOTA. 2-layer transformer로 효율적이나 depth 가변화 없음.

11. **FFN-SkipLLM: A Hidden Gem for Autoregressive Decoding with Adaptive Feed Forward Skipping** (2404.03865, 2024) — 관련성: 입력 의존적 FFN 계층 skip. per-token layer skip 계열이지만 LLM NLP 전용.

12. **Mixture-of-Recursions (MoR)** (2507.10524, 2025) — 관련성: Recursive Transformer에서 lightweight router가 token별 recursion depth를 동적 결정. 제안과 메커니즘이 유사하나 NLP 도메인.

13. **Not All Frames Deserve Full Computation: SCOPE** (2604.02979, 2026) — 관련성: autoregressive video diffusion에서 프레임 단위 선택적 연산. token별 depth 가변화 아님, RL WM 아님.

14. **Learning to Play Atari in a World of Tokens (DART)** (2406.01361, ICML 2024) — 관련성: IRIS 기반 token WM, Atari 100k 성능 개선. layer depth 가변화 없음.

---

## Claim-by-Claim Novelty 분석

### Claim 1: "input-side router가 expensive 연산 이전에 per-token depth를 결정"
- **선점도: 높음** — MoD(Raposo 2024), D-LLM(NeurIPS 2024), TSA(2605.05222)가 동일 메커니즘을 이미 구현.
- 아이디어 파일이 제시한 "entropy-after-forward 함정 해소"라는 차별화 포인트는 D-LLM/MoD가 이미 해결한 문제.
- **결론: 이 claim 자체는 선점됨. 독립 기여로 인정 불가.**

### Claim 2: "IRIS autoregressive imagination에 적용하여 FLOPs 45~60% 절감"
- **선점도: 낮음** — MoD/D-LLM은 언어 모델 NLP 도메인이며 WM imagination loop에 통합한 사례 없음.
- IRIS + VQ-VAE + autoregressive GPT-2 구조에서 discrete token prediction의 layer depth를 가변화하는 것은 미탐색 영역.
- Δ-IRIS가 delta 인코딩으로 다른 각도의 IRIS 효율화를 달성했으나, layer depth budgeting은 직접 다루지 않음.
- **결론: 이 claim은 생존. IRIS-specific integration이 차별점.**

### Claim 3: "reward-sensitivity를 기준으로 token depth를 배분하고 high-stakes token에 full depth 강제"
- **선점도: 낮음** — 문헌에서 WM reward signal로 per-token depth를 가이드하는 사례 없음.
- 단, **Sparse Imagination(2506.01392)의 반증이 결정적**: visual WM 도메인에서 saliency/importance 기반 token 선택이 random selection보다 열위였음. reward-sensitivity로 depth를 배분하는 접근의 유효성에 대한 직접 의문 제기.
- 반증의 축이 다름(token 수 감소 vs layer depth 가변화)이므로 완전 기각은 아니지만, 왜 reward-guided depth-axis가 성공할 것인지에 대한 pre-argument가 필수.
- **결론: 이 claim은 생존하나, 실험적 검증 부담 높음.**

### Claim 4: "RL imagination 환경의 token redundancy를 직접 활용"
- **선점도: 없음** — RL return objective에 종속된 depth budgeting(WM-specific 목적함수 결합)은 기존 LLM early-exit 문헌에 없음.
- **결론: 생존.**

---

## 판정 근거

**INCREMENTAL로 판정한다. NOVEL이 아닌 이유:**

1. 아이디어의 가장 핵심 메커니즘인 "input-side router + decide-before-compute + per-token depth allocation"은 이미 MoD(2024)와 D-LLM(NeurIPS 2024)이 구현 완료. 제안 아이디어가 강조하는 "entropy 사후판단이 아닌 입력단 결정"이라는 설계 포인트는 선점됨.

2. 독립적으로 가치 있는 두 가지 미탐색 각도가 존재:
   - IRIS autoregressive WM imagination loop에서의 layer depth budgeting (도메인 특화 통합)
   - reward/return-objective에 종속된 depth 배분 (WM-specific 목적함수 결합)

3. **가장 큰 리스크**: Sparse Imagination(2506.01392)이 visual WM에서 importance-based token 선택이 random보다 열등하다는 것을 보였음. 이는 reward-guided compute allocation이라는 제안의 핵심 가정에 대한 직접적 반증. depth-axis에서는 다를 수 있지만, PoC에서 이 반증을 반드시 address해야 함.

**CONDITIONAL-GO로 진행 가능. 단, 아래 두 가지를 PoC에서 먼저 검증:**
1. IRIS FLOP 분해: transformer layer가 실제로 ≥70% hot-path인지 (Amdahl gate)
2. reward-guided depth가 random depth allocation보다 WM 도메인에서 유효한지 (Sparse Imagination 반증에 대한 counter-evidence)

---

## 차별점 (최소 2개, INCREMENTAL 기준)

1. **도메인 특화 통합**: MoD/D-LLM이 LLM NLP에만 적용한 메커니즘을 IRIS 류 autoregressive WM의 imagination training loop에 통합 — VQ-discrete token, RL reward 신호, long rollout 안정성 등 WM-specific 제약을 다룸.

2. **WM-objective coupling**: depth budget을 reward-head gradient saliency와 결합하여 RL return 예측 품질을 우선 보존 — 일반 LLM의 perplexity 보존 목표와 근본적으로 다른 optimization target.

---

## 권고 사항

1. **즉시**: 아이디어 파일에 MoD, D-LLM을 핵심 선행 연구로 명시. "input-side router"를 독자적 novelty로 주장하는 프레이밍 수정 — 이를 motivation이 아닌 기반 기술로 포지셔닝.

2. **PoC 1순위**: IRIS FLOP 분해(transformer hot-path 비율 검증) — gate가 70%에 못 미치면 Amdahl 계산 전체가 무너짐.

3. **PoC 2순위**: reward-guided depth vs random depth 비교 실험 추가. Sparse Imagination의 반증을 이 아이디어가 왜 극복하는지 보여야 함. depth-axis에서 reward-sensitivity가 token-selection-axis보다 더 강한 signal을 제공한다는 근거 필요.

4. **논문 포지셔닝 권고**: "Mixture-of-Depths meets IRIS World Model"이 아니라 "RL-Objective-Guided Depth Budgeting for Autoregressive World Models"로 포지셔닝. 메커니즘 novelty보다 WM-objective coupling을 전면에 내세워야 INCREMENTAL 수준에서 accept 가능성이 높음.
