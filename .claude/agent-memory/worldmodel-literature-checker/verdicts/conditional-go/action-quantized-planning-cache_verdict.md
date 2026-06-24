---
slug: action-quantized-planning-cache
verdict: INCREMENTAL
checked-date: 2026-06-11 KST
papers-reviewed: 8
---

## 판정: INCREMENTAL

## 검색 요약
| 검색어 | 결과 수 | 관련 논문 |
|---|---|---|
| MPPI memoization caching transition model-based RL planning efficiency | 8 | 없음 (직접 적중 없음) |
| TD-MPC2 MPPI planning efficiency action quantization | 7 | DC-MPC (Discrete Codebook World Models, 2503.00653) |
| trajectory optimization caching memoization neural network MPPI CEM | 9 | 없음 (직접 적중 없음) |
| action space quantization MPPI model predictive path integral | 8 | AQuaDem (2110.10149), DC-MPC |
| LSH reinforcement learning state action space planning cache | 9 | 탐색 기반 해싱만 (exploration, 1611.04717) |
| transposition table neural network planning MCTS continuous | 8 | 없음 (직접 적중 없음) |
| MPPI duplicate sample deduplication unique trajectory | 9 | 없음 (직접 적중 없음) |
| iCEM improved CEM memory trajectory optimization | 5 | iCEM (2008.06389) — memory=elite 재사용, 시간상관 샘플링 |
| amortized planning computation reuse world model MPPI deduplication | 10 | 없음 (직접 적중 없음) |
| CEM batch deduplication unique samples trajectory optimization speedup | 8 | iCEM, CEM-GD (2112.07746) |

## 관련 논문 목록 (검토된 8개)

1. **iCEM: Sample-efficient Cross-Entropy Method for Real-time Planning** (Pinneri et al., 2020, CoRL) — 관련성: CEM/MPPI의 "memory" 추가. 단, memory는 elite 궤적의 temporal shift carry-over (warm-start)이지 (s,a) 중복 dedup 아님. speedup 2.7~22x 달성 (샘플 수 절감). **핵심 차이: iCEM의 memory = 이전 iteration elite를 다음 iteration 초기화로 재사용 (분포 warm-start). 제안 방법의 memoization = 같은 iteration 내 중복 (s,a) 입력의 transition forward skip (연산 dedup). 완전히 다른 메커니즘.**

2. **CEM-GD: Cross-Entropy Method with Gradient Descent Planner** (2021, arxiv 2112.07746) — 관련성: CEM 효율화, 100x 적은 샘플로 더 나은 성능. gradient 활용으로 zeroth-order 샘플 수 절감. (s,a) 중복 dedup이나 cache 없음.

3. **DC-MPC: Discrete Codebook World Models for Continuous Control** (2025, arxiv 2503.00653) — 관련성: TD-MPC2에 이산 codebook latent 추가해 MPPI와 결합. **action 양자화가 아닌 latent state 양자화.** MPPI transition의 연산 재사용/dedup 메커니즘 없음. MPPI는 수정된 표준 버전 사용.

4. **AQuaDem: Continuous Control with Action Quantization from Demonstrations** (Dadashi et al., 2022, ICML) — 관련성: 연속 action space의 양자화. demonstration prior로 학습된 discretization. planning loop 내 (s,a) memoization 없음. 목적이 근본적으로 다름 (imitation learning 효율화).

5. **TD-MPC2: Scalable, Robust World Models for Continuous Control** (Hansen et al., 2023, arxiv 2310.16828) — 관련성: 기반 시스템. MPPI planning loop, policy prior 5% warm-start, N×H×iters forward 구조. iteration간 momentum 제거. (s,a) 중복 dedup/cache 없음.

6. **Residual-MPPI: Online Policy Customization for Continuous Control** (2024, arxiv 2407.00898) — 관련성: MPPI 효율화 최신 연구. policy prior 통합으로 샘플 수 절감. (s,a) dedup/cache 없음.

7. **MPPI-Generic: A CUDA Library for Stochastic Optimization** (2024, arxiv 2409.07563) — 관련성: GPU 병렬 MPPI 구현. 병렬화를 통한 속도 향상. batched rollout이지만 unique (s,a) batch dedup 아님.

8. **Hashing for Exploration in RL** (Tang et al., 2017, arxiv 1707.00524; Bellemare et al., 2016, 1611.04717) — 관련성: RL에서 LSH 활용 선행 연구. exploration 목적의 state hashing. planning loop memoization 아님.

## Novelty 분석

### 제안 방법과 유사한 점
- **iCEM의 memory 개념**: 이전 iteration 결과를 다음 iteration에 활용 → warm-start 재사용이라는 넓은 의미에서 "계산 재사용"의 정신 공유.
- **DC-MPC의 latent 양자화**: latent state를 codebook으로 이산화 → 양자화 기반 discrete representation 아이디어 부분 공유.
- **AQuaDem의 action 양자화**: 연속 action space를 이산화 → action discretization 아이디어 부분 공유.
- **iCEM, Residual-MPPI**: elite trajectory의 정밀 평가와 non-elite의 근사 처리 방향 부분 공유.

### 명확히 다른 점 (차별점 3개 이상)

1. **dedup memoization 메커니즘의 부재**: 기존 논문 중 MPPI의 N samples 중 **동일 iteration 내** (s,a) 충돌을 탐지하여 transition forward call 자체를 skip하는 방법은 없음. iCEM의 memory는 iteration-간 elite warm-start이고, GPU 병렬화는 모든 sample을 그대로 forward. **중복 forward call을 batch-level에서 제거한다는 아이디어가 선행 연구에 없음.**

2. **LSH 기반 (s,a) 공동 양자화**: DC-MPC는 latent s만 codebook 이산화. AQuaDem은 action만 양자화. **latent state와 action을 공동으로 LSH/grid 양자화하여 transition 캐시 키를 구성하는 방법은 신규.**

3. **value-aware bin 해상도 + elite full-forward 안전장치**: quantization-error를 elite 정밀평가로 bound하는 2-tier 설계. 기존 양자화/approximate 논문들은 이 오차 bound 메커니즘을 planning quality 보장에 적용하지 않음.

4. **MPPI 수렴 특성(elite 집중)을 hit-rate 증가 메커니즘으로 명시 활용**: iteration 후반 분포 수렴 → (s,a) 집중 → cache hit↑ 라는 인과 연결을 명시적으로 설계에 포함한 연구 없음.

## 판정 근거

**INCREMENTAL** 판정 이유:

- **선행 연구 공백 확인**: 8개 논문 검토 결과 MPPI/CEM planning loop 내 (s,a) dedup memoization을 핵심 메커니즘으로 제안한 논문을 발견하지 못함. 직접 경쟁 논문 없음.
- **관련 조각들이 존재**: iCEM (memory/warm-start), DC-MPC (latent 양자화), AQuaDem (action 양자화), GPU-MPPI (batched rollout)의 각 요소가 분리 존재. **이들의 조합+공동 설계를 MPPI dedup에 적용한 논문은 없음** — 즉 incremental 조합이지만 연결이 비자명(non-trivial).
- **NOVEL이 아닌 이유**: 핵심 구성 요소들(iteration간 warm-start reuse, latent 양자화, action 양자화, elite selective evaluation)이 각각 선행 연구에 존재. 완전히 새로운 메커니즘이 아닌, 기존 요소의 새로운 조합.
- **NO-GO가 아닌 이유**: 동일 메커니즘을 직접 선점한 논문 없음. 차별점 3개 이상 확인. PoC 가능성 있음.

**주요 위험**: iCEM의 전체 논문을 확인하지 못함 — iCEM의 "memory" 구현 상세가 (s,a) 재사용을 포함할 가능성 낮지만, 논문 본문 확인 시 revision 필요할 수 있음.

## 권고 사항
- **아이디어 검증(worldmodel-idea-validator)으로 이동 권고.**
- PoC 전 iCEM 논문 본문 §3 methodology 확인 권고 — memory 메커니즘이 시간상관 샘플링+elite carry-over임을 재확인.
- 차별점 강조 전략: "MPPI sample간 within-iteration (s,a) dedup"을 iCEM의 "between-iteration warm-start"와 명확히 대비.
- DC-MPC와의 차별: DC-MPC는 latent space 이산화로 representation 품질 개선이 목적. 제안 방법은 transition forward 연산 절감이 목적 — orthogonal.
