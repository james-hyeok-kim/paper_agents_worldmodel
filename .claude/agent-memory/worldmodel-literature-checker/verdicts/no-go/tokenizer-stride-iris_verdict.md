---
slug: tokenizer-stride-iris
verdict: NO-GO
checked-date: 2026-06-11 KST
papers-reviewed: 7
---

## 판정: NO-GO

## 검색 요약
| 검색어 | 결과 수 | 관련 논문 |
|---|---|---|
| IRIS world model token-space reward head latent imagination decoder skip 2024 | 8 | Delta-IRIS(2406.19320), REM(2402.05643) |
| IRIS transformer world model VQVAE decoder skip imagination efficiency | 7 | REM(2402.05643), 2502.01591 |
| latent policy optimization token world model actor critic without decoding 2024 2025 | 10 | REM(2402.05643), DART(2406.01361) |
| token world model actor policy latent tokens token embeddings imagination without decoder 2024 | 8 | REM(2402.05643), DART(2406.01361), LIMT(2407.13466) |
| STORM IRIS DIAMOND imagination token space latent space actor reward value head 2024 2025 | 9 | STORM, DIAMOND |
| world model imagination frame stride periodic decoding anchor latent drift 2024 2025 | 8 | Sparse Imagination(2506.01392) |
| IRIS decoder efficiency imagination bottleneck VQVAE tokenizer profiling 2024 | 8 | REM(2402.05643), Delta-IRIS(2406.19320) |

## 관련 논문 목록

1. **REM: Improving Token-Based World Models with Parallel Observation Prediction** (Corchero Saura et al., 2024, ICML 2024) — 관련성: 핵심 선점. actor-critic이 latent token에서 직접 운영되며 imagination 중 decoder를 완전히 bypass. 논문 명시: "the controller operates on latent tokens and their embeddings, trained to maximize return entirely in imagination." IRIS pixel-controller(C_IRIS) 대비 ablation이 존재하여 제안 방법의 정확한 IRIS comparison을 이미 수행. [arXiv:2402.05643](https://arxiv.org/abs/2402.05643)

2. **DART: Learning to Play Atari in a World of Tokens** (Anand et al., 2024, ICML 2024) — 관련성: 핵심 선점. "policy observes K discrete tokens received from the world model"이며, VQ-VAE decoder는 representation learning에서만 사용되고 imagination 중에는 완전히 미호출. IRIS 대비 "CNN/LSTM policy after image reconstruction" 방식을 명시적으로 개선 포인트로 지적. [arXiv:2406.01361](https://arxiv.org/abs/2406.01361)

3. **Delta-IRIS: Efficient World Models with Context-Aware Tokenization** (Micheli et al., 2024, ICML 2024) — 관련성: 부분 선점. stochastic delta token 기반 효율화. 다만 imagination 중 decoder를 매 frame 호출하며 actor는 decoded pixel frame을 소비 — REM/DART와 달리 decoder를 유지. [arXiv:2406.19320](https://arxiv.org/abs/2406.19320)

4. **DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning** (Zhou et al., 2024, ICML 2025) — 관련성: 관련. decoder를 training objective에서 완전히 분리("decoder is entirely optional"). latent 상에서만 transition model이 운영되며 decoder 없이 planning. [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)

5. **LIMT: Language-Informed Multi-Task Visual World Models** (Ferreira et al., 2024) — 관련성: 부분. actor/critic이 "observation token embeddings as inputs"를 받아 latent imagination에서 공동 훈련됨을 명시. [arXiv:2407.13466](https://arxiv.org/abs/2407.13466)

6. **Sparse Imagination for Efficient Visual World Model Planning** (2025) — 관련성: 인접. token-level sparsity로 rollout 효율화. decoder skip이 아닌 token 수 감소 접근. [arXiv:2506.01392](https://arxiv.org/abs/2506.01392)

7. **Improving Transformer World Models for Data-Efficient RL** (Robine et al., 2025) — 관련성: 인접. IRIS 계열 개선, nearest neighbor tokenizer, block teacher forcing. [arXiv:2502.01591](https://arxiv.org/abs/2502.01591)

## Novelty 분석

### 제안 방법과 유사한 점
- **actor/critic을 token-space에서 직접 운영, decoder 미호출**: REM(2402.05643)과 DART(2406.01361)이 이미 정확히 동일한 메커니즘을 구현. 두 논문 모두 ICML 2024 채택.
- **token-space reward/value head**: REM의 controller가 token embedding에서 reward/value를 추정. LIMT도 동일 구조.
- **IRIS imagination에서 pixel 재구성 제거**: DART가 이를 IRIS 대비 개선 포인트로 명시적으로 주장하고 실험.

### 명확히 다른 점 (차별점)
- **K-frame stride decode anchor + adaptive re-anchor**: 제안 방법의 유일한 미선점 요소. K frame마다 decode→re-encode하여 token-space drift를 bound하고 stride를 adaptive하게 조정.
- 그러나 이 메커니즘은 REM/DART가 해결한 문제(token-space에서 직접 훈련)를 우회하는 방식으로, 선행 방법 대비 추가 비용을 발생시킴. drift 문제를 token-native training으로 근본 해결한 REM/DART에 비해 열위.

## 판정 근거

**NO-GO**: 제안 방법의 핵심 contribution인 "IRIS imagination에서 decoder를 skip하고 actor/reward/value를 token-space head로 운영"이 REM(arXiv:2402.05643, ICML 2024)와 DART(arXiv:2406.01361, ICML 2024) 두 논문에 의해 이미 선점되었다. 두 논문 모두 IRIS의 pixel-controller를 token-space controller로 교체하는 것을 명시적 contribution으로 주장하며 ablation study를 포함한다.

더 결정적으로, 제안 방법은 "K frame마다 1회 decode"를 유지하는 반면 REM/DART는 imagination 중 decoder call을 완전히 제거한다. 즉, 제안 방법은 선행 연구 대비 더 많은 비용을 사용하는 열위 버전이다. 유일한 차별 요소인 K-frame anchor는 REM/DART가 존재하지 않는 문제(decoder 의존 actor가 있을 때 발생하는 drift)를 처리하기 위한 장치로, 선행 방법이 근본적으로 해결한 문제에 대한 우회책이다.

아이디어 파일이 가정한 "IRIS imagination에서 decoder call이 매 frame 호출된다"는 전제 자체도 REM/DART의 존재로 인해 무효화된다.

## 권고 사항
- **BLACKLIST 추가 권고**: "token-space actor/critic without decoder in IRIS-style imagination" 방향을 BLACKLIST에 등재. BL 번호 신규 부여 필요.
- 후속 방향으로 검토 가능한 것: REM/DART가 해결하지 못한 문제, 예를 들어 (1) K-frame stride가 아닌 sparse selective reconstruction이 *필요한* 경우(다운스트림 task가 pixel fidelity를 요구하는 경우), (2) token-space actor의 generalization 한계 개선, (3) token-space reward head의 reward-sparse 환경 성능 등.
