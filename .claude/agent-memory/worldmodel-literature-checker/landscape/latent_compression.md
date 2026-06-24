# Landscape: Latent Compression

latent space 압축, quantization, delta-encoding 관련 논문 목록.

## 핵심 논문

| 논문 | venue | 핵심 기여 | 관련성 |
|---|---|---|---|
| DreamerV2 (Hafner et al., 2021) | ICLR 2021 | Categorical latent (discrete) | latent quantization baseline |
| IRIS (Micheli et al., 2023) | ICLR 2023 | Discrete tokenizer WM | token compression |
| VQ-VAE (van den Oord et al., 2017) | NeurIPS 2017 | Vector quantization | 기반 기법 |

## 검색 키워드 (검증 완료)
- "world model latent quantization" — 확인 필요
- "delta encoding latent states RL" — 확인 필요
- "low-rank latent world model" — 확인 필요

## latent-delta-rollout 검증 중 발견 논문

| 논문 | venue | 핵심 기여 | 관련성 |
|---|---|---|---|
| GateL0RD (Gumbsch et al., 2021) | NeurIPS 2021 | L0 norm penalty로 sparsely changing latent states 학습 | latent temporal sparsity. FLOPs 절감 아님, 표현력 목적 |
| DeltaRNN (Neil et al., 2016) | arXiv 2016 | temporal delta threshold로 RNN MAC 연산 skip. 최대 9-100× FLOPs 감소 | hardware-level temporal sparsity |
| Latent Bridge (2025) | arXiv 2025 | VLM output delta를 경량 predictor로 예측 후 VLM forward pass skip. 1.65-1.73× speedup | "delta predict then skip heavy compute" 패턴 |

<!-- 새 논문 발견 시 추가 -->
