# worldmodel-idea-validator Memory

PoC 검증 결과 이력.

## 인덱스

<!-- 새 판정 추가 시 아래에 한 줄 추가 -->
<!-- - [<Slug>](<passed|conditional|failed>/<slug>_validation.md) — verdict: <판정> | rollout_speedup: X.Xx | quality_delta: X.XX -->
- [latent-delta-rollout](failed/latent-delta-rollout_validation.md) — verdict: FAIL | rollout_speedup: 최대 37.6x (단독, quality 동시 불가) | quality_delta: 최솟값 1.74 (기준 0.05의 35배)
- [learned-kl-reliability-lambda](failed/learned-kl-reliability-lambda_validation.md) — verdict: FAIL | quality/sample-eff축 | signal validity PASS (critic_spread t-ctrl Spearman 0.535) BUT feasibility robust FAIL (oracle selective-bootstrap도 trivial agnostic m=1 못 이김) | 2026-06-18 (team-lead 직접, advisor-hardened, 사전등록)
