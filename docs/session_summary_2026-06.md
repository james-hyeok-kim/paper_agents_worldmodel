# Session Summary — World Model Idea Loop (2026-06-18 ~ 06-20 KST)

전체 흐름: idea 생성 → novelty → PoC → 실물 실험 funnel을 반복하며 **통과한 것만 알람**. 이 세션은 R4~R8 + base 인프라 구축.

## 결론 한 줄
**이 세션 survivable idea = 0 (R4~R8 8라운드).** funnel은 나쁜 idea를 *싸게* 죽이는 역할을 정확히 수행. 유일 paper-grade 자산 **dual-rate**(이 세션 이전, quality/sample-eff)는 논문 실험 미완(이제 완주 가능). generation은 경험적으로 소진(gen-r4 구조적 진단).

## base 인프라 (구축·작동 확인, [[base-models-infra]])
| base | 패러다임 | 상태 |
|---|---|---|
| DreamerV3 (dreamerv3-torch) | RSSM imagination | 작동(dmc_proprio), 메인 env torch2.11+cu130 |
| TD-MPC2 (baselines/tdmpc2) | MPPI planning + Q-ensemble | 작동(smoke R27→83), 격리 env `/home/jovyan/envs/tdmpc2` torch2.7.1+cu128 |
| Puppeteer (baselines/puppeteer) | hierarchical TD-MPC2 | 작동(eval corridor R11.7, HL walk 학습), tdmpc2 env + gym/sb3/matplotlib, ckpt 16개 |
| DreamerV4 | transformer/offline/video | **보류** — 공식코드 없음, 커뮤니티 구현 toy(CartPole/MNIST) |

**Infra 수리(영구)**: GL headless = `LD_LIBRARY_PATH=/home/jovyan/egl_libs`(설치 불필요, libglvnd 기존재). GPU = **B200 sm_100** → torch≥2.7+cu128 필수. ([[dmcontrol-headless-egl-fix]])

## 라운드별 결과
| Round | 방향 | 생성 | survival | 사인 |
|---|---|---|---|---|
| R4 | new WM family efficiency | 0 | 0 | 7/7 문헌 포화 |
| R4b | DreamerV3 quality/sample-eff | 1 | 0 | learned-kl-reliability-lambda FAIL (synth+실물, BL-14) |
| R5 | eval-diagnostic | 1 | 0 | decision-fidelity-atlas drop (value-equivalence 선점, thin) |
| R6 | TD-MPC2 planning eff | 2 | 0 | elite-staged FAIL (wall-clock microbench, BL-15) / adaptive-mppi-budget thin |
| R7 | TD-MPC2 non-planning | 0 | 0 | named-paper 선점(iQRL/TD-M(PC)²/RePo) |
| R8 | open-space bet | 0(+1 bet) | 0 | Puppeteer transfer bet FAIL (사전등록 게이트, compute 0) |

## BLACKLIST: BL-01~15 (생성 금지 패턴)
핵심 신규(이 세션): BL-14(imagination reliability→λ-target reweighting: 신호 valid해도 reweight 무이득, 실물 재확인), BL-15(TD-MPC2 planning value 2-stage: vmap-vectorized라 wall-clock 무이득).

## 핵심 교훈 (방법론)
1. **wall-clock ≠ FLOPs**: vectorized/vmap op는 sample 축소해도 GPU kernel-launch-bound라 무이득(BL-10/12/15, dual-rate 0.926×). efficiency idea는 **GPU+CPU wall-clock microbench를 맨 먼저**.
2. **diagnostic은 학습 전 4-bar 사전등록**(plausible/surprising/decision-changing/distinguishable)으로 무료 판정. 못 쓰면 FAIL.
3. **oracle/state-agnostic 통제 + tune/test split**으로 feasibility의 vacuous-pass 방지(BL-14).
4. **Amdahl/hot-path는 코드로 검증**, 단 forward-count(FLOPs)와 wall-clock 구분.
5. **literature-fresh는 novelty를 반증만 가능, 입증 불가** → genuinely-new setting은 search로 de-risk 불가(gen-r4 R8). 진짜 미테스트 lever = human이 정의하는 새 problem-setting.
6. **collision manufacturing 금지** — 0이면 0으로 정직 보고 + open-space 지목.

## 남은 자산 / 권고
- **dual-rate-world-model**: 검증된 유일 survivor(CONDITIONAL-GO 1.93× + 실험 GO). 논문 실험(Plan 001, 11 run)이 6-8k/100k step에서 멈춤 → 수리된 infra로 완주 가능. **최고 EV 미완 작업.**
- agent 상태: generator(gen-r4)만 정상. literature-checker/validator 에이전트는 spawn 즉시 idle(wedge) → 해당 단계는 team-lead가 직접 수행.

## 진행 중
- 2026-06-20 KST: 사용자 요청으로 새 idea loop(R9~) 시작 — 모든 누적 게이트 장착, fresh start (plan_005).
