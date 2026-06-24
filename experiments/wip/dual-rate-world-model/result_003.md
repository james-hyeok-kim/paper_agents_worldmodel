# Result 003 — Dual-Rate Ratio/K 재조정 (사전등록 kill-test 결과)

## 판정 (n=5 최종): PARITY — quality 동률, but 어느 축에서도 이득 없음

**n=2 "robust FAIL"은 럭키 베이스라인 착시였음 (advisor 적중).** 바(deter384)를 n=2→n=5로 확장하니
mean 755 → 604로 붕괴(원래 782/730은 운 좋은 빠른 seed 2개). config-A도 604 → **A/바 = 99.9% 동률.**

### n=5 결과 (85k eval_return)
| 그룹 | seed별 | mean | std | range |
|---|---|---|---|---|
| vanilla-384 (바) | 384,556,571,730,782 | 604.4 | 157.5 | [384,782] |
| config-A (128/256,K3) | 525,583,585,640,685 | 603.8 | 61.1 | [525,685] |

### 학습 곡선 (n=5 mean) — sample efficiency 없음
| step | vanilla | config-A | A/van |
|---|---|---|---|
| 25k | 231 | 232 | 100.5% |
| 45k | 434 | 354 | 81.5% (vanilla 앞섬) |
| 65k | 551 | 460 | 83.4% (vanilla 앞섬) |
| 85k | 604 | 604 | 99.9% (수렴) |

### 종합 (matched capacity 384, matched steps)
1. **최종 품질: 동률** (604=604). 원래 seed=0 +12.4%는 confound(deter 384 vs 512) 확정.
2. **sample efficiency: 이득 없음.** 중반(35-75k) vanilla가 앞섬. dual-rate가 빠르지 않음.
3. **wall-clock: 이득 없음 (구조적 추론, config-A 직접 미측정).** config-A는 작은 128 GRU만 skip → 절감 미미. (0.926×는 256/128 측정값이지 A 측정값 아님 — A는 더 적게 skip하니 절감 더 작다는 구조적 논증.)
4. **분산: A가 낮아 보임** (std 61 vs 157). **단 n=5라 underpowered** — F≈6.6은 경계선(단측 0.05 수준), vanilla의 넓은 range는 unlucky seed(384) 하나 탓일 수 있음. 평균에서 n=2가 속였듯 분산에서 n=5도 속일 수 있음. "확립된 분산 감소"가 아니라 **seed 대폭 확대 필요한 hint**로만 기록.

## 결과 (85k eval_return, walker_walk, 2-seed)

| Config | deter_slow/fast | K | seed0 | seed1 | mean | vs deter384(755.7) |
|---|---|---|---|---|---|---|
| **A** | 128/256 | 3 | 583.5 | 685.2 | **634.3** | 83.9% |
| **B** | 128/256 | 2 | 505.1 | 482.6 | 493.8 | 65.3% |
| 기존 | 256/128 | 3 | 284.5 | 465.6 | 375.1 | 49.6% (exp002) |
| **기준** baseline_deter384 | (vanilla 384) | — | 781.7 | 729.7 | **755.7** | — |

100k 최종: A=703.7, B=538.7 (참고, 대조군 deter384는 92k에서 kill되어 85k가 마지막 clean eval)

## 분석

1. **frozen mass 진단 정확**: deter_slow 256→128 (frozen mass 171→85)로 줄이자 375→634 (+69%). slow 상태가 freeze되어 dynamics를 망친다는 가설이 정량 확인됨.
2. **그러나 바 미달**: 최고 config(A) mean 634 < bar 755. A 최고 seed(685) < deter384 최저 seed(729) → 구간 비겹침, robust.
3. **K=2(B)가 K=3(A)보다 나쁨**: B=494 < A=634. frozen을 더 줄여도(64 vs 85) 품질 회복 안 됨 → frozen mass만의 문제가 아니라 slow/fast 2-GRU 분할 구조 자체가 same-capacity 단일 GRU보다 열등.
4. **separation_score 0.84 (높음)**: collapse 없음. 메커니즘은 의도대로 작동하나, 작동하는 것이 곧 이득은 아님.

## 잠정 결론 (n=2, 확인 진행 중)

## 결론: dual-rate는 paper-grade 이득 없음 (parity)

matched capacity(384) + matched steps에서 config-A는 vanilla RSSM과 **동률**.
- 품질 이득 없음 (604=604), sample-eff 이득 없음 (중반 vanilla 앞섬), wall-clock 이득 없음 (작은 GRU skip).
- 유일 차별점 = seed 분산 낮음(std 61 vs 157)이나 n=5 불확실 + 평균 이득 아님.

**efficiency-quality 트레이드오프가 근본적으로 불리:**
- deter_slow 크게+자주 skip(256/128) → 실제 compute 절감, but 품질 375로 붕괴.
- deter_slow 작게(128/256) → 품질 동률, but compute 절감 미미.
- 둘 다 만족하는 sweet spot 없음. slow 상태를 skip하는 만큼 dynamics가 망가짐.

→ **dual-rate efficiency/sample-eff 논문 불가. MWM/STORM 확장 무의미.**
   단 "broken/열등"이 아니라 "parity, no advantage". 11라운드 + 이 확장으로 proper control 하에 종결.
   핵심 자산 = **confound 제거 방법론**(size-matched baseline + n=5 seed가 n=2 착시를 잡음).

## 방향 재설정 옵션 (사용자 결정)
- (A) dual-rate 완전 종료, 새 방향.
- (B) 분산 감소(stability) 각도로 재프레이밍 — 단 n 대폭 확대 + 메커니즘 스토리 필요, 低 EV.
