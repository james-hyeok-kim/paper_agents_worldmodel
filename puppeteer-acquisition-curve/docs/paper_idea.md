# Paper Idea — Tracker Competence as a Gate for Hierarchical Policy Acquisition (R8)

## 한 줄 요약
Hierarchical world model(Puppeteer/TD-MPC2)에서 high-level 정책의 학습 가능 여부와 학습 속도가
고정된(frozen) low-level "tracker" 정책의 competence에 의해 좌우된다 — random tracker로는 학습이
거의 일어나지 않고, tracker가 어느 정도 학습되면 high-level 학습이 급격히 가능해진다(corridor,
gaps-corridor, walls-corridor 세 task 모두에서 재현, d>8). 그 이후(500k~10M) 구간은 대체로
좁은 범위에 몰려 있지만 완전히 평탄하지는 않다 — 500k와 3M 사이에는 통계적으로 유의한
국소적 차이가 있다(n=11, p=0.0008).

## 배경 — 이 아이디어에 이르게 된 경로
저장소의 아이디어 루프(`docs/idea_status.md`)에서 14라운드 이상에 걸쳐 세 카테고리가 순차적으로
문헌 선점·un-absorbability·ceiling-capped 등으로 closure됨:
- **efficiency** framing (Round 5~15c): launch-bound / call-skip / batch-reduction 등으로 환원, 6연속 closure
- **quality/sample-efficiency** framing (Round 15b): un-absorbability로 closure
- **data-collection** framing (Round 15d): ceiling-capped(Plan2Explore가 oracle과 거의 동등)로 closure

이 세 카테고리가 모두 소진된 뒤 남은 유일한 경로가 **human-defined problem-setting** —
Puppeteer acquisition-curve bet(R8, 사용자가 직접 지정)이었음. (`docs/idea_status.md` Round 9~15d 참조)

## 핵심 가설
Tracker(고정된 low-level 정책)가 자기 역할 — "high-level이 지시한 목표 위치로 몸을 실제로
움직이는 것" — 을 얼마나 잘 해내는지가, 그 위에 있는 high-level 정책이 얼마나 빨리(적은
학습량으로) 좋은 정책을 배울 수 있는지를 좌우한다. Tracker가 서투르면 high-level을 아무리
오래 학습시켜도 빨리 배우지 못한다 — 즉 tracker가 어느 정도는 "일을 잘 해야" high-level
학습이 빨라질 수 있다는 뜻이다.

(참고: tracker가 "일을 얼마나 잘 하는지"는 이 실험에서 tracker를 얼마나 오래 학습시켰는지 —
0 step(전혀 학습 안 됨, 지시를 줘도 랜덤하게 움직임) ~ 10M step(완전히 학습됨, 지시받은
위치로 정확히 도달) — 로 대신 측정했다.)

## 실험 1 (기초 확인, 통과 ✓) — 잘 훈련된 tracker vs 아무렇게나 초기화한 tracker
- 10M step 훈련된 tracker(유능한 tracker) vs random 초기화만 한 tracker(서투른 tracker)를 각각
  붙여서 비교, corridor task, 200k steps, seed 2개
- **AUC_100k(곡선 아래 넓이 — 초반에 얼마나 빨리 잘하게 됐는지의 척도) 비율 = 4.46배**
  (통과 기준 2배 이상 → 통과), 최종 점수 7.25배, 최종 점수의 절반 도달 시점: 유능한 tracker
  쪽은 20–40k steps인데 서투른 tracker 쪽은 180k 내내 도달 못함
- 결론: tracker가 서투르면 high-level 학습 자체가 안 된다 — random tracker로는 사실상 전혀
  학습되지 않음.
- 출처: `docs/experiment_summary.md` (본 폴더, 실험 1 섹션)

## 실험 2 (완료) — tracker 학습량에 따른 학습 속도 변화
Tracker 학습 step을 0 / 100k / 500k / 1M / 3M / 10M로 나누어 각 지점에서 고정된 tracker를 얹고
high-level(corridor) 정책을 학습, AUC_100k(곡선 아래 넓이)로 학습 속도를 측정하는 ablation.

**n=2 seed 결과** (2026-07-04 완료분, 잠정):

| Tracker steps | 평균 AUC_100k | 0-step(서투른 tracker) 대비 | 10M(완전 학습) 대비 |
|---|---|---|---|
| 0 (random) | 2.139 | 1.00× | 0.22× |
| 100k | 6.059 | 2.83× | 0.64× |
| 500k | 9.872 | 4.62× | **1.04×** |
| 1M | 6.569 | 3.07× | 0.69× |
| 3M | 7.434 | 3.48× | 0.78× |
| 10M | 9.536 | 4.46× | 1.00× |

n=2에서는 100k 이후 관계가 단조증가가 아니라 500k 피크/1M dip/3M 부분회복의 비단조 패턴처럼
보였으나, seed 간 spread가 커서 진짜 효과인지 seed 노이즈인지 확정 불가였음.

**n=5로 확장 후 최종 결과** (2026-07-08 06:12 KST 완료, `docs/experiment_summary.md` 상세):

| Tracker steps | 평균 AUC_100k | std | vs 0-step | vs 10M |
|---|---|---|---|---|
| 0 (n=2, 실험 1에서 재사용) | 2.139 | – | 1.00× | 0.22× |
| 500k (n=5) | 9.245 | 0.940 | 4.32× | 0.97× |
| 1M (n=5) | 8.745 | 2.716 | 4.09× | 0.92× |
| 3M (n=5) | 8.155 | 0.902 | 3.81× | 0.86× |
| 10M (n=2, 실험 1에서 재사용) | 9.536 | – | 4.46× | 1.00× |

(0-step·10M은 이번 seed 확장 대상이 아니었음 — 비단조 패턴이 의심됐던 500k/1M/3M 구간만
n=5로 늘렸고, 두 끝값은 실험 1에서 이미 확인한 값을 기준점으로 그대로 사용.)

ANOVA F=0.492, p=0.62 — 세 구간 모두 통계적으로 구분되지 않음. n=2의 비단조 패턴은 **재현되지
않았고**(1M의 신규 seed 3,5가 11.5 근방으로 높게 나오며 "dip"이 소거됨), 세 구간 모두 10M
대비 86~97%로 서로 겹쳐 있다.

- **확실한 발견**: 0→100k 구간에서 AUC_100k가 2.83× 급증 — tracker가 조금만 학습돼도 질적 전환이
  일어남 (random tracker와는 명확히 다른 체제).
- **잠정 발견 (n=5, 아래에서 수정됨)**: 100k 이후 500k~10M 구간은 plateau(포화)로 보였음 —
  tracker를 500k만 학습해도 10M(완전학습)과 통계적으로 구분 안 되는 acquisition speed를 얻는
  것처럼 보였다.

> **⚠️ 아래 "n=11 재확장" 섹션에서 이 plateau 결론이 수정됨** — 검정력 부족(n=5)이 원인이었고,
> n=11에서는 500k vs 3M가 실제로 유의한 차이로 나타남.

## 실험 2 후속 — seed 11개 재확장 + gaps/walls-corridor 일반화 검증 (완료, 2026-07-10 KST)

**동기**: n=5 결과를 사후 검정력 분석해보니 500k vs 3M 비교(d=1.18)의 검정력이 약 35%에
불과함을 발견 — 80% 검정력에는 그룹당 11 seed 필요. 동시에 지금까지 모든 검증이 `corridor`
task 하나에만 의존했음을 인지, 이미 구현되어 있던 `gaps-corridor`/`walls-corridor`로 일반화
검증을 추가. 500k/1M/3M을 seed 11개로, 10M을 seed 5개로 확장하고, gaps/walls-corridor에서
0-step vs 500k × seed 3개씩 검증 — 총 33 run, 전부 크래시/OOM 없이 완료.

| 조건 | n | 평균 AUC_100k | std |
|---|---|---|---|
| corridor 0-step | 2 | 2.139 | 0.336 |
| corridor 500k | 11 | **9.362** | 0.814 |
| corridor 1M | 11 | 8.935 | 2.362 |
| corridor 3M | 11 | 7.890 | 0.925 |
| corridor 10M | 5 | 8.411 | 1.287 |
| gaps-corridor 0-step | 3 | 1.811 | 0.024 |
| gaps-corridor 500k | 3 | 9.617 | 1.036 |
| walls-corridor 0-step | 3 | 2.375 | 0.243 |
| walls-corridor 500k | 3 | 8.976 | 1.093 |

**통계 재검정 핵심**:
- Welch's ANOVA(1M의 큰 분산을 보정한 정식 omnibus test, 500k/1M/3M): F=7.614, **p=0.0039** →
  세 구간 사이에 진짜 차이가 있음 (등분산 가정 표준 ANOVA는 p=0.086로 이 이질성 때문에 놓침)
- **500k vs 3M: p=0.0008, d=1.689 (큰 효과), Bonferroni(α=0.0167) 생존** — n=5에서는 못 봤던
  유의한 차이. 검정력(n=11 시뮬레이션): 96.3%
- 500k vs 1M, 1M vs 3M: 둘 다 유의하지 않음 (1M은 분산이 매우 커서 어느 쪽과도 안 갈림)
- 3M(n=11) vs 10M(n=5): p=0.447, 유의 아님 / 500k(n=11) vs 10M(n=5): p=0.183, d=0.884
  (중간~큰 효과지만 n=5라 검정력 부족, 확정 불가)
- **gaps-corridor 500k vs 0-step: p=0.0058, d=10.65 / walls-corridor 500k vs 0-step: p=0.0068,
  d=8.34** — corridor의 0→500k 급전환이 완전히 다른 두 task에서 극단적으로 큰 효과크기로
  재현됨

## 왜 논문이 되는가 (비자명성) — n=11 재검정 반영, 수정판

1. "tracker가 있으면 학습이 쉬워진다"는 자명하지 않음 — random tracker도 동일한 action
   space(15-dim appendage delta)를 제공하지만 학습이 전혀 일어나지 않음. tracker의 역할이 단순
   action abstraction이 아니라 물리적으로 의미있는 명령 해석임을 시사.
2. **핵심 주장 (가장 견고함, task 일반화 확인됨)**: tracker가 서투른 상태(0-step)에서 적당히
   학습된 상태(500k)로 바뀌는 순간 high-level acquisition speed가 질적으로 달라지는 급전환이
   있다. 이제 corridor뿐 아니라 **gaps-corridor, walls-corridor 두 개의 새로운 task에서도
   동일 패턴이 극단적으로 큰 효과크기(d=10.65, d=8.34)로 재현**됐다 — task를 바꿔도 유지되는
   견고한 현상이라는 게 이 연구의 가장 강한 근거.
3. **수정된 주장 (n=5 → n=11로 정정)**: "500k 이후로는 tracker를 더 학습시켜도 완전히
   포화(plateau)되어 이득이 없다"는 예전 결론은 부정확했다. n=11로 검정력을 확보하니 500k와
   3M 사이에 실제로 통계적으로 유의한 차이가 있었다(p=0.0008). 다만 이게 "500k가 정점이고
   그 이후 단조 감소"라는 뜻은 아니다 — 1M은 분산이 너무 커서 어느 쪽과도 유의하게 구분 안 되고,
   10M(n=5)은 3M보다 오히려 평균이 높아 단조 감소와도 맞지 않는다. 정확한 서술: *"500k~10M
   구간은 좁은 범위(7.9~9.4)에 대체로 몰려 있어 0→500k 전환에 비하면 부차적이지만, 완전히
   평탄하지는 않고 500k와 3M 사이에는 실제 유의한 국소적 차이가 있다. 1M/10M을 포함한 정확한
   곡선 형태는 seed 수 부족으로 아직 확정 못함."* 이 정정 자체도 흥미로운 방법론적 포인트다 —
   n=5에서 관찰한 "완전 무효과"가 검정력 부족 때문이었고 n=11에서 실제 효과가 드러난 사례.

## Validator 절차에 대한 정정 (2026-07-10 KST)

`round8-openspace-pointer.md`에 기록된 원래 "다음: worldmodel-idea-validator 정식 제출"은
2026-06-24, 실험 1(4.46× PASS)만 있던 시점에 쓰인 것이었다. 표준 `worldmodel-idea-validator`
agent는 "실제 실험 전에 30분짜리 synthetic mock PoC로 rollout_speedup>1.5×/quality_proxy<0.05를
확인해서 컴퓨트 투입 가치를 판단"하는 용도인데(agent 정의 참고), R8은 이미 **실제 데이터로
33-run을 전부 완료**해서 feasibility 질문 자체가 소진됐다 — mock 환경 벤치마크를 도는 건
이미 답이 나온 질문에 가짜 답을 만드는 것과 같다. 대신 `round8-openspace-pointer.md`가 스스로
명시한 R8의 진짜 gate는 **"non-obvious AND 결정을 바꾸는가"**이고, 이건 "human research
taste의 영역"이라 파이프라인이 자체 인증할 수 없다고 못박혀 있음 — 즉 이 판정은 사용자의
research-taste judgment call이며 아직 GO/FAIL이 확정되지 않았다.

## "당연함(obviousness)" 판정 근거 — naive 모델 대비 정량화

가장 우려되는 반박은 "고장난/서투른 low-level이 hierarchy를 망가뜨리는 건 당연하지 않은가"다.
이를 구체적으로 검증하기 위해 naive 선형-스케일링 모델(tracker 학습 진행률만큼만 acquisition
speed가 개선된다는 가정)과 실측치를 직접 비교했다 (0-step=2.139, 10M=8.411을 양 끝점으로 한
개선폭 6.272를 기준):

| tracker 학습량 | 전체(10M) 대비 | naive 선형 예측(0→10M 개선폭 중 확보 비율) | 실측 확보 비율 | 격차 |
|---|---|---|---|---|
| 100k | 1% | 1% | **62.5%** | +61.5pp |
| 500k | 5% | 5% | **115.2%** | +110.2pp |
| 1M | 10% | 10% | **108.4%** | +98.4pp |
| 3M | 30% | 30% | **91.7%** | +61.7pp |
| 10M | 100% | 100% | 100% | 0pp |

전체 학습량의 단 1%(100k step)만에 이미 개선폭의 62.5%를, 5%(500k)에서는 100%를 넘는 수준을
확보한다 — naive 선형 모델이 예측하는 것보다 60~110%p 앞서 있다. "서서히 계속 좋아지는 완만한
곡선"이 아니라 "초반 급상승 후 즉시 포화"라는 뜻이며, 이 격차의 크기 자체가 이 결과의
non-obviousness에 대한 정량적 근거다.

**세 가지 naive 가설과 데이터의 반박**:
1. *"tracker 학습량과 획득 속도는 대략 선형/연속적으로 비례한다"* → 반박: 위 표. 1%/5% 지점에서
   naive 예측(1%/5%)을 크게 앞지름(62.5%/115.2%).
2. *"high-level이 제대로 학습되려면 low-level이 (거의) 완전히 학습돼 있어야 한다"* → 반박:
   완전학습(10M)의 5%만 학습시킨 tracker(500k)가 완전학습 tracker와 통계적으로 구분 안
   되거나(평균은 오히려 더 높음) 유의하게 앞서는(3M 대비) acquisition speed를 만든다.
   hierarchical 시스템에서 흔한 "sub-skill을 수렴까지 pretrain" curriculum 관행이 이 setting
   에서는 compute 낭비일 수 있음을 시사 — 이게 "결정을 바꾸는" 부분.
3. *"서투른 tracker는 학습을 좀 느리게 만들 뿐이다"* → 반박: random tracker(0-step) 조건은
   180k step 내내 점수가 사실상 그대로였다(실험 1) — "느린 학습"이 아니라 "학습 자체가 안 됨"
   (완전한 floor). gradient degradation이 아니라 sharp on/off gate.

(방법론 주의: 100k는 n=2라 신뢰구간이 넓고, 1M은 분산이 커서(std=2.362) 중간 구간의 정확한
형태는 불확실 — 위 반박은 0-step/500k/10M 세 앵커 지점의 견고한 차이에 기반한다.)

시각 자료(곡선·cross-task 재현 그래프)는 이 대화 세션에서 artifact로 생성함
(`r8_obviousness_case.html` 소스는 세션 스크래치 디렉토리).

## 다음 단계
1. **사용자 GO/FAIL 판정 대기** — 위 obviousness 근거를 포함해 R8이 "non-obvious AND
   결정을 바꾼다" gate를 통과하는지는 human research taste 판단. GO 시 validator 보고서
   형식(judgment-based 예외 조항, agent 정의 line 206 "구조적으로 측정 불가 → validator
   판단으로 CONDITIONAL-GO, 근거 명시")으로 기록 후 worldmodel-experiment-planner로 전달.
   FAIL 시 BLACKLIST.md에 mechanism family 기록.
2. (선택, 우선순위 낮음) 1M/10M 조건 seed를 11개 수준으로 추가 확장해 500k~10M 구간의 정확한
   곡선 모양(정점/부분회복 여부) 확정
3. (선택) 100k~500k 사이(200k/300k)를 더 촘촘히 찍어 "정확히 언제 포화가 시작되는지" 좁히면
   naive-모델 반박이 더 정밀해짐

## 출처
- `/home/jovyan/workspace/paper_agents_worldmodel/docs/idea_status.md` (Round 9~15d, R8 판정 섹션)
- `docs/experiment_summary.md` (본 폴더) — 실험 1·2 상세 결과 및 n=11/n=5 통계, gaps/walls 일반화 검증
- `/home/jovyan/workspace/paper_agents_worldmodel/plans/plan_011.md` — seed 확장·일반화 검증 계획 및 완료 기록
- `src/analyze_results_final.py`, `results/final_analysis_results.json` — 재분석 스크립트 및 원본 수치
