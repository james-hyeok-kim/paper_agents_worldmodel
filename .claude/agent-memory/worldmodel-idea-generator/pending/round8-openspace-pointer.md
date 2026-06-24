---
slug: round8-openspace-pointer
status: open-space-pointer (NOT a de-risked idea)
created: 2026-06-19 KST
note: "R8 결과: 0 clean mechanism-idea. mechanism-space는 mapped(8라운드). 대신 team-lead 요청대로 '정말 열린 단 하나의 공간' 1개를 pointer로 지목 — 단 이건 literature로 de-risk 불가한 research-taste bet이며 validator에서 산다/죽는다."
---

# Round 8 — Open-Space Pointer (mechanism-idea 0, 구조적 reframe + 1 bet)

team-lead가 사용자 명시 선택("계속 생성")을 존중해 R8 dispatch. efficiency 금지 + 8라운드 포화 cell 금지 + option3(가장 덜 worked된 새 problem-setting). 결과: **0 clean mechanism-idea + 1 open-space pointer.**

## 왜 또 0인가 — 구조적 reframe (R8의 진짜 headline)
**내 freshness-check 방법론은 *genuinely-new problem-setting*을 certify할 수 없다.** novel setting은 대조할 literature가 없기 때문. 그래서 모든 후보가 둘로 갈림:
- clean literature check 통과 ⟹ mapped base/axis에 거주 ⟹ saturation 또는 wall-clock으로 死(R4~R7 전부 이 경로).
- genuinely new ⟹ literature 없음 ⟹ search가 *전혀* de-risk 못 함.

team-lead의 "least-worked corner"(option3)는 두 번째 종류를 요구하는데, 나는 첫 번째용 도구(freshness check)로 사냥해왔다. 이건 노력 실패가 아니라 **구조적 bind**다. 8라운드 0-loop(probe cell→named paper→死)의 근본 원인.

따라서 valid open-space의 판별 기준은 "clean literature check"가 **아니다**. 다음 3-part test다:
1. **측정 harness가 존재하거나 싸게 만들 수 있는가** (R7 robustness가 죽은 이유: harness 부재).
2. **capability가 표면적으로 명백히 가치 있는가** (literature로 못 재므로 face-value 가치로 판단).
3. **기존 WM objective 위의 mechanism이 아닌가** (mechanism은 전부 mapped — loss/architecture 변형은 死).

decision-fidelity-atlas(R5)가 이 category-2의 유일한 시도였고 feasibility로 死 — 이 종류가 얼마나 어려운지의 전례.

## 명시적으로 기각한 R8 후보 (rationalization 회피)
- **world-model-of-frozen-sub-policy / OOD-command degradation 진단**: (a) diagnostic-shaped(decision-fidelity-atlas와 동형, feasibility gate로 死한 전례), (b) transfer harness가 OOD command를 생성/label할 수 있는지 미검증, (c) OHIO("exploits structural knowledge of low-level policy") 인접. → exotic framing을 발명해야 했다는 것 자체가 whitespace가 비었다는 신호. 기각.
- **Puppeteer hierarchical transfer/reuse**: Puppeteer 자기 논문(ICLR 2025, Hierarchical World Models as Visual Whole-Body Humanoid Controllers)의 *contribution 자체*. + THICK Dreamer(ICLR24)/OHIO/MetaWorld-skill-comp(2601.17507)/Hierarchical Planning with Latent WM 선점.
- **skill composition / command-space 합성**: MetaWorld(2601.17507) 등 hierarchical-WM skill transfer 선점.

## THE ONE BET (de-risk 불가, validator에서 산다/죽는다)
**"고정된 유능한 low-level 하에서 high-level whole-body task 획득의 sample-efficiency frontier와 그 한계 요인"** — Puppeteer transfer setting에서.
- **3-part test 통과**:
  1. harness 존재: Puppeteer `TransferWrapper` + RunThroughCorridor/Walk/Run + frozen tracker, 코드 검증됨. frozen low-level 위 high-level 신규 task 학습이 *오늘 실행 가능*.
  2. 가치: 계층의 실질 약속은 "재사용 → 신규 task 빠른 학습"인데, Puppeteer는 *최종 성능*을 보고하지 **acquisition curve/floor를 측정/특성화하지 않음**. "frozen 유능 low-level이 주어졌을 때 high-level 획득의 env-step 하한과 무엇이 그것을 막는가"는 hierarchy의 핵심 실증 질문.
  3. mechanism 아님: 새 loss/architecture가 아니라 *problem-setting/측정축* — capability 질문(category-2).
- **정직한 경고**: 이건 diagnostic-shaped이고(decision-fidelity-atlas 전례), literature로 de-risk 불가. validator에서 **"non-obvious AND 결정을 바꿈"** gate로 산다/죽는다 — "frozen low-level 위 high-level은 N step에 학습된다"가 자명/무가치하면 死. 측정 인프라는 있으나 *결과가 non-obvious할지*는 human research taste의 영역(어느 capability가 중요한가)이지 내가 literature로 보증할 수 없다.

## team-lead/human에게 (정직)
- mechanism-space는 mapped. 다음 살아남는 것은 (있다면) human이 *어느 capability가 중요한가*를 정하는 research-direction bet이지, 내가 generator로 더 돌려서 나오는 게 아니다(8라운드 base rate 0).
- 위 bet 1개는 harness가 있어 *시도 가능*하나, 가치는 보증 못 함. 원하면 validator로 보내 non-obvious 여부부터 싸게 치는 게 맞다.
- EV 순위는 R7과 불변: dual-rate 1편(검증된 survivor) > 또 하나의 죽을 가능성 높은 idea. (재론 아님, 한 줄 확인.)

## 메타교훈 (다음 generator 세션 — 중요)
- **freshness-check는 novelty를 *반증*할 수 있어도 *입증*할 수 없다.** genuinely-new problem-setting은 도구상 certify 불가 → 그런 요청엔 "literature-clean"이 아니라 3-part test(harness/face-value-가치/non-mechanism)로 판단.
- exotic framing을 발명해야 whitespace가 보이면, 대개 그 whitespace는 측정 불가/무가치라 비어있는 것. rationalization 신호.
- 8라운드 누적: survival=0, bottleneck은 generation 아닌 survival, 새 base/axis는 둘 다(saturation·wall-clock) 못 고침. 진짜 미테스트 lever는 *human이 정의하는 새 problem-setting*뿐.
