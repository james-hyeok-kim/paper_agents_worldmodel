# BLACKLIST — 생성 금지 아이디어 패턴

worldmodel-idea-generator는 이 파일을 반드시 먼저 읽은 후 아이디어를 생성한다.
BLACKLIST 항목과 유사한 아이디어는 최소 3개 차별점이 없으면 생성 금지.

---

## BL-01: 단순 LLM KV캐시 재사용 → WM latent 캐싱
- **패턴**: Transformer WM에서 이전 토큰의 KV cache를 그대로 재사용
- **이유 기각**: TWM, STORM에서 이미 탐구. world-model-specific challenge(stochastic transition, reward head) 없음
- **차별화 필요**: stochastic latent와 결합한 selective cache invalidation, reward-conditioned cache 등

## BL-02: 단순 early stopping (fixed threshold)
- **패턴**: rollout step이 N을 넘으면 무조건 중단
- **이유 기각**: 고정 임계값은 task-agnostic이며 adaptive 요소 없음. 단순 hyperparameter
- **차별화 필요**: uncertainty 또는 value gradient 기반 dynamic threshold

## BL-03: 픽셀 레벨 압축 (JPEG/codec) 적용
- **패턴**: latent state가 아닌 픽셀 observation에 손실 압축 적용
- **이유 기각**: perception 품질 저하를 world model 효율화와 혼동. WM과 무관
- **차별화 필요**: latent space 내부에서의 압축, trajectory-level 압축

## BL-04: 단순 world model distillation (teacher→student)
- **패턴**: 큰 WM을 작은 WM으로 knowledge distillation
- **이유 기각**: 일반 모델 압축 기법이며 world model specific이 아님. ICLR/NeurIPS 기준 contribution 부족
- **차별화 필요**: latent dynamics에 특화된 distillation (stochastic state, multi-step prediction), 또는 RL policy와의 co-distillation

## BL-05: hyperparameter tuning (rollout horizon 길이 탐색)
- **패턴**: 최적 imagination horizon을 grid search로 찾기
- **이유 기각**: 단순 ablation 수준. 방법론적 novelty 없음
- **차별화 필요**: task에 따라 adaptive하게 horizon을 예측하는 learned scheduler

---

<!-- BL-06 (latent-delta-rollout): PoC inconclusive — metric bug, re-running.
     quality 측정에서 eps 공유 미적용으로 인한 RNG 노이즈 오염, speedup도 wall-clock 노이즈.
     수정된 PoC 결과 나오면 확정 업데이트 예정. 현재 BLACKLIST 효력 없음. -->

## BL-06: RSSM deterministic state carry-forward (delta-skip)
- **패턴**: GRU h가 작게 변할 때 update를 skip하고 이전 h를 carry-forward
- **이유 기각**: PoC 검증 결과, carry-forward로 인한 누적 drift가 quality_proxy ~1.9 (gate 0.05 대비 38× 초과). stochastic z가 frozen h에서 샘플링되어 diversity 유지해도 h drift가 decoder/reward 품질을 훼손.
- **차별화 필요**: drift 누적 없이 연산 절감 (예: 저차원 h projection + sparse update, 또는 delta를 보정하는 lightweight residual 적용)

## BL-07: MPPI/planning 분기 KL merge (planning branch dedup)
- **패턴**: 같은 초기 state에서 갈라지는 N개 planning 분기를 KL 기반으로 merge해 forward 횟수 절감
- **이유 기각**: PoC 검증 결과, high-action regime에서 merge rate 9.9%에 그쳐 speedup 최대 1.11× (gate 1.5× 미달). action이 latent에 실질적 영향을 주는 환경(=planning이 실제로 필요한 환경)에서 분기가 빠르게 발산해 merge 이득이 소멸.
- **차별화 필요**: merge가 아닌 tree 구조 재활용 (공통 prefix subtree의 shared computation), 또는 action-conditioned clustering

## BL-08: RSSM prior entropy 기반 adaptive horizon (learned controller)
- **패턴**: prior entropy 변화율과 reward-head 분산을 입력으로 learned MLP가 각 step에서 조기 종료 결정
- **이유 기각**: PoC 검증 결과, Type B 에피소드(post-plateau에도 significant return 존재, 35%) 때문에 controller가 항상 continue 선택 → speedup 1.0×. positive label rate 3.1%로 imbalance가 극심해 학습 실패. reward-cliff 구간을 사전에 알 수 없으면 safe stopping이 불가능.
- **차별화 필요**: stopping 대신 horizon을 task-level에서 결정(meta-learning 접근), 또는 단순 truncation이 아닌 variable-depth rollout with guaranteed value bootstrap

## BL-09: static/dynamic encoder 분리로 에피소드당 1회 amortize
- **패턴**: encoder를 static(에피소드 불변)/dynamic(매 step)으로 분리해 static을 1회만 호출
- **이유 기각**: PoC 검증 결과, encoder FLOPs 절감이 decoder+dynamics에 희석되어 end-to-end speedup 최대 1.2× (gate 1.5× 미달). DreamerV3에서 decoder(19.8k params)가 encoder(4.4k params)보다 4× 크고 매 step 실행되어 bottleneck이 아닌 부분을 최적화.
- **차별화 필요**: encoder+decoder를 함께 절감하는 방식 (예: imagination 중 decoder를 아예 skip하고 latent-only reward 추정), 또는 bottleneck인 decoder를 sparse하게 호출하는 방법

## BL-10: rank-r corrector로 RSSM GRU 연산 절감 (cheap-per-step GRU replacement)
- **패턴**: GRU를 더 작은 rank-r 연산으로 교체하거나, step별 cheap corrector로 근사해 FLOPs를 줄임
- **이유 기각**: PoC 결과, (1) GPU에서 소형 GRU(D=128 vs D=512)는 kernel launch overhead 때문에 실제 speedup이 이론치(16×)의 1/10 수준인 1.41×에 그침. (2) prior MLP가 transition의 59%를 차지해 Amdahl headroom 자체가 부족(GRU=41%). (3) corrector drift 0.25 >> gate 0.05. 핵심: GPU에서 "작은 연산 여러 번"은 "큰 연산 한 번"보다 느릴 수 있음.
- **차별화 필요**: GPU에서 실질적 speedup은 "call을 완전히 skip"할 때만 나옴(dual-rate 패턴). 연산 자체를 줄이는 방식은 GPU batch에서 이득이 없음.

## BL-11: trajectory-space selective imagination (state별 rollout 선택적 시작)
- **패턴**: latent-only proxy critic이 value-variance 높은 state를 선별해 해당 trajectory만 imagine
- **이유 기각**: PoC 결과, (1) 1-step latent proxy와 H-step return-variance의 상관 0.085 (proxy가 작동 안 함). (2) imagination이 model-based update의 37%에 불과 — encoder/decoder가 더 큰 비중. (3) 배치를 M개 sub-batch + proxy scoring으로 분리하면 오히려 느려짐(0.49×). 핵심: DreamerV3-style model-based update에서 imagination은 hot-path가 아닐 수 있음.
- **차별화 필요**: 실제 DreamerV3 imagination에서 hot-path를 먼저 프로파일링 후 타겟 설정. 또는 imagination이 실질 hot-path인 IRIS-style transformer WM에서만 적용.

## BL-12: 2-tier world model conditional routing (cheap/expensive model-tier selection)
- **패턴**: 작은 GRU와 큰 GRU 두 모델을 두고, policy-gating이 step별로 어느 모델을 쓸지 선택
- **이유 기각**: PoC 결과, (1) GPU에서 cheap GRU(D=128)가 expensive GRU(D=512)의 71%를 걸림 — FLOPs 이론치(1/16) 대신 실제 0.71×(kernel overhead 지배). (2) 70% cheap routing 시도해도 theoretical speedup 1.26×로 gate 1.5× 미달. (3) 미학습 cheap model의 value-gap이 0.73 (정상화 후)으로 gating 기준으로 쓸 수 없음(routing 가능한 state 7.4%뿐). BL-10과 동일한 GPU kernel overhead 문제.
- **차별화 필요**: GPU에서 실질 speedup을 얻으려면 call 자체를 skip해야 함. 또는 model-tier를 episode/batch 단위로 (step이 아닌) 선택해 kernel overhead를 amortize.

## BL-13: transformer layer depth budgeting for WM token rollout (IRIS early-exit)
- **패턴**: IRIS-style autoregressive WM에서 easy token은 얕은 layer에서 early-exit, hard token만 full depth
- **이유 기각**: PoC 결과, (1) IRIS 상에서 tokenizer(CNN VQVAE)가 비용의 76%를 차지해 transformer 최적화 Amdahl headroom이 24%뿐. (2) token sub-batch 분리(easy/hard) 시 GPU에서 2개 forward pass로 분리 → kernel overhead로 오히려 느려짐(0.66×). (3) 품질은 PASS(CE delta 0.0004)지만 speedup 불가능. 핵심: IRIS에서 진짜 bottleneck은 tokenizer(VQVAE encoder/decoder)이지 transformer가 아님.
- **차별화 필요**: tokenizer 자체를 절감(VQVAE encoding 비용 줄이기), 또는 transformer를 bottleneck으로 만드는 다른 WM 아키텍처에서 적용.

<!-- FAIL 판정 시 worldmodel-idea-validator가 아래에 추가 -->

## BL-14: imagined trajectory reliability/confidence로 λ-return(정책/critic target) reweighting
- **패턴**: RSSM imagination의 step별 신뢰도(학습된 KL-proxy, critic-distribution spread, prior-entropy 등)를 추정해 λ-return target의 step별 기여를 가중(soft weighting / effective discount / selective bootstrap)해 value overestimation 억제 → sample-eff 개선.
- **이유 기각 (PoC, learned-kl-reliability-lambda)**: 신호 validity 자체는 통과(critic-spread가 decision-relevant drift와 t-controlled Spearman 0.535, BL-08 prior-entropy는 -0.185로 실패). **그러나 그 신호로 target을 reweight하는 메커니즘이 무이득**: sign-unbiased selective bootstrap에서 **oracle(완벽한 true-drift 신호)조차** "무조건 가장 이른 신뢰 지점에서 bootstrap"하는 trivial **state-agnostic 균일 전략(m=1)**을 못 이김(oracle 3.942 vs agnostic 3.918, full 4.846, bar 3.722). λ-target 정확도는 critic bootstrap이 지배 → "더 잘 가중"이 "더 일찍 bootstrap"의 이득을 넘지 못함.
- **차별화 필요**: reliability 신호를 *target reweighting*이 아닌 다른 곳에 써야 함(예: imagination 조기 종료로 compute 절감 — 단 efficiency라 BL-08/GPU overhead 재검토; 또는 exploration uncertainty bonus). 핵심 교훈: **"신호가 drift를 예측한다"(validity)와 "그 신호로 target을 고치면 이득"(feasibility)은 별개** — 후자는 state-agnostic shortening/bootstrap baseline + oracle 통제로 반드시 검증.
- **실물 DreamerV3 재확인 (2026-06-19)**: 실제 DreamerV3(dmc_proprio walker_walk)에서도 FAIL. synthetic과 정반대 regime — 실물에선 신호 validity 자체가 붕괴(critic_spread t-ctrl -0.058)하고 H=15 imagination이 충분히 정확해 full rollout이 최적(agnostic m=15, oracle 무이득). 표준 DreamerV3 H=15에서 "far-horizon imagined step이 신뢰 낮다"는 전제가 성립 안 함. 이 방향 재시도 금지.

## BL-15: TD-MPC2 MPPI terminal value-ensemble 2단계화 (cheap-rank → full-elite Q)
- **패턴**: MPPI planning에서 terminal Q-ensemble 평가를 cheap-rank(Q-subset/distilled scalar)로 512 sample 순위 → full ensemble은 elite 64에만 적용해 planning 비용 절감.
- **이유 기각 (microbench, elite-staged-value-planning)**: 실물 TD-MPC2 wall-clock 마이크로벤치(best-case cheap-rank=distilled scalar)에서 **GPU 0.94×(느려짐), CPU 1.011×(flat)** — 양 device 모두 gate 1.2× 미달. 원인: TD-MPC2 Q-ensemble은 vmap **vectorized 단일 op**(`qnet(z)`가 5-head 한 번에 계산, value엔 2-head subsample) → sample 512→64 축소가 GPU에선 kernel-launch-bound(BL-10/12, dual-rate 0.926× 재확인)라 무이득 + cheap-rank launch 추가로 오히려 느림. CPU(FLOP-bound)서도 terminal이 작아 1.1%뿐 → terminal Q는 planning wall-clock bottleneck이 아님(dynamics rollout이 horizon-sequential로 지배).
- **차별화 필요**: vectorized ensemble의 sample 수 축소는 wall-clock 이득 없음. planning efficiency는 **iterations 또는 dynamics rollout(horizon-sequential)** 을 줄여야 함. 단 iteration 조기종료(adaptive-mppi-budget)는 Adaptive Online Planning(1912.01188)이 선점(novelty thin). 교훈: planning efficiency 아이디어는 **wall-clock 마이크로벤치를 GPU+CPU로 맨 먼저** 쳐서 vectorization/launch-overhead를 확인할 것(FLOPs-count Amdahl로는 부족).

## BL-16: action-quantized planning cache (MPPI 중복 (s,a) transition memoization/dedup)
- **패턴**: TD-MPC2 MPPI에서 (s,a)를 양자화해 충돌하는 transition forward를 dedup/memoize, unique만 forward하고 scatter로 복원해 planning 연산 절감.
- **이유 기각 (PoC Stage 1 wall-clock microbench, R12)**: 실제 TD-MPC2 dynamics 아키텍처(latent512+action, 2×512 MLP, SimNorm)로 **oracle best-case collision(rho=0.9, U=53/536)** 줘도 GPU 0.397×(2.5배 느림), CPU 1.209×(rho=0.9에서만, 1.5× 미달). 원인: TD-MPC2 dynamics는 작은 vectorized MLP(plan당 1.78ms, kernel-launch-bound)라 dedup 기계(hash+torch.unique+gather+scatter)가 절약하는 forward보다 더 많은 launch를 추가. BL-10/12/15와 동일 — "batch dedup(batch 축소)"도 결국 gather/scatter라 GPU launch overhead 지배. 현실 collision은 90%보다 훨씬 낮아 더 나쁨.
- **차별화 필요**: GPU에서 batch-dedup/gather/scatter 기반 절감은 작은 vectorized op에서 무이득(launch-bound). planning 절감은 dynamics rollout 자체를 *통째로 fewer-step*으로 줄여야 함(per-step gather 없이). 단 그 방향(adaptive horizon/iteration)은 BL-08/AOP 선점. 교훈 재확인: vectorized op의 batch 축소는 wall-clock 이득 없음 — gather/scatter는 절감을 상쇄.

## BL-17: value-bootstrapped depth scheduler (pre-rollout depth-level grouping, staircase rollout)
- **패턴**: DreamerV3 imagination에서 초기 state별 rollout depth를 예측해 이산 level{5,10,15}로 group, shallow group이 fewer img_step. per-step gather 없이 pre-rollout 1회 grouping(staircase: 전체 N→2N/3→N/3로 점차 축소)이라 BL-11 회피 주장.
- **이유 기각 (PoC Stage 1 wall-clock microbench, R12)**: 충실한 RSSM(stoch1024+action, GRU512, hidden512, H=15)에서 **best-case staircase조차 GPU 1.069×**(FLOPs 천장 1.5× 대비), naive 3-group은 0.541×(느림). CPU staircase 0.937×(느림). 원인: img_step이 N=1024 batch에서 launch-bound라 rollout 후반 batch 축소(10N vs 15N row-step)가 wall-clock으로 거의 환산 안 됨. BL-11(sub-batching 0.49×) 동일 벽. + Amdahl(imagination 37%)로 이득은 더 희석.
- **차별화 필요**: GPU에서 row-step/batch 축소 기반 절감은 launch-bound라 무이득. imagination 단계 자체가 hot-path가 아니고(37%), batch 작아지는 late-step 절감은 wall-clock에 안 잡힘. 교훈: FLOPs 천장(1.5×)이 있어도 launch-bound면 realized speedup은 1.0× 근처.

## BL-18: spectral multi-level RSSM (차원별 자기상관으로 L>2 band, band별 sub-GRU call-skip)
- **패턴**: RSSM latent 차원을 temporal autocorrelation으로 L개 frequency band에 배정, band b는 K_b step마다만 sub-GRU 갱신(나머지 call-skip). dual-rate(L=2)를 L>2로 일반화, "중간 스케일 차원"의 Pareto gap을 채운다 주장.
- **이유 기각 (PoC Stage 1 wall-clock microbench, R12)**: 충실한 RSSM(deter512→3 band[170,170,172], period[1,2,4], H=15)에서 GPU **0.710×**(느림). dual-rate(2-band) measured 0.926×보다 더 나쁨 — 예측대로 L>2는 더 많고 작은 sub-GRU call이라 launch overhead 가중. avg 1.75 band-call/step인데도 절감은커녕 30% 느림. + quality 측면도 dual-rate parity 종결(2026-06-22)로 "L>2가 Pareto gap 채운다" 주장의 base가 이미 무너짐(skip하는 만큼 dynamics 손상은 band 수와 무관).
- **차별화 필요**: 없음. call-skip 기반 dynamics 절감은 GPU launch-bound(dual-rate 0.926×, spectral 0.710×, BL-10/12) + quality는 parity가 천장. multi-band는 dual-rate의 strictly-worse 변형. 이 방향(RSSM call-skip 효율) 전체 종결.

## BL-19: embedding-only task onboarding (frozen backbone + single task embedding 적응)
- **패턴**: TD-MPC2 multitask checkpoint에서 backbone(encoder/dynamics/reward/pi/Q) 전부 freeze, task embedding table에 새 task용 96-dim row 하나만 추가해 학습. "96-dim이 수백만 파라미터를 대체한다" 주장.
- **이유 기각 (PoC R13, embedding-only-task-onboarding)**: collapse-proof evaluation(z_true=frozen base encoder)에서도 **embed-only가 full FT보다 3~100× 높은 eval loss** (mass=1.0 in-manifold에서도 gap=66%). dynamics function 자체가 task-specific하게 변하는 경우(질량/물리 변형), 96-dim embedding vector는 "어떤 task인지"만 알려주지 dynamics network weight 자체를 바꾸지 못함 → 근본적으로 표현력 부족. Q2(embedding 이동 거리 s2로 gap 예측)도 FAIL(rho=-0.536) — s2가 단조증가해도 absolute gap은 비단조라 상관 무효. 가장 유리한 in-manifold task에서도 comparable하지 않음 → NULL 조건 적용.
- **핵심 교훈**: frozen backbone + embedding-only는 동일 obs/action dim이어도 dynamics function 변화에 무력함. "task embedding이 충분하다" 주장은 PoC 기준 기각. s2(embedding reachability)는 absolute gap predictor로 circular하지 않지만 비단조 base-loss 때문에 무효 — 상대 gap(gap/embed)과의 상관은 양수 가능성 있으나 post-hoc이라 invalid.
- **차별화 필요**: embedding-only 아이디어 전체 금지. "몇 개 파라미터만 학습"을 핵심 주장으로 쓰려면 (1) 실제 policy return 측정, (2) LoRA/adapter 방식처럼 backbone 일부 low-rank update 허용, (3) in-manifold 정의를 embedding space proximity로 더 엄격히 통제해야 함. dynamics function이 변하는 한 embedding-only로 full FT를 따라잡는다는 주장은 무효.

## BL-20: task-conditioning locus (same-embodiment family에서 dynamics vs reward 모듈별 embedding sensitivity 분리)
- **패턴**: 동일 embodiment family (walker-{stand,walk,run}) 내에서 wrong task embedding을 주입했을 때 dynamics head는 robust하고 reward head만 fragile하다 → 두 모듈의 역할이 분리되어 있다는 주장.
- **이유 기각 (PoC R14, task-conditioning-locus)**: **(1) Tautology**: walker-stand/walk/run은 동일 physics XML을 공유 — body mass, inertia, joint structure 모두 identical, reward function parameter만 다름. dynamics function이 identical하므로 "wrong embedding에 robust"는 설계상 당연한 결과, novel finding이 아님. **(2) reward arm OOD**: velocity sweep으로 synthetic states(qvel[1] 수동 설정) 생성하면 expert policy rollout 분포 외부 — v=1에서 pred_walk=0.39 vs actual=0.83(gap 0.44). correct/wrong embedding 모두 크게 틀려 sensitivity 신호가 noise 수준. wrong embedding이 우연히 더 낮은 MSE를 주는 case(walk→run pair: ratio=+2.84 only, stand→walk/run: ratio 음수). Q1 mean ratio = -0.48, FAIL.
- **핵심 교훈**: (1) "같은 embodiment = dynamics 공유" 는 dynamics robustness의 trivial 원인이지 분리 발견이 아님. (2) reward sensitivity 테스트는 policy-generated in-distribution data가 없으면 OOD로 망가진다. (3) 흥미로운 역방향 가설("reward head가 task embedding을 underuse한다")은 in-distribution rollout이 있어야 검증 가능 — 현재 infra로는 시도 금지.
- **차별화 필요**: (a) 다른 physics를 가진 cross-embodiment(예: walker vs cheetah)에서 embedding sensitivity 비교 — embodiment가 다르면 dynamics가 진짜 달라지므로 tautology 제거. (b) reward head underuse 가설: expert policy로 rollout 데이터를 생성한 후 task-matched vs task-mismatched reward prediction 비교. 단 (a)(b) 모두 현재 mt30 checkpoint + rollout infra로 접근하기 어려움.
