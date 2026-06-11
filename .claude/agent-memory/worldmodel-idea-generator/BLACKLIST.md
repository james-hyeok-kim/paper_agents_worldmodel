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

<!-- FAIL 판정 시 worldmodel-idea-validator가 아래에 추가 -->
