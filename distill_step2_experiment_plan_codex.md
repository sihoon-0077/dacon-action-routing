# DISTILL STEP 2 EXPERIMENT PLAN — 5-Fold Teacher Logits 기반 Soft Distillation

작성일: 2026-07-04  
대상 대회: DACON AI Agent Action Routing 14-class Macro-F1  
상태: mDeBERTa 5-fold teacher 학습 완료 가정  
목표: 느린 transformer의 판단 신호를 빠른 student 모델에 압축하여, 제출 시간 제한을 피하면서 local/public 성능을 끌어올린다.

---

## 0. 현재 상황 요약

현재까지 가장 중요한 사실은 다음이다.

```text
advanced_router:
  빠르고 안정적인 base.
  local GroupShuffle/Fold 기준 약 0.7113.

mDeBERTa teacher:
  fold0/fold1에서 약 0.716~0.718까지 확인.
  다만 전체 hidden test에 긴 max_len으로 직접 돌리면 TLE 위험.

v3_spm 제출:
  public 약 0.710.
  transformer path는 서버에서 실제로 탔지만 runtime이 길다.

N4 candidate gating:
  20k candidate public이 0.7127까지 올라간 신호가 있음.
  즉 transformer signal은 유효하지만, 어느 row에 얼마만큼 쓸지와 runtime이 병목.

XLM-R:
  token audit은 통과했으나 fold0 0.697로 gate fail.

N2/N3 standalone specialists:
  inspect/communicate standalone cheap specialist는 reject.

Distillation:
  아직 제대로 실행하지 않음.
  이번 문서의 목적은 Step 2, 즉 soft-label MLP/fast student distillation을 시작하는 것.
```

---

## 1. Distillation이 필요한 이유

현재 직접 transformer inference 구조는 아래 문제가 있다.

```text
1. mDeBERTa는 일부 action boundary에서 advanced_router보다 강하다.
2. 하지만 mDeBERTa를 전체 test row에 돌리면 10분 제한에 걸릴 수 있다.
3. candidate를 8k/12k/20k로 제한하면 runtime은 줄지만 coverage가 줄어 성능 상한도 낮아진다.
4. 따라서 transformer의 decision boundary를 빠른 모델에 압축해야 한다.
```

Distillation의 목표는 다음이다.

```text
teacher = mDeBERTa 5-fold OOF logits/probs
student = 빠른 모델, 예: TF-IDF/SVD + MLP 또는 LogReg/SGD

학습 시:
  student가 true label과 teacher probability distribution을 동시에 배움.

추론 시:
  mDeBERTa는 돌리지 않음.
  student만 빠르게 전체 test row에 실행.
```

이렇게 하면 다음을 기대할 수 있다.

```text
- transformer signal을 전체 row에 적용 가능
- 제출 runtime 1~3분대 가능
- advanced_router와 blend 가능
- N4 candidate gating의 runtime 병목 완화
```

---

## 2. 핵심 원칙

### 2.1 Teacher probability는 target이지 feature가 아니다

Teacher logits/probs는 학습 target으로만 사용한다.
Inference 때 teacher는 없으므로, student 입력 feature에 teacher probability를 넣으면 안 된다.

```text
허용:
  loss = CE(true_label) + KL(student_probs, teacher_probs)

금지:
  X feature에 teacher_probs를 concat
```

### 2.2 OOF teacher를 사용해 검증 누수를 막는다

5-fold teacher가 있다면, 각 sample에 대해 해당 sample을 학습에 사용하지 않은 teacher fold의 logits를 OOF teacher target으로 사용한다.

```text
좋은 방식:
  sample i가 fold3 valid에 있었다면,
  fold3 teacher가 예측한 logits를 sample i의 teacher target으로 사용.

나쁜 방식:
  full train teacher가 train 전체를 다시 예측한 logits를 validation 평가에 사용.
```

### 2.3 Student 검증도 fold-safe로 해야 한다

Teacher가 OOF라고 해도, student를 train 전체로 학습하고 train 전체에서 평가하면 안 된다.
Student도 OOF 평가를 한다.

```text
for student_fold in 0..4:
  train student on 4 folds
  evaluate on heldout fold
```

### 2.4 Advanced router는 feature 또는 blend로 사용 가능

Advanced router prediction/probability/score는 inference 때도 계산 가능하므로 student 입력 또는 final blend에 사용할 수 있다.

가능한 사용 방식:

```text
A. Student input feature:
   advanced_pred one-hot
   advanced_group one-hot
   advanced_margin
   advanced 14 score/proba

B. Final blend:
   final_probs = w_s * student_probs + w_a * advanced_probs
```

단, class order alignment를 반드시 확인한다.

### 2.5 Soft distillation은 hard CE와 같이 쓴다

Teacher가 틀릴 수 있으므로 teacher만 따라가면 안 된다.

기본 loss:

```text
loss = lambda_hard * CrossEntropy(y_true)
     + lambda_soft * T^2 * KL(student_logits/T, teacher_probs/T)
```

추천 시작값:

```text
lambda_hard = 0.65
lambda_soft = 0.35
T = 2.0
```

---

## 3. 필요한 입력 산출물

Codex는 먼저 아래 파일의 존재 여부를 확인한다.

```text
필수:
  data/train.jsonl
  data/train_labels.csv
  folds/fold_assignments.csv 또는 pipeline_v4 fold assignments
  teacher OOF logits/probs for 5 folds
  label order / ALL_CLASSES

권장:
  advanced_router.pkl
  advanced_router OOF predictions/scores/probs
  latest serializer v2 or v2.2
  token/audit utilities
```

### 3.1 Teacher logits 파일 규약

가능하면 아래 형태로 정리한다.

```text
artifacts/teacher_oof/
├── teacher_oof_logits.npy        # shape = (70000, 14)
├── teacher_oof_probs.npy         # shape = (70000, 14), optional
├── teacher_oof_pred.npy          # shape = (70000,)
├── teacher_oof_meta.json         # fold, model, max_len, serializer, class_order
└── class_order.json              # ALL_CLASSES order
```

`teacher_oof_probs.npy`가 없으면 logits에서 softmax로 만든다.

```python
teacher_probs = softmax(teacher_logits / teacher_temperature)
```

초기에는 `teacher_temperature = 1.0`으로 두고, 이후 temperature sweep을 한다.

### 3.2 Advanced router score 파일 규약

가능하면 아래 형태로 정리한다.

```text
artifacts/advanced_oof/
├── advanced_oof_probs.npy        # shape = (70000, 14)
├── advanced_oof_pred.npy
├── advanced_oof_margin.npy
├── advanced_oof_top2.npy
└── class_order.json
```

advanced가 완전한 proba를 주기 어렵다면 score를 softmax-like로 변환해도 된다.
단, class order를 반드시 `ALL_CLASSES`에 맞춘다.

---

## 4. Step 2 Distillation 전체 실험 구조

이번 실험은 5단계로 나눈다.

```text
D2-0. Asset audit
D2-1. Student feature cache 생성
D2-2. Fast baseline students
D2-3. MLP soft distillation
D2-4. Advanced blend / class bias / calibration
D2-5. Full train + submit package
```

---

# D2-0. Asset Audit

## 목적

Teacher logits/probs와 fold assignment가 정상인지 확인한다.

## 체크리스트

```text
1. teacher_oof_logits shape == (70000, 14)
2. class order == ALL_CLASSES
3. no NaN / no inf
4. per-class teacher argmax 분포가 이상하지 않은지
5. teacher OOF Macro-F1 계산
6. teacher OOF NLL 계산
7. teacher confidence 분포 확인
8. teacher vs true class-wise F1 확인
9. teacher와 advanced의 win/loss 분석
```

## 산출물

```text
reports/distill_step2/d2_0_asset_audit/
├── teacher_metrics.json
├── teacher_classwise_f1.csv
├── teacher_confidence_hist.csv
├── teacher_vs_advanced_winloss.csv
└── summary.md
```

## 통과 기준

```text
teacher OOF Macro-F1 >= 0.710
teacher logits/probs 정상
class order mismatch 없음
NaN 없음
```

만약 teacher OOF가 0.70 아래라면 distillation 전 teacher asset부터 재점검한다.

---

# D2-1. Student Feature Cache

## 목적

Student가 빠르게 학습/추론할 수 있는 feature를 만든다.

## 1. Text feature

기본 serializer는 `v2.2`를 우선한다.
만약 v2.2가 구현되지 않았으면 현재 best v2 serializer를 사용한다.

권장 text layout:

```text
[NOW] {current_prompt}
[LAST] action={last_action} args={compact_args} result_bucket={result_buckets} result={short_result_summary}
[STATE] test={...} lint={...} edits_after_test={...} edits_after_lint={...} insp_streak={...} last_mod_ext={...} open_cnt={...}
[SEQ] actions={a1 > a2 > ...}
[META] tier={...} lang={...} ci={...} dirty={...} turn={...} budget={...} loc={...} len={...}
[OPEN] {open_files compact}
[HIST] recent compact user/action/result pairs
```

주의:

```text
- 일반 NLP 클리닝 금지
- 파일명/path/ext 보존
- PASS/FAIL/0 matches/files matched 보존
- [NOW], [LAST], [STATE] 우선 보존
```

## 2. Vectorization

첫 번째 student는 dense MLP이므로 sparse TF-IDF를 바로 넣지 않고 SVD로 압축한다.

추천 vectorizer:

```python
FeatureUnion([
  ('word', TfidfVectorizer(
      analyzer='word',
      ngram_range=(1, 2),
      min_df=2,
      max_features=180_000,
      sublinear_tf=True,
      lowercase=False,
  )),
  ('char', TfidfVectorizer(
      analyzer='char_wb',
      ngram_range=(3, 5),
      min_df=2,
      max_features=120_000,
      sublinear_tf=True,
      lowercase=False,
  )),
])
```

SVD:

```text
n_components 후보:
  512
  768
  1024

초기값:
  768
```

주의:
- 70k × 300k sparse는 가능하지만 SVD 메모리 주의.
- RAM 12GB 제출 환경도 고려해야 하므로 final artifact는 너무 크면 안 된다.
- training은 로컬에서 가능하지만 submit artifact는 vectorizer/SVD/student만 포함한다.

## 3. Numeric/categorical feature

다음 feature를 dense로 append한다.

```text
session_meta:
  user_tier one-hot
  language_pref one-hot
  last_ci_status one-hot
  git_dirty
  turn_bucket
  budget_bucket
  loc_bucket

workflow:
  last_action one-hot
  last2_action pair hash/one-hot
  last_result_bucket one-hot
  test_state one-hot
  lint_state one-hot
  edits_after_test bucket
  edits_after_lint bucket
  insp_streak bucket
  last_mod_ext one-hot
  open_cnt bucket

prompt flags:
  has_file_ext
  has_path
  has_test_word
  has_lint_word
  has_build_word
  has_web_word
  has_summary_word
  has_plan_word
  has_question_mark
  prompt_len_bucket

advanced features:
  advanced_pred one-hot
  advanced_group one-hot
  advanced_margin
  advanced_top2 one-hot/pair id
  advanced 14 probs or scores
```

Teacher probs는 dense feature로 넣지 않는다. Teacher는 target이다.

## 산출물

```text
artifacts/distill_features/
├── texts_v22.jsonl
├── y.npy
├── fold_ids.npy
├── vectorizer.pkl
├── svd.pkl
├── dense_features.npy
├── feature_meta.json
└── class_order.json
```

For OOF training, vectorizer/SVD는 fold별로 train fold에만 fit하는 것이 가장 엄밀하다.  
하지만 시간 절약을 위해 첫 gate에서는 vectorizer/SVD를 full train에 fit하고 student OOF만 fold로 나눌 수 있다.  
단, 최종 보고에는 이 shortcut 여부를 기록한다.

엄밀 모드:

```text
fold별 vectorizer/SVD fit
느리지만 leakage 최소
```

빠른 모드:

```text
full-text vectorizer/SVD fit
빠르지만 unsupervised leakage 가능성 약간 있음
대회 텍스트 feature에서는 보통 허용 가능한 수준이나, 평가 보고에 기록
```

추천:

```text
D2-quick gate: fast mode
D2-final validation: strict fold mode if 시간이 허용되면 실행
```

---

# D2-2. Fast Baseline Students

## 목적

MLP 전에 빠른 student baseline을 만든다. 여기서 전혀 안 오르면 MLP soft distillation의 기대값을 낮게 본다.

## D2-2A. Hard-label LogReg student

```text
input = SVD text + dense features + advanced features
model = LogisticRegression 또는 SGDClassifier(log_loss)
target = true label
```

비교:

```text
student alone
advanced_router
student + advanced blend
```

## D2-2B. Teacher-top1 pseudo-label weighted student

Teacher confidence가 높은 row에 teacher top1을 보조 target으로 사용한다.

방법 1. duplicate pseudo rows:

```text
original row:
  target = y_true
  weight = 1.0

pseudo row:
  target = teacher_top1
  weight = beta * teacher_confidence
```

Sweep:

```text
teacher_conf_threshold = 0.55, 0.65, 0.75
beta = 0.2, 0.4, 0.6
```

방법 2. sample weight만 조정:

```text
if teacher_top1 == y_true:
  weight += beta * teacher_confidence
```

## D2-2C. Hybrid imitation student

Teacher를 pure mDeBERTa가 아니라 현재 best hybrid로 둔다.

```text
hybrid_teacher_pred = advanced_router + mDeBERTa stronger-class override
```

Student가 이 final decision을 imitation한다.

```text
target = hybrid_teacher_pred
sample_weight = 0.5 + teacher_confidence
```

## D2-2 평가

모든 fast student에 대해:

```text
Macro-F1
Accuracy
NLL if proba available
class-wise F1
group-wise F1
advanced vs student win/loss
runtime estimate
```

## D2-2 통과 기준

```text
student alone >= advanced_router - 0.005
or
student + advanced blend >= advanced_router + 0.003
or
student + advanced blend >= current public-equivalent local baseline + 0.003
```

통과하지 못하면 MLP soft distill로 가기 전에 feature/teacher asset을 점검한다.

---

# D2-3. MLP Soft Distillation

## 목적

Teacher의 14-class probability distribution을 student가 직접 학습한다.

## 3.1 Student architecture

기본 MLP:

```python
class DistillMLP(nn.Module):
    def __init__(self, input_dim, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

Small variant:

```text
hidden 256 → 128
빠름, overfit 낮음
```

Large variant:

```text
hidden 768 → 384
성능 가능성은 있지만 overfit/시간 증가
```

초기 후보:

```text
M0: 512-256 dropout 0.15/0.10
M1: 256-128 dropout 0.10
M2: 768-384 dropout 0.20/0.10
```

## 3.2 Input feature

```text
X = concat(
  svd_text_features,
  dense_state_features,
  advanced_router_features
)
```

초기 ablation:

```text
A. text_svd only
B. text_svd + dense_state
C. text_svd + dense_state + advanced_features
```

Advanced features를 넣는 C가 기본 후보지만, ablation으로 확인한다.

## 3.3 Loss

Hard CE:

```python
loss_hard = CrossEntropyLoss()(student_logits, y_true)
```

Soft KL:

```python
T = distill_temperature
teacher_soft = softmax(teacher_logits / T)
student_log_soft = log_softmax(student_logits / T)
loss_soft = KLDivLoss(batchmean)(student_log_soft, teacher_soft) * T * T
```

Total:

```python
loss = lambda_hard * loss_hard + lambda_soft * loss_soft
```

Initial config:

```text
lambda_hard = 0.65
lambda_soft = 0.35
T = 2.0
```

Sweep small:

```text
D2-M0:
  lambda_soft=0.20, T=2.0
D2-M1:
  lambda_soft=0.35, T=2.0
D2-M2:
  lambda_soft=0.50, T=2.0
D2-M3:
  lambda_soft=0.35, T=3.0
```

## 3.4 Teacher temperature

Teacher logits may be overconfident. Try teacher softening.

```text
teacher_T = 1.0, 1.5, 2.0
student_T = same as teacher_T for KL
```

Do not over-sweep. Start with T=2.0.

## 3.5 Training setup

```text
optimizer = AdamW
lr = 1e-3, 5e-4
weight_decay = 1e-4
batch_size = 1024 or 2048
epochs = 30 max
early_stop = patience 5 on Macro-F1
seed = 42
```

If GPU is available, MLP training is fast. CPU also feasible.

## 3.6 OOF evaluation

Student OOF is mandatory.

```python
for fold in 0..4:
    train_idx = fold_id != fold
    val_idx = fold_id == fold
    train student on train_idx
    evaluate on val_idx
    save logits/probs
```

Teacher target for train_idx uses OOF teacher probs for those rows. This is conservative and avoids teacher-overfit leakage.

## 3.7 Metrics

For every MLP config:

```text
student_argmax_macro_f1
student_argmax_accuracy
student_nll
student_classwise_f1
student_groupwise_f1
advanced_blend_macro_f1
student_vs_advanced_winloss
student_runtime_estimate
```

Blend sweep:

```text
p_final = w_s * p_student + w_a * p_advanced
w_s ∈ {0.3, 0.5, 0.7, 0.9, 1.0}
w_a = 1 - w_s
```

Also try logit-space blend if advanced scores are logits:

```text
logit_final = w_s * logit_student + w_a * logit_advanced
```

But probability blend is safer first.

---

# D2-4. Calibration, Bias, and Group-local Decision

## 4.1 Temperature scaling for student

Fit temperature on student OOF logits.

```text
T_student bounds: 0.5~5.0
objective: NLL
```

Evaluate:

```text
argmax before T
argmax after T
blend before T
blend after T
```

Temperature may not change argmax alone, but improves probability blend.

## 4.2 Class bias

Macro-F1 is not optimized by argmax. Try conservative class bias on final blended probabilities.

Decision:

```python
pred = argmax(log(p_final + eps) + class_bias)
```

Bias search:

```text
range = [-0.5, +0.5]
step = 0.05
L2 penalty = 0.01
sweeps = 2
```

Strict validation:

```text
split OOF rows into halfA/halfB by session.
fit bias on A, evaluate on B.
fit bias on B, evaluate on A.
adopt only if both directions non-negative and average >= +0.001.
```

Previous class threshold/bias overfit happened, so strict rule is mandatory.

## 4.3 Group-local bias

If global bias fails, try group-local bias.

```text
inspect bias within read/grep/list/glob
execute bias within bash/tests/lint
communicate bias within ask/plan/web/respond
```

Adopt criteria stricter:

```text
average half-split gain >= +0.0015
no group F1 drop > 0.003
```

## 4.4 Weak-class recall bias

Try tiny positive bias for weak classes only.

Candidate weak classes:

```text
list_directory
glob_pattern
lint_or_typecheck
web_search
ask_user
write_file
```

Bias range:

```text
0.05, 0.10, 0.15, 0.20
```

This is risky. Adopt only with half-split stability.

---

# D2-5. Final Student Training and Submit Package

## 5.1 Choosing final config

Pick final config by this priority:

```text
1. OOF Macro-F1 of student + advanced blend
2. Half-split stability after bias
3. Runtime estimate
4. Class-wise no-collapse
5. Simplicity of submit package
```

Minimum to proceed to submit:

```text
OOF final >= advanced_router + 0.005
or
OOF final >= current local hybrid - 0.002 while runtime is much faster
```

Ideal:

```text
OOF final >= 0.720
runtime <= 2 minutes
```

## 5.2 Final training strategy

Train final student on all 70k.

Teacher target options:

### Option A. OOF teacher targets for all train rows

```text
Use teacher_oof_probs as soft target.
This is conservative and avoids teacher overfit.
```

### Option B. Full teacher probs if available

```text
If full mDeBERTa teacher has generated logits for all 70k,
try training a second final student with full teacher probs.
```

Do not use Option B for validation selection. Only for final package after Option A is validated.

## 5.3 Submit package structure

```text
submit_distill_student_v1.zip
├── model/
│   ├── student.pt
│   ├── vectorizer.pkl
│   ├── svd.pkl
│   ├── dense_feature_encoder.pkl
│   ├── advanced_router.pkl             # optional but recommended
│   ├── class_bias.json                 # optional
│   ├── temperature.json                # optional
│   ├── config.json
│   └── class_order.json
├── script.py
└── requirements.txt
```

Requirements:

```text
# Prefer empty or minimal.
# torch, sklearn, numpy, pandas, joblib are already installed in the competition environment.
```

## 5.4 Inference logic

```python
samples = load_jsonl('./data/test.jsonl')
texts = [serialize_v22(s) for s in samples]
X_text = vectorizer.transform(texts)
X_svd = svd.transform(X_text)
X_dense = build_dense_features(samples)

if advanced_router is available:
    adv_probs, adv_pred, adv_margin = advanced_predict(samples)
    X = concat([X_svd, X_dense, adv_features])
else:
    X = concat([X_svd, X_dense])

student_logits = student(X)
student_probs = softmax(student_logits / T_student)

if advanced_probs available:
    probs = w_s * student_probs + w_a * advanced_probs
else:
    probs = student_probs

pred = argmax(log(probs) + class_bias)
write submission.csv
```

## 5.5 Runtime target

```text
Target total runtime: <= 2 minutes
Hard max: <= 4 minutes
```

This should be much faster than mDeBERTa candidate inference.

---

## 6. Experiment Matrix

### Stage 1: Fast gate

| ID | Feature | Student | Teacher target | Expected time | Pass |
|---|---|---|---|---:|---|
| D2-G1 | SVD768 + dense | LogReg/SGD | hard label | 30m | baseline |
| D2-G2 | SVD768 + dense + adv | LogReg/SGD | hard label | 30m | adv feature value |
| D2-G3 | SVD768 + dense + adv | LogReg/SGD | teacher top1 pseudo | 1h | +0.003 |
| D2-G4 | SVD768 + dense + adv | LogReg/SGD | hybrid imitation | 1h | +0.003 |

### Stage 2: MLP soft distill

| ID | SVD | Features | lambda_soft | T | Hidden | Pass |
|---|---:|---|---:|---:|---|---|
| D2-M1 | 768 | text+dense+adv | 0.20 | 2.0 | 512-256 | OOF gain |
| D2-M2 | 768 | text+dense+adv | 0.35 | 2.0 | 512-256 | main |
| D2-M3 | 768 | text+dense+adv | 0.50 | 2.0 | 512-256 | compare |
| D2-M4 | 768 | text+dense+adv | 0.35 | 3.0 | 512-256 | soft teacher |
| D2-M5 | 512 | text+dense+adv | 0.35 | 2.0 | 256-128 | speed/regularize |
| D2-M6 | 1024 | text+dense+adv | 0.35 | 2.0 | 512-256 | capacity |

### Stage 3: Blend/bias

| ID | Input | Sweep | Pass |
|---|---|---|---|
| D2-B1 | best student + advanced | w_s 0.3~1.0 | +0.003 |
| D2-B2 | best blend + T | T 0.5~5 | NLL improve |
| D2-B3 | best blend + class bias | half split | stable |
| D2-B4 | best blend + weak-class bias | tiny bias | stable |

---

## 7. Detailed Scripts to Implement

### 7.1 `scripts/distill_prepare_teacher.py`

Purpose:

```text
Load 5-fold mDeBERTa logits, align class order, assemble OOF teacher probs.
```

Outputs:

```text
artifacts/teacher_oof/teacher_oof_logits.npy
artifacts/teacher_oof/teacher_oof_probs.npy
artifacts/teacher_oof/teacher_oof_pred.npy
reports/distill_step2/d2_0_asset_audit/summary.md
```

Must check:

```text
shape
NaN
class order
teacher OOF Macro-F1
teacher class-wise F1
```

---

### 7.2 `scripts/distill_build_features.py`

Purpose:

```text
Build serialized text, TF-IDF/SVD features, dense state features, advanced features.
```

Outputs:

```text
artifacts/distill_features/vectorizer.pkl
artifacts/distill_features/svd.pkl
artifacts/distill_features/X_svd.npy
artifacts/distill_features/X_dense.npy
artifacts/distill_features/advanced_features.npy
artifacts/distill_features/y.npy
artifacts/distill_features/fold_ids.npy
```

Options:

```text
--serializer v2|v22
--svd-dim 512|768|1024
--strict-fold-vectorizer true|false
```

---

### 7.3 `scripts/distill_train_fast_students.py`

Purpose:

```text
Run D2-G1~G4 fast LogReg/SGD gate.
```

Outputs:

```text
reports/distill_step2/fast_students/results.csv
reports/distill_step2/fast_students/classwise.csv
reports/distill_step2/fast_students/summary.md
```

---

### 7.4 `scripts/distill_train_mlp_oof.py`

Purpose:

```text
Train MLP student OOF for configs D2-M1~M6.
```

Outputs:

```text
artifacts/distill_oof/{config}/oof_logits.npy
artifacts/distill_oof/{config}/oof_probs.npy
reports/distill_step2/mlp_oof/{config}/metrics.json
reports/distill_step2/mlp_oof/{config}/classwise.csv
```

---

### 7.5 `scripts/distill_eval_blends.py`

Purpose:

```text
Blend student probs with advanced probs, fit temperature/bias, evaluate strict half split.
```

Outputs:

```text
reports/distill_step2/blends/blend_results.csv
reports/distill_step2/blends/bias_results.csv
reports/distill_step2/blends/best_config.json
```

---

### 7.6 `scripts/distill_train_full_student.py`

Purpose:

```text
Train final student on all 70k using selected config.
```

Outputs:

```text
model/distill_student/student.pt
model/distill_student/vectorizer.pkl
model/distill_student/svd.pkl
model/distill_student/dense_feature_encoder.pkl
model/distill_student/config.json
```

---

### 7.7 `scripts/build_submit_distill.py`

Purpose:

```text
Build submit_distill_student_v1.zip.
```

Must include:

```text
script.py
requirements.txt
model/distill_student/*
optional model/advanced_router.pkl
```

Smoke:

```text
unzip to temp
python script.py
check output/submission.csv
estimate runtime on 1000/5000 rows
```

---

## 8. Required Reports

Final distillation report:

```text
reports/distill_step2/SUMMARY.md
```

Must include:

```text
1. Teacher OOF metrics
2. Fast student results
3. MLP student OOF results
4. Advanced blend results
5. Calibration/bias stability
6. Class-wise F1 before/after
7. Group-wise F1 before/after
8. Runtime estimate
9. Submit recommendation
10. Reject/adopt decision
```

Required tables:

```text
teacher_vs_student_classwise.csv
student_vs_advanced_winloss.csv
groupwise_f1_delta.csv
blend_sweep.csv
strict_bias_report.csv
runtime_report.csv
```

---

## 9. Success / Failure Decision Tree

### Case A. Student alone is strong

```text
student OOF >= advanced_router + 0.005
```

Decision:

```text
student becomes new base.
Blend with advanced only if improves.
Build submit immediately.
```

### Case B. Student alone is weaker but blend improves

```text
student alone < advanced
advanced + student >= advanced + 0.005
```

Decision:

```text
Use probability blend.
Build submit with advanced_router included.
```

### Case C. Student does not improve advanced but improves weak classes

```text
overall flat
weak class F1 improves
```

Decision:

```text
Use group/action-specific blend or bias.
Do not submit until overall positive.
```

### Case D. Student is weak and blend does not improve

```text
student + advanced <= advanced + 0.001
```

Decision:

```text
Reject distillation v1.
Try different teacher target only if time remains.
Otherwise focus on N4/runtime and execute resolver.
```

---

## 10. Common Failure Modes

### 10.1 Teacher class order mismatch

Symptom:

```text
student collapses or class-wise F1 bizarre.
```

Check:

```text
teacher_oof_pred from logits must reproduce teacher metrics.
```

### 10.2 Teacher leakage in validation

Symptom:

```text
OOF score too high, public low.
```

Fix:

```text
Use OOF teacher logits for validation.
Student OOF training required.
```

### 10.3 Student includes teacher probs as feature

Symptom:

```text
Validation high, inference impossible or broken.
```

Fix:

```text
Teacher probs are target only.
```

### 10.4 SVD/vectorizer leakage overstated

Unsupervised vectorizer on full text is usually acceptable for quick gate, but if results are marginal, rerun strict fold vectorizer.

### 10.5 Advanced proba class alignment

Existing lessons show proba columns must align to `ALL_CLASSES`.
Always verify.

### 10.6 Overfitting class bias

Previous class threshold/bias overfit. Use strict half-split adoption only.

---

## 11. Initial Codex Execution Prompt

```text
We have completed 5-fold mDeBERTa teacher training. Start Distill Step 2.

Goal:
  Train a fast student that absorbs mDeBERTa teacher probabilities and runs without transformer inference at submit time.

Do not run new transformer training.
Do not use teacher probabilities as inference features.
Do not use test data or transductive lookup.

Step 1: Teacher audit
  - Assemble teacher_oof_logits/probs for all 70k rows.
  - Verify class order, no NaNs, teacher OOF Macro-F1, class-wise F1.

Step 2: Feature cache
  - Build serializer v2.2 texts if available; otherwise current best serializer.
  - Build TF-IDF word+char features.
  - Apply TruncatedSVD dims 768 first.
  - Build dense workflow/meta/flag features.
  - Build advanced_router features if advanced_router artifact is available.

Step 3: Fast gate
  - Train LogReg/SGD hard-label student.
  - Train teacher-top1 pseudo-label weighted student.
  - Evaluate OOF against advanced_router.

Step 4: MLP soft distill
  - Train OOF MLP with loss = 0.65 CE + 0.35 KL, T=2.0.
  - Also test lambda_soft 0.2 and 0.5.
  - Save OOF logits/probs.

Step 5: Blend and bias
  - Blend student_probs with advanced_probs.
  - Sweep student weight 0.3, 0.5, 0.7, 0.9, 1.0.
  - Fit temperature and strict half-split class bias.

Step 6: Decision
  - If best OOF >= advanced_router + 0.005, train full student and build submit zip.
  - If not, write reject reason and do not submit.

Outputs:
  reports/distill_step2/SUMMARY.md
  reports/distill_step2/*csv
  model/distill_student/* if adopted
  submit_distill_student_v1.zip if adopted
```

---

## 12. Expected Runtime

Approximate local runtime:

```text
Teacher audit:
  10~30 min

Feature cache:
  30~90 min depending on TF-IDF/SVD

Fast students:
  30~60 min

MLP OOF:
  1~3 h

Blend/bias:
  20~60 min

Full student + submit:
  30~90 min
```

Expected total:

```text
quick gate:
  2~4 h

usable MLP soft distill:
  4~8 h

full submit package:
  within 1 day
```

---

## 13. Expected Score Range

Conservative expectation:

```text
advanced_router local 0.7113
student/advanced blend +0.003~0.006
```

Good outcome:

```text
local 0.718~0.724
public 0.713~0.720
```

Great outcome:

```text
student absorbs most transformer lift
local 0.725+
public 0.72+
```

Failure:

```text
student <= advanced_router +0.001
or class-wise collapse
```

Distillation alone is not expected to jump directly to 0.79, but it is one of the few remaining paths that can improve both score and runtime simultaneously.

---

## 14. Final Recommendation

Run Distill Step 2 now because:

```text
1. 5-fold teacher is already available.
2. Transformer direct inference is TLE-prone.
3. Candidate gating helps but coverage-limited.
4. Small micro-rules only add ~0.001~0.003 each.
5. A fast student can apply teacher signal to all rows.
```

Do not spend more GPU on XLM-R or broad specialists before this distillation gate.

Primary success target:

```text
advanced + distill_student blend OOF >= advanced_router + 0.005
```

If achieved, build submit immediately.
