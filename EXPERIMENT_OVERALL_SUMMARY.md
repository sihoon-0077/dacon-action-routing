# DACON Action Routing 전체 실험 요약

작성일: 2026-07-03  
목적: 지금까지 수행한 실험의 성공/실패, 제출 결과, 병목, 다음 방향을 팀 공유용으로 정리한다.

## 1. 한 줄 결론

이 문제는 단순 `current_prompt` intent classification이 아니라 `current_prompt + history + last action/result + workspace state`로 다음 action policy를 복원하는 문제다.

현재까지 가장 믿을 만한 구조는 다음이다.

```text
advanced_router 같은 빠른 선형 router를 base로 사용
+ transformer는 전체 대체가 아니라 특정 action/class specialist로 사용
+ 제출에서는 runtime/TLE 제약 때문에 transformer inference를 매우 조심해서 써야 함
```

현재 확인된 제출 기준 최고점은 `submit_policy_v3_spm.zip`의 `0.7099979659`이다.  
로컬 검증 기준 최고 의사결정은 `advanced_router + transformer stronger-class override`의 약 `0.7217`이다.

## 2. 현재 최고 성과

| 구분 | 점수 | 실험/제출 | 판단 |
|---|---:|---|---|
| 기본 baseline | `0.4369` | current_prompt TF-IDF + LogReg | 시작점 |
| 선형 feature 계열 | `0.6664` | compact flags + transition/rule scores | 성공 |
| advanced router | `0.7113` | coarse/fine/group specialists + pair resolver | 강한 기준선 |
| transformer hybrid local | `0.7217` | advanced + calibrated transformer override | 로컬 최고권 |
| mDeBERTa v2 fold0 | `0.7176` | `v2bundle_512_5e/fold_0` | transformer specialist 후보 |
| mDeBERTa v2 fold1 | `0.7165` | `v2bundle_512_5e/fold_1` | fold 안정성 확인 |
| mDeBERTa 384 fold0 | `0.7178` | `mdeberta384_v2_384_5e/fold_0` | strong full-train gate 통과 |
| XLM-R fold0 | `0.6970` | `xlmr_state_v1_512` | gate fail, reject |
| public/hidden 확인 최고 | `0.7099979659` | `submit_policy_v3_spm.zip` | 현재 제출 최고 |

## 3. 성공한 것

### 3.1 History 중심 접근

성공:
- `current_prompt`만 보는 baseline은 `0.4369` 수준.
- 최근 `assistant_action.name`, user history, result_summary, args, workspace state를 넣으면서 `0.66+`까지 상승.
- 이 문제는 prompt intent보다 agent workflow transition 성격이 강하다는 판단이 맞았다.

결정:
- 모든 후속 모델은 history/action sequence를 핵심 feature로 둔다.

### 3.2 Compact linear router

성공:
- `compact_flags_lr_combo_a1_0.06_gw_0.08_rw_0.02`: Macro-F1 `0.666351`.
- sparse TF-IDF + compact state + 작은 transition/group/rule score가 강했다.

결정:
- RandomForest류보다 선형 sparse 모델이 이 데이터에 잘 맞는다.

### 3.3 Advanced router

성공:
- GroupShuffle Macro-F1 `0.711324`.
- 구조: compact coarse LinearSVC, group-specific fine LogReg, transition prior, pairwise resolver.
- inspect/action group 구분과 fine specialist가 성능을 올렸다.

결정:
- 현재 가장 안정적인 lightweight base model이다.
- 제출 fallback/base로 계속 사용한다.

### 3.4 Transformer serializer bug fix

성공:
- 초기 transformer가 낮았던 핵심 원인은 `[NOW] current_prompt`가 뒤쪽에 있어 truncation으로 잘리는 문제였다.
- now-first serializer 후 `[NOW]` 누락 `4393/10000 -> 0/10000`.
- mDeBERTa full run이 `0.6868`까지 올라왔고, advanced router와 결합하면 `0.7195~0.7217`까지 상승.

결정:
- transformer 입력은 항상 `[NOW]`를 맨 앞에 두고, `[LAST]`, `[STATE]`, `[SEQ]`를 앞쪽에 둔다.

### 3.5 Workflow-state flags

성공:
- Intuition validation I1: proxy delta `+0.003935`.
- execute target class delta `+0.011375`.
- half split 안정성도 통과.

결정:
- workflow flags는 serializer/state feature로 채택한다.

### 3.6 제출 엔지니어링 개선

성공:
- `token_type_ids` 때문에 v4 transformer override가 skip되는 버그 발견.
- fp16 checkpoint와 CPU/GPU dtype mismatch 버그 발견.
- `config.json` 누락 가능성 보강.
- smoke test에서 `policy_v4_transformer: selected=5/5 changed=1` 확인.

결정:
- 모든 submit zip은 extracted folder에서 offline smoke test를 반드시 돌린다.

## 4. 실패하거나 버린 것

### 4.1 RandomForest / ExtraTrees / dense tree 계열

결과:
- RF/ExtraTrees는 sparse TF-IDF 직접 투입에 부적합.
- SVD/dense로 바꿔도 단독 최고점은 낮았다.
- I3 structural ExtraTrees probe도 `0.317806`으로 실패.

결정:
- direct ensemble member로는 사용하지 않는다.

### 4.2 LightGBM / boosting

결과:
- sparse TF-IDF 직접 투입은 애매했다.
- SVD/embedding/numeric 조합에서는 diversity source 가능성은 있지만 핵심 성능원은 아니었다.

결정:
- GPU/시간 우선순위 낮음.

### 4.3 MiniLM embedding 단독

결과:
- embedding 단독은 약했다.
- TF-IDF/numeric과 concat하면 보조 신호로 약간 도움.

결정:
- 핵심 track은 아님. distillation/student 쪽에서 다시 볼 수 있음.

### 4.4 Exact lookup / replay

결과:
- train-history-only lookup은 검증에서 악화 또는 무효.
- validation/test batch self-history를 쓰면 매우 높지만 transductive/rule-risk가 크다.
- hidden 제출에서는 lookup variant가 점수 상승을 만들지 못했다.

결정:
- 공식 모델에는 넣지 않는다.
- leak diagnostic으로만 유지한다.

### 4.5 Class threshold / learned selector

결과:
- I9 learned selector: static override 대비 `-0.020104`.
- I10 class threshold는 same-val에서는 좋아 보였지만 half split에서 불안정.

결정:
- validation overfit 위험 때문에 submit logic에서 제외.

### 4.6 XLM-R

결과:
- token audit은 성공: `max_len=512`에서 `[NOW]/[LAST]/[STATE]` 100% 보존.
- 그러나 fold0 best Macro-F1은 `0.697038`.
- 사전 gate `0.710/0.720`을 못 넘김.

결정:
- XLM-R state v1은 main track에서 reject.
- fold1/full train은 생략.

## 5. 제출 결과 장부

| 제출 파일 | 결과 | 시간 | 목적 | 결론 |
|---|---:|---:|---|---|
| `submit_01_fixed_stable.zip` | `0.704342388` | `1m47s` | advanced-only 기준선 | baseline |
| `submit_02_fixed_lookup.zip` | `0.704342388` | `1m52s` | lookup 효과 확인 | hidden lift 없음 |
| `submit_policy_v3.zip` | `0.704342388` | `1m49s` | transformer override | 서버에서 transformer skip/fallback 가능성 |
| `submit_policy_v3_spm.zip` | `0.7099979659` | `6m10s` | tokenizer/SPM 호환성 보강 | 현재 제출 최고 |
| `submit_v4_fold0_debug.zip` | 제출 오류 | `10m+` | fold0 transformer 전체 실행 | TLE |
| `submit_v4_fold0_fast.zip` | diagnostic | smoke pass | 256/max 8000 candidate | 성능용보다는 생존 확인용 |
| `submit_v4_fold0_384_12k.zip` | pending | smoke pass | 384/max 12000 candidate | TLE-safe probe |

중요 해석:
- `0.704342388`로 동일하게 나오면 대체로 transformer path가 서버에서 안 돌았거나 실질 변경이 없었다는 뜻이다.
- TLE가 나면 모델 성능 이전에 제출 엔지니어링 실패다.
- transformer는 좋더라도 전체 3만 row에 직접 돌리면 10분 제한에 걸릴 수 있다.

## 6. 주요 버그와 교훈

| 문제 | 증상 | 수정/교훈 |
|---|---|---|
| `predict_proba` class alignment | class score가 wrong label로 이동 | `model.classes_ -> ALL_CLASSES` 정렬 필수 |
| `[NOW]` truncation | transformer가 baseline보다 낮음 | `[NOW]`를 항상 맨 앞에 배치 |
| fp16 dtype 문제 | NaN, unscale error, dtype mismatch | trainable weights는 fp32, autocast로 fp16 |
| tokenizer/server compatibility | 제출 점수 base와 동일 | `spm.model`, local files, config 포함 확인 |
| `token_type_ids` forward mismatch | v4 override skip | custom model forward에 `**kwargs` 허용 |
| CPU/GPU dtype mismatch | transformer override skip | mask/output dtype 맞추고 CPU `model.float()` |
| all-row 512 transformer | TLE | candidate gating, max_len/batch 조정 필요 |

## 7. 현재 병목

### 7.1 모델링 병목

가장 큰 병목은 inspect 4형제다.

```text
read_file
grep_search
list_directory
glob_pattern
```

advanced router에서도 inspect group 내부 fine confusion이 크다. modify 계열은 거의 풀렸고, execute/communicate도 0.70 근처까지는 가지만 inspect가 전체 Macro-F1을 묶고 있다.

### 7.2 제출 병목

transformer가 로컬 validation에서는 도움이 되지만:
- 서버에서 로딩 실패하면 fallback으로 base만 돈다.
- 전체 hidden test에 직접 돌리면 TLE가 난다.
- candidate gating을 너무 줄이면 성능 기여가 약해진다.

즉 다음 핵심은 `성능 좋은 transformer 신호를 10분 안에 쓰는 방법`이다.

## 8. 현재 유지할 자산

유지:
- `advanced_router.pkl`
- `pipeline_v4` fold/serializer/calibration scaffold
- now-first serializer
- workflow-state flags
- mDeBERTa fold0/fold1 logits/checkpoints
- submission smoke infrastructure
- `research.md` hypothesis/submission ledger

버림:
- XLM-R main track
- exact lookup main track
- learned selector
- class threshold overfit table
- structural-only ExtraTrees
- full transformer replacement

## 9. 다음 추천 실험

### N1. Transformer distillation

가설:
- mDeBERTa logits/strong-class decisions를 teacher로 쓰고, fast student model이 그 신호를 흡수하면 제출 runtime을 줄이면서 transformer lift를 유지할 수 있다.

싼 검증:
- OOF/fold logits로 advanced feature + teacher logits를 넣은 linear/LightGBM/student proxy.

통과 기준:
- advanced router `0.7113` 대비 `+0.005` 이상.
- inference는 transformer 없이 2분 이내 예상.

### N2. Inspect specialist

가설:
- 전체 Macro-F1 병목은 inspect group fine confusion이다.

싼 검증:
- inspect-only validation에서 file/path/result/open_files feature 강화.

통과 기준:
- inspect isolated Macro-F1 `+0.008`.
- 전체 Macro-F1 proxy `+0.003`.

### N3. mDeBERTa specialist 개선

현재 상태:
- `max_len=384` fold0이 Macro-F1 `0.7178`로 strong full-train gate를 통과했다.
- 따라서 full 70k 384 학습을 진행하고, ep3/ep5 `cand8000` 제출 후보를 만든다.

검증:
- full 70k `max_len=384`, epoch3/epoch5 checkpoint submit.

통과 기준:
- public `>=0.70999`이면 v3_spm 대비 유효.
- public `>=0.715`이면 384 full track을 메인 후보로 승격.

### N4. 제출 runtime profiling

가설:
- candidate 수와 max_len 사이에 public score/runtime 최적점이 있다.

검증:
- 384/12k, 320/16k, 256/20k 같은 제출 probe.

통과 기준:
- 10분 이내 실행 + public이 `0.70999` 이상.

## 10. 운영 규칙

앞으로 모든 실험은 아래 형식으로 시작한다.

| ID | 가설 | 싼 검증 | 비싼 검증 | 통과 기준 | 결과 | 결정 |
|---|---|---|---|---|---|---|
| Hxx | 실험 전 작성 | Tier A/B | Tier C/D | 실험 전 수치로 고정 | pass/fail | adopt/reject |

규칙:
- public leaderboard는 최종 판단 기준이 아니라 검증 도구다.
- negative result는 반드시 기록한다.
- fold gate를 못 넘기면 full train 금지.
- offline smoke 없는 zip은 제출 금지.
- TLE가 난 transformer는 teacher/specialist로만 쓴다.
