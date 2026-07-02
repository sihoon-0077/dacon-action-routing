# Phase 0 Bugfix Report

## Verdict

- Primary root cause found earlier: `[NOW] current_prompt` was placed at the tail in the legacy serializer, so long samples could lose the target prompt under head-only truncation.
- v3 mitigation: `serialize_policy_v3` exposes `[META]`, `[OPEN]`, `[FLAG]`, and `[NOW]` explicitly; training code must preserve `[NOW]` during truncation.

## P0-1 Label Order

- transformer script label order matches `ACTIONS`: `True`
- transformer classes: `['read_file', 'grep_search', 'list_directory', 'glob_pattern', 'edit_file', 'write_file', 'apply_patch', 'run_bash', 'run_tests', 'lint_or_typecheck', 'ask_user', 'plan_task', 'web_search', 'respond_only']`

## P0-2 Loss Curve

- parsed epochs: `3`
- train losses: `[1.161018, 0.777006, 0.703958]`
- loss decreases: `True`

## P0-3 Serialization Samples

Legacy serializer puts `[NOW]` at the end:

```text
[META] tier=enterprise lang=en ci=passed dirty=True turn=TURN_7_10 budget=BUDGET_HIGH
[OPEN] src/schemas/types.py pyproject.toml
[MIX] py:0.82 yaml:0.10 dockerfile:0.05
[H1] U: schemas cmd should print the go runtime schemas too. show me the current impl, cheers >> A: read_file path=pyproject.toml => ok; read pyproject.toml (173L)
[H2] U: quick one — hmm fernet key is blank. list what's in plugins real quick >> A: grep_search pattern=Config => 14 matches in 5 files
[H3] U: drop in a REST_FRAMEWORK throttle config, scoped rate for anon vs user >> A: apply_patch n_files=2 => ok; patched 2 files (115+/10-)
[H4] U: base image is fine, just cuda. rebuild if possible >> A: run_tests target=all => PASS: 53 tests passed
[H5] U: let me actually confirm the render count dropped, fire up the app and watch the perf monitor if you can >> A: run_tests target=all => PASS: 66 tests passed
[H6] U: when you're free, also HomeScreen probably needs the dependency array tightened on that effect. touch both in one go >> A: apply_patch n_files=6 => ok; patched 6 files (16+/3-)
[NOW] bundle's fine. now run the profile tests to make sure i didn't break the happy path!
```

Fixed now-first serializer puts `[NOW]` first:

```text
[NOW] bundle's fine. now run the profile tests to make sure i didn't break the happy path!
[META] tier=enterprise lang=en ci=passed dirty=True turn=TURN_7_10 budget=BUDGET_HIGH
[OPEN] src/schemas/types.py pyproject.toml
[MIX] py:0.82 yaml:0.10 dockerfile:0.05
[H1] U: when you're free, also HomeScreen probably needs the dependency array tightened on that effect. touch both in one go >> A: apply_patch n_files=6 => ok; patched 6 files (16+/3-)
[H2] U: let me actually confirm the render count dropped, fire up the app and watch the perf monitor if you can >> A: run_tests target=all => PASS: 66 tests passed
[H3] U: base image is fine, just cuda. rebuild if possible >> A: run_tests target=all => PASS: 53 tests passed
[H4] U: drop in a REST_FRAMEWORK throttle config, scoped rate for anon vs user >> A: apply_patch n_files=2 => ok; patched 2 files (115+/10-)
[H5] U: quick one — hmm fernet key is blank. list what's in plugins real quick >> A: grep_search pattern=Config => 14 matches in 5 files
[H6] U: schemas cmd should print the go runtime schemas too. show me the current impl, cheers >> A: read_file path=pyproject.toml => ok; read pyproject.toml (173L)
```

v3 serializer sample:

```text
[META] tier=enterprise lang=en ci=passed dirty=True turn=8 budget=b3 loc=l2
[OPEN] src/schemas/types.py pyproject.toml
[MIX] py:0.82 yaml:0.10
[FLAG] pf=0 pf_open=0 pf_seen=0
[H1] U: schemas cmd should print the go runtime schemas too. show me the current impl, cheers
[H1] A: read_file(path=pyproject.toml) -> ok; read pyproject.toml (173L)
[H2] U: quick one — hmm fernet key is blank. list what's in plugins real quick
[H2] A: grep_search(pattern=Config) -> 14 matches in 5 files
[H3] U: drop in a REST_FRAMEWORK throttle config, scoped rate for anon vs user
[H3] A: apply_patch(n_files=2) -> ok; patched 2 files (115+/10-)
[H4] U: base image is fine, just cuda. rebuild if possible
[H4] A: run_tests(target=all) -> PASS: 53 tests passed
[H5] U: let me actually confirm the render count dropped, fire up the app and watch the perf monitor if you can
[H5] A: run_tests(target=all) -> PASS: 66 tests passed
[H6] U: when you're free, also HomeScreen probably needs the dependency array tightened on that effect. touch both in one go
[H6] A: apply_patch(n_files=6) -> ok; patched 6 files (16+/3-)
[NOW] bundle's fine. now run the profile tests to make sure i didn't break the happy path!
```
