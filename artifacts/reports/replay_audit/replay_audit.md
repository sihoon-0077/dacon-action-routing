# Replay Lookup Audit

## session_scoped

- train internal coverage: `60548/70000`
- train internal precision: `1.000000`
- GroupSplit train-to-valid coverage: `0/14106`
- GroupSplit train-to-valid precision: `0.000000`
- transductive valid coverage: `12215/14106`
- transductive valid precision: `1.000000`
- placeholder test hits: `5/5`

## global_prompt

- train internal coverage: `55818/70000`
- train internal precision: `0.995432`
- GroupSplit train-to-valid coverage: `1065/14106`
- GroupSplit train-to-valid precision: `0.432864`
- transductive valid coverage: `11268/14106`
- transductive valid precision: `0.995474`
- placeholder test hits: `5/5`

## Recommendation

- Keep replay variants separated from safe submissions.
- Session-scoped test self-history replay is diagnostic/transductive and should only be used after rule confirmation.