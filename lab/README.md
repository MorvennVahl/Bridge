# Lab notebook

`notebook.jsonl` is append-only. One JSON object per line, two event types:

- `register` — an experiment declared before it ran (hypothesis, approach, label, features)
- `complete` — its outcome (metrics, findings, artifacts, next_steps)

Never edit or delete a line. To correct a result, append a new `complete` entry whose
`supersedes` field names the superseded experiment id.

Read it with `bridge.labnotebook.read()`. Rank with `leaderboard(label=...)`. Check how
burned the validation set is with `validate_evaluations()`.

`TEST_SET_SEAL.txt` does not exist and must be created by a human, not an agent. Its
presence unseals the test set for one single final evaluation. See AGENT.md §3.
