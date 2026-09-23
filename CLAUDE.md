# Bridge — Repository Conventions

This file is loaded by the `code-review` plugin's compliance agents. Each rule is a hard requirement — flag any PR that violates it.

## Modal patterns

- **Async context**: use `.aio()` variants (`await modal.Function.remote.aio(...)`, `await modal.Sandbox.create.aio(...)`) whenever the caller is async. Never mix sync `.remote()` and `.spawn()` inside an `async def`.
- **Image reproducibility**: `modal.Image.debian_slim()` / `.from_registry()` chains must pin package versions (`pip_install("pandas==2.2.0")`, not `pip_install("pandas")`). Unpinned installs are a review-blocker.
- **Secrets**: never hardcode API keys, tokens, or credentials in source. Use `modal.Secret.from_name("name")` and reference via `secrets=[...]` on the function decorator.
- **Timeouts**: every `@app.function()` that does network I/O, GPU work, or unbounded compute must set an explicit `timeout=` argument. Runaway functions burn credits.
- **GPU sizing**: only request GPU on functions that need it. Specify type (`gpu="A10G"`, `gpu="H100"`) — do not use unadorned `gpu=True`.
- **App naming**: `modal.App("stable-name")` — no random strings, no environment concatenation. Stable names keep deployments observable.
- **Cold-start awareness**: heavy imports at module scope inflate cold-start time. Prefer lazy imports inside function bodies for anything not needed at deploy time.
- **Batching**: functions handling high-throughput per-item workloads should use `@modal.batched(max_batch_size=..., wait_ms=...)` rather than looping externally.

## Python style

- **Type hints**: every function signature (params + return) must have annotations. `def f(x)` is a review-blocker. Use `from __future__ import annotations` only if targeting <3.10; otherwise native syntax.
- **No wildcard imports**: `from foo import *` is banned. Enumerate names.
- **No mutable default arguments**: `def f(x=[])` is a bug. Use `None` and coerce inside.
- **Logging over print**: production code uses `logging`, not `print`. `print` is allowed only in `@app.local_entrypoint()` or scripts.
- **f-strings**: prefer f-strings over `.format()` or `%` formatting.
- **Path handling**: use `pathlib.Path`, not string concatenation.

## Repo conventions

- **Data files**: anything under `data/` is Git LFS-tracked (see `.gitattributes`). Do not commit raw large files outside `data/`.
- **Ignored regenerable artifacts**: `.agents/` (Modal skill install), `.venv/`, `__pycache__/`, `.idea/` — never stage these.
- **Dep management**: `uv` only. Never edit `uv.lock` by hand. New deps go through `uv add` (runtime) or `uv add --group dev` (dev).
- **CI is authoritative**: `ruff check`, `ruff format --check`, and `pyright` must pass. Do not disable rules to silence warnings — fix the code or justify with an inline `# noqa: <code>` comment explaining why.
- **No AI attribution.** Never add `Co-Authored-By: Claude ...`, `Generated with Claude Code`, or any equivalent to a commit message or a pull request description. The history and the PR record are the team's own. Enforced by `.github/workflows/attribution.yml`; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Review focus

When reviewing a PR, prioritize in this order:
1. Correctness bugs introduced by the diff (not pre-existing)
2. Modal-specific pitfalls from the list above
3. Missing/wrong type hints
4. Secret or credential leaks
5. Cost implications (missing timeouts, unnecessary GPUs, unbounded parallelism)

Do not flag: style nits already caught by ruff, pre-existing issues, or hypothetical future-proofing concerns.
