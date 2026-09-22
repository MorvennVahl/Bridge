---
name: modal-compute
description: How to run compute on Modal (ablack3 workspace) from this repo — CLI invocation, app/function structure, CPU vs GPU, secrets. Use when a task needs to run code in the cloud (heavy fetch, GPU training/inference, parallel batch jobs) rather than locally.
---

# Modal compute

## Token

Auth is already set up on this machine: a token pair lives in `~/.modal.toml` under profile
`ablack3`. Nothing further is needed to run jobs.

If the token is ever missing or invalid, get a new one at
[modal.com/settings/tokens](https://modal.com/settings/tokens) (workspace `ablack3`), or
regenerate it from this machine with:

```bash
python3 -m modal token new
```

`python3 -m modal setup` does the same thing via a browser login flow. Never print or paste
the token itself into chat, a file, or a URL — only confirm it works by running a job.

## Running jobs

The `modal` CLI script isn't on PATH here. Always invoke via `python3 -m modal`:

```bash
python3 -m modal run path/to/script.py       # run once, tear down after (use for scripts, testing)
python3 -m modal deploy path/to/script.py    # deploy persistently (use for scheduled/served functions)
```

To call a deployed function's API from other code instead of the CLI:

```python
import modal

fn = modal.Function.from_name("app-name", "function-name")
result = fn.remote(x)          # sync call, blocks for the result
call = fn.spawn(x)             # async call, returns a FunctionCall handle
result = call.get()            # block on that handle later
```

`from_name` looks up a function in an already-`deploy`ed app by app name and function name —
use it when calling Modal compute from code that isn't itself the app definition (e.g. a
script in this repo invoking a deployed pipeline).

## Minimal app

```python
import modal

app = modal.App("stable-name")  # no random strings, no env concatenation

image = modal.Image.debian_slim().pip_install("pandas==2.2.0")  # pin versions


@app.function(image=image, timeout=600)
def work(x: int) -> int:
    return x * 2


@app.local_entrypoint()
def main() -> None:
    print(work.remote(21))
```

## Rules (see repo `CLAUDE.md` for the full list; highlights below)

- Every `@app.function()` doing network I/O, GPU, or unbounded compute needs an explicit
  `timeout=`.
- GPU only on functions that need it, and typed: `gpu="A10G"`, not `gpu=True`.
- Pin all `pip_install` versions.
- Secrets via `modal.Secret.from_name("name")`, referenced in `secrets=[...]` — never
  hardcoded.
- In an `async def` caller, use `.aio()` variants (`await work.remote.aio(...)`); never mix
  sync `.remote()`/`.spawn()` inside async code.
- High-throughput per-item workloads: `@modal.batched(max_batch_size=..., wait_ms=...)`
  instead of looping externally.
- `print()` is fine only in `@app.local_entrypoint()` or scripts; use `logging` elsewhere.

## Verifying a run

`modal run` prints a dashboard URL (`https://modal.com/apps/ablack3/...`) — check it for logs
and cost if a run behaves unexpectedly.
