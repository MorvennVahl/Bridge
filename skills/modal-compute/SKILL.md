---
name: modal-compute
description: How to run compute on Modal (ablack3 workspace) from this repo — CLI invocation, app/function structure, CPU vs GPU, secrets. Use when a task needs to run code in the cloud (heavy fetch, GPU training/inference, parallel batch jobs) rather than locally.
---

# Modal compute

Auth is already set up: token in `~/.modal.toml`, profile `ablack3`. No `modal setup` needed.

## Running

The `modal` CLI script isn't on PATH. Always invoke via:

```bash
python3 -m modal run path/to/script.py
python3 -m modal deploy path/to/script.py
```

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
