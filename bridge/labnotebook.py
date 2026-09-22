"""Shared, append-only lab notebook for the Bridge experiment agents.

Every experiment any agent runs is registered here before it runs and completed
here after it runs. The notebook is the only durable memory shared between
agents: an agent that does not read it will repeat someone else's dead end, and
an agent that does not write to it has wasted the compute.

Storage is one JSON object per line in lab/notebook.jsonl. Append-only: never
edit or delete a line. A superseded result is corrected by appending a new entry
whose `supersedes` names the old id.

Typical use inside an experiment
--------------------------------
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="agent-07",
        title="Pathway-overlap features + gradient boosting",
        hypothesis="Reactome overlap between drug target genes and condition "
                   "genes predicts FAERS harm signal beyond drug/condition degree.",
        approach="LightGBM on 14 overlap features plus degree nuisance terms.",
        label="y_faers_signal",
        features=["pathway_overlap_jaccard", "shared_gene_count", "drug_degree",
                  "condition_degree"],
        split="train -> grouped 5-fold CV; single validate evaluation at the end",
    )

    ...fit on train, select with grouped CV inside train...

    ln.complete(exp, metrics={"validate_average_precision": 0.31,
                              "validate_roc_auc": 0.78,
                              "train_cv_average_precision": 0.33},
                findings="Overlap adds +0.04 AP over the degree-only baseline. "
                         "Gain is concentrated in conditions with >=5 curated genes; "
                         "no gain at all for phenotype-arm conditions.",
                artifacts=["results/exp_0007_calibration.png"],
                next_steps="Try restricting to OMOP2OBO manual-tier mappings only.")

Validation discipline is enforced socially, not technically: `validate_evaluations`
counts how many times each agent has touched the validation set, and
`leaderboard()` shows it. A hypothesis tuned against 40 validation evaluations is
not a finding, and the count makes that visible to reviewers.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "lab"
NOTEBOOK = LAB / "notebook.jsonl"
TEST_SEAL = LAB / "TEST_SET_SEAL.txt"

VALID_LABELS = {
    "y_faers_signal",
    "y_semmeddb_causes",
    "y_semmeddb_treats",
    "y_any_harm",
    "faers_prr",
    "other",
}


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def _append(record: dict[str, Any]) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read() -> list[dict[str, Any]]:
    """Every entry, oldest first. Read this before proposing an experiment."""
    if not NOTEBOOK.exists():
        return []
    out = []
    with open(NOTEBOOK) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def note(agent: str, kind: str, title: str, body: str, data: dict[str, Any] | None = None) -> str:
    """Append a durable non-experiment entry: dataset state, a decision, a caveat.

    This is the lab's shared memory for things that are true about the data rather
    than results of an experiment — what exists, what it contains, what is known
    to be wrong with it. Agents should read these before designing an experiment
    so they do not rediscover a documented limitation.

    `kind` is free text; the ones in use are "dataset_state", "decision",
    "caveat", "round_plan".
    """
    note_id = f"note_{datetime.now(UTC):%Y%m%d}_{uuid.uuid4().hex[:6]}"
    _append(
        {
            "schema": 1,
            "event": "note",
            "id": note_id,
            "agent": agent,
            "kind": kind,
            "title": title,
            "body": body,
            "data": data or {},
            "git_commit": _git_commit(),
            "ts": datetime.now(UTC).isoformat(),
        }
    )
    return note_id


def notes(kind: str | None = None) -> list[dict[str, Any]]:
    """Every note entry, oldest first, optionally filtered by kind."""
    return [
        e for e in read() if e.get("event") == "note" and (kind is None or e.get("kind") == kind)
    ]


def register(
    agent: str,
    title: str,
    hypothesis: str,
    approach: str,
    label: str,
    features: list[str] | None = None,
    split: str = "",
    notes: str = "",
) -> str:
    """Record an experiment BEFORE running it. Returns its experiment id.

    Registering first is what makes a negative result trustworthy: the
    hypothesis is on record before the numbers are known, so a null finding
    cannot be quietly reframed as a different question that happened to work.
    """
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}, got {label!r}")
    exp_id = f"exp_{datetime.now(UTC):%Y%m%d}_{uuid.uuid4().hex[:6]}"
    _append(
        {
            "schema": 1,
            "event": "register",
            "id": exp_id,
            "agent": agent,
            "title": title,
            "hypothesis": hypothesis,
            "approach": approach,
            "label": label,
            "features": features or [],
            "split": split,
            "notes": notes,
            "git_commit": _git_commit(),
            "host": platform.node() if os.environ.get("BRIDGE_LOG_HOST") else None,
            "ts": datetime.now(UTC).isoformat(),
        }
    )
    return exp_id


def complete(
    exp_id: str,
    metrics: dict[str, float],
    findings: str,
    artifacts: list[str] | None = None,
    next_steps: str = "",
    used_test_set: bool = False,
    supersedes: str | None = None,
    failed: bool = False,
) -> None:
    """Record the outcome. Call this even when the experiment failed or was null.

    A null result that is logged saves every later agent the same run. A null
    result that is not logged costs the lab that run again, repeatedly.
    """
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty dict, even for a null result")
    if not findings.strip():
        raise ValueError("findings must say what was learned, including 'nothing'")
    _append(
        {
            "schema": 1,
            "event": "complete",
            "id": exp_id,
            "metrics": metrics,
            "findings": findings,
            "artifacts": artifacts or [],
            "next_steps": next_steps,
            "used_test_set": bool(used_test_set),
            "supersedes": supersedes,
            "failed": bool(failed),
            "git_commit": _git_commit(),
            "ts": datetime.now(UTC).isoformat(),
        }
    )


def validate_evaluations(agent: str | None = None) -> int:
    """How many completed experiments have reported a validation metric.

    High counts mean the validation set has been queried many times and is
    partially burned; treat small differences between models as noise.
    """
    n = 0
    reg = {e["id"]: e for e in read() if e.get("event") == "register"}
    for e in read():
        if e.get("event") != "complete":
            continue
        if agent and reg.get(e["id"], {}).get("agent") != agent:
            continue
        if any("validate" in k for k in (e.get("metrics") or {})):
            n += 1
    return n


def leaderboard(label: str | None = None, metric: str = "validate_average_precision"):
    """Completed experiments ranked by one validation metric, best first."""
    reg = {e["id"]: e for e in read() if e.get("event") == "register"}
    rows = []
    for e in read():
        if e.get("event") != "complete" or e.get("failed"):
            continue
        r = reg.get(e["id"], {})
        if label and r.get("label") != label:
            continue
        if metric in (e.get("metrics") or {}):
            rows.append(
                {
                    "id": e["id"],
                    "agent": r.get("agent"),
                    "title": r.get("title"),
                    "label": r.get("label"),
                    metric: e["metrics"][metric],
                    "used_test_set": e.get("used_test_set"),
                }
            )
    rows.sort(key=lambda r: r[metric], reverse=True)
    return rows


def check_test_seal() -> None:
    """Raise unless a human has unsealed the test set for one final evaluation.

    The test set answers exactly one question: how well does the finally-chosen
    model do on drugs whose target families it has never seen. Every touch
    before that point converts it into another validation set and destroys the
    only unbiased estimate the project has.

    To unseal, a human writes lab/TEST_SET_SEAL.txt containing the experiment id
    being promoted and the reason. Do not create this file yourself.
    """
    if not TEST_SEAL.exists():
        raise PermissionError(
            "The test set is sealed. Use train for fitting and validate for "
            "model selection. If you believe the project is genuinely ready for "
            "its single final evaluation, say so in your report and ask the "
            "human to unseal it — do not create lab/TEST_SET_SEAL.txt yourself."
        )
