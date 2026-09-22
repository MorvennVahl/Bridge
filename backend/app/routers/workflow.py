"""Continuous-improvement workflow: data -> features -> train -> evaluate -> review -> iterate.

Each stage is a handler that runs real work and appends a run record to
``data/experiments/runs.jsonl``. Nothing here is mocked — a stage that isn't
wired yet returns status ``'not_implemented'`` explicitly so the UI can show
an honest state.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.data import DATA_DIR

router = APIRouter(prefix="/api/workflow", tags=["workflow"])

EXPERIMENTS_DIR: Path = DATA_DIR / "experiments"
RUNS_LOG: Path = EXPERIMENTS_DIR / "runs.jsonl"
_append_lock = Lock()

STAGES: list[str] = [
    "data_check",
    "build_features",
    "iterate",
]


class TriggerRequest(BaseModel):
    stage: str = Field(description=" | ".join(STAGES))
    notes: str | None = None


class Run(BaseModel):
    id: str
    stage: str
    status: str  # ok | failed | not_implemented
    started_at: str
    finished_at: str
    elapsed_ms: int
    metrics: dict[str, Any]
    notes: str | None = None
    error: str | None = None


class KPIs(BaseModel):
    total_runs: int
    last_stage: str | None
    last_stage_at: str | None
    stages_run: dict[str, int]
    conditions_total: int
    conditions_mapped: int
    conditions_with_genes: int
    conditions_with_drug_target_overlap: int
    swarm_runs: int
    latest_metrics: dict[str, Any]


def _read_runs() -> list[Run]:
    if not RUNS_LOG.is_file():
        return []
    out: list[Run] = []
    with RUNS_LOG.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Run.model_validate_json(line))
            except Exception:
                continue
    return out


def _append_run(run: Run) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with _append_lock, RUNS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(run.model_dump_json() + "\n")


# ---------- stage handlers ----------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _handle_data_check() -> tuple[str, dict[str, Any], str | None]:
    required = {
        "cem_ingredients.csv": DATA_DIR / "cem_ingredients.csv",
        "cem_ingredient_condition_associations.csv": DATA_DIR
        / "cem_ingredient_condition_associations.csv",
        "ingredient_features.csv": DATA_DIR / "ingredient_features.csv",
        "ref/omop/concept.csv": DATA_DIR / "ref" / "omop" / "concept.csv",
        "ref/mondo.sssom.tsv": DATA_DIR / "ref" / "mondo.sssom.tsv",
        "ref/hpo__hp.obo": DATA_DIR / "ref" / "hpo__hp.obo",
    }
    present = {k: p.is_file() for k, p in required.items()}
    total_bytes = sum(p.stat().st_size for p in required.values() if p.is_file())
    all_present = all(present.values())
    return (
        "ok" if all_present else "failed",
        {
            "files_checked": len(required),
            "files_present": sum(present.values()),
            "total_bytes": total_bytes,
            "present": present,
        },
        None if all_present else "one or more required files missing",
    )


def _handle_build_features() -> tuple[str, dict[str, Any], str | None]:
    metrics: dict[str, Any] = {}
    for module in (
        "bridge.build_hpo_features",
        "bridge.build_disease_features",
        "bridge.build_condition_features",
    ):
        proc = subprocess.run(
            ["uv", "run", "python", "-m", module],
            capture_output=True,
            text=True,
            timeout=180,
        )
        metrics[module] = {
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr.strip().splitlines()[-5:] if proc.stderr else [],
        }
        if proc.returncode != 0:
            return "failed", metrics, f"{module} exited {proc.returncode}"

    derived = DATA_DIR / "derived"
    parquets = sorted(derived.glob("*.parquet")) if derived.is_dir() else []
    metrics["outputs"] = [{"name": p.name, "size_bytes": p.stat().st_size} for p in parquets]
    return "ok", metrics, None


def _swarm_snapshot() -> list[dict[str, Any]]:
    """Return last 10 swarm entries as plain dicts."""
    swarm_log = EXPERIMENTS_DIR / "swarm.jsonl"
    if not swarm_log.is_file():
        return []
    out: list[dict[str, Any]] = []
    with swarm_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-10:]


def _handle_iterate() -> tuple[str, dict[str, Any], str | None]:
    """One turn of the improvement loop.

    Reads the last ~10 swarm runs, asks Claude to synthesise what patterns are
    visible and what to score next, and appends a review to the run's metrics.
    The updated KPI panel picks up the growing swarm count and any new
    consensus stats automatically on the next refresh.

    Loop is the swarm side — this handler doesn't kick it off, only reviews it.
    """
    try:
        import os

        import anthropic
    except ImportError:
        return (
            "failed",
            {"note": "anthropic SDK not installed"},
            "add anthropic to deps",
        )

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return (
            "failed",
            {"note": "ANTHROPIC_API_KEY not set"},
            "export ANTHROPIC_API_KEY=... and restart uvicorn",
        )

    swarm_recent = _swarm_snapshot()
    kpis_now = get_kpis()

    if not swarm_recent:
        return (
            "failed",
            {"note": "no swarm runs yet — start the swarm loop first"},
            "swarm.jsonl is empty",
        )

    # Compact summary the model can reason over.
    def _one_line(sr: dict[str, Any]) -> str:
        c = sr.get("consensus", {})
        return (
            f"- {sr.get('ingredient_name')} -> {sr.get('condition_name')} :: "
            f"treats={c.get('mean_treats')}  causes={c.get('mean_causes')}  "
            f"direction={c.get('direction')}  stdev={max(c.get('stdev_treats') or 0, c.get('stdev_causes') or 0):.2f}"
        )

    swarm_summary = "\n".join(_one_line(r) for r in swarm_recent)

    client = anthropic.Anthropic(api_key=key)
    prompt = (
        "You are the orchestrator of a drug-condition prediction pipeline. "
        "The 5-agent swarm has just scored these pairs:\n\n"
        f"{swarm_summary}\n\n"
        "Pipeline KPIs right now:\n"
        f"- conditions_mapped: {kpis_now.conditions_mapped}/{kpis_now.conditions_total}\n"
        f"- conditions_with_drug_target_overlap (model ceiling): "
        f"{kpis_now.conditions_with_drug_target_overlap}\n"
        f"- swarm_runs so far: {kpis_now.swarm_runs}\n\n"
        "In 4-6 tight bullets, (a) name the strongest pattern you see across "
        "the recent scoring, (b) name any pair whose consensus deserves manual "
        "follow-up (high stdev or contrarian split), (c) suggest which class of "
        "pairs the next batch should target and why. Be concrete. No hedging."
    )

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return "failed", {"error_class": type(e).__name__}, f"anthropic call: {e}"

    review = "\n".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )

    direction_counts: dict[str, int] = {}
    for r in swarm_recent:
        d = (r.get("consensus") or {}).get("direction", "unknown")
        direction_counts[d] = direction_counts.get(d, 0) + 1

    metrics: dict[str, Any] = {
        "model": resp.model,
        "n_swarm_runs_reviewed": len(swarm_recent),
        "direction_counts": direction_counts,
        "review": review,
    }
    return "ok", metrics, None


_HANDLERS = {
    "data_check": _handle_data_check,
    "build_features": _handle_build_features,
    "iterate": _handle_iterate,
}


# ---------- endpoints ----------


@router.get("/runs", response_model=list[Run])
def list_runs(limit: int = 100) -> list[Run]:
    return list(reversed(_read_runs()))[:limit]


def _read_coverage_from_condition_features() -> tuple[int, int, int, int]:
    """Return (total, mapped, has_genes, drug_target_overlap) from Frank's derived tables.

    - mapped: match_tier is not '0_unmatched'
    - has_genes: n_genes > 0
    - drug_target_overlap: distinct conditions in condition_genes ⋈ ingredient_target_long
      joined on gene_symbol
    """
    features = DATA_DIR / "derived" / "condition_features.parquet"
    if not features.is_file():
        return 0, 0, 0, 0
    try:
        import pandas as pd
        import pyarrow.parquet as pq

        fdf = pq.read_table(features).to_pandas()
        total = len(fdf)
        mapped = int((fdf["match_tier"].astype(str) != "0_unmatched").sum())
        has_genes = int((fdf["n_genes"].fillna(0).astype(int) > 0).sum())

        drug_overlap = 0
        cgenes_path = DATA_DIR / "derived" / "condition_genes.parquet"
        drug_path = DATA_DIR / "ingredient_target_long.csv"
        if cgenes_path.is_file() and drug_path.is_file():
            cgenes = pq.read_table(cgenes_path).to_pandas()
            drug_targets = pd.read_csv(drug_path, usecols=lambda c: c in {"gene_symbol"})
            drug_symbols = set(drug_targets["gene_symbol"].dropna().astype(str))
            if "gene_symbol" in cgenes.columns:
                hits = cgenes[cgenes["gene_symbol"].astype(str).isin(drug_symbols)]
                drug_overlap = int(hits["condition_concept_id"].nunique())
        return total, mapped, has_genes, drug_overlap
    except Exception:
        return 0, 0, 0, 0


def _read_swarm_count() -> int:
    swarm = EXPERIMENTS_DIR / "swarm.jsonl"
    if not swarm.is_file():
        return 0
    with swarm.open("rb") as fh:
        return sum(1 for _ in fh)


@router.get("/kpis", response_model=KPIs)
def get_kpis() -> KPIs:
    runs = _read_runs()
    stages_run: dict[str, int] = dict.fromkeys(STAGES, 0)
    for r in runs:
        if r.stage in stages_run:
            stages_run[r.stage] += 1
    latest_ok = next((r for r in reversed(runs) if r.status == "ok"), None)
    total, mapped, has_genes, drug_overlap = _read_coverage_from_condition_features()
    return KPIs(
        total_runs=len(runs),
        last_stage=runs[-1].stage if runs else None,
        last_stage_at=runs[-1].started_at if runs else None,
        stages_run=stages_run,
        conditions_total=total,
        conditions_mapped=mapped,
        conditions_with_genes=has_genes,
        conditions_with_drug_target_overlap=drug_overlap,
        swarm_runs=_read_swarm_count(),
        latest_metrics=(latest_ok.metrics if latest_ok else {}),
    )


@router.post("/trigger", response_model=Run)
def trigger(req: TriggerRequest) -> Run:
    if req.stage not in _HANDLERS:
        raise HTTPException(status_code=422, detail=f"unknown stage {req.stage!r}")

    started = _now_iso()
    t0 = time.monotonic()
    try:
        status, metrics, error = _HANDLERS[req.stage]()
    except Exception as e:
        status, metrics, error = "failed", {}, f"{type(e).__name__}: {e}"
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    run = Run(
        id=uuid.uuid4().hex[:12],
        stage=req.stage,
        status=status,
        started_at=started,
        finished_at=_now_iso(),
        elapsed_ms=elapsed_ms,
        metrics=metrics,
        notes=req.notes,
        error=error,
    )
    _append_run(run)
    return run
