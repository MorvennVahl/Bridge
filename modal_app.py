"""Modal deploy for the Bridge backend + built frontend.

Serves:
- FastAPI API at ``/api/*``
- Built React SPA at ``/`` (index.html + assets)

Also defines the ``drug_evidence_summary`` function that the
``/api/compute/drug-summary`` endpoint invokes via ``modal.Function.from_name``.

Deploy with::

    cd frontend && npm install && npm run build && cd ..
    uv run modal deploy modal_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi[standard]==0.115.0",
        "pandas==2.2.3",
        "pyarrow==17.0.0",
    )
    .add_local_dir(REPO_ROOT / "backend", remote_path="/root/backend")
    .add_local_dir(REPO_ROOT / "bridge", remote_path="/root/bridge")
    .add_local_dir(REPO_ROOT / "data", remote_path="/root/data")
    .add_local_dir(REPO_ROOT / "frontend" / "dist", remote_path="/root/frontend/dist")
)

app = modal.App("bridge", image=image)


@app.function(timeout=600, min_containers=1)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root")
    from backend.app.main import app as fastapi_app

    return fastapi_app


@app.function(timeout=300)
def drug_evidence_summary(ingredient_concept_id: int, top: int = 20) -> dict[str, Any]:
    """Scan the 1.45M-row CEM associations for one ingredient.

    Returns per-condition evidence rollups. Real data, no mocks — raises if
    the file is missing.
    """
    import sys

    sys.path.insert(0, "/root")
    import pandas as pd

    from backend.app.data import load_associations

    df = load_associations()
    if df.empty:
        raise RuntimeError("associations file empty or missing on the Modal container")

    slice_ = df.loc[df["ingredient_concept_id"] == ingredient_concept_id]
    rows_scanned = len(df)
    faers_pairs = (
        int(slice_["in_faers"].fillna(False).astype(bool).sum())
        if "in_faers" in slice_.columns
        else 0
    )
    semmeddb_pairs = (
        int(slice_["in_semmeddb"].fillna(False).astype(bool).sum())
        if "in_semmeddb" in slice_.columns
        else 0
    )

    prr = (
        pd.to_numeric(slice_["faers_prr"], errors="coerce").fillna(0.0)
        if "faers_prr" in slice_.columns
        else pd.Series(0.0, index=slice_.index)
    )
    ranked = slice_.assign(_prr=prr).sort_values("_prr", ascending=False).head(top)
    top_rows = [
        {
            "condition_concept_id": int(r["condition_concept_id"]),
            "condition_name": str(r.get("condition_name") or ""),
            "faers_prr": (
                float(r["_prr"]) if pd.notna(r["_prr"]) and float(r["_prr"]) > 0 else None
            ),
            "semmeddb_relationships": (
                str(r["semmeddb_relationships"])
                if "semmeddb_relationships" in r and pd.notna(r["semmeddb_relationships"])
                else None
            ),
        }
        for _, r in ranked.iterrows()
    ]

    return {
        "rows_scanned": rows_scanned,
        "conditions": int(slice_["condition_concept_id"].nunique()) if not slice_.empty else 0,
        "faers_pairs": faers_pairs,
        "semmeddb_pairs": semmeddb_pairs,
        "top": top_rows,
    }
