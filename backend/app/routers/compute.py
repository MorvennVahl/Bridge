"""Compute endpoints — hand off to Modal functions.

No fallback, no mock: if Modal is unreachable / not deployed / not authed,
the endpoint returns 503 with the underlying error so the frontend can show
an honest failure.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from backend.app.models import DrugSummaryResponse

router = APIRouter(prefix="/api/compute", tags=["compute"])

MODAL_APP_NAME = "bridge"
MODAL_FUNCTION_NAME = "drug_evidence_summary"


@router.post("/drug-summary", response_model=DrugSummaryResponse)
async def drug_summary(ingredient_concept_id: int, top: int = 20) -> DrugSummaryResponse:
    try:
        import modal
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"modal SDK not installed: {e}") from e

    try:
        fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modal function {MODAL_APP_NAME}.{MODAL_FUNCTION_NAME} not found — "
                f"run `uv run modal deploy modal_app.py` first ({e})"
            ),
        ) from e

    started = time.monotonic()
    try:
        result = await fn.remote.aio(ingredient_concept_id=ingredient_concept_id, top=top)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Modal call failed: {e}") from e
    elapsed_ms = int((time.monotonic() - started) * 1000)

    return DrugSummaryResponse(
        ingredient_concept_id=ingredient_concept_id,
        rows_scanned=int(result.get("rows_scanned", 0)),
        conditions=int(result.get("conditions", 0)),
        faers_pairs=int(result.get("faers_pairs", 0)),
        semmeddb_pairs=int(result.get("semmeddb_pairs", 0)),
        top=result.get("top", []),
        elapsed_ms=elapsed_ms,
        executor="modal",
    )
