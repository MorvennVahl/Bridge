"""Prediction endpoint. Returns real pair info + real evidence rows.

Scores are NULL until a trained model is wired in — no fake values, ever.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app.data import load_associations, load_conditions, load_ingredients
from backend.app.models import EvidenceRow, PredictResponse

router = APIRouter(prefix="/api", tags=["predict"])


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if pd.notna(f) else None


def _safe_str(v: object) -> str | None:
    if v is None:
        return None
    try:
        if not pd.notna(v):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s or None


@router.get("/predict", response_model=PredictResponse)
def predict(ingredient_concept_id: int, condition_concept_id: int) -> PredictResponse:
    ings = load_ingredients()
    conds = load_conditions()

    ing_row = (
        ings[ings["ingredient_concept_id"] == ingredient_concept_id] if not ings.empty else ings
    )
    if ing_row.empty:
        raise HTTPException(status_code=404, detail=f"Unknown ingredient {ingredient_concept_id}")
    cond_row = (
        conds[conds["condition_concept_id"] == condition_concept_id] if not conds.empty else conds
    )
    if cond_row.empty:
        raise HTTPException(status_code=404, detail=f"Unknown condition {condition_concept_id}")

    ing_name = _safe_str(ing_row.iloc[0]["ingredient_name"]) or f"concept_{ingredient_concept_id}"
    cond_name = _safe_str(cond_row.iloc[0]["condition_name"]) or f"concept_{condition_concept_id}"

    evidence: list[EvidenceRow] = []
    associations = load_associations()
    if not associations.empty:
        try:
            match = associations[
                (associations["ingredient_concept_id"] == ingredient_concept_id)
                & (associations["condition_concept_id"] == condition_concept_id)
            ]
        except KeyError:
            match = associations.iloc[0:0]

        if not match.empty:
            row = match.iloc[0]
            if bool(row.get("in_faers", False)):
                evidence.append(
                    EvidenceRow(
                        source="faers",
                        relationship="disproportionality",
                        value=_safe_float(row.get("faers_prr")),
                    )
                )
            if bool(row.get("in_semmeddb", False)):
                evidence.append(
                    EvidenceRow(
                        source="semmeddb",
                        relationship=_safe_str(row.get("semmeddb_relationships")),
                        value=None,
                    )
                )

    return PredictResponse(
        ingredient_concept_id=ingredient_concept_id,
        ingredient_name=ing_name,
        condition_concept_id=condition_concept_id,
        condition_name=cond_name,
        treats_score=None,
        causes_score=None,
        confidence=None,
        evidence=evidence,
        status="model_not_trained",
        model=None,
    )
