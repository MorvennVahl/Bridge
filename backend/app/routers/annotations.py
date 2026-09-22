"""Append-only CRU for user-curated drug-condition annotations.

Persists to ``data/annotations.csv``. No delete endpoint. Reads catalog to
resolve names — if either concept id is unknown, the write is rejected.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException

from backend.app.data import DATA_DIR, load_conditions, load_ingredients
from backend.app.models import Annotation, AnnotationCreate

router = APIRouter(prefix="/api", tags=["annotations"])

ANNOTATIONS_PATH: Path = DATA_DIR / "annotations.csv"
_write_lock = Lock()

_FIELDS = [
    "ingredient_concept_id",
    "ingredient_name",
    "condition_concept_id",
    "condition_name",
    "assertion",
    "notes",
    "added_by",
    "added_at",
]


def _ensure_file() -> None:
    if ANNOTATIONS_PATH.is_file():
        return
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ANNOTATIONS_PATH.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=_FIELDS).writeheader()


def _read_all() -> list[Annotation]:
    if not ANNOTATIONS_PATH.is_file():
        return []
    out: list[Annotation] = []
    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            try:
                out.append(
                    Annotation(
                        ingredient_concept_id=int(raw["ingredient_concept_id"]),
                        ingredient_name=raw["ingredient_name"],
                        condition_concept_id=int(raw["condition_concept_id"]),
                        condition_name=raw["condition_name"],
                        assertion=raw["assertion"],
                        notes=raw["notes"] or None,
                        added_by=raw["added_by"] or None,
                        added_at=raw["added_at"],
                    )
                )
            except (KeyError, ValueError):
                continue
    return out


@router.get("/annotations", response_model=list[Annotation])
def list_annotations() -> list[Annotation]:
    return _read_all()


@router.post("/annotations", response_model=Annotation, status_code=201)
def create_annotation(body: AnnotationCreate) -> Annotation:
    if body.assertion not in {"treats", "causes"}:
        raise HTTPException(status_code=422, detail="assertion must be 'treats' or 'causes'")

    ings = load_ingredients()
    conds = load_conditions()
    ing_match = (
        ings[ings["ingredient_concept_id"] == body.ingredient_concept_id]
        if not ings.empty
        else ings
    )
    cond_match = (
        conds[conds["condition_concept_id"] == body.condition_concept_id]
        if not conds.empty
        else conds
    )
    if ing_match.empty:
        raise HTTPException(
            status_code=404, detail=f"Unknown ingredient {body.ingredient_concept_id}"
        )
    if cond_match.empty:
        raise HTTPException(
            status_code=404, detail=f"Unknown condition {body.condition_concept_id}"
        )

    row = Annotation(
        ingredient_concept_id=body.ingredient_concept_id,
        ingredient_name=str(ing_match.iloc[0]["ingredient_name"]),
        condition_concept_id=body.condition_concept_id,
        condition_name=str(cond_match.iloc[0]["condition_name"]),
        assertion=body.assertion,
        notes=body.notes,
        added_by=body.added_by,
        added_at=datetime.now(UTC).isoformat(),
    )

    with _write_lock:
        _ensure_file()
        with ANNOTATIONS_PATH.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_FIELDS).writerow(row.model_dump())

    return row
