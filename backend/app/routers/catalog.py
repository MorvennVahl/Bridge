"""Catalog endpoints — ingredient and condition listings for the UI selectors."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.data import load_conditions, load_ingredients
from backend.app.models import Condition, Ingredient

router = APIRouter(prefix="/api", tags=["catalog"])


@router.get("/ingredients", response_model=list[Ingredient])
def list_ingredients(
    q: str | None = Query(None, description="Case-insensitive substring filter on name"),
    limit: int = Query(50, ge=1, le=1000),
) -> list[Ingredient]:
    df = load_ingredients()
    if q:
        df = df[df["ingredient_name"].str.contains(q, case=False, na=False)]
    df = df.head(limit)
    return [Ingredient(**row) for row in df.to_dict(orient="records")]


@router.get("/conditions", response_model=list[Condition])
def list_conditions(
    q: str | None = Query(None, description="Case-insensitive substring filter on name"),
    limit: int = Query(50, ge=1, le=1000),
) -> list[Condition]:
    df = load_conditions()
    if q:
        df = df[df["condition_name"].str.contains(q, case=False, na=False)]
    df = df.head(limit)
    return [Condition(**row) for row in df.to_dict(orient="records")]
