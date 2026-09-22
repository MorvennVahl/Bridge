"""Generic tabular CRU over any file in ``data/``.

Fully dynamic — the registry is discovered on demand by walking ``data/``
recursively for ``.csv``, ``.tsv``, and ``.parquet`` files. Row identity is
the file's zero-based row index. New rows are appended.

Deletes are intentionally not exposed. Existing rows can be edited or new rows
appended; nothing is removed.
"""

from __future__ import annotations

import contextlib
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.data import DATA_DIR, load_associations, load_conditions, load_ingredients

router = APIRouter(prefix="/api/tables", tags=["tables"])

_write_locks: dict[str, Lock] = {}

# Extensions we can round-trip (read + write).
_TEXT_EXTS = {".csv", ".tsv"}
_BINARY_EXTS = {".parquet"}
_ALL_EXTS = _TEXT_EXTS | _BINARY_EXTS

# Slashes are unsafe in URL path params. We encode nested table names by
# swapping ``/`` for ``__``.
_SEP = "__"


def _lock_for(name: str) -> Lock:
    if name not in _write_locks:
        _write_locks[name] = Lock()
    return _write_locks[name]


@dataclass(frozen=True)
class TableSpec:
    name: str
    path: Path
    fmt: str
    delimiter: str


def _slug(relpath: Path) -> str:
    return relpath.with_suffix("").as_posix().replace("/", _SEP)


def _count_text_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def _count_parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    try:
        return int(pq.read_metadata(str(path)).num_rows)
    except Exception:
        return 0


def _row_count(path: Path, fmt: str) -> int:
    if not path.is_file():
        return 0
    if fmt == "parquet":
        return _count_parquet_rows(path)
    return _count_text_rows(path)


def _discover_tables() -> dict[str, TableSpec]:
    out: dict[str, TableSpec] = {}
    if not DATA_DIR.is_dir():
        return out
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in _ALL_EXTS:
            continue
        rel = path.relative_to(DATA_DIR)
        name = _slug(rel)
        fmt = ext.lstrip(".")
        delim = "\t" if ext == ".tsv" else ","
        out[name] = TableSpec(name=name, path=path, fmt=fmt, delimiter=delim)

    # Always surface annotations even if the file hasn't been created yet.
    if "annotations" not in out:
        out["annotations"] = TableSpec(
            name="annotations",
            path=DATA_DIR / "annotations.csv",
            fmt="csv",
            delimiter=",",
        )
    return out


def _get_tables() -> dict[str, TableSpec]:
    """Rediscover on every call so newly-added files show up immediately."""
    return _discover_tables()


# Cache-invalidation targets when a table is committed. Keeps the /api/ingredients
# and /api/conditions selectors fresh after edits.
_CACHE_INVALIDATORS = {
    "cem_ingredients": (load_ingredients,),
    "cem_ingredient_condition_associations": (load_conditions, load_associations),
}


class TableMeta(BaseModel):
    name: str
    path: str
    fmt: str
    columns: list[str]
    row_count: int
    editable: bool
    warn: str | None = None


class TablePage(BaseModel):
    name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    offset: int
    limit: int
    total: int


class RowEdit(BaseModel):
    index: int
    values: dict[str, Any]


class RowAdd(BaseModel):
    values: dict[str, Any]


class Commit(BaseModel):
    edits: list[RowEdit] = []
    adds: list[RowAdd] = []


class CommitResult(BaseModel):
    edited: int
    added: int
    total: int


# ---------- read/write adapters per format ----------


def _read_all(spec: TableSpec) -> tuple[list[str], list[dict[str, str]]]:
    if not spec.path.is_file():
        return [], []
    if spec.fmt == "parquet":
        import pandas as pd

        df = pd.read_parquet(spec.path)
        columns = [str(c) for c in df.columns]
        cast = df.astype("string").fillna("")
        return columns, [
            {c: str(v) for c, v in row.items()} for row in cast.to_dict(orient="records")
        ]
    with spec.path.open("r", encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh, delimiter=spec.delimiter)
        columns = list(r.fieldnames or [])
        rows = [dict(row) for row in r]
    return columns, rows


def _read_header(spec: TableSpec) -> list[str]:
    if not spec.path.is_file():
        return []
    if spec.fmt == "parquet":
        import pyarrow.parquet as pq

        try:
            return [str(f.name) for f in pq.read_schema(str(spec.path))]
        except Exception:
            return []
    with spec.path.open("r", encoding="utf-8", newline="") as fh:
        r = csv.reader(fh, delimiter=spec.delimiter)
        try:
            return next(r)
        except StopIteration:
            return []


def _write_all(spec: TableSpec, columns: list[str], rows: list[dict[str, Any]]) -> None:
    tmp = spec.path.with_suffix(spec.path.suffix + ".tmp")
    if spec.fmt == "parquet":
        import pandas as pd

        normalized = [{c: ("" if r.get(c) is None else r.get(c)) for c in columns} for r in rows]
        pd.DataFrame(normalized, columns=columns).to_parquet(tmp, index=False)
    else:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh, fieldnames=columns, extrasaction="ignore", delimiter=spec.delimiter
            )
            w.writeheader()
            for row in rows:
                w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
    shutil.move(str(tmp), str(spec.path))


# ---------- endpoints ----------


@router.get("", response_model=list[TableMeta])
def list_tables() -> list[TableMeta]:
    tables = _get_tables()
    out: list[TableMeta] = []
    for spec in tables.values():
        columns = _read_header(spec)
        row_count = _row_count(spec.path, spec.fmt)
        warn: str | None = None
        if row_count > 100_000:
            warn = f"{row_count:,} rows — full-file rewrite on save is slow"
        out.append(
            TableMeta(
                name=spec.name,
                path=str(spec.path.relative_to(DATA_DIR.parent)),
                fmt=spec.fmt,
                columns=columns,
                row_count=row_count,
                editable=True,
                warn=warn,
            )
        )
    return out


@router.get("/{name}", response_model=TablePage)
def read_table(name: str, offset: int = 0, limit: int = 50) -> TablePage:
    tables = _get_tables()
    if name not in tables:
        raise HTTPException(status_code=404, detail=f"unknown table {name!r}")
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be in [1, 500]")
    spec = tables[name]
    columns, all_rows = _read_all(spec)
    total = len(all_rows)
    slice_ = all_rows[offset : offset + limit]
    rows_with_idx = [{**r, "__index__": offset + i} for i, r in enumerate(slice_)]
    return TablePage(
        name=name,
        columns=columns,
        rows=rows_with_idx,
        offset=offset,
        limit=limit,
        total=total,
    )


@router.post("/{name}/commit", response_model=CommitResult)
def commit_table(name: str, body: Commit) -> CommitResult:
    tables = _get_tables()
    if name not in tables:
        raise HTTPException(status_code=404, detail=f"unknown table {name!r}")
    spec = tables[name]

    with _lock_for(name):
        columns, rows = _read_all(spec)
        if not columns:
            if not body.adds:
                raise HTTPException(status_code=400, detail="file has no header and no adds")
            columns = list(body.adds[0].values.keys())

        edit_count = 0
        for edit in body.edits:
            if edit.index < 0 or edit.index >= len(rows):
                raise HTTPException(status_code=422, detail=f"edit index {edit.index} out of range")
            for col, val in edit.values.items():
                if col not in columns:
                    continue
                rows[edit.index][col] = "" if val is None else str(val)
            edit_count += 1

        for add in body.adds:
            new_row = {c: str(add.values.get(c, "") or "") for c in columns}
            rows.append(new_row)

        _write_all(spec, columns, rows)

    # Invalidate cached loaders keyed by the underlying file stem.
    file_stem = spec.path.stem
    for loader in _CACHE_INVALIDATORS.get(file_stem, ()):
        with contextlib.suppress(AttributeError):
            loader.cache_clear()  # type: ignore[attr-defined]

    return CommitResult(edited=edit_count, added=len(body.adds), total=len(rows))
