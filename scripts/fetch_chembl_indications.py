"""Fetch ChEMBL's curated drug_indication table — a real indication layer.

AGENT.md section 4: `in_eu_label` is false on all 1.45M CEM rows, so the efficacy side of
the label rests on ~7,900 SemMedDB literature assertions and nothing else. This brings in
a curated alternative.

Each ChEMBL indication record carries a molecule, a disease in two independent ontologies
(EFO and MeSH), the highest clinical phase reached for that specific indication, and the
references behind it — DailyMed, EMA, ClinicalTrials. `max_phase_for_ind = 4` is an
approved indication rather than a trialled one, which is the distinction the efficacy
label has been missing.

LEAKAGE WARNING. Open Targets' `known_drug` / `dt_clinical` evidence is derived from this
same ChEMBL table. Any design using an indication label from here must exclude that
Open Targets datatype, or it is reading the answer off its own input. See the decision
note in lab/notebook.jsonl.

Run with `uv run python scripts/fetch_chembl_indications.py`.
"""

import datetime
import json
import time
from pathlib import Path

import pandas as pd
import requests

ENDPOINT = "https://www.ebi.ac.uk/chembl/api/data/drug_indication.json"
PAGE_SIZE = 1000
RETRIES = 4

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "ref" / "chembl"
OUT_PATH = OUT_DIR / "drug_indication.csv"
PROVENANCE = OUT_DIR / "drug_indication_provenance.json"

FIELDS = [
    "drugind_id",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "efo_id",
    "efo_term",
    "mesh_id",
    "mesh_heading",
    "max_phase_for_ind",
]


def fetch_page(session: requests.Session, offset: int) -> dict:
    params = {"limit": PAGE_SIZE, "offset": offset}
    for attempt in range(RETRIES):
        response = session.get(ENDPOINT, params=params, timeout=180)
        if response.status_code == 200:
            return response.json()
        if attempt == RETRIES - 1:
            response.raise_for_status()
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def flatten(record: dict) -> dict:
    """One row per indication, with reference types collapsed to a sorted summary."""
    refs = record.get("indication_refs") or []
    row = {field: record.get(field) for field in FIELDS}
    row["n_refs"] = len(refs)
    row["ref_types"] = ";".join(sorted({r.get("ref_type", "") for r in refs if r.get("ref_type")}))
    return row


def main() -> None:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    first = fetch_page(session, 0)
    total = int(first["page_meta"]["total_count"])
    print(f"{total:,} drug_indication records, {PAGE_SIZE} per page")

    rows = [flatten(r) for r in first["drug_indications"]]
    offset = PAGE_SIZE
    while offset < total:
        payload = fetch_page(session, offset)
        rows.extend(flatten(r) for r in payload["drug_indications"])
        offset += PAGE_SIZE
        print(f"  {min(offset, total):,}/{total:,}", flush=True)

    frame = pd.DataFrame(rows)
    frame["max_phase_for_ind"] = pd.to_numeric(frame["max_phase_for_ind"], errors="coerce")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)

    approved = int((frame["max_phase_for_ind"] >= 4).sum())
    print(f"\nwrote {len(frame):,} rows -> {OUT_PATH}")
    print(f"  distinct molecules  {frame['molecule_chembl_id'].nunique():,}")
    print(f"  with an EFO id      {frame['efo_id'].notna().sum():,}")
    print(f"  with a MeSH id      {frame['mesh_id'].notna().sum():,}")
    print(f"  max_phase 4 (approved) {approved:,}")
    print("\nmax_phase_for_ind distribution:")
    print(frame["max_phase_for_ind"].value_counts(dropna=False).sort_index().to_string())

    PROVENANCE.write_text(
        json.dumps(
            {
                "source": "ChEMBL",
                "endpoint": ENDPOINT,
                "n_rows": len(frame),
                "n_molecules": int(frame["molecule_chembl_id"].nunique()),
                "n_max_phase_4": approved,
                "fields": [*FIELDS, "n_refs", "ref_types"],
                "leakage_note": (
                    "Open Targets known_drug / dt_clinical evidence is derived from this "
                    "same table; do not use both in one design."
                ),
                "retrieved_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"provenance -> {PROVENANCE}")


if __name__ == "__main__":
    main()
