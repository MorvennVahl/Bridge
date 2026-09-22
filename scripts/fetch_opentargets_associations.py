"""Fetch Open Targets target-disease associations, filtered to the CEM condition set.

DESIGN.md calls the weighted gene-to-condition edge "the edge that carries most of the
causal information", and it was the one input we did not have. The stand-in built from
HPO's OMIM/Orphanet links runs 8,499 Mendelian to 646 polygenic, while FAERS reports
overwhelmingly on common polygenic disease.

The full dataset is ~2.8 GB across 41 parquet parts, but we never transfer it. The
`timeseries` column is roughly 95% of those bytes and we have no use for per-year scores,
so the parts are read remotely with a column projection: polars fetches only the column
chunks it needs. That turns a multi-hour download into a few minutes, and the rows are
filtered to diseases a CEM condition actually maps to before anything is written.

The `indirect` variant is used deliberately: it propagates evidence up the disease
ontology, so a gene associated with a specific disease also counts for its parents. Many
CEM conditions are broad ("Disorder of eye"), and the direct variant would leave them empty.

`known_drug` evidence is kept in the output so the reference file stays faithful to the
source; it is excluded where features are built, because it encodes drug-disease outcomes
and would leak the label. See `bridge.conditions.LEAKY_DATATYPES`.

Run with `uv run python scripts/fetch_opentargets_associations.py`.
"""

import argparse
import datetime
import json
import re
from pathlib import Path

import polars as pl
import requests

FTP_ROOT = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
DATASET = "association_by_datatype_indirect"
DEFAULT_VERSION = "26.06"

KEEP_COLUMNS = ["diseaseId", "targetId", "aggregationValue", "associationScore", "evidenceCount"]

REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITION_MAP = REPO_ROOT / "data" / "derived" / "condition_ontology_map.parquet"
OUT_PATH = REPO_ROOT / "data" / "ref" / "ot" / f"{DATASET}__filtered.parquet"
PROVENANCE = REPO_ROOT / "data" / "ref" / "ot" / f"{DATASET}__provenance.json"

_PART = re.compile(r"part-\d+-[0-9a-f-]+-c\d+\.snappy\.parquet")


def wanted_diseases() -> list[str]:
    """Open Targets disease ids that at least one CEM condition maps to.

    Ids are stored colon-separated (`MONDO:0009061`) by the build; Open Targets uses an
    underscore (`MONDO_0009061`).
    """
    if not CONDITION_MAP.exists():
        raise SystemExit(
            f"{CONDITION_MAP} not found. Run `uv run python -m bridge.build_condition_features` "
            "first — this script filters to the diseases those conditions map to."
        )
    frame = pl.read_parquet(CONDITION_MAP, columns=["ontology_id", "arm"]).filter(
        (pl.col("arm") == "disease") & pl.col("ontology_id").is_not_null()
    )
    return sorted({v.replace(":", "_", 1) for v in frame["ontology_id"].unique().to_list()})


def part_urls(version: str) -> list[str]:
    base = f"{FTP_ROOT}/{version}/output/{DATASET}/"
    response = requests.get(base, timeout=120)
    response.raise_for_status()
    return [base + part for part in sorted(set(_PART.findall(response.text)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Open Targets platform release")
    args = parser.parse_args()

    diseases = wanted_diseases()
    urls = part_urls(args.version)
    print(f"{len(urls)} parquet parts in {DATASET} for release {args.version}")
    print(f"filtering to {len(diseases)} diseases reachable from a CEM condition")

    associations = (
        pl.scan_parquet(urls)
        .select(KEEP_COLUMNS)
        .filter(pl.col("diseaseId").is_in(diseases))
        .rename({"aggregationValue": "datatype", "targetId": "ensembl_gene_id"})
        .unique()
        .sort("diseaseId", "ensembl_gene_id", "datatype")
        .collect()
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    associations.write_parquet(OUT_PATH)

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nwrote {associations.height:,} rows ({size_mb:.1f} MB) -> {OUT_PATH}")
    print(
        f"{associations['diseaseId'].n_unique():,} diseases, "
        f"{associations['ensembl_gene_id'].n_unique():,} genes"
    )
    print("\nrows by datatype:")
    for datatype, count in (
        associations.group_by("datatype").len().sort("len", descending=True).iter_rows()
    ):
        print(f"  {datatype:24s} {count:>9,}")

    PROVENANCE.write_text(
        json.dumps(
            {
                "source": "Open Targets Platform",
                "release": args.version,
                "dataset": DATASET,
                "url": f"{FTP_ROOT}/{args.version}/output/{DATASET}/",
                "n_parts": len(urls),
                "columns_kept": KEEP_COLUMNS,
                "columns_dropped": ["timeseries", "currentNovelty", "aggregationType"],
                "filtered_to": "diseases reachable from a CEM condition via the disease arm",
                "n_diseases_requested": len(diseases),
                "n_diseases_matched": associations["diseaseId"].n_unique(),
                "n_rows_kept": associations.height,
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
