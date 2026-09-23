"""Build the indication label layer and report how far it reaches.

Writes `data/derived/pair_indications.parquet`: one row per OMOP ingredient-condition pair
that ChEMBL records an indication for, with the highest clinical phase reached.

The number that matters is how many of these pairs are also in the CEM table, because only
those can be joined to the existing features and splits. Reported against the 7,223
SemMedDB efficacy positives this is meant to supplement.

Run with `uv run python -m bridge.build_indications`.
"""

import logging

import polars as pl

from bridge import indications, paths

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths.DERIVED.mkdir(parents=True, exist_ok=True)

    records = indications.load_indications()
    logger.info("%d ChEMBL indication records", records.height)

    pairs = indications.build_pairs(records)
    pairs.write_parquet(paths.PAIR_INDICATIONS_OUT)
    logger.info("pairs  %d rows -> %s", pairs.height, paths.PAIR_INDICATIONS_OUT.name)

    _report(pairs)


def _report(pairs: pl.DataFrame) -> None:
    approved = pairs.filter(pl.col("is_approved"))
    logger.info(
        "%d pairs, %d approved (max_phase 4), over %d ingredients and %d conditions",
        pairs.height,
        approved.height,
        pairs["ingredient_concept_id"].n_unique(),
        pairs["condition_concept_id"].n_unique(),
    )
    logger.info("matched via:")
    for route, count in (
        pairs.group_by("matched_via").len().sort("len", descending=True).iter_rows()
    ):
        logger.info("  %-14s %6d", route, count)

    cem = pl.read_csv(
        paths.CEM_ASSOCIATIONS,
        columns=["ingredient_concept_id", "condition_concept_id"],
        infer_schema_length=0,
    ).with_columns(
        pl.col("ingredient_concept_id").cast(pl.Int64),
        pl.col("condition_concept_id").cast(pl.Int64),
    )
    in_cem = pairs.join(cem.unique(), on=["ingredient_concept_id", "condition_concept_id"])
    in_cem_approved = in_cem.filter(pl.col("is_approved"))

    logger.info(
        "%d/%d indication pairs (%.1f%%) are also CEM pairs -- only these join to the "
        "existing features and splits",
        in_cem.height,
        pairs.height,
        100 * in_cem.height / pairs.height,
    )
    logger.info("  of those, %d are approved indications", in_cem_approved.height)

    if paths.PAIR_LABELS_OUT.exists():
        labels = pl.read_parquet(paths.PAIR_LABELS_OUT)
        semmeddb_positive = labels.filter(pl.col("efficacy_label") == 1).height
        overlap = in_cem.join(
            labels.filter(pl.col("efficacy_label") == 1),
            on=["ingredient_concept_id", "condition_concept_id"],
        ).height
        logger.info(
            "against the SemMedDB efficacy label: %d positives there, %d overlap with this "
            "layer, so %d CEM pairs gain an efficacy signal they did not have",
            semmeddb_positive,
            overlap,
            in_cem.height - overlap,
        )
    else:
        logger.warning(
            "%s absent; run `python -m bridge.build_labels` to compare against SemMedDB",
            paths.PAIR_LABELS_OUT.name,
        )


if __name__ == "__main__":
    main()
