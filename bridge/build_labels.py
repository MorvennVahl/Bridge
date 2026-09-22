"""Build the label layer for drug-condition pairs.

Writes to `data/derived/`:

- `pair_labels.parquet`          — directional labels and FAERS signal, one row per pair
- `pair_nuisance_label_adjacent.parquet` — ingredient and condition degree

The nuisance file is named for the repo convention that anything derived from the label
table is not a predictor. Degree is computed from the CEM association table itself, so
joining it in as a feature would leak.

Run with `uv run python -m bridge.build_labels`.
"""

import logging

import polars as pl

from bridge import labels, paths

logger = logging.getLogger(__name__)

KEEP = [
    "ingredient_concept_id",
    "condition_concept_id",
    "efficacy_asserted",
    "efficacy_negated",
    "efficacy_label",
    "efficacy_contradicted",
    "harm_asserted",
    "harm_negated",
    "harm_label",
    "harm_contradicted",
    "nondirectional_asserted",
    "treats_and_causes",
    "faers_case_count",
    "faers_prr",
    "faers_chi_square",
    "faers_signal",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths.DERIVED.mkdir(parents=True, exist_ok=True)

    cem = pl.read_csv(paths.CEM_ASSOCIATIONS, infer_schema_length=0)
    logger.info("read %d CEM pairs", cem.height)

    built = labels.build_labels(cem)
    pair_labels = built.select(
        pl.col("ingredient_concept_id").cast(pl.Int64),
        pl.col("condition_concept_id").cast(pl.Int64),
        *[pl.col(c) for c in KEEP[2:12]],
        pl.col("faers_case_count").cast(pl.Int64),
        pl.col("faers_prr").cast(pl.Float64),
        pl.col("faers_chi_square").cast(pl.Float64),
        pl.col("faers_signal"),
    )
    pair_labels.write_parquet(paths.PAIR_LABELS_OUT)

    ingredient, condition = labels.degrees(cem)
    nuisance = (
        pair_labels.select("ingredient_concept_id", "condition_concept_id")
        .join(
            ingredient.with_columns(pl.col("ingredient_concept_id").cast(pl.Int64)),
            on="ingredient_concept_id",
        )
        .join(
            condition.with_columns(pl.col("condition_concept_id").cast(pl.Int64)),
            on="condition_concept_id",
        )
    )
    nuisance.write_parquet(paths.PAIR_NUISANCE_OUT)

    _report(pair_labels, nuisance)


def _report(pair_labels: pl.DataFrame, nuisance: pl.DataFrame) -> None:
    total = pair_labels.height
    logger.info("labels     %d rows -> %s", total, paths.PAIR_LABELS_OUT.name)
    logger.info("nuisance   %d rows -> %s", nuisance.height, paths.PAIR_NUISANCE_OUT.name)

    for name in ("efficacy", "harm"):
        positive = pair_labels.filter(pl.col(f"{name}_label") == 1).height
        negative = pair_labels.filter(pl.col(f"{name}_label") == 0).height
        contradicted = pair_labels.filter(pl.col(f"{name}_contradicted")).height
        logger.info(
            "%-8s positive %5d   explicit negative %4d   contradicted (label null) %3d",
            name,
            positive,
            negative,
            contradicted,
        )

    both = pair_labels.filter(pl.col("treats_and_causes")).height
    logger.info("%d pairs are asserted to both treat and cause -- legitimate, not an error", both)

    reported = pair_labels.filter(pl.col("faers_signal").is_not_null()).height
    signal = pair_labels.filter(pl.col("faers_signal")).height
    logger.info(
        "FAERS: %d pairs reported, %d clear all three thresholds (%.1f%%); the rest are "
        "reported-without-signal, which is not the same as absent",
        reported,
        signal,
        100 * signal / reported,
    )

    directional = pair_labels.filter(
        pl.col("efficacy_label").is_not_null() | pl.col("harm_label").is_not_null()
    ).height
    logger.info(
        "%d/%d pairs (%.2f%%) carry a directional label -- the rest have FAERS counts only",
        directional,
        total,
        100 * directional / total,
    )


if __name__ == "__main__":
    main()
