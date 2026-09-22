"""Build the condition-side feature tables, keyed by `condition_concept_id`.

This is the join the rest of the plan waits on: the drug side is keyed by ingredient and the
ontology layer by MONDO/HPO id, and nothing connected them to an OMOP condition until now.

Writes four tables to `data/derived/`:

- `condition_ontology_map.parquet` — every (condition, ontology term) match with its tier
- `condition_features.parquet`     — one row per condition, scalar features
- `condition_genes.parquet`        — condition -> gene, the pivot to the drug side
- `condition_ancestors.parquet`    — condition -> SNOMED ancestor, for hierarchy features

Run with `uv run python -m bridge.build_condition_features`.
"""

import logging

import polars as pl

from bridge import conditions, paths
from bridge.genes import symbol_to_ensembl

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths.DERIVED.mkdir(parents=True, exist_ok=True)

    logger.info("mapping conditions onto ontology terms (this reads the full HPO and OT indexes)")
    mapped = conditions.run_mapping()
    matched = mapped.filter(pl.col("ontology_id").is_not_null())
    best = conditions.best_tier_matches(matched)

    genes = conditions.condition_genes(best).with_columns(
        pl.col("gene_symbol")
        .replace_strict(symbol_to_ensembl(), default=None, return_dtype=pl.String)
        .alias("ensembl_gene_id")
    )

    scalars, ancestors = conditions.omop_context()
    n_total = scalars.height

    tiers = best.group_by("condition_concept_id").agg(
        pl.col("match_tier").first(),
        pl.col("arm").first(),
        pl.col("ontology").first(),
        pl.col("ontology_id").n_unique().alias("n_ontology_terms"),
    )
    gene_counts = genes.group_by("condition_concept_id").agg(
        pl.col("ncbi_gene_id").n_unique().alias("n_genes"),
        pl.col("ensembl_gene_id").drop_nulls().n_unique().alias("n_genes_with_ensembl"),
    )

    features = (
        scalars.join(tiers, on="condition_concept_id", how="left")
        .join(conditions.ontology_scalars(best), on="condition_concept_id", how="left")
        .join(conditions.phenotype_counts(best), on="condition_concept_id", how="left")
        .join(gene_counts, on="condition_concept_id", how="left")
        .with_columns(
            pl.col("match_tier").fill_null("0_unmatched"),
            pl.col("n_ontology_terms").fill_null(0),
            pl.col("n_genes").fill_null(0),
            pl.col("n_genes_with_ensembl").fill_null(0),
            pl.col("n_phenotypes").fill_null(0),
        )
        .sort("condition_concept_id")
    )

    mapped.write_parquet(paths.CONDITION_MAP_OUT)
    features.write_parquet(paths.CONDITION_FEATURES_OUT)
    genes.write_parquet(paths.CONDITION_GENES_OUT)
    ancestors.write_parquet(paths.CONDITION_ANCESTORS_OUT)

    _report(features, genes, n_total)


def _report(features: pl.DataFrame, genes: pl.DataFrame, n_total: int) -> None:
    logger.info("features   %d rows -> %s", features.height, paths.CONDITION_FEATURES_OUT.name)
    logger.info("genes      %d rows -> %s", genes.height, paths.CONDITION_GENES_OUT.name)

    logger.info("match tier breakdown:")
    for tier, count in features.group_by("match_tier").len().sort("match_tier").iter_rows():
        logger.info("  %-14s %5d  (%.1f%%)", tier, count, 100 * count / n_total)

    with_genes = features.filter(pl.col("n_genes") > 0).height
    logger.info(
        "%d/%d conditions (%.1f%%) carry at least one gene",
        with_genes,
        n_total,
        100 * with_genes / n_total,
    )

    drug_genes = conditions.drug_side_reachable_genes()
    if drug_genes:
        joinable = (
            genes.filter(pl.col("gene_symbol").is_in(list(drug_genes)))["condition_concept_id"]
            .unique()
            .len()
        )
        logger.info(
            "%d/%d conditions (%.1f%%) share at least one gene with a drug target -- "
            "these are the pairs the model can actually reason about",
            joinable,
            n_total,
            100 * joinable / n_total,
        )


if __name__ == "__main__":
    main()
