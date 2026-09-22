"""Build the condition-side feature tables, keyed by `condition_concept_id`.

This is the join the rest of the plan waits on: the drug side is keyed by ingredient and the
ontology layer by MONDO/HPO id, and nothing connected them to an OMOP condition until now.

Writes four tables to `data/derived/`:

- `condition_ontology_map.parquet` — every (condition, ontology term) match with its tier
- `condition_features.parquet`     — one row per condition, scalar features
- `condition_genes.parquet`        — condition -> gene, the pivot to the drug side
- `condition_ancestors.parquet`    — condition -> SNOMED ancestor, for hierarchy features
- `condition_gene_evidence.parquet` — condition -> gene per Open Targets evidence type,
  with association scores. Excludes `known_drug`; see `conditions.LEAKY_DATATYPES`.

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

    evidence = conditions.gene_evidence(best)
    genes = conditions.condition_genes(best, evidence).with_columns(
        # Curated genes arrive without an Ensembl id; Open Targets ones already have one.
        pl.col("ensembl_gene_id").fill_null(
            pl.col("gene_symbol").replace_strict(
                symbol_to_ensembl(), default=None, return_dtype=pl.String
            )
        )
    )

    scalars, ancestors = conditions.omop_context()
    n_total = scalars.height

    tiers = best.group_by("condition_concept_id").agg(
        pl.col("match_tier").first(),
        # A condition can match terms in both arms at the same tier, so `arm` reports every
        # arm it drew from ("disease", "phenotype" or "disease+phenotype") rather than an
        # arbitrary first one.
        pl.col("arm").unique().sort().str.join("+").alias("arm"),
        pl.col("ontology").unique().sort().str.join("+").alias("ontology"),
        pl.col("ontology_id").n_unique().alias("n_ontology_terms"),
    )
    gene_counts = genes.group_by("condition_concept_id").agg(
        pl.col("gene_symbol").n_unique().alias("n_genes"),
        pl.col("ensembl_gene_id").drop_nulls().n_unique().alias("n_genes_with_ensembl"),
        pl.col("gene_symbol").filter(pl.col("from_curated")).n_unique().alias("n_genes_curated"),
        pl.col("gene_symbol")
        .filter(pl.col("from_open_targets"))
        .n_unique()
        .alias("n_genes_open_targets"),
        pl.col("ot_genetic_score").max().alias("max_genetic_score"),
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
            pl.col("n_genes_curated").fill_null(0),
            pl.col("n_genes_open_targets").fill_null(0),
            pl.col("n_phenotypes").fill_null(0),
        )
        .sort("condition_concept_id")
    )

    mapped.write_parquet(paths.CONDITION_MAP_OUT)
    features.write_parquet(paths.CONDITION_FEATURES_OUT)
    genes.write_parquet(paths.CONDITION_GENES_OUT)
    ancestors.write_parquet(paths.CONDITION_ANCESTORS_OUT)
    evidence.write_parquet(paths.CONDITION_GENE_EVIDENCE_OUT)

    _report(features, genes, evidence, n_total)


def _report(
    features: pl.DataFrame, genes: pl.DataFrame, evidence: pl.DataFrame, n_total: int
) -> None:
    logger.info("features   %d rows -> %s", features.height, paths.CONDITION_FEATURES_OUT.name)
    logger.info("genes      %d rows -> %s", genes.height, paths.CONDITION_GENES_OUT.name)
    logger.info("evidence   %d rows -> %s", evidence.height, paths.CONDITION_GENE_EVIDENCE_OUT.name)

    logger.info("match tier breakdown:")
    for tier, count in features.group_by("match_tier").len().sort("match_tier").iter_rows():
        logger.info("  %-14s %5d  (%.1f%%)", tier, count, 100 * count / n_total)

    for column, label in (
        ("n_genes", "any source"),
        ("n_genes_curated", "HPO/OMIM/Orphanet only"),
        ("n_genes_open_targets", "Open Targets only"),
    ):
        count = features.filter(pl.col(column) > 0).height
        logger.info(
            "  %-24s %5d/%d conditions carry a gene (%.1f%%)",
            label,
            count,
            n_total,
            100 * count / n_total,
        )
    if evidence.height:
        logger.info(
            "Open Targets evidence rows by datatype (%s excluded as label-leaking):",
            ", ".join(sorted(conditions.LEAKY_DATATYPES)),
        )
        for datatype, count in (
            evidence.group_by("datatype").len().sort("len", descending=True).iter_rows()
        ):
            logger.info("    %-22s %8d", datatype, count)

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
