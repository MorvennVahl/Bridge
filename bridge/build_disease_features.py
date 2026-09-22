"""Build the disease half of the condition feature layer.

Writes four tables to `data/derived/`:

- `disease_terms.parquet`       — one row per disease: label, hierarchy, therapeutic areas
- `disease_xrefs.parquet`       — disease to SNOMED/OMIM/Orphanet/MeSH codes, the OMOP key
- `disease_genes.parquet`       — disease to NCBI gene, via OMIM and Orphanet
- `disease_phenotypes.parquet`  — disease to HPO term, joining this half to `bridge.hpo`

Run with `uv run python -m bridge.build_disease_features`.
"""

import logging

import polars as pl

from bridge import paths
from bridge.disease import (
    DiseaseTerm,
    add_sssom_terms,
    build_terms,
    coverage,
    disease_phenotype_pairs,
    genes_by_disease,
    xrefs_from_open_targets,
    xrefs_from_sssom,
)
from bridge.genes import load_gene_symbols
from bridge.xrefs import OMOP_REACHABLE

logger = logging.getLogger(__name__)


def load_sssom() -> pl.DataFrame:
    return pl.read_csv(paths.MONDO_SSSOM, separator="\t", comment_prefix="#", infer_schema_length=0)


def build_terms_table(terms: dict[str, DiseaseTerm]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "disease_id": term.disease_id,
                "name": term.name,
                "in_open_targets": term.in_open_targets,
                "n_parents": len(term.parents),
                "n_ancestors": len(term.ancestors),
                "n_descendants": len(term.descendants),
                "therapeutic_areas": term.therapeutic_areas,
            }
            for term in terms.values()
        ],
        schema={
            "disease_id": pl.String,
            "name": pl.String,
            "in_open_targets": pl.Boolean,
            "n_parents": pl.Int32,
            "n_ancestors": pl.Int32,
            "n_descendants": pl.Int32,
            "therapeutic_areas": pl.List(pl.String),
        },
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths.DERIVED.mkdir(parents=True, exist_ok=True)

    ot = pl.read_parquet(
        paths.OT_DISEASE,
        columns=["id", "name", "dbXRefs", "parents", "ancestors", "therapeuticAreas"],
    )
    sssom = load_sssom()

    terms = build_terms(
        ot["id"].to_list(),
        ot["name"].to_list(),
        ot["parents"].to_list(),
        ot["ancestors"].to_list(),
        ot["therapeuticAreas"].to_list(),
    )
    added = add_sssom_terms(terms, sssom["subject_id"].to_list(), sssom["subject_label"].to_list())
    logger.info("%d disease terms (%d added from SSSOM alone)", len(terms), added)

    xrefs = xrefs_from_open_targets(ot["id"].to_list(), ot["dbXRefs"].to_list()) + xrefs_from_sssom(
        sssom["subject_id"].to_list(),
        sssom["object_id"].to_list(),
        sssom["predicate_id"].to_list(),
    )

    g2d = pl.read_csv(paths.HPO_GENES_TO_DISEASE, separator="\t", infer_schema_length=0)
    genes = genes_by_disease(
        xrefs,
        g2d["disease_id"].to_list(),
        [int(v.removeprefix("NCBIGene:")) for v in g2d["ncbi_gene_id"].to_list()],
        g2d["association_type"].to_list(),
    )

    dp = pl.read_parquet(paths.OT_DISEASE_PHENOTYPE, columns=["disease", "phenotype"])
    pairs = disease_phenotype_pairs(dp["disease"].to_list(), dp["phenotype"].to_list())

    terms_table = build_terms_table(terms)
    xrefs_table = pl.DataFrame(
        xrefs,
        schema={"disease_id": pl.String, "source": pl.String, "code": pl.String},
        orient="row",
    ).unique()
    symbols = load_gene_symbols()
    genes_table = (
        pl.DataFrame(
            genes,
            schema={
                "disease_id": pl.String,
                "ncbi_gene_id": pl.Int64,
                "association_type": pl.String,
            },
            orient="row",
        )
        .unique()
        # The drug-side tables key on symbol, not NCBI id, so carry both.
        .with_columns(
            pl.col("ncbi_gene_id")
            .replace_strict(symbols, default=None, return_dtype=pl.String)
            .alias("gene_symbol")
        )
    )
    pheno_table = pl.DataFrame(
        pairs, schema={"disease_id": pl.String, "hpo_id": pl.String}, orient="row"
    ).unique()

    terms_table.write_parquet(paths.DISEASE_TERMS_OUT)
    xrefs_table.write_parquet(paths.DISEASE_XREFS_OUT)
    genes_table.write_parquet(paths.DISEASE_GENES_OUT)
    pheno_table.write_parquet(paths.DISEASE_PHENOTYPES_OUT)

    stats = coverage(terms, xrefs)
    logger.info("terms            %d rows -> %s", terms_table.height, paths.DISEASE_TERMS_OUT.name)
    logger.info("xrefs            %d rows -> %s", xrefs_table.height, paths.DISEASE_XREFS_OUT.name)
    logger.info("genes            %d rows -> %s", genes_table.height, paths.DISEASE_GENES_OUT.name)
    logger.info(
        "phenotypes       %d rows -> %s", pheno_table.height, paths.DISEASE_PHENOTYPES_OUT.name
    )

    n = stats["n_terms"]
    logger.info(
        "%d/%d terms come from Open Targets and have a hierarchy", stats["n_open_targets"], n
    )
    for source, count in stats["by_source"].items():
        marker = "*" if source in OMOP_REACHABLE else " "
        logger.info("  %s %-10s %5d terms (%.1f%%)", marker, source, count, 100 * count / n)
    logger.info("(* = reachable from OMOP, so usable as a condition join key)")

    joinable = {d for d, s, _ in xrefs if s in OMOP_REACHABLE}
    logger.info(
        "%d/%d terms (%.1f%%) carry at least one OMOP-reachable code",
        len(joinable),
        n,
        100 * len(joinable) / n,
    )
    logger.info("genes cover %d distinct diseases", genes_table["disease_id"].n_unique())
    logger.info("phenotype links cover %d distinct diseases", pheno_table["disease_id"].n_unique())


if __name__ == "__main__":
    main()
