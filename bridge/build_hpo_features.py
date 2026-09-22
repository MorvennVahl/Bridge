"""Build the HPO half of the condition feature layer.

Writes three tables to `data/derived/`:

- `hpo_terms.parquet`  — one row per HPO term: label, synonyms, position in the ontology
- `hpo_xrefs.parquet`  — HPO term to SNOMED/UMLS/MeSH codes, the join key to OMOP
- `hpo_genes.parquet`  — HPO term to NCBI gene, flagged by whether the link is specific

Run with `uv run python -m bridge.build_hpo_features`.
"""

import logging

import polars as pl

from bridge import paths
from bridge.hpo import (
    HpoTerm,
    compute_ancestors,
    compute_depths,
    extract_xrefs,
    information_content,
    invert,
    parse_obo,
    specific_genes,
)

logger = logging.getLogger(__name__)


def load_annotated_genes() -> dict[str, set[int]]:
    """HPO term to its annotated NCBI gene ids, as shipped (already rolled up the tree)."""
    frame = pl.read_csv(
        paths.HPO_PHENOTYPE_TO_GENES,
        separator="\t",
        columns=["hpo_id", "ncbi_gene_id"],
        schema_overrides={"hpo_id": pl.String, "ncbi_gene_id": pl.Int64},
    ).drop_nulls()

    annotated: dict[str, set[int]] = {}
    for hpo_id, gene in frame.iter_rows():
        annotated.setdefault(hpo_id, set()).add(gene)
    logger.info("%d gene annotations over %d HPO terms", frame.height, len(annotated))
    return annotated


def build_terms_table(
    terms: dict[str, HpoTerm],
    ancestors: dict[str, set[str]],
    descendants: dict[str, set[str]],
    depths: dict[str, int],
    annotated: dict[str, set[int]],
    specific: dict[str, set[int]],
    ic: dict[str, float],
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "hpo_id": term.hpo_id,
                "name": term.name,
                "is_obsolete": term.is_obsolete,
                "replaced_by": term.replaced_by,
                "depth": depths.get(term.hpo_id),
                "n_parents": len(term.parents),
                "n_ancestors": len(ancestors.get(term.hpo_id, ())),
                "n_descendants": len(descendants.get(term.hpo_id, ())),
                "n_genes": len(annotated.get(term.hpo_id, ())),
                "n_genes_specific": len(specific.get(term.hpo_id, ())),
                "information_content": ic.get(term.hpo_id),
                "synonyms": term.synonyms,
            }
            for term in terms.values()
        ],
        schema={
            "hpo_id": pl.String,
            "name": pl.String,
            "is_obsolete": pl.Boolean,
            "replaced_by": pl.String,
            "depth": pl.Int32,
            "n_parents": pl.Int32,
            "n_ancestors": pl.Int32,
            "n_descendants": pl.Int32,
            "n_genes": pl.Int32,
            "n_genes_specific": pl.Int32,
            "information_content": pl.Float64,
            "synonyms": pl.List(pl.String),
        },
    )


def build_xrefs_table() -> pl.DataFrame:
    frame = pl.read_parquet(paths.OT_DISEASE_HPO, columns=["id", "dbXRefs"])
    triples = extract_xrefs(frame["id"].to_list(), frame["dbXRefs"].to_list())
    return pl.DataFrame(
        triples,
        schema={"hpo_id": pl.String, "source": pl.String, "code": pl.String},
        orient="row",
    ).unique()


def build_genes_table(
    annotated: dict[str, set[int]], specific: dict[str, set[int]]
) -> pl.DataFrame:
    """One row per (term, gene), flagging genes attached at this term rather than below it."""
    rows = [
        {
            "hpo_id": hpo_id,
            "ncbi_gene_id": gene,
            "is_specific": gene in specific.get(hpo_id, ()),
        }
        for hpo_id, genes in annotated.items()
        for gene in genes
    ]
    return pl.DataFrame(
        rows,
        schema={"hpo_id": pl.String, "ncbi_gene_id": pl.Int64, "is_specific": pl.Boolean},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths.DERIVED.mkdir(parents=True, exist_ok=True)

    terms = parse_obo(paths.HPO_OBO)
    ancestors = compute_ancestors(terms)
    descendants = invert(ancestors)
    depths = compute_depths(terms)

    annotated = load_annotated_genes()
    specific = specific_genes(annotated, terms)
    ic = information_content(annotated)

    terms_table = build_terms_table(terms, ancestors, descendants, depths, annotated, specific, ic)
    xrefs_table = build_xrefs_table()
    genes_table = build_genes_table(annotated, specific)

    terms_table.write_parquet(paths.HPO_TERMS_OUT)
    xrefs_table.write_parquet(paths.HPO_XREFS_OUT)
    genes_table.write_parquet(paths.HPO_GENES_OUT)

    n_terms = terms_table.height
    unreachable = terms_table.filter(pl.col("depth").is_null() & ~pl.col("is_obsolete")).height
    snomed = xrefs_table.filter(pl.col("source") == "SNOMED")

    logger.info("terms            %d rows -> %s", n_terms, paths.HPO_TERMS_OUT.name)
    logger.info("xrefs            %d rows -> %s", xrefs_table.height, paths.HPO_XREFS_OUT.name)
    logger.info("genes            %d rows -> %s", genes_table.height, paths.HPO_GENES_OUT.name)
    logger.info(
        "coverage: %d/%d terms have a SNOMED code (%.1f%%), %d have any xref (%.1f%%)",
        snomed["hpo_id"].n_unique(),
        n_terms,
        100 * snomed["hpo_id"].n_unique() / n_terms,
        xrefs_table["hpo_id"].n_unique(),
        100 * xrefs_table["hpo_id"].n_unique() / n_terms,
    )
    logger.info(
        "coverage: %d/%d terms carry a gene annotation (%.1f%%); %d of those have at least "
        "one gene not inherited from a child",
        len(annotated),
        n_terms,
        100 * len(annotated) / n_terms,
        sum(1 for genes in specific.values() if genes),
    )
    if unreachable:
        logger.warning("%d non-obsolete terms are unreachable from the root", unreachable)


if __name__ == "__main__":
    main()
