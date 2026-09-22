"""Join the ontology feature layer onto OMOP condition concepts.

`bridge.hpo` and `bridge.disease` produce features keyed by ontology id; `condition_map`
maps OMOP conditions onto those ids. This module is the join between them, and it is what
finally gives the condition side of a drug-condition pair a feature vector.

The output that matters most is `condition_genes`: condition -> gene. The drug side already
has ingredient -> target -> gene, so genes are the pivot on which the two halves of the
model meet.

One row per condition is kept at the best available match tier. A condition can map to
several ontology terms at that tier, in which case its genes are the union over them and
its scalar features are the maximum — a condition is at least as specific, and at least as
gene-rich, as the best term it maps to.
"""

import logging

import pandas as pd
import polars as pl

from bridge import condition_map, paths
from bridge.genes import ensembl_to_symbol
from bridge.xrefs import normalise_id

logger = logging.getLogger(__name__)

PHENOTYPE_ARM = "phenotype"
DISEASE_ARM = "disease"

#: Open Targets evidence types that must never become model features.
#:
#: `known_drug` is derived from ChEMBL drug indications and clinical trials: it says
#: "a drug hitting this target is developed for this disease". Our label is whether a drug
#: treats or causes a condition, so feeding it in would let the model read the answer off
#: its own input. The other datatypes (genetic_association, animal_model, literature,
#: somatic_mutation, rna_expression, affected_pathway, genetic_literature) describe
#: gene-disease biology rather than drug-disease outcomes, and are safe.
LEAKY_DATATYPES = frozenset({"known_drug"})


def run_mapping() -> pl.DataFrame:
    """Run the tiered mapper over the 5,631 CEM conditions.

    Returns every (condition, ontology term) match with its tier, ontology ids normalised to
    the colon form (`MONDO:0009061`) that the derived tables use.
    """
    ot = condition_map.load_open_targets_diseases()
    hpo = condition_map.load_hpo_terms()
    sssom = condition_map.load_mondo_sssom()
    codes = condition_map.condition_concept_codes()

    # Built column by column rather than by slicing, so the result is unambiguously a
    # DataFrame; pandas' __getitem__ widens to Series | DataFrame under pyright.
    raw = condition_map.load_condition_concepts()
    concepts = pd.DataFrame(
        {
            "condition_concept_id": raw["condition_concept_id"],
            "condition_name": raw["concept_name"],
        }
    )

    label, token, sctid, meta = condition_map.build_indexes(ot, hpo, sssom)
    mapped = condition_map.map_conditions(
        concepts,
        label,
        token,
        sctid,
        meta,
        concept_codes=codes,
        synonyms=condition_map.condition_synonyms(),
    )

    frame = pl.from_pandas(mapped.astype({"condition_concept_id": "int64"}))
    return frame.with_columns(
        pl.col("ontology_id")
        .map_elements(lambda v: normalise_id(v) if v else None, return_dtype=pl.String)
        .alias("ontology_id")
    )


def best_tier_matches(mapped: pl.DataFrame) -> pl.DataFrame:
    """Keep every match at each condition's best tier, dropping worse tiers for that condition.

    `condition_map.coverage_report` keeps only one row per condition; here all terms at the
    winning tier are retained, because a condition that maps to three MONDO terms should
    draw genes from all three.
    """
    best = mapped.group_by("condition_concept_id").agg(pl.col("match_tier").min().alias("_best"))
    return (
        mapped.join(best, on="condition_concept_id")
        .filter(pl.col("match_tier") == pl.col("_best"))
        .drop("_best")
    )


def gene_evidence(matches: pl.DataFrame) -> pl.DataFrame:
    """Condition -> gene weighted by Open Targets evidence, one row per datatype.

    Returns an empty frame when the Open Targets association file has not been fetched, so
    the rest of the build still runs; see `scripts/fetch_opentargets_associations.py`.

    `known_drug` is dropped here rather than at fetch time: the reference file stays
    faithful to Open Targets, and the exclusion lives next to the reason for it.
    """
    schema = {
        "condition_concept_id": pl.Int64,
        "ensembl_gene_id": pl.String,
        "datatype": pl.String,
        "association_score": pl.Float64,
        "evidence_count": pl.Int64,
    }
    if not paths.OT_ASSOCIATIONS.exists():
        logger.warning(
            "%s not found; condition genes will come from HPO only. Run "
            "scripts/fetch_opentargets_associations.py to add the weighted edge.",
            paths.OT_ASSOCIATIONS.name,
        )
        return pl.DataFrame(schema=schema)

    associations = pl.read_parquet(paths.OT_ASSOCIATIONS).with_columns(
        pl.col("diseaseId").str.replace("_", ":").alias("ontology_id")
    )
    return (
        matches.filter(pl.col("arm") == DISEASE_ARM)
        .join(associations, on="ontology_id")
        .filter(~pl.col("datatype").is_in(list(LEAKY_DATATYPES)))
        .group_by("condition_concept_id", "ensembl_gene_id", "datatype")
        .agg(
            pl.col("associationScore").max().alias("association_score"),
            pl.col("evidenceCount").sum().alias("evidence_count"),
        )
        .cast(schema)  # type: ignore[arg-type]
    )


def condition_genes(matches: pl.DataFrame, evidence: pl.DataFrame) -> pl.DataFrame:
    """Condition -> gene, unioned across every source that implicates the gene.

    `gene_symbol` is the canonical key because it is the only identifier both this side and
    the drug side carry. `ncbi_gene_id` is null for genes that only Open Targets knows
    about, and `ot_max_score` is null for genes only HPO knows about.
    """
    hpo_genes = pl.read_parquet(paths.HPO_GENES_OUT)
    disease_genes = pl.read_parquet(paths.DISEASE_GENES_OUT)

    phenotype = (
        matches.filter(pl.col("arm") == PHENOTYPE_ARM)
        .join(hpo_genes, left_on="ontology_id", right_on="hpo_id")
        .select("condition_concept_id", "ncbi_gene_id", "gene_symbol")
    )
    disease = (
        matches.filter(pl.col("arm") == DISEASE_ARM)
        .join(disease_genes, left_on="ontology_id", right_on="disease_id")
        .select("condition_concept_id", "ncbi_gene_id", "gene_symbol")
    )
    curated = (
        pl.concat([phenotype, disease])
        .drop_nulls("gene_symbol")
        .unique(subset=["condition_concept_id", "gene_symbol"], keep="first")
        .with_columns(pl.lit(True).alias("from_curated"))
    )

    scored = (
        evidence.group_by("condition_concept_id", "ensembl_gene_id")
        .agg(
            pl.col("association_score").max().alias("ot_max_score"),
            pl.col("association_score")
            .filter(pl.col("datatype") == "genetic_association")
            .max()
            .alias("ot_genetic_score"),
            pl.col("datatype").n_unique().alias("n_ot_datatypes"),
        )
        .with_columns(
            pl.col("ensembl_gene_id")
            .replace_strict(ensembl_to_symbol(), default=None, return_dtype=pl.String)
            .alias("gene_symbol")
        )
        .drop_nulls("gene_symbol")
        # Several Ensembl ids can share an approved symbol, so collapse again on the symbol
        # or the join below multiplies rows.
        .group_by("condition_concept_id", "gene_symbol")
        .agg(
            pl.col("ensembl_gene_id").sort().first(),
            pl.col("ot_max_score").max(),
            pl.col("ot_genetic_score").max(),
            pl.col("n_ot_datatypes").max(),
        )
        .with_columns(pl.lit(True).alias("from_open_targets"))
    )

    return (
        curated.join(scored, on=["condition_concept_id", "gene_symbol"], how="full", coalesce=True)
        .with_columns(
            pl.col("from_curated").fill_null(False),
            pl.col("from_open_targets").fill_null(False),
            pl.col("n_ot_datatypes").fill_null(0),
        )
        .select(
            "condition_concept_id",
            "gene_symbol",
            "ncbi_gene_id",
            "ensembl_gene_id",
            "from_curated",
            "from_open_targets",
            "ot_max_score",
            "ot_genetic_score",
            "n_ot_datatypes",
        )
    )


def ontology_scalars(matches: pl.DataFrame) -> pl.DataFrame:
    """Per-condition scalars taken from the mapped ontology terms."""
    hpo_terms = pl.read_parquet(paths.HPO_TERMS_OUT).select(
        pl.col("hpo_id").alias("ontology_id"),
        pl.col("depth").alias("ontology_depth"),
        pl.col("n_ancestors").alias("n_ontology_ancestors"),
        pl.col("n_descendants").alias("n_ontology_descendants"),
        "information_content",
    )
    disease_terms = pl.read_parquet(paths.DISEASE_TERMS_OUT).select(
        pl.col("disease_id").alias("ontology_id"),
        pl.lit(None, dtype=pl.Int32).alias("ontology_depth"),
        pl.col("n_ancestors").alias("n_ontology_ancestors"),
        pl.col("n_descendants").alias("n_ontology_descendants"),
        pl.lit(None, dtype=pl.Float64).alias("information_content"),
    )
    terms = pl.concat([hpo_terms, disease_terms])

    return (
        matches.join(terms, on="ontology_id", how="left")
        .group_by("condition_concept_id")
        .agg(
            pl.col("ontology_depth").max(),
            pl.col("n_ontology_ancestors").max(),
            pl.col("n_ontology_descendants").max(),
            pl.col("information_content").max(),
        )
    )


def phenotype_counts(matches: pl.DataFrame) -> pl.DataFrame:
    """How many HPO phenotypes each disease-arm condition inherits, via `disease_phenotypes`."""
    links = pl.read_parquet(paths.DISEASE_PHENOTYPES_OUT)
    return (
        matches.filter(pl.col("arm") == DISEASE_ARM)
        .join(links, left_on="ontology_id", right_on="disease_id")
        .group_by("condition_concept_id")
        .agg(pl.col("hpo_id").n_unique().alias("n_phenotypes"))
    )


def omop_context() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-condition OMOP scalars and the long condition-to-ancestor table.

    `concept_ancestor` includes each concept as its own ancestor at distance zero; that row
    is dropped so `n_omop_ancestors` counts genuine ancestors only.
    """
    concepts = pl.read_csv(paths.OMOP_CONCEPT, infer_schema_length=0).select(
        pl.col("condition_concept_id").cast(pl.Int64),
        "concept_name",
        "concept_code",
        "concept_class_id",
    )
    ancestors = (
        pl.read_csv(paths.OMOP_CONCEPT_ANCESTOR, infer_schema_length=0)
        .with_columns(
            pl.col("condition_concept_id").cast(pl.Int64),
            pl.col("ancestor_concept_id").cast(pl.Int64),
            pl.col("min_levels_of_separation").cast(pl.Int32),
        )
        .filter(pl.col("min_levels_of_separation") > 0)
    )
    synonyms = (
        pl.read_csv(paths.OMOP_CONCEPT_SYNONYM, infer_schema_length=0)
        .with_columns(pl.col("condition_concept_id").cast(pl.Int64))
        .group_by("condition_concept_id")
        .agg(pl.len().alias("n_synonyms"))
    )
    summary = ancestors.group_by("condition_concept_id").agg(
        pl.len().alias("n_omop_ancestors"),
        pl.col("min_levels_of_separation").max().alias("omop_depth"),
    )
    scalars = concepts.join(summary, on="condition_concept_id", how="left").join(
        synonyms, on="condition_concept_id", how="left"
    )
    return scalars, ancestors.select(
        "condition_concept_id",
        "ancestor_concept_id",
        "ancestor_name",
        "min_levels_of_separation",
    )


def drug_side_reachable_genes() -> set[str]:
    """Gene symbols that appear on the drug side, for measuring how much of it can join."""
    if not paths.INGREDIENT_TARGETS.exists():
        logger.warning("%s missing; cannot measure drug-side overlap", paths.INGREDIENT_TARGETS)
        return set()
    frame = pl.read_csv(
        paths.INGREDIENT_TARGETS, columns=["gene_symbol"], infer_schema_length=0
    ).drop_nulls()
    return set(frame["gene_symbol"].to_list())
