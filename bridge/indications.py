"""Map ChEMBL drug indications onto OMOP ingredient-condition pairs.

The efficacy half of this project has been running on ~7,900 SemMedDB literature
assertions because `in_eu_label` is empty on every CEM row. ChEMBL's curated
`drug_indication` table is a real alternative: 60,055 records over 10,073 molecules, of
which 8,683 reached `max_phase_for_ind = 4`, meaning an approved indication rather than
a trialled one.

Both sides need mapping, and each has two independent routes so that neither depends on a
single fragile join:

- **drug** — ChEMBL molecule id against `chembl_id`, then against `chembl_parent_id`,
  since CEM ingredients are parent molecules while indications are often recorded on a
  salt or specific form.
- **condition** — the indication's ontology id (EFO, MONDO, HP, Orphanet) against the
  condition ontology map, and the indication's MeSH id against both the OMOP source-code
  export and MONDO's MeSH cross-references.

LEAKAGE. Open Targets' `known_drug` / `dt_clinical` evidence is built from this same
ChEMBL table. A design must not use both.
"""

import logging

import polars as pl

from bridge import hpo, paths

logger = logging.getLogger(__name__)

APPROVED_PHASE = 4.0

#: Above this many descendants, an ontology term is a category rather than a disease, and
#: an indication joined through it lands on everything beneath it.
#:
#: AGENT.md trap 5 warns that OMOP2OBO's one-to-many rows attach a concept to broad
#: parents. Filtering on the tier alone is the wrong cut: it drops atorvastatin's false
#: "Soft tissue infection" (via MeSH "Cardiovascular Diseases") but also warfarin's correct
#: deep-vein-thrombosis mapping, which is a legitimate one-to-many. Fan-out does not
#: separate them either — "cardiovascular disease" is claimed by 3 conditions while the
#: sound "thrombotic disease" is claimed by 9. Generality of the term itself does.
#:
#: This is a default, not a verdict. Every row carries `ontology_n_descendants`, so a
#: consumer can set its own threshold or ignore this one.
MAX_TERM_DESCENDANTS = 200


def _normalise(expr: pl.Expr) -> pl.Expr:
    """`EFO:0003898` -> `EFO_0003898`, the form the condition ontology map uses."""
    return expr.str.replace(":", "_")


def load_indications() -> pl.DataFrame:
    return pl.read_csv(paths.CHEMBL_INDICATIONS, infer_schema_length=0).with_columns(
        pl.col("max_phase_for_ind").cast(pl.Float64),
        pl.col("n_refs").cast(pl.Int32),
        _normalise(pl.col("efo_id")).alias("ontology_id"),
    )


def drug_map() -> pl.DataFrame:
    """ChEMBL molecule id -> `ingredient_concept_id`, via both the id and the parent id."""
    features = pl.read_csv(
        paths.INGREDIENT_FEATURES,
        columns=["omop_concept_id", "chembl_id", "chembl_parent_id"],
        infer_schema_length=0,
    ).with_columns(pl.col("omop_concept_id").cast(pl.Float64).cast(pl.Int64))

    direct = features.select(
        pl.col("chembl_id").alias("molecule_chembl_id"),
        pl.col("omop_concept_id").alias("ingredient_concept_id"),
    )
    parent = features.select(
        pl.col("chembl_parent_id").alias("molecule_chembl_id"),
        pl.col("omop_concept_id").alias("ingredient_concept_id"),
    )
    return pl.concat([direct, parent]).drop_nulls().unique()


def term_generality() -> pl.DataFrame:
    """Ontology term -> number of descendants, the measure of how broad the term is.

    MONDO/EFO/Orphanet counts come from Open Targets' disease index, HPO counts from
    hp.obo. A term with thousands of descendants is a branch of the ontology, not a
    diagnosis, and an indication attached to it belongs to none of the conditions
    underneath in particular.
    """
    ot = pl.read_parquet(paths.OT_DISEASE, columns=["id", "descendants"]).select(
        pl.col("id").alias("ontology_id"),
        pl.col("descendants").list.len().fill_null(0).cast(pl.Int32).alias("n_descendants"),
    )

    terms = hpo.parse_obo(paths.HPO_OBO)
    descendants = hpo.invert(hpo.compute_ancestors(terms))
    hpo_counts = pl.DataFrame(
        {
            "ontology_id": [t.replace(":", "_") for t in terms],
            "n_descendants": [len(descendants.get(t, ())) for t in terms],
        },
        schema={"ontology_id": pl.String, "n_descendants": pl.Int32},
    )
    return pl.concat([ot, hpo_counts]).unique(subset=["ontology_id"], keep="first")


def condition_by_ontology() -> pl.DataFrame:
    """Ontology term -> `condition_concept_id`, with the term's generality attached.

    Terms broader than `MAX_TERM_DESCENDANTS` are dropped: joining an indication through
    "cardiovascular disease" puts it on every condition that lists it as a category.
    """
    frame = (
        pl.read_csv(paths.CONDITION_ONTOLOGY_MAP, infer_schema_length=0)
        .select(
            pl.col("condition_concept_id").cast(pl.Int64),
            "ontology_id",
            pl.col("match_tier").alias("condition_match_tier"),
        )
        .drop_nulls("ontology_id")
    )

    joined = frame.join(term_generality(), on="ontology_id", how="left")
    kept = joined.filter(
        pl.col("n_descendants").is_null() | (pl.col("n_descendants") <= MAX_TERM_DESCENDANTS)
    )
    logger.info(
        "condition ontology map: %d rows, %d kept after dropping terms with >%d descendants",
        frame.height,
        kept.height,
        MAX_TERM_DESCENDANTS,
    )
    return kept.rename({"n_descendants": "ontology_n_descendants"}).unique()


def condition_by_mesh() -> pl.DataFrame:
    """MeSH id -> `condition_concept_id`, from the OMOP export and MONDO's xrefs.

    Two sources because neither alone is enough: the OMOP source-code export carries MeSH
    for only 2,117 conditions, while MONDO's MeSH cross-references reach 8,183 terms but
    have to be walked back through the ontology map to reach a condition.
    """
    from_omop = (
        pl.read_csv(paths.OMOP_SOURCE_CODES, infer_schema_length=0)
        .filter(pl.col("source_vocabulary_id") == "MeSH")
        .select(
            pl.col("condition_concept_id").cast(pl.Int64),
            pl.col("source_concept_code").alias("mesh_id"),
        )
    )

    sssom = (
        pl.read_csv(paths.MONDO_SSSOM, separator="\t", comment_prefix="#", infer_schema_length=0)
        .filter(
            (pl.col("predicate_id") == "skos:exactMatch")
            & pl.col("object_id").str.starts_with("mesh:")
        )
        .select(
            _normalise(pl.col("subject_id")).alias("ontology_id"),
            pl.col("object_id").str.replace("mesh:", "").alias("mesh_id"),
        )
    )
    from_mondo = (
        sssom.join(condition_by_ontology(), on="ontology_id")
        .select("condition_concept_id", "mesh_id")
        .unique()
    )

    return pl.concat([from_omop, from_mondo]).drop_nulls().unique()


def build_pairs(indications: pl.DataFrame) -> pl.DataFrame:
    """One row per (ingredient, condition) with an indication, tagged by how it matched."""
    drugs = drug_map()
    with_drug = indications.join(drugs, on="molecule_chembl_id")
    logger.info(
        "%d of %d indication records reach an OMOP ingredient",
        with_drug.height,
        indications.height,
    )

    via_ontology = (
        with_drug.join(condition_by_ontology(), on="ontology_id")
        .select(
            "ingredient_concept_id",
            "condition_concept_id",
            "max_phase_for_ind",
            "n_refs",
            "ontology_n_descendants",
        )
        .with_columns(pl.lit("ontology").alias("route"))
    )
    via_mesh = (
        with_drug.join(condition_by_mesh(), on="mesh_id")
        .select(
            "ingredient_concept_id",
            "condition_concept_id",
            "max_phase_for_ind",
            "n_refs",
            pl.lit(None, dtype=pl.Int32).alias("ontology_n_descendants"),
        )
        .with_columns(pl.lit("mesh").alias("route"))
    )
    logger.info("ontology route %d rows, mesh route %d rows", via_ontology.height, via_mesh.height)

    # A pair can be reached by both routes; keep the strongest phase and record which
    # routes found it, since agreement across two independent ontologies is worth knowing.
    return (
        pl.concat([via_ontology, via_mesh])
        .group_by("ingredient_concept_id", "condition_concept_id")
        .agg(
            pl.col("max_phase_for_ind").max().alias("max_phase"),
            pl.col("n_refs").max().alias("n_refs"),
            pl.col("route").unique().sort().str.join("+").alias("matched_via"),
            pl.col("ontology_n_descendants").min(),
        )
        .with_columns((pl.col("max_phase") >= APPROVED_PHASE).alias("is_approved"))
        .sort("ingredient_concept_id", "condition_concept_id")
    )
