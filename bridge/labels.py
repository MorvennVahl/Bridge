"""Turn the CEM association file into directional labels for drug-condition pairs.

The CEM file mixes two very different things in one table, and they become two different
labels:

- **SemMedDB** carries the only *directional* signal — whether the literature asserts a
  drug treats a condition or causes it. It is thin (10,394 of 1.45M pairs) but it is the
  only thing that can answer "will this drug help".
- **FAERS** carries disproportionality statistics for 1.44M pairs. These are a safety
  signal only, and mostly noise at the row level: the median case count is 2 and the
  median PRR 1.36.

Three rules shape what this module will and will not emit.

**Absence is not evidence of safety.** A pair missing from the file can mean "no effect",
"nobody ever prescribed it" or "nobody reported it". Only pairs with evidence get a row;
negative sampling is a modelling decision and is deliberately left to the caller. The
explicit `NEG_*` assertions are the only genuine negatives in the data.

**Contradictions are surfaced, not resolved.** A pair can carry both `TREATS` and
`NEG_TREATS`. Rather than voting on sentence counts — which measure how often something was
written, not how true it is — such pairs get a null label and a flag.

**Treating and causing are not opposites.** A drug can legitimately both treat a condition
and cause it, so efficacy and harm are separate labels rather than one three-way class.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)

EFFICACY_POSITIVE = ("TREATS", "PREVENTS")
EFFICACY_NEGATED = ("NEG_TREATS", "NEG_PREVENTS")
HARM_POSITIVE = ("CAUSES", "PREDISPOSES", "COMPLICATES")
HARM_NEGATED = ("NEG_CAUSES", "NEG_PREDISPOSES", "NEG_COMPLICATES")
NONDIRECTIONAL = ("AFFECTS", "ASSOCIATED_WITH", "DISRUPTS")
NONDIRECTIONAL_NEGATED = ("NEG_AFFECTS", "NEG_ASSOCIATED_WITH", "NEG_DISRUPTS")

ALL_RELATIONSHIPS = (
    *EFFICACY_POSITIVE,
    *EFFICACY_NEGATED,
    *HARM_POSITIVE,
    *HARM_NEGATED,
    *NONDIRECTIONAL,
    *NONDIRECTIONAL_NEGATED,
)

#: Disproportionality thresholds for calling a FAERS signal (Evans et al. 2001): at least
#: three cases, PRR at or above 2, chi-square at or above 4. Applying all three is what
#: separates a signal from the long tail of single-case pairs that dominate the file.
MIN_CASES = 3
MIN_PRR = 2.0
MIN_CHI_SQUARE = 4.0


def _count_expr(relationship: str) -> pl.Expr:
    """Per-pair sentence count for one relationship type.

    The `(?:^|;)` anchor matters: without it, `TREATS` would also match inside
    `NEG_TREATS` and every negation would be counted as an assertion.
    """
    return (
        pl.col("semmeddb_relationships")
        .str.extract(rf"(?:^|;){relationship}=(\d+)", 1)
        .cast(pl.Int32)
        .fill_null(0)
        .alias(f"n_{relationship.lower()}")
    )


def _group_total(relationships: tuple[str, ...]) -> pl.Expr:
    total = pl.lit(0, dtype=pl.Int32)
    for relationship in relationships:
        total = total + pl.col(f"n_{relationship.lower()}")
    return total


def _directional_label(
    positive: tuple[str, ...], negated: tuple[str, ...], name: str
) -> list[pl.Expr]:
    """Asserted / negated counts, a label, and a contradiction flag for one direction.

    The label is 1 only when the direction is asserted and never negated, and 0 only when
    it is negated and never asserted. Pairs with both are left null and flagged, because
    choosing between them would mean trusting sentence counts as evidence strength.
    """
    # The expressions are reused rather than referenced as columns: polars evaluates
    # everything in one `with_columns` against the input frame, so a column created in the
    # same call is not yet visible.
    asserted = _group_total(positive)
    negated_total = _group_total(negated)
    return [
        asserted.alias(f"{name}_asserted"),
        negated_total.alias(f"{name}_negated"),
        pl.when((asserted > 0) & (negated_total == 0))
        .then(1)
        .when((negated_total > 0) & (asserted == 0))
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
        .alias(f"{name}_label"),
        ((asserted > 0) & (negated_total > 0)).alias(f"{name}_contradicted"),
    ]


def build_labels(cem: pl.DataFrame) -> pl.DataFrame:
    """One row per drug-condition pair that carries any evidence at all."""
    counts = cem.with_columns([_count_expr(r) for r in ALL_RELATIONSHIPS])

    labelled = counts.with_columns(
        _directional_label(EFFICACY_POSITIVE, EFFICACY_NEGATED, "efficacy")
    )
    labelled = labelled.with_columns(_directional_label(HARM_POSITIVE, HARM_NEGATED, "harm"))

    return labelled.with_columns(
        _group_total(NONDIRECTIONAL).alias("nondirectional_asserted"),
        faers_signal(),
        ((pl.col("efficacy_label") == 1) & (pl.col("harm_label") == 1)).alias("treats_and_causes"),
    )


def faers_signal() -> pl.Expr:
    """Whether a pair clears all three disproportionality thresholds.

    Null rather than false when the pair has no FAERS evidence at all: "not reported" and
    "reported without a signal" are different statements, and collapsing them would make
    absence look like safety.
    """
    return (
        pl.when(pl.col("in_faers") != "t")
        .then(None)
        .otherwise(
            (pl.col("faers_case_count").cast(pl.Float64) >= MIN_CASES)
            & (pl.col("faers_prr").cast(pl.Float64) >= MIN_PRR)
            & (pl.col("faers_chi_square").cast(pl.Float64) >= MIN_CHI_SQUARE)
        )
        .alias("faers_signal")
    )


def degrees(cem: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ingredient and condition degree over the whole CEM table.

    These are the nuisance terms DESIGN.md calls for: the top conditions by degree are
    Respiratory finding, Nausea, Pain, Fever — reporting artefacts, not pharmacology. A
    model without them spends its capacity learning which events get reported a lot.

    Derived from the label table, so they are label-adjacent and must never be joined into
    model inputs as predictors.
    """
    ingredient = (
        cem.group_by("ingredient_concept_id")
        .agg(pl.len().alias("ingredient_degree"))
        .with_columns(pl.col("ingredient_degree").log1p().alias("log_ingredient_degree"))
    )
    condition = (
        cem.group_by("condition_concept_id")
        .agg(pl.len().alias("condition_degree"))
        .with_columns(pl.col("condition_degree").log1p().alias("log_condition_degree"))
    )
    return ingredient, condition
