"""Restructure the Bridge data into input/ and output/, and split 50/30/20.

Layout produced under data/:

    input/drug/          ingredient- and target-level features (the drug side)
    input/condition/     condition-level features and the ontology map
    output/pair_labels.csv   one row per ingredient-condition pair, labels only
    splits/              fold assignment + train/validate/test pair keys

Why the split is grouped rather than random
-------------------------------------------
The question this dataset exists to answer is "what will this drug do, given that
it has no real-world data of its own". A random split over the 1.45M pairs would
put the same ingredient in train and test with different conditions, so a model
could score well by memorising the drug rather than learning anything from the
biology. Folds are therefore assigned to whole GROUPS of ingredients, and no
ingredient's pairs are ever divided.

Two grouping schemes:

  primary_gene (default)  ingredients sharing a primary target gene stay together.
                          363 groups over 1,541 targeted ingredients, largest 55.
  target_component        connected components of the ingredient-target bipartite
                          graph. Strictly safer, but promiscuous targets merge 638
                          ingredients (41% of targeted ones) into a single
                          component, so a 50/30/20 split is not achievable and the
                          fold sizes will be badly unbalanced. Offered for a
                          leakage-worst-case check, not as the default.

Ingredients with no known target form singleton groups under both schemes; they
cannot leak through a shared target because they have none recorded.

Labels
------
Nothing here is a causal effect estimate. These are the CEM evidence flags, and
they carry the biases documented in DESIGN.md — FAERS reporting bias above all.
Raw statistics are kept alongside the derived flags so a downstream model can use
magnitudes or re-threshold.

  y_faers_signal        FAERS disproportionality signal by the Evans criteria:
                        PRR >= 2 AND chi-square >= 4 AND case count >= 3
                        (Evans SJ, Waller PC, Davis S. Pharmacoepidemiol Drug Saf
                        2001;10:483-6). This is a screening convention, not proof
                        of causation.
  y_semmeddb_causes     SemMedDB asserts CAUSES, PREDISPOSES or COMPLICATES
  y_semmeddb_treats     SemMedDB asserts TREATS or PREVENTS
  y_any_harm            y_faers_signal OR y_semmeddb_causes

NEG_* SemMedDB predicates are negations and are counted separately, never as
positives.

IMPORTANT: absence of a pair from the CEM file is NOT a negative. It may mean no
effect, or that the drug was never prescribed, or that nobody reported it. The
pair table covers only observed pairs; constructing negatives needs the CEM
negative-control lists, which is a separate job.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --scheme target_component --seed 7
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PROPORTIONS = {"train": 0.50, "validate": 0.30, "test": 0.20}

DRUG_FILES = [
    "ingredient_features.csv",
    "target_features.csv",
    "ingredient_target_long.csv",
    "ingredient_features_label_adjacent.csv",
]
CONDITION_FILES = ["condition_ontology_map.csv"]

HARM_PREDICATES = {"CAUSES", "PREDISPOSES", "COMPLICATES"}
BENEFIT_PREDICATES = {"TREATS", "PREVENTS"}


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def parse_semmeddb(cell: str) -> dict[str, int]:
    """'CAUSES=3;TREATS=1' -> {'CAUSES': 3, 'TREATS': 1}."""
    out: dict[str, int] = {}
    if not isinstance(cell, str) or not cell:
        return out
    for token in cell.split(";"):
        key, _, val = token.partition("=")
        try:
            out[key.strip()] = int(val)
        except ValueError:
            out[key.strip()] = 0
    return out


def build_labels(cem: pd.DataFrame) -> pd.DataFrame:
    df = cem.copy()
    for col in ["in_faers", "in_eu_label", "in_semmeddb"]:
        df[col] = df[col].astype(str).eq("t")

    preds = df.semmeddb_relationships.map(parse_semmeddb)
    df["semmeddb_harm_sentences"] = preds.map(
        lambda d: sum(v for k, v in d.items() if k in HARM_PREDICATES)
    )
    df["semmeddb_benefit_sentences"] = preds.map(
        lambda d: sum(v for k, v in d.items() if k in BENEFIT_PREDICATES)
    )
    df["semmeddb_negated_sentences"] = preds.map(
        lambda d: sum(v for k, v in d.items() if k.startswith("NEG_"))
    )

    df["y_faers_signal"] = (
        df.faers_prr.ge(2) & df.faers_chi_square.ge(4) & df.faers_case_count.ge(3)
    ).fillna(False)
    df["y_semmeddb_causes"] = df.semmeddb_harm_sentences.gt(0)
    df["y_semmeddb_treats"] = df.semmeddb_benefit_sentences.gt(0)
    df["y_any_harm"] = df.y_faers_signal | df.y_semmeddb_causes

    keep = [
        "ingredient_concept_id",
        "condition_concept_id",
        "in_faers",
        "in_semmeddb",
        "in_eu_label",
        "faers_case_count",
        "faers_prr",
        "faers_chi_square",
        "semmeddb_harm_sentences",
        "semmeddb_benefit_sentences",
        "semmeddb_negated_sentences",
        "y_faers_signal",
        "y_semmeddb_causes",
        "y_semmeddb_treats",
        "y_any_harm",
    ]
    return df[keep]


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #


def primary_gene_groups(itl: pd.DataFrame, ingredients: list[int]) -> pd.Series:
    """ingredient -> group key, using the highest-confidence mechanism gene."""
    tgt = itl[itl.gene_symbol.notna()]
    prim = (
        tgt.sort_values(
            ["omop_concept_id", "disease_efficacy", "direct_interaction"],
            ascending=[True, False, False],
        )
        .groupby("omop_concept_id")
        .gene_symbol.first()
    )
    return pd.Series(
        {i: f"gene:{prim[i]}" if i in prim.index else f"ing:{i}" for i in ingredients},
        name="group_key",
    )


def target_component_groups(itl: pd.DataFrame, ingredients: list[int]) -> pd.Series:
    """ingredient -> connected component of the ingredient-target graph."""
    import networkx as nx

    g = nx.Graph()
    for row in (
        itl[itl.gene_symbol.notna()][["omop_concept_id", "gene_symbol"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        g.add_edge(("ing", row.omop_concept_id), ("gene", row.gene_symbol))
    lookup: dict[int, str] = {}
    for n, comp in enumerate(nx.connected_components(g)):
        for node in comp:
            if node[0] == "ing":
                lookup[node[1]] = f"component:{n}"
    return pd.Series({i: lookup.get(i, f"ing:{i}") for i in ingredients}, name="group_key")


def assign_folds(group_sizes: pd.Series, seed: int) -> dict[str, str]:
    """Greedy largest-group-first assignment to whichever fold is furthest behind.

    Balances the number of PAIRS per fold, not the number of groups, because pair
    count is what the model actually trains on. Largest-first keeps a single big
    group from overshooting a fold late in the assignment.
    """
    total = group_sizes.sum()
    targets = {f: p * total for f, p in PROPORTIONS.items()}
    current = dict.fromkeys(PROPORTIONS, 0.0)
    assignment: dict[str, str] = {}

    shuffled = group_sizes.sample(frac=1.0, random_state=seed)
    for key, size in shuffled.sort_values(ascending=False, kind="stable").items():
        fold = max(PROPORTIONS, key=lambda f: (targets[f] - current[f]) / targets[f])
        assignment[key] = fold
        current[fold] += size
    return assignment


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scheme", choices=["primary_gene", "target_component"], default="primary_gene"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_drug, in_cond = DATA / "input" / "drug", DATA / "input" / "condition"
    out_dir, split_dir = DATA / "output", DATA / "splits"
    for d in (in_drug, in_cond, out_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Copy rather than move: other sessions read these paths concurrently.
    for name in DRUG_FILES:
        if (DATA / name).exists():
            shutil.copy2(DATA / name, in_drug / name)
    for name in CONDITION_FILES:
        if (DATA / name).exists():
            shutil.copy2(DATA / name, in_cond / name)

    cem = pd.read_csv(
        DATA / "cem_ingredient_condition_associations.csv",
        dtype={"ingredient_concept_id": "int64", "condition_concept_id": "int64"},
    )
    labels = build_labels(cem)
    labels.to_csv(out_dir / "pair_labels.csv", index=False)
    print(f"pair_labels: {len(labels)} rows", flush=True)

    itl = pd.read_csv(DATA / "ingredient_target_long.csv")
    ingredients = sorted(labels.ingredient_concept_id.unique().tolist())
    groups = (
        primary_gene_groups(itl, ingredients)
        if args.scheme == "primary_gene"
        else target_component_groups(itl, ingredients)
    )

    pairs_per_ing = labels.groupby("ingredient_concept_id").size()
    grp = pd.DataFrame({"group_key": groups, "n_pairs": pairs_per_ing}).fillna({"n_pairs": 0})
    fold_of_group = assign_folds(grp.groupby("group_key").n_pairs.sum(), args.seed)
    grp["fold"] = grp.group_key.map(fold_of_group)

    assignment = grp.reset_index(names="ingredient_concept_id")
    assignment["scheme"] = args.scheme
    assignment.to_csv(split_dir / "split_assignment.csv", index=False)

    labelled = labels.merge(
        assignment[["ingredient_concept_id", "fold", "group_key"]],
        on="ingredient_concept_id",
        how="left",
    )
    for fold in PROPORTIONS:
        sub = labelled[labelled.fold == fold]
        sub.to_csv(split_dir / f"{fold}.csv", index=False)

    summary = []
    for fold in PROPORTIONS:
        sub = labelled[labelled.fold == fold]
        summary.append(
            {
                "fold": fold,
                "target_share": PROPORTIONS[fold],
                "pairs": len(sub),
                "pair_share": round(len(sub) / len(labelled), 4),
                "ingredients": sub.ingredient_concept_id.nunique(),
                "groups": sub.group_key.nunique(),
                "conditions": sub.condition_concept_id.nunique(),
                "rate_faers_signal": round(sub.y_faers_signal.mean(), 4),
                "rate_semmeddb_causes": round(sub.y_semmeddb_causes.mean(), 4),
                "rate_semmeddb_treats": round(sub.y_semmeddb_treats.mean(), 4),
            }
        )
    summ = pd.DataFrame(summary)
    summ.to_csv(split_dir / "split_summary.csv", index=False)
    print(summ.to_string(index=False), flush=True)

    # leakage assertions: no ingredient and no group may appear in two folds
    assert labelled.groupby("ingredient_concept_id").fold.nunique().max() == 1, (
        "an ingredient appears in more than one fold"
    )
    assert labelled.groupby("group_key").fold.nunique().max() == 1, (
        "a group appears in more than one fold"
    )

    (split_dir / "split_provenance.json").write_text(
        json.dumps(
            {
                "scheme": args.scheme,
                "seed": args.seed,
                "proportions": PROPORTIONS,
                "grouping_unit": (
                    "primary target gene from ingredient_target_long.csv; "
                    "untargeted ingredients are singletons"
                ),
                "balanced_on": "number of pairs per fold",
                "faers_signal_criteria": "PRR>=2 & chi_square>=4 & case_count>=3 (Evans et al. 2001)",
                "n_pairs": len(labelled),
                "n_ingredients": int(labelled.ingredient_concept_id.nunique()),
                "n_conditions": int(labelled.condition_concept_id.nunique()),
                "caveat": (
                    "pairs absent from the CEM file are unobserved, not negative; "
                    "labels are evidence flags, not causal effect estimates"
                ),
            },
            indent=2,
        )
    )
    print("wrote", split_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
