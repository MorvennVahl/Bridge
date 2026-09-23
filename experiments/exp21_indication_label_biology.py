"""exp21 — Does biology predict APPROVED indications?

Registered as exp_20260922_132b7b.

exp14 tested gene overlap against a harm label. exp05 tested mechanism features against
7,223 SemMedDB literature assertions. Neither tested biology against a *curated efficacy*
label, because none existed — exp11's adjudicated reference set failed its precondition
with 3 drugs having >=5 reference pairs against a required 30.

The ChEMBL indication layer supplies 106 drugs with >=5 approved-indication pairs and >=5
non-indication pairs. This asks whether gene overlap ranks a drug's approved indications
above its non-indications, against a floor measured on the same subset.

LEAKAGE, load-bearing. Open Targets `dt_clinical` is derived from the same ChEMBL table as
this label, and `ot_score` blends it in. Overlap is weighted by `dt_genetic_association`
only. The leak check at the end includes `dt_clinical` deliberately, to show the size of
the effect it would fabricate — that number is a demonstration, not a result.

Run: uv run python experiments/exp21_indication_label_biology.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXP_ID = "exp_20260922_132b7b"

MIN_PAIRS_PER_DRUG = 20  # experiments/METRIC.md eligibility
RNG_SEEDS = (0, 1, 2)


def drug_macro_auc(df: pd.DataFrame, score_col: str, label_col: str = "y") -> tuple[float, int]:
    """Mean per-drug ROC-AUC over eligible drugs, per experiments/METRIC.md."""
    aucs = []
    for _, grp in df.groupby("ingredient_concept_id"):
        y = grp[label_col]
        if len(grp) < MIN_PAIRS_PER_DRUG or y.nunique() < 2:
            continue
        aucs.append(roc_auc_score(y, grp[score_col]))
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def drug_macro_p10(df: pd.DataFrame, score_col: str, label_col: str = "y") -> float:
    vals = []
    for _, grp in df.groupby("ingredient_concept_id"):
        if len(grp) < MIN_PAIRS_PER_DRUG or grp[label_col].nunique() < 2:
            continue
        vals.append(grp.nlargest(10, score_col)[label_col].mean())
    return float(np.mean(vals)) if vals else float("nan")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["ingredient_concept_id", "condition_concept_id", "group_key"]
    train = pd.read_csv(ROOT / "data/splits/train.csv", usecols=cols)
    validate = pd.read_csv(ROOT / "data/splits/validate.csv", usecols=cols)

    ind = pd.read_parquet(ROOT / "data/derived/pair_indications.parquet")
    approved = ind[ind["is_approved"]][["ingredient_concept_id", "condition_concept_id"]].assign(
        y=1
    )

    out = []
    for df in (train, validate):
        merged = df.merge(
            approved, on=["ingredient_concept_id", "condition_concept_id"], how="left"
        )
        merged["y"] = merged["y"].fillna(0).astype(int)
        # Restrict to drugs ChEMBL curates indications for. A condition absent from a
        # curated drug's indication list is a far better negative than absence from
        # spontaneous reporting; for a drug with no curated indications at all, absence
        # says nothing, so those drugs are excluded rather than labelled negative.
        curated = merged.groupby("ingredient_concept_id")["y"].transform("sum") > 0
        out.append(merged[curated].copy())
    return out[0], out[1]


def gene_overlap(pairs: pd.DataFrame, *, include_clinical: bool) -> np.ndarray:
    """Genetic-evidence-weighted overlap between drug target genes and condition genes."""
    targets = (
        pd.read_csv(
            ROOT / "data/input/drug/ingredient_target_long.csv",
            usecols=["omop_concept_id", "gene_symbol"],
        )
        .dropna()
        .rename(columns={"omop_concept_id": "ingredient_concept_id"})
        .drop_duplicates()
    )

    cg = pd.read_csv(
        ROOT / "data/input/condition/condition_gene_ot_long.csv",
        usecols=["condition_concept_id", "gene_symbol", "dt_genetic_association", "dt_clinical"],
    ).dropna(subset=["gene_symbol"])
    weight = cg["dt_genetic_association"].fillna(0.0)
    if include_clinical:
        weight = weight + cg["dt_clinical"].fillna(0.0)
    cg = cg.assign(w=weight)[["condition_concept_id", "gene_symbol", "w"]]
    cg = cg.groupby(["condition_concept_id", "gene_symbol"], as_index=False)["w"].max()

    per_pair = (
        targets.merge(cg, on="gene_symbol")
        .groupby(["ingredient_concept_id", "condition_concept_id"], as_index=False)["w"]
        .sum()
    )
    keyed = pairs.merge(per_pair, on=["ingredient_concept_id", "condition_concept_id"], how="left")
    return keyed["w"].fillna(0.0).to_numpy()


def build_features(
    train: pd.DataFrame, df: pd.DataFrame, *, include_clinical: bool
) -> pd.DataFrame:
    """Train-derived p_c and degree terms, plus the overlap feature."""
    p_c = train.groupby("condition_concept_id")["y"].mean()
    drug_deg = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    cond_deg = train.groupby("condition_concept_id")["ingredient_concept_id"].nunique()

    out = pd.DataFrame(index=df.index)
    out["p_c"] = df["condition_concept_id"].map(p_c).fillna(p_c.median())
    out["log_drug_degree"] = np.log1p(
        df["ingredient_concept_id"].map(drug_deg).fillna(drug_deg.median())
    )
    out["log_condition_degree"] = np.log1p(
        df["condition_concept_id"].map(cond_deg).fillna(cond_deg.median())
    )
    out["gene_overlap"] = gene_overlap(df, include_clinical=include_clinical)
    return out


def fit_score(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    cols: list[str],
    *,
    include_clinical: bool = False,
) -> np.ndarray:
    x_tr = build_features(train, train, include_clinical=include_clinical)[cols]
    x_va = build_features(train, validate, include_clinical=include_clinical)[cols]
    preds = []
    for seed in RNG_SEEDS:
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31, random_state=seed
        )
        model.fit(x_tr, train["y"])
        preds.append(model.predict_proba(x_va)[:, 1])
    return np.mean(preds, axis=0)


DEGREE_COLS = ["p_c", "log_drug_degree", "log_condition_degree"]
UNION_COLS = [*DEGREE_COLS, "gene_overlap"]


def paired_drug_aucs(df: pd.DataFrame, score_a: str, score_b: str) -> np.ndarray:
    """Per-drug AUC differences (a - b), over drugs eligible under METRIC.md."""
    diffs = []
    for _, grp in df.groupby("ingredient_concept_id"):
        if len(grp) < MIN_PAIRS_PER_DRUG or grp["y"].nunique() < 2:
            continue
        diffs.append(roc_auc_score(grp["y"], grp[score_a]) - roc_auc_score(grp["y"], grp[score_b]))
    return np.asarray(diffs)


def bootstrap_ci(per_drug: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for the macro-averaged difference, resampling drugs."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_drug, size=(n, per_drug.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    train, validate = load()
    print(f"train {len(train):,} pairs / {train.ingredient_concept_id.nunique()} curated drugs")
    print(
        f"validate {len(validate):,} pairs / {validate.ingredient_concept_id.nunique()} curated drugs"
    )
    print(f"validate positive rate {validate['y'].mean():.4f}")

    feats_va = build_features(train, validate, include_clinical=False)
    validate = validate.assign(
        s_pc=feats_va["p_c"].to_numpy(), overlap=feats_va["gene_overlap"].to_numpy()
    )

    results: dict[str, float] = {}

    # Floors, measured on THIS subset — exp13's correction.
    results["floor_pc"], n_drugs = drug_macro_auc(validate, "s_pc")
    results["n_drugs_scored"] = float(n_drugs)
    validate["s_degree_pc"] = fit_score(train, validate, DEGREE_COLS)
    results["floor_degree_pc"], _ = drug_macro_auc(validate, "s_degree_pc")

    # The test.
    validate["s_union"] = fit_score(train, validate, UNION_COLS)
    results["union_with_gene_overlap"], _ = drug_macro_auc(validate, "s_union")
    results["overlap_alone"], _ = drug_macro_auc(validate, "overlap")
    # The increment is measured against the BEST floor, which on this label is the p_c
    # lookup, not the fitted degree model. METRIC.md says so and exp13 caught exp09
    # comparing to a weaker floor; the fitted model is reported too, but it is not the bar.
    results["increment_over_pc"] = results["union_with_gene_overlap"] - results["floor_pc"]
    results["increment_over_degree_pc"] = (
        results["union_with_gene_overlap"] - results["floor_degree_pc"]
    )
    results["share_pairs_with_overlap"] = float((validate["overlap"] > 0).mean())
    results["drug_macro_p10_union"] = drug_macro_p10(validate, "s_union")
    results["drug_macro_p10_floor_pc"] = drug_macro_p10(validate, "s_pc")
    results["pooled_ap_union"] = float(average_precision_score(validate["y"], validate["s_union"]))
    results["pooled_ap_floor_pc"] = float(average_precision_score(validate["y"], validate["s_pc"]))

    # Bootstrap the increment over drugs, against the p_c floor. Resampling the per-drug
    # AUCs is identical to resampling drugs and re-macro-averaging, and avoids refiltering
    # the frame 2,000 times.
    per_drug = paired_drug_aucs(validate, "s_union", "s_pc")
    per_drug_vs_fitted = paired_drug_aucs(validate, "s_union", "s_degree_pc")
    results["increment_ci_lo"], results["increment_ci_hi"] = bootstrap_ci(per_drug)
    results["increment_vs_fitted_ci_lo"], results["increment_vs_fitted_ci_hi"] = bootstrap_ci(
        per_drug_vs_fitted
    )

    # Leak demonstration: what dt_clinical would fabricate.
    validate["s_leak"] = fit_score(train, validate, UNION_COLS, include_clinical=True)
    results["LEAK_union_with_dt_clinical"], _ = drug_macro_auc(validate, "s_leak")
    results["LEAK_inflation"] = (
        results["LEAK_union_with_dt_clinical"] - results["union_with_gene_overlap"]
    )

    print("\n=== results ===")
    for key, value in results.items():
        print(f"  {key:34s} {value:.4f}")

    (RESULTS / f"{EXP_ID}_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / f'{EXP_ID}_metrics.json'}")


if __name__ == "__main__":
    main()
