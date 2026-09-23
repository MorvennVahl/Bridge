"""exp22 — Does the indication-label biology advantage survive target-space distance?

Registered as exp_20260923_8f4c2f. Follows exp21 (exp_20260922_132b7b), which found gene
overlap beating the p_c floor by +0.0148 drug_macro_auc on approved ChEMBL indications.

exp12 ran this ladder on the FAERS harm label and the advantage collapsed to the floor by
D1-D2. Nobody has run it on an efficacy label, because until the ChEMBL indication layer
there was no curated one with enough drugs. The project's claim is transportability to a
molecule with no real-world data, so an advantage that exists only at D0 — where the drug
already shares a target gene with a training drug — is not the claim.

Rungs follow exp12 exactly so the two ladders are comparable:

    D0  shares a target gene with a train drug
    D1  shares a ChEMBL protein-class leaf
    D2  shares a ChEMBL protein-class L1
    D3  has target annotation, no shared class
    D4  no ChEMBL target annotation

Each rung gets its OWN p_c floor measured on that rung's rows — exp13's correction, and
the thing exp09 got wrong. The model is fit once on all of train, as exp12 does, because
the question is about the evaluation population rather than per-rung refitting.

Run: uv run python experiments/exp22_indication_transportability.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXP_ID = "exp_20260923_8f4c2f"

MIN_PAIRS_PER_DRUG = 20  # experiments/METRIC.md eligibility
RNG_SEEDS = (0, 1, 2)
RUNGS = ("D0", "D1", "D2", "D3", "D4")


def load_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validate pairs labelled by approved ChEMBL indication, curated drugs only."""
    cols = ["ingredient_concept_id", "condition_concept_id", "group_key"]
    frames = []
    approved = (
        pd.read_parquet(ROOT / "data/derived/pair_indications.parquet")
        .query("is_approved")[["ingredient_concept_id", "condition_concept_id"]]
        .assign(y=1)
    )
    for name in ("train", "validate"):
        df = pd.read_csv(ROOT / f"data/splits/{name}.csv", usecols=cols)
        df = df.merge(approved, on=["ingredient_concept_id", "condition_concept_id"], how="left")
        df["y"] = df["y"].fillna(0).astype(int)
        curated = df.groupby("ingredient_concept_id")["y"].transform("sum") > 0
        frames.append(df[curated].copy())
    return frames[0], frames[1]


def target_table() -> pd.DataFrame:
    return (
        pd.read_csv(
            ROOT / "data/input/drug/ingredient_target_long.csv",
            usecols=[
                "omop_concept_id",
                "gene_symbol",
                "chembl_protein_class_leaf",
                "chembl_protein_class_L1",
            ],
        )
        .rename(columns={"omop_concept_id": "ingredient_concept_id"})
        .drop_duplicates()
    )


def assign_rungs(train: pd.DataFrame, validate: pd.DataFrame) -> pd.Series:
    """exp12's five-rung target-space distance, measured against train drugs."""
    targets = target_table()
    train_drugs = set(train["ingredient_concept_id"].unique())
    train_targets = targets[targets["ingredient_concept_id"].isin(train_drugs)]

    seen_genes = set(train_targets["gene_symbol"].dropna())
    seen_leaf = set(train_targets["chembl_protein_class_leaf"].dropna())
    seen_l1 = set(train_targets["chembl_protein_class_L1"].dropna())

    by_drug = targets.groupby("ingredient_concept_id").agg(
        genes=("gene_symbol", lambda s: set(s.dropna())),
        leaf=("chembl_protein_class_leaf", lambda s: set(s.dropna())),
        l1=("chembl_protein_class_L1", lambda s: set(s.dropna())),
    )

    def rung_for(drug: int) -> str:
        if drug not in by_drug.index:
            return "D4"
        row = by_drug.loc[drug]
        if row["genes"] & seen_genes:
            return "D0"
        if row["leaf"] & seen_leaf:
            return "D1"
        if row["l1"] & seen_l1:
            return "D2"
        if row["genes"] or row["leaf"] or row["l1"]:
            return "D3"
        return "D4"

    return validate["ingredient_concept_id"].map(rung_for)


def gene_overlap(pairs: pd.DataFrame) -> np.ndarray:
    """Genetic-evidence-weighted overlap. dt_clinical excluded: it derives from the label."""
    targets = target_table()[["ingredient_concept_id", "gene_symbol"]].dropna().drop_duplicates()
    cg = pd.read_csv(
        ROOT / "data/input/condition/condition_gene_ot_long.csv",
        usecols=["condition_concept_id", "gene_symbol", "dt_genetic_association"],
    ).dropna(subset=["gene_symbol"])
    cg = cg.assign(w=cg["dt_genetic_association"].fillna(0.0))
    cg = cg.groupby(["condition_concept_id", "gene_symbol"], as_index=False)["w"].max()

    per_pair = (
        targets.merge(cg, on="gene_symbol")
        .groupby(["ingredient_concept_id", "condition_concept_id"], as_index=False)["w"]
        .sum()
    )
    merged = pairs.merge(per_pair, on=["ingredient_concept_id", "condition_concept_id"], how="left")
    return merged["w"].fillna(0.0).to_numpy()


ABLATION_FEATURES = ["p_c", "log_drug_degree", "log_condition_degree"]
FEATURES = [*ABLATION_FEATURES, "gene_overlap"]


def build_features(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
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
    out["gene_overlap"] = gene_overlap(df)
    return out


def per_drug_aucs(df: pd.DataFrame, score_a: str, score_b: str) -> np.ndarray:
    """Per-drug AUC differences (a - b) over drugs eligible under METRIC.md."""
    diffs = []
    for _, grp in df.groupby("ingredient_concept_id"):
        if len(grp) < MIN_PAIRS_PER_DRUG or grp["y"].nunique() < 2:
            continue
        diffs.append(roc_auc_score(grp["y"], grp[score_a]) - roc_auc_score(grp["y"], grp[score_b]))
    return np.asarray(diffs)


def macro_auc(df: pd.DataFrame, score: str) -> tuple[float, int]:
    aucs = []
    for _, grp in df.groupby("ingredient_concept_id"):
        if len(grp) < MIN_PAIRS_PER_DRUG or grp["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(grp["y"], grp[score]))
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    train, validate = load_pairs()
    print(f"train {len(train):,} pairs / {train.ingredient_concept_id.nunique()} curated drugs")
    print(f"validate {len(validate):,} pairs / {validate.ingredient_concept_id.nunique()} drugs")

    x_train = build_features(train, train)
    x_val = build_features(train, validate)

    # One fitted model, scored twice: once as-is, once with gene_overlap forced to zero.
    #
    # Refitting without the feature does NOT work here, and the placebo proves it. Adding
    # a feature changes the fitted trees globally, so a refit model differs from the union
    # model even on rows where the feature is constant. Scored that way, D4 — where no
    # drug has any target and gene_overlap is identically zero, so biology cannot act —
    # showed a margin of +0.0675 with a CI excluding zero. That is the same trap exp15
    # caught in exp08: a difference between two differently-fitted models, not a null.
    #
    # Zeroing the feature at prediction time inside one model isolates its contribution,
    # and makes D4 an exact placebo: the two scores must agree there by construction.
    x_val_zeroed = x_val.copy()
    x_val_zeroed["gene_overlap"] = 0.0

    preds, preds_zeroed = [], []
    for seed in RNG_SEEDS:
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31, random_state=seed
        )
        model.fit(x_train[FEATURES], train["y"])
        preds.append(model.predict_proba(x_val[FEATURES])[:, 1])
        preds_zeroed.append(model.predict_proba(x_val_zeroed[FEATURES])[:, 1])

    validate = validate.assign(
        s_union=np.mean(preds, axis=0),
        s_ablation=np.mean(preds_zeroed, axis=0),
        s_pc=x_val["p_c"].to_numpy(),
        overlap=x_val["gene_overlap"].to_numpy(),
        rung=assign_rungs(train, validate).to_numpy(),
    )

    overall_union, n_all = macro_auc(validate, "s_union")
    overall_floor, _ = macro_auc(validate, "s_pc")
    print(f"\noverall: union {overall_union:.4f} | p_c floor {overall_floor:.4f} | {n_all} drugs")

    rows = []
    header = (
        f"\n{'rung':5s} {'pairs':>8s} {'drg':>4s} {'ovlp%':>6s} {'floor':>7s} "
        f"{'abltn':>7s} {'union':>7s} {'vs_pc':>8s} {'vs_abl':>8s}  biology CI"
    )
    print(header)
    for rung in RUNGS:
        sub = validate[validate["rung"] == rung]
        if sub.empty:
            continue
        floor, _ = macro_auc(sub, "s_pc")
        ablation, _ = macro_auc(sub, "s_ablation")
        union, n_drugs = macro_auc(sub, "s_union")
        bio_lo, bio_hi = bootstrap_ci(per_drug_aucs(sub, "s_union", "s_ablation"))
        pc_lo, pc_hi = bootstrap_ci(per_drug_aucs(sub, "s_union", "s_pc"))
        rows.append(
            {
                "rung": rung,
                "n_pairs": len(sub),
                "n_drugs_in_rung": int(sub["ingredient_concept_id"].nunique()),
                "n_drugs_scored": int(n_drugs),
                "share_pairs_with_overlap": float((sub["overlap"] > 0).mean()),
                "floor_pc": floor,
                "ablation_degree_pc": ablation,
                "union": union,
                "margin_vs_pc": union - floor,
                "margin_vs_pc_ci_lo": pc_lo,
                "margin_vs_pc_ci_hi": pc_hi,
                "margin_vs_ablation": union - ablation,
                "margin_vs_ablation_ci_lo": bio_lo,
                "margin_vs_ablation_ci_hi": bio_hi,
            }
        )
        r = rows[-1]
        print(
            f"{rung:5s} {len(sub):8,d} {n_drugs:4d} {100 * r['share_pairs_with_overlap']:5.1f}% "
            f"{floor:7.4f} {ablation:7.4f} {union:7.4f} "
            f"{r['margin_vs_pc']:+8.4f} {r['margin_vs_ablation']:+8.4f}  "
            f"[{bio_lo:+.4f}, {bio_hi:+.4f}]"
        )

    ladder = pd.DataFrame(rows)
    ladder.to_csv(RESULTS / f"{EXP_ID}_ladder.csv", index=False)

    overall_ablation, _ = macro_auc(validate, "s_ablation")
    overall_bio_lo, overall_bio_hi = bootstrap_ci(per_drug_aucs(validate, "s_union", "s_ablation"))
    summary = {
        "overall_union": overall_union,
        "overall_floor_pc": overall_floor,
        "overall_ablation_degree_pc": overall_ablation,
        "overall_margin_vs_pc": overall_union - overall_floor,
        "overall_margin_vs_ablation": overall_union - overall_ablation,
        "overall_margin_vs_ablation_ci_lo": overall_bio_lo,
        "overall_margin_vs_ablation_ci_hi": overall_bio_hi,
        "n_drugs_scored_overall": float(n_all),
        **{f"{r['rung']}_margin_vs_ablation": r["margin_vs_ablation"] for r in rows},
        **{f"{r['rung']}_bio_ci_lo": r["margin_vs_ablation_ci_lo"] for r in rows},
        **{f"{r['rung']}_bio_ci_hi": r["margin_vs_ablation_ci_hi"] for r in rows},
        **{f"{r['rung']}_margin_vs_pc": r["margin_vs_pc"] for r in rows},
        **{f"{r['rung']}_share_overlap": r["share_pairs_with_overlap"] for r in rows},
        **{f"{r['rung']}_n_drugs_scored": float(r["n_drugs_scored"]) for r in rows},
    }
    (RESULTS / f"{EXP_ID}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / f'{EXP_ID}_ladder.csv'} and _metrics.json")


if __name__ == "__main__":
    main()
