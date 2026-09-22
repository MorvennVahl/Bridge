"""exp06 — Why exp_20260922_3f4a30's residual Spearman flips sign on validate.

exp04 reported train CV rho=+0.489 against validate rho=-0.306 for the same model on the
same target. A reversal that large is usually a construction error rather than a weak
signal, and this asks which.

The suspect is `exp04_disproportionality_residual.py:232-239`:

    d_eff = df["ingredient_concept_id"].map(drug_effect).fillna(0.0)
    ...
    df["residual"] = df["z"] - grand_mean - df["drug_offset"] - df["condition_offset"]

`drug_effect` is fitted on train. The splits hold out whole ingredients, so no validate
drug appears in train, every validate row takes `drug_offset = 0`, and the validate target
keeps the drug effect that the training target had removed. Model and target would then be
measuring different quantities.

Runs locally against data/splits/. Does not touch test.csv.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"

CASE_COUNT_FLOOR = 3
DEMEAN_TOL = 1e-6
DEMEAN_MAX_ITER = 50


def alternating_demean(
    df: pd.DataFrame, value_col: str, drug_col: str, cond_col: str
) -> tuple[pd.Series, pd.Series, float]:
    """exp04's two-way fixed effects by Gauss-Seidel demeaning, train only."""
    grand_mean = float(df[value_col].mean())
    work = pd.DataFrame(
        {
            drug_col: df[drug_col].to_numpy(),
            cond_col: df[cond_col].to_numpy(),
            "z": (df[value_col] - grand_mean).to_numpy(),
        }
    )
    drug_effect = pd.Series(0.0, index=pd.Index(work[drug_col].unique()))
    cond_effect = pd.Series(0.0, index=pd.Index(work[cond_col].unique()))

    for _ in range(DEMEAN_MAX_ITER):
        resid = work["z"] - work[cond_col].map(cond_effect).to_numpy()
        new_drug = resid.groupby(work[drug_col]).mean()
        shift_d = (new_drug - drug_effect.reindex(new_drug.index).fillna(0.0)).abs().max()
        drug_effect = new_drug

        resid = work["z"] - work[drug_col].map(drug_effect).to_numpy()
        new_cond = resid.groupby(work[cond_col]).mean()
        shift_c = (new_cond - cond_effect.reindex(new_cond.index).fillna(0.0)).abs().max()
        cond_effect = new_cond

        if float(np.nanmax([shift_d, shift_c])) < DEMEAN_TOL:
            break
    return drug_effect, cond_effect, grand_mean


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS / f"{name}.csv")
    df = df[(df["faers_case_count"] >= CASE_COUNT_FLOOR) & (df["faers_prr"] > 0)].copy()
    df["z"] = np.log(df["faers_prr"])
    return df


def build_features(df: pd.DataFrame, train: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    """Train-derived degree terms plus condition record_count, as in the degree floor."""
    drug_degree = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    cond_degree = train.groupby("condition_concept_id")["ingredient_concept_id"].nunique()
    record = cond.set_index("condition_concept_id")["record_count"]

    out = pd.DataFrame(index=df.index)
    out["log_drug_degree"] = np.log1p(
        df["ingredient_concept_id"].map(drug_degree).fillna(drug_degree.median())
    )
    out["log_condition_degree"] = np.log1p(
        df["condition_concept_id"].map(cond_degree).fillna(cond_degree.median())
    )
    out["log_condition_record_count"] = np.log1p(
        df["condition_concept_id"].map(record).fillna(record.median())
    )
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    exp_id = "exp_20260922_7a4500"

    train = load_split("train")
    validate = load_split("validate")
    cond = pd.read_csv(ROOT / "data" / "input" / "condition" / "condition_features_basic.csv")
    print(f"train {len(train):,} rows | validate {len(validate):,} rows (case_count>=3, prr>0)")

    drug_effect, cond_effect, grand_mean = alternating_demean(
        train, "z", "ingredient_concept_id", "condition_concept_id"
    )

    for df in (train, validate):
        df["drug_offset"] = df["ingredient_concept_id"].map(drug_effect).fillna(0.0)
        df["condition_offset"] = df["condition_concept_id"].map(cond_effect).fillna(0.0)
        df["drug_unseen"] = ~df["ingredient_concept_id"].isin(drug_effect.index)
        # exp04's target, reproduced exactly.
        df["residual"] = df["z"] - grand_mean - df["drug_offset"] - df["condition_offset"]
        # An estimand that is well defined for a drug never seen in training.
        df["residual_cond_only"] = df["z"] - grand_mean - df["condition_offset"]

    # ---- Prediction 1: is every validate drug unseen? --------------------------------
    share_unseen_val = float(validate["drug_unseen"].mean())
    share_unseen_train = float(train["drug_unseen"].mean())
    print(f"\n[1] validate rows with drug_unseen: {share_unseen_val:.1%}")
    print(f"    train rows with drug_unseen:    {share_unseen_train:.1%}")

    # ---- Prediction 2: how much of the validate target is the omitted drug effect? ---
    # The true per-drug mean of the exp04 target on validate. Under a correct
    # construction this is ~0; what it actually is, is the offset that was never removed.
    val_drug_mean = validate.groupby("ingredient_concept_id")["residual"].transform("mean")
    var_total = float(validate["residual"].var())
    var_between_drug = float(val_drug_mean.var())
    print(f"\n[2] validate target variance {var_total:.4f}")
    print(
        f"    between-drug component     {var_between_drug:.4f} "
        f"({var_between_drug / var_total:.1%} of total)"
    )
    train_drug_mean = train.groupby("ingredient_concept_id")["residual"].transform("mean")
    print(
        f"    same on train              {float(train_drug_mean.var()) / float(train['residual'].var()):.1%} "
        "(should be ~0: the drug effect was removed there)"
    )

    # ---- Fit exactly as exp04 did -----------------------------------------------------
    x_train = build_features(train, train, cond)
    x_val = build_features(validate, train, cond)
    y_train = train["residual"].to_numpy()

    gkf = GroupKFold(n_splits=3)
    cv_rhos = []
    for tr_idx, te_idx in gkf.split(x_train, y_train, groups=train["group_key"]):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=63)
        m.fit(x_train.iloc[tr_idx], y_train[tr_idx])
        rho, _ = spearmanr(m.predict(x_train.iloc[te_idx]), y_train[te_idx])
        cv_rhos.append(rho)
    train_cv_rho = float(np.mean(cv_rhos))

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=63)
    model.fit(x_train, y_train)
    pred_val = model.predict(x_val)

    # ---- Prediction 3: three evaluations of the same predictions ----------------------
    rho_exp04, _ = spearmanr(pred_val, validate["residual"])

    # Within-drug: invariant to any constant per-drug offset, so the missing drug effect
    # cannot contribute. Pooled over drugs with enough rows to rank.
    validate["_pred"] = pred_val
    within = []
    for _, grp in validate.groupby("ingredient_concept_id"):
        if len(grp) >= 20:
            r, _ = spearmanr(grp["_pred"], grp["residual"])
            if not np.isnan(r):
                within.append(r)
    rho_within = float(np.mean(within))

    # `residual_cond_only` is identical to `residual` on validate, because drug_offset is
    # 0 on every row — keeping the check only to make that degeneracy explicit.
    cond_only_identical = bool(
        np.allclose(validate["residual"].to_numpy(), validate["residual_cond_only"].to_numpy())
    )

    # What actually produces the negative sign: the omitted per-drug offset is 40% of the
    # validate target's variance, so if predictions anti-correlate with it, the pooled
    # correlation follows it rather than the within-drug ranking.
    rho_pred_vs_omitted, _ = spearmanr(pred_val, val_drug_mean)

    print(f"\n[3] train CV rho (exp04 target)         {train_cv_rho:+.4f}")
    print(f"    validate rho, exp04 as written      {rho_exp04:+.4f}")
    print(f"    validate rho, within-drug (n={len(within)})   {rho_within:+.4f}")
    print(f"    condition-only target identical on validate: {cond_only_identical}")
    print(f"    rho(prediction, omitted drug offset) {rho_pred_vs_omitted:+.4f}")

    metrics = {
        "share_validate_drug_unseen": share_unseen_val,
        "share_train_drug_unseen": share_unseen_train,
        "validate_target_variance": var_total,
        "validate_between_drug_variance": var_between_drug,
        "validate_between_drug_share": var_between_drug / var_total,
        "train_cv_spearman": train_cv_rho,
        "validate_spearman_as_written": float(rho_exp04),
        "validate_spearman_within_drug": rho_within,
        "spearman_prediction_vs_omitted_drug_offset": float(rho_pred_vs_omitted),
        "n_drugs_within_drug_eval": float(len(within)),
    }
    (RESULTS / f"{exp_id}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / f'{exp_id}_metrics.json'}")


if __name__ == "__main__":
    main()
