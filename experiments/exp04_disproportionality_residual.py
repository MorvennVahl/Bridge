"""Modal experiment: is the two-way-demeaned log(faers_prr) residual a better
modelling target than the raw binary FAERS signal flag?

Implements experiments/exp04_disproportionality_residual.md exactly.
"""

from __future__ import annotations

import logging
import pathlib

import modal

app = modal.App("bridge-exp04")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pandas==2.2.3",
    "numpy==2.1.3",
    "scikit-learn==1.5.2",
    "lightgbm==4.5.0",
    "pyarrow==17.0.0",
    "scipy==1.14.1",
    "matplotlib==3.9.2",
)

data = modal.Volume.from_name("bridge-data")
results = modal.Volume.from_name("bridge-results", create_if_missing=True)

CASE_COUNT_FLOOR = 3
EXPECTED_TRAIN_SUBSET_ROWS = 344_653
EXPECTED_VALIDATE_SUBSET_ROWS = 206_586
EXPECTED_LOG_PRR_MEAN = 0.283
EXPECTED_LOG_PRR_SD = 1.052
TOLERANCE_ROWS = 0.02  # 2% relative tolerance on row counts
TOLERANCE_MOMENT = 0.05  # absolute tolerance on mean/sd of log(faers_prr)

DEMEAN_TOL = 1e-6
DEMEAN_MAX_ITER = 50

EXCLUDED_BLOCKS = {"identity"}
EXCLUDED_COLUMNS = {
    "in_cem_list",
    "in_indication_roster",
    "roster_name",
    "indication",
    "target_chembl_ids",
    "target_genes",
    "target_gene_symbols",
    "mechanisms_of_action",
    "action_types",
    "warning_types",
    "warning_classes",
    "atc_codes",
    "atc_l3",
    "chembl_metab_enzymes",
    "chembl_metabolite_names",
    "kegg_metab_enzymes",
    "kegg_transporters",
    "kegg_interaction_genes",
    "reactome_top_level_terms",
    "target_safety_events",
    "usan_stem_definition",
    "indication_class",
}
INCLUDED_BLOCKS = {
    "development",
    "safety",
    "pharmacology",
    "chemistry",
    "exposure",
    "metabolism",
    "mechanism",
    "target-biology",
    "interactions",
}


def _assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"Precondition failed: {name} = {actual} (expected {expected} +/- {tol})"
        )


def _assert_row_count(name: str, actual: int, expected: int, tol_frac: float) -> None:
    if abs(actual - expected) > tol_frac * expected:
        raise AssertionError(f"Precondition failed: {name} = {actual} rows (expected ~{expected})")


def _select_drug_feature_columns(data_dict: pd.DataFrame) -> list[str]:  # noqa: F821
    dd = data_dict[data_dict["table"] == "ingredient_features.csv"]
    dd = dd[dd["block"].isin(INCLUDED_BLOCKS)]
    cols = [c for c in dd["column"].tolist() if c not in EXCLUDED_COLUMNS]
    return cols


def _alternating_demean(
    df: pd.DataFrame,  # noqa: F821
    value_col: str,
    drug_col: str,
    cond_col: str,
    tol: float = DEMEAN_TOL,
    max_iter: int = DEMEAN_MAX_ITER,
) -> tuple[pd.Series, pd.Series, float, int]:  # noqa: F821
    """Alternating-demeaning (Gauss-Seidel) two-way fixed effects.

    Returns (drug_effect, condition_effect, grand_mean, n_iterations), all
    computed TRAIN-ONLY. drug_effect and condition_effect are deviations from
    the grand mean (i.e. sum to ~0 within each), so:
        z ~= grand_mean + drug_effect[d] + condition_effect[c] + residual
    """
    import numpy as np
    import pandas as pd

    grand_mean = float(df[value_col].mean())
    z_centered = df[value_col] - grand_mean

    drug_effect = pd.Series(0.0, index=df[drug_col].unique())
    cond_effect = pd.Series(0.0, index=df[cond_col].unique())

    work = pd.DataFrame(
        {
            drug_col: df[drug_col].to_numpy(),
            cond_col: df[cond_col].to_numpy(),
            "z": z_centered.to_numpy(),
        }
    )

    n_iter = 0
    for i in range(max_iter):
        n_iter = i + 1
        # residual after removing current condition effect, demean by drug
        resid_for_drug = work["z"] - work[cond_col].map(cond_effect).to_numpy()
        new_drug_effect = resid_for_drug.groupby(work[drug_col]).mean()
        drug_shift = (
            (new_drug_effect - drug_effect.reindex(new_drug_effect.index).fillna(0.0)).abs().max()
        )
        drug_effect = new_drug_effect

        # residual after removing updated drug effect, demean by condition
        resid_for_cond = work["z"] - work[drug_col].map(drug_effect).to_numpy()
        new_cond_effect = resid_for_cond.groupby(work[cond_col]).mean()
        cond_shift = (
            (new_cond_effect - cond_effect.reindex(new_cond_effect.index).fillna(0.0)).abs().max()
        )
        cond_effect = new_cond_effect

        max_shift = float(np.nanmax([drug_shift, cond_shift]))
        if max_shift < tol:
            break

    return drug_effect, cond_effect, grand_mean, n_iter


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp04")

    import matplotlib
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    findings_fallback_notes: list[str] = []

    # ---- Step 1: preconditions ----------------------------------------------------
    required_paths = [
        "/data/splits/train.csv",
        "/data/splits/validate.csv",
        "/data/drug/ingredient_features.csv",
        "/data/condition/condition_features_basic.csv",
        "/data/condition/condition_group_long.csv",
        "/data/data_dictionary.csv",
    ]
    for p in required_paths:
        if not pathlib.Path(p).exists():
            msg = f"Missing required input: {p}"
            log.error(msg)
            raise FileNotFoundError(msg)

    log.info("Reading train/validate splits")
    train = pd.read_csv("/data/splits/train.csv")
    validate = pd.read_csv("/data/splits/validate.csv")

    # ---- Step 2: restrict to case_count >= 3 and faers_prr > 0 --------------------
    train_sub = train[
        (train["faers_case_count"] >= CASE_COUNT_FLOOR) & (train["faers_prr"] > 0)
    ].copy()
    validate_sub = validate[
        (validate["faers_case_count"] >= CASE_COUNT_FLOOR) & (validate["faers_prr"] > 0)
    ].copy()

    log.info("train_sub=%d validate_sub=%d", len(train_sub), len(validate_sub))
    _assert_row_count("n_train_rows", len(train_sub), EXPECTED_TRAIN_SUBSET_ROWS, TOLERANCE_ROWS)
    _assert_row_count(
        "n_validate_rows", len(validate_sub), EXPECTED_VALIDATE_SUBSET_ROWS, TOLERANCE_ROWS
    )

    train_sub["z"] = np.log(train_sub["faers_prr"])
    validate_sub["z"] = np.log(validate_sub["faers_prr"])

    log_prr_mean = float(train_sub["z"].mean())
    log_prr_sd = float(train_sub["z"].std())
    log.info("log(faers_prr) train mean=%.4f sd=%.4f", log_prr_mean, log_prr_sd)
    _assert_close("log(faers_prr) mean", log_prr_mean, EXPECTED_LOG_PRR_MEAN, TOLERANCE_MOMENT)
    _assert_close("log(faers_prr) sd", log_prr_sd, EXPECTED_LOG_PRR_SD, TOLERANCE_MOMENT)

    # ---- Step 3: alternating demeaning, TRAIN ONLY ---------------------------------
    log.info("Fitting two-way fixed effects by alternating demeaning (train only)")
    drug_effect, cond_effect, grand_mean, n_iter = _alternating_demean(
        train_sub, value_col="z", drug_col="ingredient_concept_id", cond_col="condition_concept_id"
    )
    log.info("Alternating demeaning converged in %d iterations", n_iter)

    def _apply_offsets(df: pd.DataFrame) -> pd.DataFrame:
        d_eff = df["ingredient_concept_id"].map(drug_effect).fillna(0.0)
        c_eff = df["condition_concept_id"].map(cond_effect).fillna(0.0)
        df = df.copy()
        df["drug_offset"] = d_eff.to_numpy()
        df["condition_offset"] = c_eff.to_numpy()
        df["drug_unseen"] = ~df["ingredient_concept_id"].isin(drug_effect.index)
        df["condition_unseen"] = ~df["condition_concept_id"].isin(cond_effect.index)
        df["residual"] = df["z"] - grand_mean - df["drug_offset"] - df["condition_offset"]
        return df

    train_sub = _apply_offsets(train_sub)
    # ---- Step 4: apply train offsets to validate (never re-estimate) --------------
    validate_sub = _apply_offsets(validate_sub)

    # ---- Step 7: variance of z explained by the two-way structure (train) ---------
    fitted_train = grand_mean + train_sub["drug_offset"] + train_sub["condition_offset"]
    ss_res_twoway = float(((train_sub["z"] - fitted_train) ** 2).sum())
    ss_tot_twoway = float(((train_sub["z"] - train_sub["z"].mean()) ** 2).sum())
    twoway_r2 = 1.0 - ss_res_twoway / ss_tot_twoway if ss_tot_twoway > 0 else float("nan")
    log.info("Two-way structure R^2 on train z = %.4f", twoway_r2)

    # ---- Offsets deliverable --------------------------------------------------------
    drug_n = train_sub.groupby("ingredient_concept_id").size()
    cond_n = train_sub.groupby("condition_concept_id").size()
    offsets_drug = pd.DataFrame(
        {"entity_type": "drug", "entity_id": drug_effect.index, "offset": drug_effect.values}
    )
    offsets_drug["n"] = offsets_drug["entity_id"].map(drug_n).to_numpy()
    offsets_cond = pd.DataFrame(
        {"entity_type": "condition", "entity_id": cond_effect.index, "offset": cond_effect.values}
    )
    offsets_cond["n"] = offsets_cond["entity_id"].map(cond_n).to_numpy()
    offsets_df = pd.concat([offsets_drug, offsets_cond], ignore_index=True)
    offsets_df.to_csv(out / f"{exp_id}_offsets.csv", index=False)

    twoway_fit_df = pd.DataFrame(
        [
            {
                "grand_mean": grand_mean,
                "n_iterations": n_iter,
                "twoway_r2": twoway_r2,
                "ss_res": ss_res_twoway,
                "ss_tot": ss_tot_twoway,
                "n_train_rows": len(train_sub),
            }
        ]
    )
    twoway_fit_df.to_csv(out / f"{exp_id}_twoway_fit.csv", index=False)

    # ---- Feature assembly -----------------------------------------------------------
    log.info("Reading feature tables")
    data_dict = pd.read_csv("/data/data_dictionary.csv")
    drug_feat_cols = _select_drug_feature_columns(data_dict)
    ingredient_features = pd.read_csv("/data/drug/ingredient_features.csv")

    available_drug_cols = [c for c in drug_feat_cols if c in ingredient_features.columns]
    missing_drug_cols = sorted(set(drug_feat_cols) - set(available_drug_cols))
    if missing_drug_cols:
        log.warning(
            "data_dictionary lists %d drug columns not present in ingredient_features.csv: %s",
            len(missing_drug_cols),
            missing_drug_cols[:10],
        )

    drug_features = ingredient_features[["omop_concept_id", *available_drug_cols]].copy()
    drug_features["has_chembl_match"] = (
        ingredient_features["chembl_id"].notna()
        if "chembl_id" in ingredient_features.columns
        else False
    )
    drug_features = drug_features.rename(columns={"omop_concept_id": "ingredient_concept_id"})

    condition_features = pd.read_csv("/data/condition/condition_features_basic.csv")
    condition_group_long = pd.read_csv("/data/condition/condition_group_long.csv")
    group_onehot = condition_group_long.assign(val=1).pivot_table(
        index="condition_concept_id",
        columns=["group_source", "group_label"],
        values="val",
        aggfunc="max",
        fill_value=0,
    )
    group_onehot.columns = [f"group__{src}__{lbl}" for src, lbl in group_onehot.columns]
    group_onehot = group_onehot.reset_index()

    # degree terms (train-only, applied to validate; unseen -> train median)
    drug_degree_train = train.groupby("ingredient_concept_id").size().rename("drug_degree")
    cond_degree_train = train.groupby("condition_concept_id").size().rename("condition_degree")
    drug_degree_median = float(drug_degree_train.median())
    cond_degree_median = float(cond_degree_train.median())

    def _add_common_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(drug_features, on="ingredient_concept_id", how="left")
        df = df.merge(condition_features, on="condition_concept_id", how="left")
        df = df.merge(group_onehot, on="condition_concept_id", how="left")
        df["drug_degree"] = df["ingredient_concept_id"].map(drug_degree_train)
        df["drug_degree"] = df["drug_degree"].fillna(drug_degree_median)
        df["condition_degree"] = df["condition_concept_id"].map(cond_degree_train)
        df["condition_degree"] = df["condition_degree"].fillna(cond_degree_median)
        return df

    train_sub = _add_common_features(train_sub)
    validate_sub = _add_common_features(validate_sub)

    group_cols = [c for c in group_onehot.columns if c != "condition_concept_id"]
    non_feature_cols = {
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
        "fold",
        "group_key",
        "z",
        "drug_offset",
        "condition_offset",
        "drug_unseen",
        "condition_unseen",
        "residual",
        "condition_name",
    }
    feature_cols = [c for c in train_sub.columns if c not in non_feature_cols]
    # Keep only numeric / boolean columns; drop remaining object/free-text columns.
    numeric_feature_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_sub[c]) or pd.api.types.is_bool_dtype(train_sub[c]):
            numeric_feature_cols.append(c)
    feature_cols = numeric_feature_cols
    log.info("Using %d feature columns (%d group one-hots)", len(feature_cols), len(group_cols))

    X_train = train_sub[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")  # noqa: N806 -- conventional sklearn feature-matrix name
    X_validate = validate_sub[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")  # noqa: N806
    y_train_res = train_sub["residual"].to_numpy()
    y_validate_res = validate_sub["residual"].to_numpy()
    groups_train = train_sub["group_key"].to_numpy()

    # ---- Step 5: regressor on residual, grouped 3-fold CV + validate --------------
    log.info("Fitting HistGradientBoostingRegressor on residual with grouped 3-fold CV")
    gkf = GroupKFold(n_splits=3)
    cv_spearman = []
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_train, y_train_res, groups=groups_train)):
        reg = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
            loss="squared_error",
        )
        reg.fit(X_train.iloc[tr_idx], y_train_res[tr_idx])
        pred = reg.predict(X_train.iloc[te_idx])
        rho, _ = spearmanr(pred, y_train_res[te_idx])
        cv_spearman.append(rho)
        log.info("CV fold %d spearman rho = %.4f", fold_i, rho)

    train_cv_spearman_mean = float(np.mean(cv_spearman))
    train_cv_spearman_spread = float(np.std(cv_spearman))

    reg_full = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=63,
        early_stopping=False,
        random_state=0,
        loss="squared_error",
    )
    reg_full.fit(X_train, y_train_res)
    pred_validate_res = reg_full.predict(X_validate)

    validate_spearman_residual, _ = spearmanr(pred_validate_res, y_validate_res)
    validate_pearson_r, _ = pearsonr(pred_validate_res, y_validate_res)
    ss_res_model = float(((y_validate_res - pred_validate_res) ** 2).sum())
    ss_tot_model = float(((y_validate_res - y_validate_res.mean()) ** 2).sum())
    validate_r2 = 1.0 - ss_res_model / ss_tot_model if ss_tot_model > 0 else float("nan")

    log.info(
        "Validate: spearman=%.4f pearson_r=%.4f r2=%.4f",
        validate_spearman_residual,
        validate_pearson_r,
        validate_r2,
    )

    residual_model_df = pd.DataFrame(
        [
            {
                "train_cv_spearman_mean": train_cv_spearman_mean,
                "train_cv_spearman_spread": train_cv_spearman_spread,
                "validate_spearman": float(validate_spearman_residual),
                "validate_pearson_r": float(validate_pearson_r),
                "validate_r2": float(validate_r2),
                "n_train_rows": len(train_sub),
                "n_validate_rows": len(validate_sub),
            }
        ]
    )
    residual_model_df.to_csv(out / f"{exp_id}_residual_model.csv", index=False)

    # ---- Step 6: binary classifier comparison on identical subset/features --------
    y_train_bin = train_sub["y_faers_signal"].to_numpy()
    y_validate_bin = validate_sub["y_faers_signal"].to_numpy()
    prevalence = float(y_validate_bin.mean())

    budget_risk = False
    try:
        log.info("Fitting HistGradientBoostingClassifier on y_faers_signal with grouped 3-fold CV")
        cv_ap = []
        for fold_i, (tr_idx, te_idx) in enumerate(
            gkf.split(X_train, y_train_bin, groups=groups_train)
        ):
            clf = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.06,
                max_leaf_nodes=63,
                early_stopping=False,
                random_state=0,
            )
            clf.fit(X_train.iloc[tr_idx], y_train_bin[tr_idx])
            proba = clf.predict_proba(X_train.iloc[te_idx])[:, 1]
            ap = average_precision_score(y_train_bin[te_idx], proba)
            cv_ap.append(ap)
            log.info("Classifier CV fold %d AP = %.4f", fold_i, ap)
        train_cv_ap_mean = float(np.mean(cv_ap))
    except Exception as exc:  # pragma: no cover - budget fallback path
        budget_risk = True
        findings_fallback_notes.append(
            f"Dropped classifier grouped CV due to budget risk ({exc}); kept single train fit + validate pass."
        )
        train_cv_ap_mean = float("nan")

    clf_full = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=63,
        early_stopping=False,
        random_state=0,
    )
    clf_full.fit(X_train, y_train_bin)
    pred_validate_bin_proba = clf_full.predict_proba(X_validate)[:, 1]
    validate_ap = average_precision_score(y_validate_bin, pred_validate_bin_proba)

    # degree-only baseline AP on this subset
    degree_only_cols = ["drug_degree", "condition_degree"]
    clf_degree = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=63,
        early_stopping=False,
        random_state=0,
    )
    clf_degree.fit(X_train[degree_only_cols], y_train_bin)
    pred_degree_proba = clf_degree.predict_proba(X_validate[degree_only_cols])[:, 1]
    baseline_degree_only_ap = average_precision_score(y_validate_bin, pred_degree_proba)

    # four-cell comparison table:
    #   rows = model trained on {residual, binary}; cols = evaluated against {residual (spearman), binary (AP)}
    resid_model_vs_residual_rho, _ = spearmanr(pred_validate_res, y_validate_res)
    resid_model_vs_binary_ap = average_precision_score(y_validate_bin, pred_validate_res)
    bin_model_vs_residual_rho, _ = spearmanr(pred_validate_bin_proba, y_validate_res)
    bin_model_vs_binary_ap = average_precision_score(y_validate_bin, pred_validate_bin_proba)

    target_comparison_df = pd.DataFrame(
        [
            {
                "trained_on": "residual",
                "evaluated_against": "residual",
                "metric": "spearman_rho",
                "value": resid_model_vs_residual_rho,
            },
            {
                "trained_on": "residual",
                "evaluated_against": "binary_flag",
                "metric": "average_precision",
                "value": resid_model_vs_binary_ap,
            },
            {
                "trained_on": "binary_flag",
                "evaluated_against": "residual",
                "metric": "spearman_rho",
                "value": bin_model_vs_residual_rho,
            },
            {
                "trained_on": "binary_flag",
                "evaluated_against": "binary_flag",
                "metric": "average_precision",
                "value": bin_model_vs_binary_ap,
            },
        ]
    )
    target_comparison_df.to_csv(out / f"{exp_id}_target_comparison.csv", index=False)

    # ---- Step 7 (cont'd): heteroscedasticity by case-count strata + subgroups -----
    def _stratum(cc: int) -> str:
        if cc <= 5:
            return "3-5"
        if cc <= 20:
            return "6-20"
        return "21+"

    validate_sub["case_count_stratum"] = validate_sub["faers_case_count"].apply(_stratum)
    validate_sub["pred_residual"] = pred_validate_res

    subgroup_rows = []
    for stratum, grp in validate_sub.groupby("case_count_stratum"):
        if len(grp) > 1:
            rho, _ = spearmanr(grp["pred_residual"], grp["residual"])
        else:
            rho = float("nan")
        subgroup_rows.append(
            {
                "subgroup_type": "case_count_stratum",
                "subgroup": stratum,
                "n": len(grp),
                "validate_spearman": rho,
            }
        )

    if "is_mapped" in validate_sub.columns:
        for val, grp in validate_sub.groupby("is_mapped"):
            rho, _ = (
                spearmanr(grp["pred_residual"], grp["residual"])
                if len(grp) > 1
                else (float("nan"), None)
            )
            subgroup_rows.append(
                {
                    "subgroup_type": "is_mapped",
                    "subgroup": str(val),
                    "n": len(grp),
                    "validate_spearman": rho,
                }
            )

    if "arm" in validate_sub.columns:
        for val, grp in validate_sub.groupby("arm"):
            rho, _ = (
                spearmanr(grp["pred_residual"], grp["residual"])
                if len(grp) > 1
                else (float("nan"), None)
            )
            subgroup_rows.append(
                {
                    "subgroup_type": "arm",
                    "subgroup": str(val),
                    "n": len(grp),
                    "validate_spearman": rho,
                }
            )

    for val, grp in validate_sub.groupby("drug_unseen"):
        rho, _ = (
            spearmanr(grp["pred_residual"], grp["residual"])
            if len(grp) > 1
            else (float("nan"), None)
        )
        subgroup_rows.append(
            {
                "subgroup_type": "drug_unseen_in_train",
                "subgroup": str(val),
                "n": len(grp),
                "validate_spearman": rho,
            }
        )

    for val, grp in validate_sub.groupby("condition_unseen"):
        rho, _ = (
            spearmanr(grp["pred_residual"], grp["residual"])
            if len(grp) > 1
            else (float("nan"), None)
        )
        subgroup_rows.append(
            {
                "subgroup_type": "condition_unseen_in_train",
                "subgroup": str(val),
                "n": len(grp),
                "validate_spearman": rho,
            }
        )

    subgroups_df = pd.DataFrame(subgroup_rows)
    subgroups_df.to_csv(out / f"{exp_id}_subgroups.csv", index=False)

    # ---- Figure: predicted vs actual residual hexbin ------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(pred_validate_res, y_validate_res, gridsize=50, cmap="viridis", mincnt=1)
    ax.set_xlabel("predicted residual")
    ax.set_ylabel("actual residual")
    ax.set_title(
        f"exp04 predicted vs actual residual (validate)\nSpearman rho = {validate_spearman_residual:.3f}"
    )
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_residual_scatter.png", dpi=150)
    plt.close(fig)

    metrics = {
        "validate_spearman_residual": float(validate_spearman_residual),
        "train_cv_spearman_residual": train_cv_spearman_mean,
        "validate_average_precision": float(validate_ap),
        "baseline_degree_only_ap": float(baseline_degree_only_ap),
        "twoway_r2": float(twoway_r2),
        "n_train_rows": len(train_sub),
        "n_validate_rows": len(validate_sub),
    }

    if findings_fallback_notes:
        log.warning("Budget fallback taken: %s", "; ".join(findings_fallback_notes))

    metrics_meta_path = out / f"{exp_id}_run_notes.txt"
    metrics_meta_path.write_text(
        "\n".join(
            [
                f"n_iter_alternating_demeaning={n_iter}",
                f"prevalence_y_faers_signal_validate={prevalence:.4f}",
                f"train_cv_ap_mean_classifier={train_cv_ap_mean}",
                f"budget_fallback_taken={budget_risk}",
                *findings_fallback_notes,
            ]
        )
    )

    results.commit()
    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp04",
        title="Two-way-demeaned log PRR residual as the modelling target",
        hypothesis=(
            "Drug and condition intrinsic features predict the degree-free "
            "log(faers_prr) residual with Spearman rho >= 0.10 on validate, a larger "
            "effect relative to its null than the same features achieve on the binary "
            "Evans flag."
        ),
        approach=(
            "Restrict to faers_case_count >= 3. Two-way fixed effects on log PRR by "
            "alternating demeaning, fitted on train and applied to validate. HistGBM "
            "regressor on the residual, plus the binary classifier on identical rows for "
            "a four-cell prediction-vs-target comparison."
        ),
        label="other",
        features=["degree", "drug_intrinsic", "condition_intrinsic"],
        split="train/validate, grouped by primary target gene, case_count >= 3 subset",
    )

    try:
        metrics = run.remote(exp)
    except Exception as e:
        ln.complete(
            exp,
            metrics={"precondition_failed": 1.0},
            findings=f"Run raised an exception: {e!r}",
            failed=True,
        )
        raise
    print(metrics)

    validate_rho = metrics["validate_spearman_residual"]
    twoway_r2 = metrics["twoway_r2"]
    validate_ap = metrics["validate_average_precision"]
    baseline_ap = metrics["baseline_degree_only_ap"]

    findings = (
        f"Residual model: validate Spearman rho={validate_rho:.4f} on n={metrics['n_validate_rows']} "
        f"rows (null rho=0). Two-way fixed-effects structure removed R^2={twoway_r2:.4f} of the "
        f"variance in log(faers_prr) on train, fit by alternating demeaning (train only, applied "
        f"unchanged to validate; unseen drugs/conditions get offset 0 and are reported as a "
        f"separate subgroup in <exp_id>_subgroups.csv). "
        f"Binary comparison on the identical case_count>=3 subset: classifier validate AP={validate_ap:.4f} "
        f"vs degree-only baseline AP={baseline_ap:.4f} (prevalence stated in <exp_id>_run_notes.txt). "
        f"Four-cell trained-on x evaluated-against table is in <exp_id>_target_comparison.csv: compare "
        f"the residual model's AP against the binary flag and the binary model's Spearman rho against "
        f"the residual to judge whether the residual model also ranks the flag competitively "
        f"(the strong result) or only wins on its own target. "
        f"Recommendation: see <exp_id>_target_comparison.csv and the rho/AP values above for whether "
        f"later rounds should switch target -- if validate_spearman_residual is near zero while the "
        f"binary AP clears its baseline by a wide margin, the recommendation is to NOT switch, since "
        f"the features are predicting reporting structure rather than a degree-free pharmacological "
        f"signal; if rho clears 0.10 and the residual model also ranks the binary flag competitively, "
        f"the recommendation is to switch later rounds to the residual target. This subset (47.6% of "
        f"rows, case_count>=3) is not a random subsample -- it is biased toward well-reported pairs, "
        f"correlated with drug age and market size, so these numbers are not directly comparable to "
        f"exp02's headline AP on all rows."
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"results/{exp}/{exp}_residual_model.csv",
            f"results/{exp}/{exp}_target_comparison.csv",
            f"results/{exp}/{exp}_offsets.csv",
            f"results/{exp}/{exp}_twoway_fit.csv",
            f"results/{exp}/{exp}_residual_scatter.png",
            f"results/{exp}/{exp}_subgroups.csv",
        ],
        next_steps=(
            "Pull results/<exp_id>/ from the bridge-results volume and inspect "
            "<exp_id>_subgroups.csv for the case-count-stratum breakdown before generalizing the "
            "headline rho -- if signal is concentrated in the 21+ stratum, the 'closer to causal' "
            "claim needs that qualifier per the spec's traps section."
        ),
    )
