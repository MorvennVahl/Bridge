"""Modal experiment: one-way condition-demeaned log(faers_prr) residual, evaluated by
macro within-drug Spearman correlation -- a target and metric pair that both exist on
held-out (drug-grouped) validate rows.

Implements experiments/exp09_within_drug_residual.md exactly, reusing exp04's
data-loading / feature-block-selection / HistGBM patterns where they still apply.
Supersedes exp04 (exp_20260922_3f4a30): exp04's two-way (drug + condition) fixed
effects imputed drug offset 0 for every held-out drug under the drug-grouped split,
corrupting the comparison. This experiment removes only the condition level and
evaluates within drug, so the unlearnable drug level is differenced out of the metric
instead of imputed into the target.
"""

from __future__ import annotations

import logging
import pathlib

import modal

app = modal.App("bridge-exp09")

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
EXPECTED_TRAIN_SUBSET_ROWS = 344_651
EXPECTED_VALIDATE_SUBSET_ROWS = 206_585
EXPECTED_LOG_PRR_MEAN = 0.283
EXPECTED_LOG_PRR_SD = 1.052
TOLERANCE_ROWS = 0.02  # 2% relative tolerance on row counts
TOLERANCE_MOMENT = 0.05  # absolute tolerance on mean/sd of log(faers_prr)

# exp04's two-way fixed-effects R^2 on the same train subset -- hardcoded reference
# point, used only to derive "the drug level's share" of variance (see spec).
EXP04_TWOWAY_R2 = 0.4700
EXP04_SUPERSEDED_EXP_ID = "exp_20260922_3f4a30"

MIN_ROWS_PER_DRUG_FOR_RHO = 5  # Spearman on fewer points than this is unstable.
MIN_PAIRS_FOR_DRUG_AUC = 20  # matches METRIC.md's drug_macro_auc eligibility rule.

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

# Columns that must never appear in the feature matrix: the target's own ingredients
# plus every other label/outcome column. Checked explicitly at runtime (leak audit).
LEAK_BLACKLIST = {
    "faers_case_count",
    "faers_prr",
    "faers_ror",
    "faers_chi_square",
    "y_faers_signal",
    "y_semmeddb_causes",
    "y_semmeddb_treats",
    "y_any_harm",
    "in_faers",
    "in_semmeddb",
    "in_eu_label",
    "semmeddb_harm_sentences",
    "semmeddb_benefit_sentences",
    "semmeddb_negated_sentences",
}


def _assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"Precondition failed: {name} = {actual} (expected {expected} +/- {tol})"
        )


def _assert_row_count(name: str, actual: int, expected: int, tol_frac: float) -> None:
    if abs(actual - expected) > tol_frac * expected:
        raise AssertionError(f"Precondition failed: {name} = {actual} rows (expected ~{expected})")


def _select_drug_feature_columns(data_dict) -> list[str]:  # type: ignore[no-untyped-def]
    dd = data_dict[data_dict["table"] == "ingredient_features.csv"]
    dd = dd[dd["block"].isin(INCLUDED_BLOCKS)]
    cols = [c for c in dd["column"].tolist() if c not in EXCLUDED_COLUMNS]
    return cols


def _macro_within_drug_spearman(
    df,  # type: ignore[no-untyped-def]
    drug_col: str,
    target_col: str,
    pred_col: str,
    min_rows: int,
):
    """Macro-average Spearman rho of pred vs target within each drug with
    >= min_rows rows. Returns (macro_rho, n_drugs_scored, per_drug_df)."""
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    rows = []
    for drug_id, grp in df.groupby(drug_col):
        if len(grp) < min_rows:
            continue
        if grp[target_col].nunique() < 2 or grp[pred_col].nunique() < 2:
            continue
        rho, _ = spearmanr(grp[pred_col], grp[target_col])
        if np.isnan(rho):
            continue
        rows.append({"drug_id": drug_id, "n_rows": len(grp), "spearman_rho": rho})

    per_drug_df = pd.DataFrame(rows)
    if len(per_drug_df) == 0:
        return float("nan"), 0, per_drug_df
    macro_rho = float(per_drug_df["spearman_rho"].mean())
    return macro_rho, len(per_drug_df), per_drug_df


def _drug_macro_auc(
    df,  # type: ignore[no-untyped-def]
    drug_col: str,
    y_col: str,
    score_col: str,
    min_pairs: int,
):
    """Macro-average ROC-AUC of score vs binary y within each drug with
    >= min_pairs rows and both classes present."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    aucs = []
    for _drug_id, grp in df.groupby(drug_col):
        if len(grp) < min_pairs:
            continue
        y = grp[y_col].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, grp[score_col].to_numpy())
        aucs.append(auc)
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp09")

    import matplotlib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

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
            return {
                "precondition_failed": 1.0,
                "precondition_error_message": msg,
            }

    try:
        log.info("Reading train/validate splits")
        train = pd.read_csv("/data/splits/train.csv")
        validate = pd.read_csv("/data/splits/validate.csv")

        # ---- Step 2: restrict to case_count >= 3 and faers_prr > 0 ----------------
        train_sub = train[
            (train["faers_case_count"] >= CASE_COUNT_FLOOR) & (train["faers_prr"] > 0)
        ].copy()
        validate_sub = validate[
            (validate["faers_case_count"] >= CASE_COUNT_FLOOR) & (validate["faers_prr"] > 0)
        ].copy()

        log.info("train_sub=%d validate_sub=%d", len(train_sub), len(validate_sub))
        _assert_row_count(
            "n_train_rows", len(train_sub), EXPECTED_TRAIN_SUBSET_ROWS, TOLERANCE_ROWS
        )
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
    except AssertionError as exc:
        log.error("Precondition failed: %s", exc)
        return {"precondition_failed": 1.0, "precondition_error_message": str(exc)}

    # ---- Step 3: one-way condition demeaning, TRAIN ONLY ---------------------------
    log.info("Fitting one-way condition fixed effects (train only, simple groupby mean)")
    grand_mean = float(train_sub["z"].mean())
    condition_offset = train_sub.groupby("condition_concept_id")["z"].mean() - grand_mean
    condition_n = train_sub.groupby("condition_concept_id").size()

    def _apply_condition_offset(df):  # type: ignore[no-untyped-def]
        df = df.copy()
        c_off = df["condition_concept_id"].map(condition_offset)
        df["condition_unseen"] = ~df["condition_concept_id"].isin(condition_offset.index)
        df["condition_offset"] = c_off.fillna(0.0).to_numpy()
        df["r"] = df["z"] - grand_mean - df["condition_offset"]
        return df

    train_sub = _apply_condition_offset(train_sub)
    # ---- Step 4: apply train offsets to validate (never re-estimate) --------------
    validate_sub = _apply_condition_offset(validate_sub)

    # ---- Variance share removed by condition offsets alone (train) ----------------
    z_centered = train_sub["z"] - grand_mean
    ss_tot = float((z_centered**2).sum())
    ss_res_condition_only = float((train_sub["r"] ** 2).sum())
    condition_r2 = 1.0 - ss_res_condition_only / ss_tot if ss_tot > 0 else float("nan")
    log.info("Condition-only one-way structure R^2 on train z = %.4f", condition_r2)

    # exp04's two-way R^2 minus this one-way R^2 = "the drug level's share".
    drug_r2_share = EXP04_TWOWAY_R2 - condition_r2
    residual_share = max(0.0, 1.0 - condition_r2 - drug_r2_share)

    variance_decomposition_df = pd.DataFrame(
        [
            {"component": "condition_level", "r2_share": condition_r2},
            {
                "component": "drug_level",
                "r2_share": drug_r2_share,
                "note": (
                    f"exp04 two-way R2 ({EXP04_TWOWAY_R2:.4f}, hardcoded reference, "
                    f"cite {EXP04_SUPERSEDED_EXP_ID}) minus this one-way condition R2"
                ),
            },
            {"component": "residual", "r2_share": residual_share},
        ]
    )
    variance_decomposition_df.to_csv(out / f"{exp_id}_variance_decomposition.csv", index=False)

    condition_offsets_df = pd.DataFrame(
        {
            "condition_concept_id": condition_offset.index,
            "condition_offset": condition_offset.values,
        }
    )
    condition_offsets_df["n"] = (
        condition_offsets_df["condition_concept_id"].map(condition_n).to_numpy()
    )
    condition_offsets_df.to_csv(out / f"{exp_id}_condition_offsets.csv", index=False)

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

    # degree terms (train-only, applied to validate; unseen -> train median).
    # Do NOT use faers_case_count/faers_prr/faers_chi_square here: the target's own
    # ingredients. record_count comes from condition_features_basic.csv (p_c below).
    drug_degree_train = train.groupby("ingredient_concept_id").size().rename("drug_degree")
    cond_degree_train = train.groupby("condition_concept_id").size().rename("condition_degree")
    drug_degree_median = float(drug_degree_train.median())
    cond_degree_median = float(cond_degree_train.median())

    # p_c: condition's train flag rate for y_faers_signal -- explicit baseline column.
    p_c_train = train.groupby("condition_concept_id")["y_faers_signal"].mean().rename("p_c")
    p_c_median = float(p_c_train.median())

    def _add_common_features(df):  # type: ignore[no-untyped-def]
        df = df.merge(drug_features, on="ingredient_concept_id", how="left")
        df = df.merge(condition_features, on="condition_concept_id", how="left")
        df = df.merge(group_onehot, on="condition_concept_id", how="left")
        df["drug_degree"] = np.log1p(
            df["ingredient_concept_id"].map(drug_degree_train).fillna(drug_degree_median)
        )
        df["condition_degree"] = np.log1p(
            df["condition_concept_id"].map(cond_degree_train).fillna(cond_degree_median)
        )
        if "record_count" in df.columns:
            df["record_count"] = np.log1p(df["record_count"].fillna(0.0))
        df["p_c"] = df["condition_concept_id"].map(p_c_train).fillna(p_c_median)
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
        "condition_offset",
        "condition_unseen",
        "r",
        "condition_name",
    }
    feature_cols = [c for c in train_sub.columns if c not in non_feature_cols]
    numeric_feature_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_sub[c]) or pd.api.types.is_bool_dtype(train_sub[c]):
            numeric_feature_cols.append(c)
    feature_cols = numeric_feature_cols
    log.info("Using %d feature columns (%d group one-hots)", len(feature_cols), len(group_cols))

    # ---- Leak audit: assert none of the blacklisted columns made it into features -
    leaked = sorted(set(feature_cols) & LEAK_BLACKLIST)
    if leaked:
        msg = f"Leak audit failed: blacklisted columns present in feature matrix: {leaked}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}
    log.info("Leak audit passed: no blacklisted columns in %d feature columns", len(feature_cols))

    X_train = (  # noqa: N806 -- conventional sklearn name
        train_sub[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    )
    X_validate = (  # noqa: N806 -- conventional sklearn name
        validate_sub[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    )
    y_train_res = train_sub["r"].to_numpy()
    y_validate_res = validate_sub["r"].to_numpy()
    y_train_bin = train_sub["y_faers_signal"].to_numpy()
    y_validate_bin = validate_sub["y_faers_signal"].to_numpy()
    groups_train = train_sub["group_key"].to_numpy()

    # ---- Step 5: regressor on residual, grouped 3-fold CV + validate --------------
    log.info("Fitting HistGradientBoostingRegressor on residual with grouped 3-fold CV")
    gkf = GroupKFold(n_splits=3)
    cv_rho = []
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_train, y_train_res, groups=groups_train)):
        reg = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
        )
        reg.fit(X_train.iloc[tr_idx], y_train_res[tr_idx])
        pred = reg.predict(X_train.iloc[te_idx])
        fold_df = train_sub.iloc[te_idx][["ingredient_concept_id"]].copy()
        fold_df["r"] = y_train_res[te_idx]
        fold_df["pred"] = pred
        rho, n_drugs, _ = _macro_within_drug_spearman(
            fold_df, "ingredient_concept_id", "r", "pred", MIN_ROWS_PER_DRUG_FOR_RHO
        )
        cv_rho.append(rho)
        log.info("CV fold %d macro within-drug rho = %.4f (n_drugs=%d)", fold_i, rho, n_drugs)

    train_cv_macro_rho = float(np.nanmean(cv_rho))

    reg_full = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=63,
        early_stopping=False,
        random_state=0,
    )
    reg_full.fit(X_train, y_train_res)
    pred_validate_res = reg_full.predict(X_validate)
    validate_sub["pred_residual"] = pred_validate_res

    macro_rho_validate, n_drugs_scored, per_drug_df = _macro_within_drug_spearman(
        validate_sub, "ingredient_concept_id", "r", "pred_residual", MIN_ROWS_PER_DRUG_FOR_RHO
    )
    log.info(
        "Validate macro within-drug rho = %.4f (n_drugs_scored=%d, min_rows=%d)",
        macro_rho_validate,
        n_drugs_scored,
        MIN_ROWS_PER_DRUG_FOR_RHO,
    )
    per_drug_df.to_csv(out / f"{exp_id}_per_drug_rho.csv", index=False)

    residual_model_df = pd.DataFrame(
        [
            {
                "train_cv_macro_within_drug_spearman": train_cv_macro_rho,
                "validate_macro_within_drug_spearman": macro_rho_validate,
                "n_drugs_scored": n_drugs_scored,
                "min_rows_per_drug": MIN_ROWS_PER_DRUG_FOR_RHO,
                "n_train_rows": len(train_sub),
                "n_validate_rows": len(validate_sub),
            }
        ]
    )
    residual_model_df.to_csv(out / f"{exp_id}_residual_model.csv", index=False)

    # ---- Step 6: binary classifier on identical rows/features ---------------------
    log.info("Fitting HistGradientBoostingClassifier on y_faers_signal (identical rows/features)")
    budget_risk = False
    cv_ap_notes: list[str] = []
    try:
        cv_clf_auc = []
        for fold_i, (tr_idx, te_idx) in enumerate(
            gkf.split(X_train, y_train_bin, groups=groups_train)
        ):
            clf_cv = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.06,
                max_leaf_nodes=63,
                early_stopping=False,
                random_state=0,
            )
            clf_cv.fit(X_train.iloc[tr_idx], y_train_bin[tr_idx])
            proba = clf_cv.predict_proba(X_train.iloc[te_idx])[:, 1]
            fold_df = train_sub.iloc[te_idx][["ingredient_concept_id"]].copy()
            fold_df["y"] = y_train_bin[te_idx]
            fold_df["proba"] = proba
            auc, _n = _drug_macro_auc(
                fold_df, "ingredient_concept_id", "y", "proba", MIN_PAIRS_FOR_DRUG_AUC
            )
            cv_clf_auc.append(auc)
            log.info("Classifier CV fold %d drug_macro_auc = %.4f", fold_i, auc)
    except Exception as exc:  # pragma: no cover -- budget fallback path
        budget_risk = True
        cv_ap_notes.append(
            f"Dropped classifier grouped CV due to budget risk ({exc}); kept single train fit + validate pass."
        )

    clf_full = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=63,
        early_stopping=False,
        random_state=0,
    )
    clf_full.fit(X_train, y_train_bin)
    pred_validate_bin_proba = clf_full.predict_proba(X_validate)[:, 1]
    validate_sub["pred_classifier_proba"] = pred_validate_bin_proba

    # ---- Four-cell comparison: {regressor, classifier} x {within-drug rho, drug_macro_auc}
    # Cell 1: regressor's continuous output, scored by within-drug Spearman against r.
    resid_model_vs_rho, _n1, _ = _macro_within_drug_spearman(
        validate_sub, "ingredient_concept_id", "r", "pred_residual", MIN_ROWS_PER_DRUG_FOR_RHO
    )
    # Cell 2: regressor's continuous output used directly as ranking score for AUC.
    validate_sub["y_faers_signal"] = y_validate_bin
    resid_model_vs_auc, n_auc_resid = _drug_macro_auc(
        validate_sub,
        "ingredient_concept_id",
        "y_faers_signal",
        "pred_residual",
        MIN_PAIRS_FOR_DRUG_AUC,
    )
    # Cell 3: classifier's predicted probability correlated against r within drug.
    clf_model_vs_rho, _n3, _ = _macro_within_drug_spearman(
        validate_sub,
        "ingredient_concept_id",
        "r",
        "pred_classifier_proba",
        MIN_ROWS_PER_DRUG_FOR_RHO,
    )
    # Cell 4: classifier's predicted probability scored by drug_macro_auc (its own target).
    clf_model_vs_auc, n_auc_clf = _drug_macro_auc(
        validate_sub,
        "ingredient_concept_id",
        "y_faers_signal",
        "pred_classifier_proba",
        MIN_PAIRS_FOR_DRUG_AUC,
    )

    target_comparison_df = pd.DataFrame(
        [
            {
                "trained_on": "residual",
                "evaluated_by": "macro_within_drug_spearman",
                "value": resid_model_vs_rho,
            },
            {
                "trained_on": "residual",
                "evaluated_by": "drug_macro_auc",
                "value": resid_model_vs_auc,
                "n_drugs_scored": n_auc_resid,
            },
            {
                "trained_on": "classifier",
                "evaluated_by": "macro_within_drug_spearman",
                "value": clf_model_vs_rho,
            },
            {
                "trained_on": "classifier",
                "evaluated_by": "drug_macro_auc",
                "value": clf_model_vs_auc,
                "n_drugs_scored": n_auc_clf,
            },
        ]
    )
    target_comparison_df.to_csv(out / f"{exp_id}_target_comparison.csv", index=False)

    # ---- Heteroscedasticity by case-count strata -----------------------------------
    def _stratum(cc: int) -> str:
        if cc <= 5:
            return "3-5"
        if cc <= 20:
            return "6-20"
        return "21+"

    validate_sub["case_count_stratum"] = validate_sub["faers_case_count"].apply(_stratum)
    strata_rows = []
    for stratum, grp in validate_sub.groupby("case_count_stratum"):
        rho, n_drugs, _ = _macro_within_drug_spearman(
            grp, "ingredient_concept_id", "r", "pred_residual", MIN_ROWS_PER_DRUG_FOR_RHO
        )
        strata_rows.append(
            {
                "case_count_stratum": stratum,
                "n_rows": len(grp),
                "n_drugs_scored": n_drugs,
                "macro_within_drug_spearman": rho,
            }
        )
    strata_df = pd.DataFrame(strata_rows)
    strata_df.to_csv(out / f"{exp_id}_strata.csv", index=False)

    # ---- Figure: predicted vs actual residual hexbin -------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(pred_validate_res, y_validate_res, gridsize=50, cmap="viridis", mincnt=1)
    ax.set_xlabel("predicted residual")
    ax.set_ylabel("actual residual (condition-demeaned log PRR)")
    ax.set_title(
        f"exp09 predicted vs actual within-drug residual (validate)\n"
        f"macro within-drug Spearman rho = {macro_rho_validate:.3f}"
    )
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_residual_scatter.png", dpi=150)
    plt.close(fig)

    metrics = {
        "macro_within_drug_spearman": float(macro_rho_validate),
        "train_cv_macro_within_drug_spearman": float(train_cv_macro_rho),
        "drug_macro_auc_from_residual_model": float(resid_model_vs_auc),
        "drug_macro_auc_from_classifier": float(clf_model_vs_auc),
        "condition_variance_share": float(condition_r2),
        "drug_variance_share": float(drug_r2_share),
        "n_train_rows": len(train_sub),
        "n_validate_rows": len(validate_sub),
        "n_drugs_scored": n_drugs_scored,
    }

    if cv_ap_notes:
        log.warning("Budget fallback taken: %s", "; ".join(cv_ap_notes))

    run_notes_path = out / f"{exp_id}_run_notes.txt"
    run_notes_path.write_text(
        "\n".join(
            [
                f"min_rows_per_drug_for_rho={MIN_ROWS_PER_DRUG_FOR_RHO}",
                f"min_pairs_for_drug_auc={MIN_PAIRS_FOR_DRUG_AUC}",
                f"budget_fallback_taken={budget_risk}",
                *cv_ap_notes,
            ]
        )
    )

    results.commit()
    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp09",
        title="Within-drug log-PRR residual: a target that exists on held-out drugs",
        hypothesis=(
            "Removing only the condition level and evaluating within drug yields macro "
            "within-drug Spearman >= 0.10 on validate, and that ranking scores "
            "drug_macro_auc on the Evans flag competitively with the intrinsic union."
        ),
        approach=(
            "One-way condition demeaning on train applied to validate; within-drug rank "
            "correlation as the metric so the unlearnable drug level is differenced out of "
            "the metric rather than imputed to zero in the target; four-cell cross-target "
            "comparison on identical rows; case-count strata."
        ),
        label="other",
        features=["degree", "p_c", "drug_intrinsic", "condition_intrinsic"],
        split="train/validate, grouped by primary target gene, case_count >= 3 subset",
    )

    metrics = run.remote(exp)
    print(metrics)

    if metrics.get("precondition_failed"):
        ln.complete(
            exp,
            metrics={"precondition_failed": 1.0},
            findings=f"Run failed a precondition: {metrics.get('precondition_error_message')}",
            failed=True,
        )
        return

    macro_rho = metrics["macro_within_drug_spearman"]
    train_cv_rho = metrics["train_cv_macro_within_drug_spearman"]
    auc_resid = metrics["drug_macro_auc_from_residual_model"]
    auc_clf = metrics["drug_macro_auc_from_classifier"]
    cond_share = metrics["condition_variance_share"]
    drug_share = metrics["drug_variance_share"]

    recommendation = (
        "model the continuous residual in later rounds"
        if macro_rho >= 0.10 and auc_resid >= auc_clf - 0.02
        else "stay with the binary flag; the continuous target does not clear its bar here"
    )

    findings = (
        f"exp04's validate Spearman rho=-0.3064 was a target-mismatch artefact: its two-way "
        f"drug+condition fixed effects imputed drug offset 0 for every held-out drug under "
        f"the drug-grouped split, so train and validate scored two different quantities. "
        f"This entry ({exp}) supersedes {EXP04_SUPERSEDED_EXP_ID} with a one-way "
        f"condition-only demeaned target evaluated by macro within-drug Spearman correlation, "
        f"which is invariant to any additive per-drug constant and therefore estimable on "
        f"held-out drugs.\n\n"
        f"Macro within-drug Spearman rho on validate = {macro_rho:.4f} "
        f"(n_drugs_scored={metrics['n_drugs_scored']}, min {MIN_ROWS_PER_DRUG_FOR_RHO} rows/drug; "
        f"train CV macro rho = {train_cv_rho:.4f}).\n\n"
        f"Four-cell comparison (see <exp_id>_target_comparison.csv): regressor's continuous "
        f"output scores drug_macro_auc={auc_resid:.4f} on the binary Evans flag; classifier "
        f"scores drug_macro_auc={auc_clf:.4f} on its own target. Cross-scoring rows are also "
        f"in that file (classifier proba vs within-drug rho, regressor pred vs rho).\n\n"
        f"Variance decomposition on train (<exp_id>_variance_decomposition.csv): condition "
        f"level R^2={cond_share:.4f}; drug level (exp04's two-way R^2={EXP04_TWOWAY_R2:.4f} "
        f"minus this one-way R^2) share={drug_share:.4f}.\n\n"
        f"Strata breakdown in <exp_id>_strata.csv (case_count 3-5 / 6-20 / 21+) shows whether "
        f"signal concentrates in higher-count, lower-noise pairs.\n\n"
        f"Recommendation: {recommendation} (rho={macro_rho:.4f} vs the 0.10 hypothesis floor; "
        f"residual-model AUC {auc_resid:.4f} vs classifier AUC {auc_clf:.4f})."
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"results/{exp}/{exp}_residual_model.csv",
            f"results/{exp}/{exp}_target_comparison.csv",
            f"results/{exp}/{exp}_per_drug_rho.csv",
            f"results/{exp}/{exp}_condition_offsets.csv",
            f"results/{exp}/{exp}_variance_decomposition.csv",
            f"results/{exp}/{exp}_strata.csv",
            f"results/{exp}/{exp}_residual_scatter.png",
        ],
        next_steps=(
            "Pull results/<exp_id>/ from the bridge-results volume; inspect "
            "<exp_id>_strata.csv for whether signal is concentrated in the 21+ case-count "
            "stratum before generalizing the headline rho, and <exp_id>_per_drug_rho.csv for "
            "per-drug spread before treating the macro average as uniform across drugs."
        ),
        supersedes=EXP04_SUPERSEDED_EXP_ID,
    )
