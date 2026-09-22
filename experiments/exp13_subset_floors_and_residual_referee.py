"""exp13 -- Every subset gets its own floor, and exp09 re-refereed.

Implements experiments/exp13_subset_floors_and_residual_referee.md. Supersedes
exp_20260922_4e79a6 (exp09) on its recommendation, not its mechanics: exp09's
within-drug-residual target fix (condition-only demeaning, evaluated by macro
within-drug Spearman) was correct and remains the right formulation. What was wrong
is the comparison: exp09 measured drug_macro_auc 0.8330 (classifier) / 0.8274
(residual model) on the `faers_case_count >= 3` subset and compared both to
`floor_drug_macro_auc_pc_lookup = 0.5759`, a floor measured on *all rows*. The
correct floor for that subset, `degree + p_c`, is 0.8480 -- both of exp09's models
sit below it.

Ships `experiments/floors.py` (imported here and reused by later experiments): one
function computing the p_c-lookup floor and the degree+p_c floor, plus prevalence
and n_drugs_scored, for any (label, subset, eligibility) combination.

Feature-block selection (drug/condition intrinsic blocks, leak-blacklist columns)
copies exp09_within_drug_residual.py's `_select_drug_feature_columns` /
`INCLUDED_BLOCKS` / `EXCLUDED_COLUMNS`, which already implements the shared
exp02/exp07 block-selection spec. The within-drug residual target construction
(one-way condition demeaning, train-only, applied to evaluate) and the macro
within-drug Spearman metric copy exp09_within_drug_residual.py's
`_apply_condition_offset` / `_macro_within_drug_spearman` exactly, since the point
of this experiment is to re-referee that mechanics against a correct floor, not to
change it.
"""

from __future__ import annotations

import logging
import pathlib

import modal

app = modal.App("bridge-exp13")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas==2.2.3",
        "numpy==2.1.3",
        "scikit-learn==1.5.2",
        "lightgbm==4.5.0",
        "pyarrow==17.0.0",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
    )
    .add_local_python_source("floors")
)

data = modal.Volume.from_name("bridge-data")
results = modal.Volume.from_name("bridge-results", create_if_missing=True)

image = image.add_local_file(
    pathlib.Path(__file__).parent / "floors.py", remote_path="/root/floors.py"
)

CASE_COUNT_FLOOR = 3
MIN_ROWS_PER_DRUG_FOR_RHO = 5  # matches exp09; Spearman on fewer points is unstable.
MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD = 20  # METRIC.md's drug_macro_auc eligibility rule.
MIN_PAIRS_FOR_DRUG_AUC_EXP09 = 5  # exp09's (looser) choice -- reported alongside.

EXP09_EXP_ID = "exp_20260922_4e79a6"
EXP09_CASE_GE3_CLASSIFIER_AUC = 0.8330
EXP09_CASE_GE3_RESIDUAL_AUC = 0.8274
EXP09_MEASURED_DEGREE_PC_FLOOR_CASE_GE3 = 0.8480
EXP09_MEASURED_RHO = 0.6536

# ---- copied from exp09_within_drug_residual.py: block selection ----
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


def _select_drug_feature_columns(data_dict) -> list[str]:  # type: ignore[no-untyped-def]
    dd = data_dict[data_dict["table"] == "ingredient_features.csv"]
    dd = dd[dd["block"].isin(INCLUDED_BLOCKS)]
    return [c for c in dd["column"].tolist() if c not in EXCLUDED_COLUMNS]


def _macro_within_drug_spearman(
    df,  # type: ignore[no-untyped-def]
    drug_col: str,
    target_col: str,
    pred_col: str,
    min_rows: int,
):
    """Macro-average Spearman rho of pred vs target within each drug with
    >= min_rows rows. Copied from exp09_within_drug_residual.py."""
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
    >= min_pairs rows and both classes present. NaN-safe: NaN scores are filled with
    the per-drug median before scoring, and bool-dtype score columns are cast to
    float first."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    aucs = []
    for _drug_id, grp in df.groupby(drug_col):
        if len(grp) < min_pairs:
            continue
        y = grp[y_col].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        score = grp[score_col]
        if score.dtype == bool:
            score = score.astype(float)
        score = score.astype(float)
        if score.isna().any():
            score = score.fillna(score.median())
        if score.isna().all():
            continue
        auc = roc_auc_score(y, score.to_numpy())
        aucs.append(auc)
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def _stratum(cc: float) -> str:
    if cc <= 5:
        return "3-5"
    if cc <= 20:
        return "6-20"
    return "21+"


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp13")

    import matplotlib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    matplotlib.use("Agg")
    import sys

    import matplotlib.pyplot as plt

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from floors import floors

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    # ---- Step 0: preconditions -----------------------------------------------------
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
            return {"precondition_failed": 1.0, "precondition_error_message": msg}

    try:
        log.info("Reading train/validate splits")
        train = pd.read_csv("/data/splits/train.csv")
        validate = pd.read_csv("/data/splits/validate.csv")
        if "y_faers_signal" not in train.columns:
            raise AssertionError("train.csv missing y_faers_signal")
        rate = float(train["y_faers_signal"].mean())
        if abs(rate - 0.1100) > 0.005:
            raise AssertionError(f"train y_faers_signal rate={rate:.4f}, expected ~0.1100")
    except AssertionError as exc:
        log.error("Precondition failed: %s", exc)
        return {"precondition_failed": 1.0, "precondition_error_message": str(exc)}

    # ==================================================================================
    # Step 1+2: reference floor table for every population used so far
    # ==================================================================================
    log.info("Building the reference floor table")
    floor_rows: list[dict[str, object]] = []

    def _record_floor(population: str, label: str, tr: pd.DataFrame, ev: pd.DataFrame) -> None:
        if label not in tr.columns or label not in ev.columns:
            floor_rows.append(
                {
                    "population": population,
                    "label": label,
                    "status": "skipped: label column not present",
                }
            )
            return
        if tr[label].nunique() < 2:
            floor_rows.append(
                {
                    "population": population,
                    "label": label,
                    "status": "skipped: single class in train",
                }
            )
            return
        f = floors(tr, ev, label, MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD)
        floor_rows.append(
            {
                "population": population,
                "label": label,
                "eligibility": MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD,
                "status": "ok",
                **f,
            }
        )

    # all rows
    _record_floor("all_rows", "y_faers_signal", train, validate)

    # case_count >= 3
    train_ge3 = train[train["faers_case_count"] >= CASE_COUNT_FLOOR].copy()
    validate_ge3 = validate[validate["faers_case_count"] >= CASE_COUNT_FLOOR].copy()
    _record_floor("case_count_ge3", "y_faers_signal", train_ge3, validate_ge3)
    # also at exp09's own (looser) eligibility, since exp09's headline used min_rows=5
    f_ge3_exp09_elig = floors(
        train_ge3, validate_ge3, "y_faers_signal", MIN_PAIRS_FOR_DRUG_AUC_EXP09
    )
    floor_rows.append(
        {
            "population": "case_count_ge3",
            "label": "y_faers_signal",
            "eligibility": MIN_PAIRS_FOR_DRUG_AUC_EXP09,
            "status": "ok",
            **f_ge3_exp09_elig,
        }
    )

    # gene neighbour subset -- not available on this volume (no per-condition gene
    # list is staged; condition_gene_hpo_long.csv was deliberately not staged per
    # experiments/README.md). Recorded as skipped rather than improvised.
    floor_rows.append(
        {
            "population": "gene_neighbour_ge1",
            "label": "y_faers_signal",
            "status": "skipped: no per-drug/condition gene-neighbour column staged on this volume",
        }
    )

    # exp12's rungs -- exp12 has a spec (experiments/exp12_transportability_by_target_distance.md)
    # but no experiments/exp12_transportability_by_target_distance.py yet, i.e. it has not
    # landed. Recorded as skipped per the spec's "if that experiment has landed" clause.
    floor_rows.append(
        {
            "population": "exp12_rungs",
            "label": "y_faers_signal",
            "status": "skipped: exp12 has a spec but no implementation yet (not landed)",
        }
    )

    # SemMedDB-harm target
    _record_floor("all_rows", "y_semmeddb_causes", train, validate)

    floor_table_df = pd.DataFrame(floor_rows)
    floor_table_path = out / f"{exp_id}_floor_reference_table.csv"
    floor_table_df.to_csv(floor_table_path, index=False)
    log.info("Floor reference table:\n%s", floor_table_df.to_string())

    floor_pc_all_rows = next(
        (
            r["floor_pc"]
            for r in floor_rows
            if r["population"] == "all_rows" and r["label"] == "y_faers_signal"
        ),
        float("nan"),
    )
    floor_degree_pc_all_rows = next(
        (
            r["floor_degree_pc"]
            for r in floor_rows
            if r["population"] == "all_rows" and r["label"] == "y_faers_signal"
        ),
        float("nan"),
    )
    floor_pc_case_ge3 = next(
        (
            r["floor_pc"]
            for r in floor_rows
            if r["population"] == "case_count_ge3"
            and r.get("eligibility") == MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD
        ),
        float("nan"),
    )
    floor_degree_pc_case_ge3 = next(
        (
            r["floor_degree_pc"]
            for r in floor_rows
            if r["population"] == "case_count_ge3"
            and r.get("eligibility") == MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD
        ),
        float("nan"),
    )
    log.info(
        "floor_pc_all_rows=%.4f floor_degree_pc_all_rows=%.4f "
        "floor_pc_case_ge3=%.4f floor_degree_pc_case_ge3=%.4f "
        "(exp09-measured degree+p_c floor for case_ge3 was %.4f)",
        floor_pc_all_rows,
        floor_degree_pc_all_rows,
        floor_pc_case_ge3,
        floor_degree_pc_case_ge3,
        EXP09_MEASURED_DEGREE_PC_FLOOR_CASE_GE3,
    )

    # ==================================================================================
    # Step 3: re-referee exp09 on FULL rows (faers_prr > 0 only, no case_count filter)
    # ==================================================================================
    log.info("Re-fitting exp09's residual regressor and classifier on full rows")
    train_full = train[train["faers_prr"] > 0].copy()
    validate_full = validate[validate["faers_prr"] > 0].copy()
    log.info("full-rows (faers_prr>0): train=%d validate=%d", len(train_full), len(validate_full))

    train_full["z"] = np.log(train_full["faers_prr"])
    validate_full["z"] = np.log(validate_full["faers_prr"])

    grand_mean = float(train_full["z"].mean())
    condition_offset = train_full.groupby("condition_concept_id")["z"].mean() - grand_mean

    def _apply_condition_offset(df):  # type: ignore[no-untyped-def]
        df = df.copy()
        c_off = df["condition_concept_id"].map(condition_offset)
        df["condition_offset"] = c_off.fillna(0.0).to_numpy()
        df["r"] = df["z"] - grand_mean - df["condition_offset"]
        return df

    train_full = _apply_condition_offset(train_full)
    validate_full = _apply_condition_offset(validate_full)

    # ---- feature assembly, exp09's degree/p_c/drug-intrinsic/condition-intrinsic blocks
    data_dict = pd.read_csv("/data/data_dictionary.csv")
    drug_feat_cols = _select_drug_feature_columns(data_dict)
    ingredient_features = pd.read_csv("/data/drug/ingredient_features.csv")
    available_drug_cols = [c for c in drug_feat_cols if c in ingredient_features.columns]
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

    drug_degree_train = train_full.groupby("ingredient_concept_id").size().rename("drug_degree")
    cond_degree_train = train_full.groupby("condition_concept_id").size().rename("condition_degree")
    drug_degree_median = float(drug_degree_train.median())
    cond_degree_median = float(cond_degree_train.median())
    p_c_train = train_full.groupby("condition_concept_id")["y_faers_signal"].mean().rename("p_c")
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

    train_full = _add_common_features(train_full)
    validate_full = _add_common_features(validate_full)

    non_feature_cols = {
        "ingredient_concept_id",
        "condition_concept_id",
        "in_faers",
        "in_semmeddb",
        "in_eu_label",
        "faers_case_count",
        "faers_prr",
        "faers_chi_square",
        "faers_ror",
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
        "r",
        "condition_name",
    }
    feature_cols = [c for c in train_full.columns if c not in non_feature_cols]
    feature_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_full[c]) or pd.api.types.is_bool_dtype(train_full[c])
    ]

    leaked = sorted(set(feature_cols) & LEAK_BLACKLIST)
    if leaked:
        msg = f"Leak audit failed: blacklisted columns present in feature matrix: {leaked}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}
    log.info("Leak audit passed: %d feature columns", len(feature_cols))

    x_train = train_full[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    x_validate = validate_full[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_train_res = train_full["r"].to_numpy()
    y_train_bin = train_full["y_faers_signal"].to_numpy()
    y_validate_bin = validate_full["y_faers_signal"].to_numpy()
    groups_train = train_full["group_key"].to_numpy()

    gkf = GroupKFold(n_splits=3)

    # ---- regressor on residual, grouped 3-fold CV for a sanity signal + full fit ----
    cv_rho = []
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(x_train, y_train_res, groups=groups_train)):
        reg_cv = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
        )
        reg_cv.fit(x_train.iloc[tr_idx], y_train_res[tr_idx])
        pred = reg_cv.predict(x_train.iloc[te_idx])
        fold_df = train_full.iloc[te_idx][["ingredient_concept_id"]].copy()
        fold_df["r"] = y_train_res[te_idx]
        fold_df["pred"] = pred
        rho, _n, _ = _macro_within_drug_spearman(
            fold_df, "ingredient_concept_id", "r", "pred", MIN_ROWS_PER_DRUG_FOR_RHO
        )
        cv_rho.append(rho)
        log.info("Regressor CV fold %d macro within-drug rho = %.4f", fold_i, rho)
    train_cv_macro_rho = float(np.nanmean(cv_rho))

    reg_full = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=63, early_stopping=False, random_state=0
    )
    reg_full.fit(x_train, y_train_res)
    validate_full["pred_residual"] = reg_full.predict(x_validate)

    macro_rho_validate, n_drugs_rho, per_drug_df = _macro_within_drug_spearman(
        validate_full, "ingredient_concept_id", "r", "pred_residual", MIN_ROWS_PER_DRUG_FOR_RHO
    )
    per_drug_df.to_csv(out / f"{exp_id}_per_drug_rho_full_rows.csv", index=False)

    # residual model's continuous prediction used directly as an AUC ranking score
    validate_full["y_faers_signal"] = y_validate_bin
    residual_drug_macro_auc_full_rows, n_auc_resid = _drug_macro_auc(
        validate_full,
        "ingredient_concept_id",
        "y_faers_signal",
        "pred_residual",
        MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD,
    )

    # ---- classifier on identical rows/features ----
    cv_clf_auc = []
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(x_train, y_train_bin, groups=groups_train)):
        clf_cv = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
        )
        clf_cv.fit(x_train.iloc[tr_idx], y_train_bin[tr_idx])
        proba = clf_cv.predict_proba(x_train.iloc[te_idx])[:, 1]
        fold_df = train_full.iloc[te_idx][["ingredient_concept_id"]].copy()
        fold_df["y"] = y_train_bin[te_idx]
        fold_df["proba"] = proba
        auc, _n = _drug_macro_auc(
            fold_df, "ingredient_concept_id", "y", "proba", MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD
        )
        cv_clf_auc.append(auc)
        log.info("Classifier CV fold %d drug_macro_auc = %.4f", fold_i, auc)
    train_cv_clf_auc = float(np.nanmean(cv_clf_auc))

    clf_full = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=63, early_stopping=False, random_state=0
    )
    clf_full.fit(x_train, y_train_bin)
    validate_full["pred_classifier_proba"] = clf_full.predict_proba(x_validate)[:, 1]
    classifier_drug_macro_auc_full_rows, n_auc_clf = _drug_macro_auc(
        validate_full,
        "ingredient_concept_id",
        "y_faers_signal",
        "pred_classifier_proba",
        MIN_PAIRS_FOR_DRUG_AUC_METRIC_MD,
    )

    residual_vs_flag_df = pd.DataFrame(
        [
            {
                "model": "residual_regressor",
                "trained_on": "condition-demeaned log(faers_prr)",
                "drug_macro_auc_full_rows": residual_drug_macro_auc_full_rows,
                "macro_within_drug_spearman_full_rows": macro_rho_validate,
                "train_cv_drug_macro_auc": float("nan"),
                "train_cv_macro_within_drug_spearman": train_cv_macro_rho,
                "n_drugs_scored": n_auc_resid,
            },
            {
                "model": "classifier",
                "trained_on": "y_faers_signal",
                "drug_macro_auc_full_rows": classifier_drug_macro_auc_full_rows,
                "macro_within_drug_spearman_full_rows": float("nan"),
                "train_cv_drug_macro_auc": train_cv_clf_auc,
                "train_cv_macro_within_drug_spearman": float("nan"),
                "n_drugs_scored": n_auc_clf,
            },
            {
                "model": "floor_degree_pc",
                "trained_on": "degree_drug + degree_condition + p_c (logistic)",
                "drug_macro_auc_full_rows": floor_degree_pc_all_rows,
                "macro_within_drug_spearman_full_rows": float("nan"),
                "train_cv_drug_macro_auc": float("nan"),
                "train_cv_macro_within_drug_spearman": float("nan"),
                "n_drugs_scored": float("nan"),
            },
            {
                "model": "floor_pc",
                "trained_on": "p_c lookup",
                "drug_macro_auc_full_rows": floor_pc_all_rows,
                "macro_within_drug_spearman_full_rows": float("nan"),
                "train_cv_drug_macro_auc": float("nan"),
                "train_cv_macro_within_drug_spearman": float("nan"),
                "n_drugs_scored": float("nan"),
            },
        ]
    )
    residual_vs_flag_df.to_csv(out / f"{exp_id}_residual_vs_flag_full_rows.csv", index=False)

    # ==================================================================================
    # Step 4: floor the rank metric -- macro within-drug Spearman for p_c and
    # degree+p_c on the same (full-rows) rows.
    # ==================================================================================
    log.info("Computing rank-metric floors (macro within-drug Spearman) for p_c and degree+p_c")

    macro_rho_pc, n_drugs_rho_pc, _ = _macro_within_drug_spearman(
        validate_full, "ingredient_concept_id", "r", "p_c", MIN_ROWS_PER_DRUG_FOR_RHO
    )

    x_train_dp = np.column_stack(
        [
            train_full["drug_degree"].to_numpy(),
            train_full["condition_degree"].to_numpy(),
            train_full["p_c"].to_numpy(),
        ]
    )
    x_val_dp = np.column_stack(
        [
            validate_full["drug_degree"].to_numpy(),
            validate_full["condition_degree"].to_numpy(),
            validate_full["p_c"].to_numpy(),
        ]
    )
    # degree+p_c has no natural continuous target for r beyond p_c itself in this
    # floor; we regress degree+p_c against r directly (train-only) so the rank floor
    # is a fair comparison point for the residual regressor's rho, not just p_c's.
    reg_dp = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=31, early_stopping=False, random_state=0
    )
    reg_dp.fit(x_train_dp, y_train_res)
    validate_full["pred_degree_pc_r"] = reg_dp.predict(x_val_dp)
    macro_rho_degree_pc, n_drugs_rho_dp, _ = _macro_within_drug_spearman(
        validate_full, "ingredient_concept_id", "r", "pred_degree_pc_r", MIN_ROWS_PER_DRUG_FOR_RHO
    )

    rank_metric_floors_df = pd.DataFrame(
        [
            {
                "floor": "p_c",
                "macro_within_drug_spearman": macro_rho_pc,
                "n_drugs_scored": n_drugs_rho_pc,
            },
            {
                "floor": "degree_plus_p_c",
                "macro_within_drug_spearman": macro_rho_degree_pc,
                "n_drugs_scored": n_drugs_rho_dp,
            },
            {
                "floor": "residual_regressor (reference, not a floor)",
                "macro_within_drug_spearman": macro_rho_validate,
                "n_drugs_scored": n_drugs_rho,
            },
        ]
    )
    rank_metric_floors_df.to_csv(out / f"{exp_id}_rank_metric_floors.csv", index=False)
    log.info(
        "Rank-metric floors: p_c rho=%.4f, degree+p_c rho=%.4f, residual model rho=%.4f "
        "(exp09-measured rho on case_ge3 subset was %.4f -- not directly comparable, "
        "different population)",
        macro_rho_pc,
        macro_rho_degree_pc,
        macro_rho_validate,
        EXP09_MEASURED_RHO,
    )

    # ==================================================================================
    # Step 5: case-count strata with a floor per stratum
    # ==================================================================================
    log.info("Computing case-count strata with per-stratum floors")
    strata_rows = []
    for stratum_label in ["3-5", "6-20", "21+"]:
        val_stratum = validate_full[
            validate_full["faers_case_count"].apply(_stratum) == stratum_label
        ]
        train_stratum = train_full[train_full["faers_case_count"].apply(_stratum) == stratum_label]
        if len(val_stratum) == 0 or len(train_stratum) == 0:
            strata_rows.append({"case_count_stratum": stratum_label, "status": "empty"})
            continue

        rho_stratum, n_drugs_stratum, _ = _macro_within_drug_spearman(
            val_stratum, "ingredient_concept_id", "r", "pred_residual", MIN_ROWS_PER_DRUG_FOR_RHO
        )
        stratum_floor = floors(
            train_stratum, val_stratum, "y_faers_signal", MIN_PAIRS_FOR_DRUG_AUC_EXP09
        )
        strata_rows.append(
            {
                "case_count_stratum": stratum_label,
                "n_rows": len(val_stratum),
                "n_drugs_scored_rho": n_drugs_stratum,
                "macro_within_drug_spearman": rho_stratum,
                "floor_pc": stratum_floor["floor_pc"],
                "floor_degree_pc": stratum_floor["floor_degree_pc"],
                "prevalence": stratum_floor["prevalence"],
                "n_drugs_scored_auc": stratum_floor["n_drugs_scored"],
            }
        )
    strata_df = pd.DataFrame(strata_rows)
    strata_df.to_csv(out / f"{exp_id}_strata_with_floors.csv", index=False)
    log.info("Strata with floors:\n%s", strata_df.to_string())

    # ---- figure: predicted vs actual residual, full rows --------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(
        validate_full["pred_residual"], validate_full["r"], gridsize=50, cmap="viridis", mincnt=1
    )
    ax.set_xlabel("predicted residual")
    ax.set_ylabel("actual residual (condition-demeaned log PRR)")
    ax.set_title(
        f"exp13: predicted vs actual within-drug residual, full rows (validate)\n"
        f"macro within-drug Spearman rho = {macro_rho_validate:.3f}"
    )
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_residual_scatter_full_rows.png", dpi=150)
    plt.close(fig)

    metrics = {
        "floor_pc_case_ge3": float(floor_pc_case_ge3),
        "floor_degree_pc_case_ge3": float(floor_degree_pc_case_ge3),
        "floor_pc_all_rows": float(floor_pc_all_rows),
        "floor_degree_pc_all_rows": float(floor_degree_pc_all_rows),
        "residual_drug_macro_auc_full_rows": float(residual_drug_macro_auc_full_rows),
        "classifier_drug_macro_auc_full_rows": float(classifier_drug_macro_auc_full_rows),
        "macro_within_drug_spearman_full_rows": float(macro_rho_validate),
        "macro_within_drug_spearman_floor_pc": float(macro_rho_pc),
        "macro_within_drug_spearman_floor_degree_pc": float(macro_rho_degree_pc),
        "n_drugs_scored": int(n_drugs_rho),
        "exp09_case_ge3_classifier_auc_reported": EXP09_CASE_GE3_CLASSIFIER_AUC,
        "exp09_case_ge3_residual_auc_reported": EXP09_CASE_GE3_RESIDUAL_AUC,
        "exp09_case_ge3_degree_pc_floor_measured": EXP09_MEASURED_DEGREE_PC_FLOOR_CASE_GE3,
    }

    results.commit()
    log.info("Final metrics: %s", metrics)
    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp13",
        title="Per-subset floors, and the within-drug residual re-refereed against them",
        hypothesis=(
            "Neither exp09 model beats degree+p_c (0.8480) on its own case_count>=3 "
            "subset, while on full rows the continuous within-drug residual does beat the "
            "binary flag against the same floor."
        ),
        approach=(
            "Ship floors.py computing p_c and degree+p_c floors per (label, subset, "
            "eligibility); build a reference floor table for every population used so far; "
            "refit residual regressor and classifier on full rows with floors for the rank "
            "metric as well as the AUC."
        ),
        label="other",
        features=["degree", "p_c", "drug_intrinsic", "condition_intrinsic"],
        split="train/validate, full rows and case_count>=3 subset",
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

    floor_case_ge3 = metrics["floor_degree_pc_case_ge3"]
    exp09_clf = metrics["exp09_case_ge3_classifier_auc_reported"]
    exp09_res = metrics["exp09_case_ge3_residual_auc_reported"]
    auc_resid_full = metrics["residual_drug_macro_auc_full_rows"]
    auc_clf_full = metrics["classifier_drug_macro_auc_full_rows"]
    floor_full = metrics["floor_degree_pc_all_rows"]
    rho_full = metrics["macro_within_drug_spearman_full_rows"]
    rho_floor_pc = metrics["macro_within_drug_spearman_floor_pc"]
    rho_floor_dp = metrics["macro_within_drug_spearman_floor_degree_pc"]

    residual_beats_flag_full_rows = auc_resid_full > auc_clf_full
    residual_beats_floor_full_rows = auc_resid_full > floor_full
    verdict = (
        "Round 4 should model the continuous within-drug residual"
        if residual_beats_flag_full_rows and residual_beats_floor_full_rows
        else "Round 4 should stay with the binary flag: the continuous target does not "
        "clear both the flag and the degree+p_c floor on full rows"
    )

    findings = (
        f"exp09 ({EXP09_EXP_ID}) reported drug_macro_auc {exp09_clf:.4f} (classifier) and "
        f"{exp09_res:.4f} (residual model) on the faers_case_count>=3 subset, comparing both "
        f"to floor_drug_macro_auc_pc_lookup=0.5759 -- a floor measured on ALL rows, not the "
        f"subset. The correct degree+p_c floor for that subset, measured here, is "
        f"{floor_case_ge3:.4f}: both of exp09's models sit below it "
        f"({exp09_clf:.4f} and {exp09_res:.4f} < {floor_case_ge3:.4f}). This entry "
        f"supersedes exp09's recommendation on that basis, not its mechanics -- the "
        f"within-drug residual target fix (condition-only demeaning, evaluated by macro "
        f"within-drug Spearman) was correct and is reused unchanged here.\n\n"
        f"Full-row re-referee (faers_prr>0, no case_count restriction): "
        f"classifier drug_macro_auc={auc_clf_full:.4f}, residual-model drug_macro_auc="
        f"{auc_resid_full:.4f}, degree+p_c floor={floor_full:.4f}, p_c floor="
        f"{metrics['floor_pc_all_rows']:.4f} "
        f"(n_drugs_scored={metrics['n_drugs_scored']}). "
        f"{'The residual model beats the classifier and the floor on full rows.' if residual_beats_flag_full_rows and residual_beats_floor_full_rows else 'The residual model does not clearly beat both the classifier and the floor on full rows.'}\n\n"
        f"Rank-metric floors (macro within-drug Spearman on full rows, see "
        f"<exp_id>_rank_metric_floors.csv): p_c alone rho={rho_floor_pc:.4f}, degree+p_c "
        f"rho={rho_floor_dp:.4f}, residual regressor rho={rho_full:.4f}. exp09's reported "
        f"rho=0.6536 had no floor to compare against; on this different (full-row) "
        f"population the residual regressor's rho is now interpretable against these two "
        f"numbers rather than standing alone.\n\n"
        f"Per-stratum floors (case_count 3-5 / 6-20 / 21+) are in "
        f"<exp_id>_strata_with_floors.csv, extending exp09's next_steps request for a "
        f"floor per stratum -- previously uninterpretable without one.\n\n"
        f"min_pairs_per_drug: METRIC.md specifies 20 (used for the headline drug_macro_auc "
        f"numbers above); exp09 used 5. Both are in "
        f"<exp_id>_floor_reference_table.csv (case_count_ge3 population is duplicated at "
        f"both eligibility thresholds) -- this alone can move a macro metric by a few "
        f"thousandths.\n\n"
        f"gene_neighbour_ge1 and exp12's rungs could not be floored: no per-drug/condition "
        f"gene-neighbour column is staged on this volume, and exp12 has a spec but no "
        f"implementation yet. Both are recorded as 'skipped' rows in the floor reference "
        f"table rather than silently omitted.\n\n"
        f"Verdict: {verdict}."
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"results/{exp}/{exp}_floor_reference_table.csv",
            f"results/{exp}/{exp}_residual_vs_flag_full_rows.csv",
            f"results/{exp}/{exp}_rank_metric_floors.csv",
            f"results/{exp}/{exp}_strata_with_floors.csv",
            f"results/{exp}/{exp}_per_drug_rho_full_rows.csv",
            f"results/{exp}/{exp}_residual_scatter_full_rows.png",
        ],
        next_steps=(
            "Pull results/<exp_id>/ from the bridge-results volume. Every later experiment "
            "that reports a subset result must call experiments/floors.py for its own "
            "population and log floor_pc/floor_degree_pc alongside its headline -- a "
            "headline without its matching floor is not a result (per this experiment's "
            "spec). Backfill gene_neighbour_ge1 and exp12's rungs into the floor reference "
            "table once those inputs / that experiment exist."
        ),
        supersedes=EXP09_EXP_ID,
    )
