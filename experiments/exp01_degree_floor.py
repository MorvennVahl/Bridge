"""exp01 â€” The degree floor, and whether it survives the label definition.

Implements experiments/exp01_degree_floor.md. See that file for the full method,
budget, deliverables, and reporting requirements.
"""

from __future__ import annotations

import logging
import pathlib

import modal

app = modal.App("bridge-exp01")  # stable name, no random suffix

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

LABEL_DEFS: dict[str, str] = {
    "evans": "y_faers_signal",
    "prr2": "prr2",
    "prr4_c5": "prr4_c5",
    "cases10": "cases10",
    "any_harm": "y_any_harm",
}

EXPECTED_TRAIN_RATES: dict[str, float] = {
    "evans": 0.1100,
    "prr2": 0.3629,
    "prr4_c5": 0.0355,
    "cases10": 0.2161,
    "any_harm": 0.1110,
}

RATE_TOL = 0.001


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, float]:
    import numpy as np
    import pandas as pd
    from scipy.stats import kendalltau
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp01")

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    # ---- load splits -----------------------------------------------------
    train_path = pathlib.Path("/data/splits/train.csv")
    validate_path = pathlib.Path("/data/splits/validate.csv")
    cond_features_path = pathlib.Path("/data/condition/condition_features_basic.csv")

    log.info("reading train/validate splits")
    train = pd.read_csv(train_path)
    validate = pd.read_csv(validate_path)

    # ---- build label variants (all derived from split columns only) ------
    train = train.copy()
    validate = validate.copy()
    for df in (train, validate):
        df["prr2"] = (df["faers_prr"] >= 2).astype(int)
        df["prr4_c5"] = ((df["faers_prr"] >= 4) & (df["faers_case_count"] >= 5)).astype(int)
        df["cases10"] = (df["faers_case_count"] >= 10).astype(int)

    # ---- precondition check: train positive rates -------------------------
    rate_report: dict[str, float] = {}
    for label_id, col in LABEL_DEFS.items():
        observed = float(train[col].mean())
        rate_report[label_id] = observed
        expected = EXPECTED_TRAIN_RATES[label_id]
        if abs(observed - expected) > RATE_TOL:
            raise ValueError(
                f"train positive rate for label '{label_id}' is {observed:.4f}, "
                f"expected {expected:.4f} (+/- {RATE_TOL}). The split was likely "
                f"rebuilt underneath this experiment."
            )
        log.info(
            "label %s: train positive rate %.4f (expected %.4f) OK", label_id, observed, expected
        )

    # ---- degree terms, TRAIN ONLY, then applied to validate ---------------
    log.info("computing degree terms on train only")
    drug_degree_train = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    condition_degree_train = train.groupby("condition_concept_id")[
        "ingredient_concept_id"
    ].nunique()

    drug_degree_median = float(drug_degree_train.median())
    condition_degree_median = float(condition_degree_train.median())

    def map_drug_degree(df: pd.DataFrame) -> pd.Series:
        mapped = df["ingredient_concept_id"].map(drug_degree_train)
        return mapped.fillna(drug_degree_median)

    def map_condition_degree(df: pd.DataFrame) -> pd.Series:
        mapped = df["condition_concept_id"].map(condition_degree_train)
        return mapped.fillna(condition_degree_median)

    for df in (train, validate):
        df["drug_degree_train"] = map_drug_degree(df)
        df["condition_degree_train"] = map_condition_degree(df)
        df["log_drug_degree"] = np.log1p(df["drug_degree_train"])
        df["log_condition_degree"] = np.log1p(df["condition_degree_train"])

    # ---- condition_features_basic.csv (record_count), with fallback -------
    have_condition_features = cond_features_path.exists()
    feature_cols = ["log_drug_degree", "log_condition_degree"]
    single_feature_cols = ["log_drug_degree", "log_condition_degree"]
    fallback_note = ""

    if have_condition_features:
        log.info("condition_features_basic.csv found; joining record_count")
        cond_feat = pd.read_csv(cond_features_path)
        record_count_map = cond_feat.set_index("condition_concept_id")["record_count"]
        record_count_median = float(record_count_map.median())
        for df in (train, validate):
            df["condition_record_count"] = df["condition_concept_id"].map(record_count_map)
            df["condition_record_count"] = df["condition_record_count"].fillna(record_count_median)
            df["log_condition_record_count"] = np.log1p(df["condition_record_count"])
        feature_cols = [*feature_cols, "log_condition_record_count"]
        single_feature_cols = [*single_feature_cols, "log_condition_record_count"]

        for df in (train, validate):
            for c in ["is_mapped", "arm", "best_match_tier"]:
                if c in cond_feat.columns:
                    df[c] = df["condition_concept_id"].map(
                        cond_feat.set_index("condition_concept_id")[c]
                    )
    else:
        log.warning(
            "condition_features_basic.csv is ABSENT â€” running with the two "
            "train-derived degree terms only, per the spec's fallback instruction. "
            "The record_count row of results tables will be left empty."
        )
        fallback_note = (
            "condition_features_basic.csv was absent at runtime; ran with the two "
            "train-derived degree terms only (drug_degree, condition_degree). "
            "record_count is omitted from all results tables as instructed by the "
            "spec's fallback (not treated as a failure)."
        )

    single_feature_names = {
        "log_drug_degree": "drug_degree",
        "log_condition_degree": "condition_degree",
        "log_condition_record_count": "condition_record_count",
    }

    # ---- fit/eval helpers ---------------------------------------------------
    def make_logreg() -> LogisticRegression:
        return LogisticRegression(max_iter=1000)

    def make_histgbm() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=63, early_stopping=False
        )

    def grouped_cv_ap(
        X: pd.DataFrame,  # noqa: N803 -- conventional sklearn feature-matrix name
        y: pd.Series,
        groups: pd.Series,
        model_fn,
    ) -> tuple[float, float]:
        gkf = GroupKFold(n_splits=3)
        aps = []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            m = model_fn()
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = m.predict_proba(X.iloc[test_idx])[:, 1]
            aps.append(average_precision_score(y.iloc[test_idx], proba))
        aps_arr = np.array(aps)
        return float(aps_arr.mean()), float(aps_arr.std())

    ap_rows: list[dict[str, object]] = []
    ranking_rows: dict[str, dict[str, float]] = {}
    label_prevalence_validate: dict[str, float] = {}

    logreg_evans_model = None
    histgbm_evans_model = None
    evans_train_cv_ap = None

    for label_id, col in LABEL_DEFS.items():
        log.info("processing label %s (%s)", label_id, col)
        y_train = train[col]
        y_val = validate[col]
        groups_train = train["group_key"]

        prevalence_val = float(y_val.mean())
        label_prevalence_validate[label_id] = prevalence_val

        X_train_full = train[feature_cols]  # noqa: N806 -- conventional sklearn feature-matrix name
        X_val_full = validate[feature_cols]  # noqa: N806

        for model_name, model_fn in (("logreg", make_logreg), ("histgbm", make_histgbm)):
            cv_ap_mean, cv_ap_std = grouped_cv_ap(X_train_full, y_train, groups_train, model_fn)

            m = model_fn()
            m.fit(X_train_full, y_train)
            proba_val = m.predict_proba(X_val_full)[:, 1]
            val_ap = average_precision_score(y_val, proba_val)
            val_auc = roc_auc_score(y_val, proba_val)

            ap_rows.append(
                {
                    "label": label_id,
                    "model": model_name,
                    "feature_set": "all_degree_terms",
                    "validate_average_precision": val_ap,
                    "validate_roc_auc": val_auc,
                    "validate_prevalence": prevalence_val,
                    "ap_lift_over_prevalence": val_ap - prevalence_val,
                    "train_cv_average_precision_mean": cv_ap_mean,
                    "train_cv_average_precision_std": cv_ap_std,
                }
            )

            if label_id == "evans" and model_name == "logreg":
                logreg_evans_model = m
            if label_id == "evans" and model_name == "histgbm":
                histgbm_evans_model = m
                evans_train_cv_ap = cv_ap_mean

        # single-feature fits (logreg only, one feature at a time)
        single_feature_aps: dict[str, float] = {}
        for feat_col in single_feature_cols:
            X_train_single = train[[feat_col]]  # noqa: N806
            X_val_single = validate[[feat_col]]  # noqa: N806
            m = make_logreg()
            m.fit(X_train_single, y_train)
            proba_val = m.predict_proba(X_val_single)[:, 1]
            val_ap = average_precision_score(y_val, proba_val)
            cv_ap_mean, cv_ap_std = grouped_cv_ap(
                X_train_single, y_train, groups_train, make_logreg
            )

            feat_name = single_feature_names[feat_col]
            single_feature_aps[feat_name] = val_ap

            ap_rows.append(
                {
                    "label": label_id,
                    "model": "logreg",
                    "feature_set": feat_name,
                    "validate_average_precision": val_ap,
                    "validate_roc_auc": roc_auc_score(y_val, proba_val),
                    "validate_prevalence": prevalence_val,
                    "ap_lift_over_prevalence": val_ap - prevalence_val,
                    "train_cv_average_precision_mean": cv_ap_mean,
                    "train_cv_average_precision_std": cv_ap_std,
                }
            )

        ranking_rows[label_id] = single_feature_aps

    # ---- ap_table.csv -------------------------------------------------------
    ap_table = pd.DataFrame(ap_rows)
    if "condition_record_count" not in single_feature_names.values():
        pass
    if not have_condition_features:
        ap_table.loc[ap_table["feature_set"] == "condition_record_count", :] = np.nan
    ap_table_path = out / f"{exp_id}_ap_table.csv"
    ap_table.to_csv(ap_table_path, index=False)

    # ---- feature_ranking.csv -------------------------------------------------
    ranking_df = pd.DataFrame(ranking_rows)  # rows: feature name, cols: label id
    if not have_condition_features:
        ranking_df = ranking_df.reindex(
            index=list(ranking_df.index)
            + (
                ["condition_record_count"]
                if "condition_record_count" not in ranking_df.index
                else []
            )
        )
    label_ids = list(LABEL_DEFS.keys())
    tau_records = []
    for i in range(len(label_ids)):
        for j in range(i + 1, len(label_ids)):
            a, b = label_ids[i], label_ids[j]
            common = ranking_df[[a, b]].dropna()
            if len(common) >= 2:
                tau, _ = kendalltau(common[a], common[b])
            else:
                tau = np.nan
            tau_records.append({"label_a": a, "label_b": b, "kendall_tau": tau})
    tau_df = pd.DataFrame(tau_records)

    ranking_out_path = out / f"{exp_id}_feature_ranking.csv"
    with open(ranking_out_path, "w") as fh:
        ranking_df.to_csv(fh)
        fh.write("\n# pairwise kendall tau between label columns (rank stability)\n")
        tau_df.to_csv(fh, index=False)

    # rank stability: does the argmax feature agree across all 5 labels?
    top_feature_per_label = ranking_df.idxmax(axis=0)
    ranking_stable = bool(top_feature_per_label.nunique(dropna=True) == 1)
    mean_tau = float(tau_df["kendall_tau"].mean()) if not tau_df.empty else float("nan")

    # ---- calibration plot (Evans label, both models, 20 bins) ---------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    y_val_evans = validate[LABEL_DEFS["evans"]]
    for name, model in (("logreg", logreg_evans_model), ("histgbm", histgbm_evans_model)):
        proba = model.predict_proba(validate[feature_cols])[:, 1]
        bins = np.linspace(0.0, 1.0, 21)
        bin_idx = np.digitize(proba, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 19)
        mean_pred = []
        mean_obs = []
        for b in range(20):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            mean_pred.append(proba[mask].mean())
            mean_obs.append(y_val_evans[mask].mean())
        ax.plot(mean_pred, mean_obs, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title(f"Calibration â€” Evans label (y_faers_signal), validate, {exp_id}")
    ax.legend()
    calibration_path = out / f"{exp_id}_calibration.png"
    fig.savefig(calibration_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- subgroups.csv (validate AP by is_mapped, arm, best_match_tier) -----
    subgroup_rows: list[dict[str, object]] = []
    if have_condition_features:
        proba_histgbm_evans = histgbm_evans_model.predict_proba(validate[feature_cols])[:, 1]
        val_eval = validate.copy()
        val_eval["_proba"] = proba_histgbm_evans
        for subgroup_col in ["is_mapped", "arm", "best_match_tier"]:
            if subgroup_col not in val_eval.columns:
                continue
            for value, grp in val_eval.groupby(subgroup_col, dropna=False):
                if grp[LABEL_DEFS["evans"]].nunique() < 2:
                    ap = float("nan")
                else:
                    ap = average_precision_score(grp[LABEL_DEFS["evans"]], grp["_proba"])
                subgroup_rows.append(
                    {
                        "subgroup_col": subgroup_col,
                        "subgroup_value": value,
                        "n": len(grp),
                        "prevalence": float(grp[LABEL_DEFS["evans"]].mean()),
                        "validate_average_precision": ap,
                    }
                )
    else:
        log.warning(
            "condition_features_basic.csv absent: is_mapped/arm/best_match_tier "
            "unavailable, subgroups.csv will be empty."
        )
    subgroups_df = pd.DataFrame(subgroup_rows)
    subgroups_path = out / f"{exp_id}_subgroups.csv"
    subgroups_df.to_csv(subgroups_path, index=False)

    results.commit()

    # ---- metrics dict for ln.complete ---------------------------------------
    evans_histgbm_rows = ap_table[
        (ap_table["label"] == "evans")
        & (ap_table["model"] == "histgbm")
        & (ap_table["feature_set"] == "all_degree_terms")
    ]
    validate_ap_evans = float(evans_histgbm_rows["validate_average_precision"].iloc[0])
    validate_prevalence_evans = float(evans_histgbm_rows["validate_prevalence"].iloc[0])

    metrics: dict[str, float] = {
        "validate_average_precision": validate_ap_evans,
        "train_cv_average_precision": float(evans_train_cv_ap)
        if evans_train_cv_ap is not None
        else float("nan"),
        "baseline_degree_only_ap": validate_ap_evans,
        "validate_prevalence_evans": validate_prevalence_evans,
        "kendall_tau_mean": mean_tau,
        "feature_ranking_stable": float(ranking_stable),
    }
    for label_id in LABEL_DEFS:
        rows = ap_table[
            (ap_table["label"] == label_id)
            & (ap_table["model"] == "histgbm")
            & (ap_table["feature_set"] == "all_degree_terms")
        ]
        metrics[f"ap_{label_id}"] = float(rows["validate_average_precision"].iloc[0])

    log.info("metrics: %s", metrics)
    log.info("fallback note: %s", fallback_note or "(none)")

    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp01",
        title="Degree floor across five FAERS label definitions",
        hypothesis=(
            "Drug degree, condition degree and condition record_count reach AP well "
            "above prevalence on y_faers_signal, and their AP ranking is stable across "
            "five label definitions."
        ),
        approach=(
            "Train-derived degree terms only. Logistic regression and HistGBM per label "
            "variant, plus single-feature fits. Grouped 3-fold CV in train, one validate "
            "evaluation."
        ),
        label="y_faers_signal",
        features=["drug_degree_train", "condition_degree_train", "condition_record_count"],
        split="train/validate, grouped by primary target gene",
    )
    metrics = run.remote(exp)
    print(metrics)

    validate_ap = metrics.get("validate_average_precision", float("nan"))
    prevalence = metrics.get("validate_prevalence_evans", float("nan"))
    stable = bool(metrics.get("feature_ranking_stable", 0.0))
    stability_text = (
        "The single-feature AP ranking held across all five label definitions; later "
        "experiments may report against a single label."
        if stable
        else "The single-feature AP ranking did NOT hold across all five label "
        "definitions (see feature_ranking.csv for the per-label values and pairwise "
        "Kendall tau); the label-definition sensitivity is now mandatory for downstream "
        "reporting."
    )
    findings = (
        f"Evans-label (y_faers_signal) validate AP is {validate_ap:.4f} against a "
        f"validate prevalence of {prevalence:.4f} (spec states 11.7%). "
        f"See {exp}_feature_ranking.csv for which single count (drug_degree, "
        f"condition_degree, or condition_record_count) carries the most signal per "
        f"label, and {exp}_ap_table.csv for the logistic-vs-HistGBM AP gap per label "
        f"(the gap measures how much non-linearity the three counts can support). "
        f"{stability_text}"
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"/results/{exp}/{exp}_ap_table.csv",
            f"/results/{exp}/{exp}_feature_ranking.csv",
            f"/results/{exp}/{exp}_calibration.png",
            f"/results/{exp}/{exp}_subgroups.csv",
        ],
        next_steps=(
            "If the ranking held, proceed to exp02 (intrinsic feature blocks) reporting "
            "against the Evans label only. If it flipped, carry all five label "
            "definitions through exp02-exp05."
        ),
    )
