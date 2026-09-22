"""exp02 — Does either side's intrinsic biology add anything on top of degree?

Implements experiments/exp02_intrinsic_blocks.md exactly. See that file for the full
method, feature-block definitions, budget, and reporting requirements.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import pandas as pd

app = modal.App("bridge-exp02")  # stable name, no random suffix

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

# Free-text / high-cardinality list columns to drop from ingredient_features.csv
# (spec, Drug-intrinsic section). Keep any n_* derived counts.
DRUG_LIST_COLUMNS: list[str] = [
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
]

# identity-block columns to drop explicitly even though `identity` block is dropped wholesale.
IDENTITY_EXTRA_DROP: list[str] = [
    "in_cem_list",
    "in_indication_roster",
    "roster_name",
    "indication",
]

DRUG_BLOCKS_WANTED: list[str] = [
    "development",
    "safety",
    "pharmacology",
    "chemistry",
    "exposure",
    "metabolism",
    "mechanism",
    "target biology",
    "interactions",
]

CARDINALITY_CAP: int = 30


def _normalize_block(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _cap_and_onehot(series: pd.Series, prefix: str, cap: int = CARDINALITY_CAP) -> pd.DataFrame:
    """One-hot encode a categorical series, bucketing tail categories beyond `cap` as 'other'."""
    import pandas as pd

    counts = series.value_counts(dropna=True)
    keep = set(counts.index[:cap])
    bucketed = series.where(series.isin(keep) | series.isna(), other="other")
    dummies = pd.get_dummies(bucketed, prefix=prefix, dummy_na=False)
    return dummies


def _check_preconditions(
    train_path: pathlib.Path,
    validate_path: pathlib.Path,
    drug_path: pathlib.Path,
    cond_path: pathlib.Path,
    group_path: pathlib.Path,
    dict_path: pathlib.Path,
) -> str | None:
    """Return an error message naming what's wrong, or None if everything checks out."""
    import pandas as pd

    expected_rows = {
        train_path: 723_586,
        validate_path: 434_151,
        drug_path: 4_280,
        cond_path: 5_631,
        group_path: 10_554,
        dict_path: 245,
    }
    for path, expected in expected_rows.items():
        if not path.exists():
            return f"missing required input: {path}"
        with path.open() as fh:
            n = sum(1 for _ in fh) - 1
        if n != expected:
            return f"row count mismatch for {path}: expected {expected}, got {n}"

    train = pd.read_csv(train_path, usecols=["y_faers_signal"])
    rate = float(train["y_faers_signal"].mean())
    if abs(rate - 0.1100) > 0.001:
        return f"train positive rate for y_faers_signal is {rate:.4f}, expected 0.1100 +/- 0.001"
    return None


def _build_degree_features(
    train: pd.DataFrame, validate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-derived degree features (drug degree, condition degree, pair-frequency-free).

    Computed on TRAIN ONLY; the same mapping (with train median for unseen keys) is
    applied to validate. As exp01: log1p(count of observed pairs per key).
    """
    import numpy as np
    import pandas as pd

    drug_degree = train.groupby("ingredient_concept_id").size()
    cond_degree = train.groupby("condition_concept_id").size()
    train_case_count_median = train["faers_case_count"].median()

    drug_degree_median = float(drug_degree.median())
    cond_degree_median = float(cond_degree.median())

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["degree_drug"] = np.log1p(
            df["ingredient_concept_id"].map(drug_degree).fillna(drug_degree_median)
        )
        out["degree_condition"] = np.log1p(
            df["condition_concept_id"].map(cond_degree).fillna(cond_degree_median)
        )
        out["degree_faers_case_count"] = np.log1p(
            df["faers_case_count"].fillna(train_case_count_median)
        )
        return out

    return _apply(train), _apply(validate)


def _load_drug_features(
    drug_path: pathlib.Path, dict_path: pathlib.Path
) -> tuple[pd.DataFrame, dict]:
    """Load ingredient_features.csv, select block columns via data_dictionary, drop
    forbidden columns, add has_chembl_match. Returns (df, diagnostics)."""
    import pandas as pd

    ddict = pd.read_csv(dict_path)
    ddict_ing = ddict[ddict["table"] == "ingredient_features.csv"].copy()
    ddict_ing["block_norm"] = ddict_ing["block"].map(_normalize_block)

    wanted_norm = {_normalize_block(b) for b in DRUG_BLOCKS_WANTED}
    present_blocks = sorted(ddict_ing["block_norm"].unique())
    matched_blocks = sorted(wanted_norm & set(present_blocks))
    unmatched_wanted = sorted(wanted_norm - set(present_blocks))

    selected_cols = ddict_ing.loc[ddict_ing["block_norm"].isin(matched_blocks), "column"].tolist()

    drug = pd.read_csv(drug_path)
    keep_cols = [c for c in selected_cols if c in drug.columns]

    drop_cols = set(DRUG_LIST_COLUMNS) | set(IDENTITY_EXTRA_DROP)
    drop_cols |= {c for c in drug.columns if c.startswith("chembl_fuzzy_")}
    keep_cols = [c for c in keep_cols if c not in drop_cols]

    # always need join key + chembl_id (for has_chembl_match), even if not in selected blocks
    join_col = "omop_concept_id"
    out = drug[
        [join_col, "chembl_id"] + [c for c in keep_cols if c != join_col and c != "chembl_id"]
    ].copy()
    out["has_chembl_match"] = out["chembl_id"].notna()
    out = out.drop(columns=["chembl_id"])

    diagnostics = {
        "present_blocks_in_dictionary": present_blocks,
        "matched_blocks": matched_blocks,
        "unmatched_wanted_blocks": unmatched_wanted,
        "n_drug_columns_selected": len(keep_cols),
    }
    return out, diagnostics


def _build_drug_design(drug_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot / categorical-encode the drug block; return (feature_df, categorical_cols)."""
    import pandas as pd

    df = drug_raw.copy()
    join_col = "omop_concept_id"
    feature_cols = [c for c in df.columns if c != join_col]

    cat_cols: list[str] = []
    out_parts = [df[[join_col]]]
    for col in feature_cols:
        if df[col].dtype == object:
            out_parts.append(_cap_and_onehot(df[col], prefix=col))
        elif df[col].dtype == bool:
            out_parts.append(df[[col]].astype(float))
        else:
            out_parts.append(df[[col]])

    design = pd.concat(out_parts, axis=1)
    return design, cat_cols


def _load_condition_features(cond_path: pathlib.Path, group_path: pathlib.Path) -> pd.DataFrame:
    """Build the condition-intrinsic design matrix per the spec's Condition-intrinsic block."""
    import numpy as np
    import pandas as pd

    cond = pd.read_csv(cond_path)

    base_cols = [
        "condition_concept_id",
        "record_count",
        "concept_class_id",
        "n_omop_ancestors",
        "arm",
        "is_mapped",
        "best_match_tier",
        "n_ontology_terms",
        "n_hpo_genes",
        "n_groups",
    ]
    cond_base = cond[base_cols].copy()
    cond_base["record_count"] = np.log1p(cond_base["record_count"])
    cond_base["is_mapped"] = cond_base["is_mapped"].astype(float)

    # IDF-style specificity transform for n_hpo_genes: this table only has n_hpo_genes as an
    # aggregate count per condition, so the IDF weighting is approximated at the condition
    # level using the count distribution: log(n_conditions / n_conditions with >= this gene
    # count) is not computable without the per-gene list (deliberately not staged, per
    # experiments/README.md). Instead compute an IDF-style transform over the *count itself*:
    # conditions with a rarer (smaller) gene count are more specific, so weight by
    # log(n_conditions / n_conditions sharing this count), which is well-defined here.
    n = len(cond_base)
    count_freq = cond_base["n_hpo_genes"].value_counts()
    cond_base["n_hpo_genes_idf"] = cond_base["n_hpo_genes"].map(
        lambda v: np.log(n / count_freq.get(v, 1))
    )

    for col in ["concept_class_id", "arm", "best_match_tier"]:
        cond_base = pd.concat(
            [cond_base.drop(columns=[col]), _cap_and_onehot(cond_base[col], prefix=col)],
            axis=1,
        )

    group_long = pd.read_csv(group_path)
    group_long["col_label"] = group_long["group_source"] + "__" + group_long["group_label"]
    group_wide = (
        group_long.pivot_table(
            index="condition_concept_id", columns="col_label", values="group_id", aggfunc="count"
        )
        .fillna(0)
        .clip(upper=1)
    )
    group_wide.columns = [f"group_{c}" for c in group_wide.columns]
    group_wide = group_wide.reset_index()

    out = cond_base.merge(group_wide, on="condition_concept_id", how="left")
    group_cols = [c for c in out.columns if c.startswith("group_")]
    out[group_cols] = out[group_cols].fillna(0.0)

    return out


def _fit_score(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    n_folds: int = 3,
) -> dict:
    """Grouped CV inside train, then one fit on all of train + one validate scoring."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    def _make_model() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
        )

    gkf = GroupKFold(n_splits=n_folds)
    cv_aps: list[float] = []
    for train_idx, test_idx in gkf.split(x_train, y_train, groups=groups_train):
        model = _make_model()
        model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
        proba = model.predict_proba(x_train.iloc[test_idx])[:, 1]
        cv_aps.append(float(average_precision_score(y_train.iloc[test_idx], proba)))

    final_model = _make_model()
    final_model.fit(x_train, y_train)
    val_proba = final_model.predict_proba(x_val)[:, 1]
    val_ap = float(average_precision_score(y_val, val_proba))
    val_auc = float(roc_auc_score(y_val, val_proba))

    return {
        "model": final_model,
        "cv_aps": cv_aps,
        "cv_ap_mean": float(np.mean(cv_aps)),
        "cv_ap_min": float(np.min(cv_aps)),
        "cv_ap_max": float(np.max(cv_aps)),
        "val_ap": val_ap,
        "val_auc": val_auc,
        "val_proba": val_proba,
    }


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp02")

    import matplotlib
    import numpy as np
    import pandas as pd
    from sklearn.calibration import calibration_curve

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    data_root = pathlib.Path("/data")
    train_path = data_root / "splits" / "train.csv"
    validate_path = data_root / "splits" / "validate.csv"
    drug_path = data_root / "drug" / "ingredient_features.csv"
    cond_path = data_root / "condition" / "condition_features_basic.csv"
    group_path = data_root / "condition" / "condition_group_long.csv"
    dict_path = data_root / "data_dictionary.csv"

    # Step 1: preconditions.
    err = _check_preconditions(
        train_path, validate_path, drug_path, cond_path, group_path, dict_path
    )
    if err is not None:
        log.error("precondition check failed: %s", err)
        return {"precondition_failed": 1.0, "precondition_error_message": err}

    log.info("preconditions passed; loading train/validate")
    train = pd.read_csv(train_path)
    validate = pd.read_csv(validate_path)

    fallback_notes: list[str] = []

    # Step 2/degree: build degree design.
    degree_train, degree_val = _build_degree_features(train, validate)

    # Drug-intrinsic block.
    log.info("loading drug-intrinsic block")
    drug_raw, drug_diag = _load_drug_features(drug_path, dict_path)
    log.info("drug block diagnostics: %s", drug_diag)
    drug_design, _ = _build_drug_design(drug_raw)
    drug_design = drug_design.rename(columns={"omop_concept_id": "ingredient_concept_id"})

    # Condition-intrinsic block.
    log.info("loading condition-intrinsic block")
    cond_design = _load_condition_features(cond_path, group_path)

    def _assemble(base_df: pd.DataFrame, degree_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        idx = base_df.index
        d = degree_df.set_axis(idx)
        merged_drug = base_df[["ingredient_concept_id"]].merge(
            drug_design, on="ingredient_concept_id", how="left"
        )
        merged_drug.index = idx
        merged_drug = merged_drug.drop(columns=["ingredient_concept_id"])

        merged_cond = base_df[["condition_concept_id"]].merge(
            cond_design, on="condition_concept_id", how="left"
        )
        merged_cond.index = idx
        merged_cond = merged_cond.drop(columns=["condition_concept_id"])

        designs = {
            "D": d.copy(),
            "D+drug": pd.concat([d, merged_drug], axis=1),
            "D+cond": pd.concat([d, merged_cond], axis=1),
            "D+drug+cond": pd.concat([d, merged_drug, merged_cond], axis=1),
        }
        return designs

    train_designs = _assemble(train, degree_train)
    val_designs = _assemble(validate, degree_val)

    y_train = train["y_faers_signal"]
    y_val = validate["y_faers_signal"]
    groups_train = train["group_key"]

    design_order = ["D", "D+drug", "D+cond", "D+drug+cond"]
    fit_results: dict[str, dict] = {}

    for name in design_order:
        n_folds = 3
        # Budget fallback trigger point: drop to 2 folds if needed. We run 3 by default; this
        # is only relaxed if the design is unexpectedly wide (budget risk at runtime).
        if name == "D+drug+cond" and train_designs[name].shape[1] > 400:
            n_folds = 2
            fallback_notes.append(
                "Dropped CV to 2 folds for D+drug+cond due to feature-count budget risk."
            )
        log.info(
            "fitting design=%s n_features=%d n_folds=%d",
            name,
            train_designs[name].shape[1],
            n_folds,
        )
        res = _fit_score(
            train_designs[name], y_train, groups_train, val_designs[name], y_val, n_folds=n_folds
        )
        fit_results[name] = res
        log.info(
            "design=%s cv_ap_mean=%.4f val_ap=%.4f val_auc=%.4f",
            name,
            res["cv_ap_mean"],
            res["val_ap"],
            res["val_auc"],
        )

    # Deliverable: ablation table.
    ablation_rows = []
    prev_ap = None
    for name in design_order:
        res = fit_results[name]
        increment = None if prev_ap is None else res["val_ap"] - prev_ap
        ablation_rows.append(
            {
                "design": name,
                "n_features": train_designs[name].shape[1],
                "train_cv_ap_mean": res["cv_ap_mean"],
                "train_cv_ap_min": res["cv_ap_min"],
                "train_cv_ap_max": res["cv_ap_max"],
                "validate_ap": res["val_ap"],
                "validate_roc_auc": res["val_auc"],
                "increment_over_previous": increment,
            }
        )
        prev_ap = res["val_ap"]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = out / f"{exp_id}_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)

    # Deliverable: importances, top 25 per design.
    importance_rows = []
    for name in design_order:
        cols = train_designs[name].columns
        # HistGradientBoostingClassifier does not expose per-column gain importance directly;
        # sklearn HistGBM has no feature_importances_. Use permutation-free proxy: since the
        # spec explicitly forbids permutation importance (too costly for the budget) and asks
        # for "gain importances", we approximate using the model's built-in term contributions
        # via `partial_dependence`-free route is also costly, so instead we surface the only
        # cheap signal sklearn's HistGBM exposes at this version: none directly. We therefore
        # fit a lightweight LightGBM (already in the image) purely for gain-importance ranking,
        # mirroring the same design matrix, without changing the reported AP/AUC (which stay
        # from HistGBM per the spec's fixed model).
        import re

        import lightgbm as lgb

        # LightGBM rejects special JSON characters in feature names (one-hot columns can
        # carry them via category values like commas/colons/brackets). Sanitize for this
        # fit only; map importances back to the original column names via position.
        x_lgb = train_designs[name].copy()
        seen: dict[str, int] = {}
        sanitized: list[str] = []
        for c in x_lgb.columns:
            base = re.sub(r"[^0-9A-Za-z_]", "_", str(c))
            n = seen.get(base, 0)
            seen[base] = n + 1
            sanitized.append(base if n == 0 else f"{base}_{n}")
        x_lgb.columns = sanitized

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.06,
            num_leaves=63,
            random_state=0,
            verbosity=-1,
        )
        lgb_model.fit(x_lgb, y_train)
        importances = lgb_model.booster_.feature_importance(importance_type="gain")
        top_idx = np.argsort(importances)[::-1][:25]
        for rank, i in enumerate(top_idx, start=1):
            importance_rows.append(
                {
                    "design": name,
                    "rank": rank,
                    "feature": cols[i],
                    "gain": float(importances[i]),
                }
            )
    importance_df = pd.DataFrame(importance_rows)
    importance_path = out / f"{exp_id}_importance_top25.csv"
    importance_df.to_csv(importance_path, index=False)

    # Deliverable: subgroup breakdown.
    subgroup_rows = []
    cond_meta = pd.read_csv(cond_path)[
        ["condition_concept_id", "is_mapped", "arm", "best_match_tier"]
    ]
    val_meta = validate.merge(cond_meta, on="condition_concept_id", how="left")
    val_meta = val_meta.merge(
        drug_raw[["omop_concept_id", "has_chembl_match"]],
        left_on="ingredient_concept_id",
        right_on="omop_concept_id",
        how="left",
    )
    assert len(val_meta) == len(validate), "subgroup merge must not change row count/order"

    from sklearn.metrics import average_precision_score as _ap

    for name in design_order:
        proba = fit_results[name]["val_proba"]
        for subgroup_col in ["is_mapped", "arm", "best_match_tier", "has_chembl_match"]:
            for level, grp in val_meta.groupby(subgroup_col, dropna=False):
                mask = grp.index
                if grp[y_val.name].nunique() < 2:
                    continue
                subgroup_rows.append(
                    {
                        "design": name,
                        "subgroup_col": subgroup_col,
                        "level": str(level),
                        "n": len(mask),
                        "validate_ap": float(_ap(y_val.loc[mask], proba[mask])),
                    }
                )
    subgroups_df = pd.DataFrame(subgroup_rows)
    subgroups_path = out / f"{exp_id}_subgroups.csv"
    subgroups_df.to_csv(subgroups_path, index=False)

    # Deliverable: calibration plot, all four designs.
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in design_order:
        proba = fit_results[name]["val_proba"]
        frac_pos, mean_pred = calibration_curve(y_val, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title(f"{exp_id} — reliability curves")
    ax.legend()
    fig.tight_layout()
    calibration_path = out / f"{exp_id}_calibration.png"
    fig.savefig(calibration_path, dpi=150)
    plt.close(fig)

    # Step 6: missingness attribution — restrict rows to mechanism-target ingredients only.
    drug_full = pd.read_csv(drug_path)
    mechanism_ingredients = (
        set(drug_full.loc[drug_full["target_chembl_ids"].notna(), "omop_concept_id"])
        if "target_chembl_ids" in drug_full.columns
        else set()
    )
    if not mechanism_ingredients and "mechanisms_of_action" in drug_full.columns:
        mechanism_ingredients = set(
            drug_full.loc[drug_full["mechanisms_of_action"].notna(), "omop_concept_id"]
        )

    train_mech_mask = train["ingredient_concept_id"].isin(mechanism_ingredients)
    val_mech_mask = validate["ingredient_concept_id"].isin(mechanism_ingredients)

    log.info(
        "mechanism-subset restriction: train rows %d/%d, validate rows %d/%d",
        train_mech_mask.sum(),
        len(train),
        val_mech_mask.sum(),
        len(validate),
    )

    missingness_rows = []
    if train_mech_mask.sum() > 0 and val_mech_mask.sum() > 0:
        x_train_mech = train_designs["D+drug"].loc[train_mech_mask]
        y_train_mech = y_train.loc[train_mech_mask]
        groups_train_mech = groups_train.loc[train_mech_mask]
        x_val_mech = val_designs["D+drug"].loc[val_mech_mask]
        y_val_mech = y_val.loc[val_mech_mask]

        mech_res = _fit_score(
            x_train_mech, y_train_mech, groups_train_mech, x_val_mech, y_val_mech, n_folds=3
        )
        missingness_rows.append(
            {
                "design": "D+drug (mechanism subset)",
                "n_train_rows": int(train_mech_mask.sum()),
                "n_validate_rows": int(val_mech_mask.sum()),
                "train_cv_ap_mean": mech_res["cv_ap_mean"],
                "validate_ap": mech_res["val_ap"],
                "validate_roc_auc": mech_res["val_auc"],
                "full_D_drug_validate_ap": fit_results["D+drug"]["val_ap"],
                "full_D_validate_ap": fit_results["D"]["val_ap"],
            }
        )
        ap_union_mechanism_subset = mech_res["val_ap"]
    else:
        fallback_notes.append(
            "Mechanism-target subset had zero rows in train or validate; step 6 could not be scored."
        )
        ap_union_mechanism_subset = float("nan")

    missingness_df = pd.DataFrame(missingness_rows)
    missingness_path = out / f"{exp_id}_missingness_attribution.csv"
    missingness_df.to_csv(missingness_path, index=False)

    results.commit()

    baseline_degree_only_ap = fit_results["D"]["val_ap"]
    ap_drug_only_increment = fit_results["D+drug"]["val_ap"] - baseline_degree_only_ap
    ap_condition_only_increment = fit_results["D+cond"]["val_ap"] - baseline_degree_only_ap
    ap_union = fit_results["D+drug+cond"]["val_ap"]

    metrics = {
        "validate_average_precision": ap_union,
        "train_cv_average_precision": fit_results["D+drug+cond"]["cv_ap_mean"],
        "baseline_degree_only_ap": baseline_degree_only_ap,
        "ap_drug_only_increment": ap_drug_only_increment,
        "ap_condition_only_increment": ap_condition_only_increment,
        "ap_union": ap_union,
        "ap_union_mechanism_subset": ap_union_mechanism_subset,
    }

    log.info("fallback notes: %s", fallback_notes)
    log.info("final metrics: %s", metrics)

    # Stash extras for the local entrypoint to build findings text.
    extras_path = out / f"{exp_id}_run_extras.csv"
    pd.DataFrame(
        [
            {
                "cv_ap_spread_D": fit_results["D"]["cv_ap_max"] - fit_results["D"]["cv_ap_min"],
                "cv_ap_spread_Ddrug": fit_results["D+drug"]["cv_ap_max"]
                - fit_results["D+drug"]["cv_ap_min"],
                "cv_ap_spread_Dcond": fit_results["D+cond"]["cv_ap_max"]
                - fit_results["D+cond"]["cv_ap_min"],
                "cv_ap_spread_union": fit_results["D+drug+cond"]["cv_ap_max"]
                - fit_results["D+drug+cond"]["cv_ap_min"],
                "fallback_notes": "; ".join(fallback_notes) if fallback_notes else "none",
                "matched_drug_blocks": ",".join(drug_diag["matched_blocks"]),
                "unmatched_wanted_blocks": ",".join(drug_diag["unmatched_wanted_blocks"]),
            }
        ]
    ).to_csv(extras_path, index=False)
    results.commit()

    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp02",
        title="Nested ablation: degree, drug-intrinsic, condition-intrinsic, union",
        hypothesis=(
            "Drug-intrinsic features add >=0.02 AP over the degree baseline, condition-"
            "intrinsic features add a further increment, and the no-interaction union "
            "falls short of pair-level biology."
        ),
        approach=(
            "Four nested feature designs, one fixed HistGBM, grouped 3-fold CV in train, "
            "one validate pass per design, plus a mechanism-subset refit to separate "
            "ChEMBL missingness from pharmacology."
        ),
        label="y_faers_signal",
        features=["degree", "drug_intrinsic", "condition_intrinsic", "union"],
        split="train/validate, grouped by primary target gene",
    )
    metrics = run.remote(exp)
    print(metrics)

    if metrics.get("precondition_failed"):
        ln.complete(
            exp,
            metrics={"precondition_failed": 1.0},
            findings=f"Precondition check failed: {metrics.get('precondition_error_message')}",
            failed=True,
        )
        return

    findings = (
        f"validate AP: degree-only={metrics['baseline_degree_only_ap']:.4f}, "
        f"drug increment={metrics['ap_drug_only_increment']:+.4f}, "
        f"condition increment={metrics['ap_condition_only_increment']:+.4f}, "
        f"union={metrics['ap_union']:.4f}, "
        f"union on mechanism-target subset={metrics['ap_union_mechanism_subset']:.4f}. "
        "See <exp_id>_ablation.csv for per-design CV fold spread (min/max) to judge whether "
        "each increment exceeds fold-to-fold noise, <exp_id>_subgroups.csv for where the "
        "increment concentrates (is_mapped / arm / best_match_tier / has_chembl_match), and "
        "<exp_id>_missingness_attribution.csv for whether the drug-side increment survives "
        "restriction to the 1,716 mechanism-target ingredients (pharmacology) or collapses "
        "(ChEMBL-missingness indicator only). Implication for Round 2's pair-level features "
        "stated in that comparison: if the union barely exceeds the degree floor, pair-level "
        "biology work is well-motivated; if the union already captures most of the signal, "
        "degree plus intrinsics may be essentially all of it."
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"results/{exp}/{exp}_ablation.csv",
            f"results/{exp}/{exp}_importance_top25.csv",
            f"results/{exp}/{exp}_subgroups.csv",
            f"results/{exp}/{exp}_calibration.png",
            f"results/{exp}/{exp}_missingness_attribution.csv",
        ],
        next_steps=(
            "Pull results/<exp_id>/ from the bridge-results volume and inspect the ablation "
            "table and importances before deciding whether Round 2's pathway-overlap and "
            "graph work is warranted."
        ),
        failed=bool(metrics.get("precondition_failed")),
    )
