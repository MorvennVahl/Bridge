"""exp07 — Honest nested intrinsic ablation on drug_macro_auc, with a leak audit.

Implements experiments/exp07_ablation_rerun_leak_audit.md exactly, scoring per
experiments/METRIC.md's drug_macro_auc definition. Reruns exp02's four nested designs with
exp02's label leak fixed (the degree block no longer includes log1p(faers_case_count), one
of the three clauses of y_faers_signal itself), adds p_c (condition train flag rate) as an
explicit baseline column in every design, and applies the leak-audit checks defined in
experiments/leak_audit.py (also duplicated inline here so this script does not depend on
Modal's mounting of a second local file).

Column-selection / feature-block logic (drug-intrinsic block selection via the data
dictionary, condition-intrinsic block from condition_features_basic.csv +
condition_group_long.csv one-hots, one-hot cardinality capping) is copied from
experiments/exp02_intrinsic_blocks.py, which already implements this per the shared
exp02_intrinsic_blocks.md §Feature blocks spec.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

app = modal.App("bridge-exp07")

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

# ---- copied from exp02_intrinsic_blocks.py (see that file for the block-selection spec) ----

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

# ---- leak audit, inlined (also written standalone at experiments/leak_audit.py) ----

LEAK_BLACKLIST_EXACT: set[str] = {
    "faers_case_count",
    "faers_prr",
    "faers_ror",
    "faers_chi_square",
    "in_faers",
    "in_semmeddb",
    "in_eu_label",
}
LEAK_BLACKLIST_PREFIXES: tuple[str, ...] = ("semmeddb_", "y_")


def _audit_columns(columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if c in LEAK_BLACKLIST_EXACT or any(c.startswith(p) for p in LEAK_BLACKLIST_PREFIXES)
    ]


def _per_feature_auc_screen(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    y_arr = y.to_numpy()
    rows: list[dict[str, float | str]] = []
    for col in x.columns:
        series = x[col]
        if series.dtype == object or series.dtype == bool:
            try:
                series = series.astype(float)
            except (TypeError, ValueError):
                continue
        series = series.astype(float)
        if series.notna().sum() == 0:
            continue
        filled = series.fillna(series.median())
        if filled.nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y_arr, filled.to_numpy())
        except ValueError:
            continue
        auc = max(auc, 1.0 - auc)
        rows.append({"feature": col, "pooled_auc": float(auc)})
    screen = pd.DataFrame(rows, columns=["feature", "pooled_auc"])
    return screen.sort_values("pooled_auc", ascending=False).reset_index(drop=True)


def _check_train_validate_gap(
    train_cv_metric: float, validate_metric: float, max_gap: float = 0.15
) -> None:
    gap = train_cv_metric - validate_metric
    if gap > max_gap:
        raise AssertionError(
            f"train-validate gap {gap:.4f} exceeds max_gap={max_gap:.4f} "
            f"(train_cv={train_cv_metric:.4f}, validate={validate_metric:.4f})"
        )


def _normalize_block(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _cap_and_onehot(series: pd.Series, prefix: str, cap: int = CARDINALITY_CAP) -> pd.DataFrame:
    import pandas as pd

    counts = series.value_counts(dropna=True)
    keep = set(counts.index[:cap])
    bucketed = series.where(series.isin(keep) | series.isna(), other="other")
    return pd.get_dummies(bucketed, prefix=prefix, dummy_na=False)


def _check_preconditions(
    train_path: pathlib.Path,
    validate_path: pathlib.Path,
    drug_path: pathlib.Path,
    cond_path: pathlib.Path,
    group_path: pathlib.Path,
    dict_path: pathlib.Path,
) -> str | None:
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


def _build_degree_and_pc_features(
    train: pd.DataFrame, validate: pd.DataFrame, cond_basic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train-only degree + p_c block. Bug fix vs exp02: no faers_case_count/prr/chi_square.

    Degree: drug degree, condition degree, condition record_count (all log1p, train-only).
    p_c: condition's TRAIN flag rate for y_faers_signal, mapped to both splits; conditions
    unseen in train fall back to the train-wide mean y_faers_signal rate (noted below).
    """
    import numpy as np
    import pandas as pd

    fallback_notes: list[str] = []

    drug_degree = train.groupby("ingredient_concept_id").size()
    cond_degree = train.groupby("condition_concept_id").size()
    drug_degree_median = float(drug_degree.median())
    cond_degree_median = float(cond_degree.median())

    record_count_map = cond_basic.set_index("condition_concept_id")["record_count"]
    record_count_median = float(record_count_map.median())

    p_c_map = train.groupby("condition_concept_id")["y_faers_signal"].mean()
    train_wide_mean = float(train["y_faers_signal"].mean())
    n_unseen_val = int((~validate["condition_concept_id"].isin(p_c_map.index)).sum())
    if n_unseen_val:
        fallback_notes.append(
            f"{n_unseen_val} validate rows had a condition unseen in train; p_c fell back "
            f"to the train-wide y_faers_signal rate ({train_wide_mean:.4f})."
        )

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["degree_drug"] = np.log1p(
            df["ingredient_concept_id"].map(drug_degree).fillna(drug_degree_median)
        )
        out["degree_condition"] = np.log1p(
            df["condition_concept_id"].map(cond_degree).fillna(cond_degree_median)
        )
        out["degree_condition_record_count"] = np.log1p(
            df["condition_concept_id"].map(record_count_map).fillna(record_count_median)
        )
        out["p_c"] = df["condition_concept_id"].map(p_c_map).fillna(train_wide_mean)
        return out

    return _apply(train), _apply(validate), fallback_notes


def _load_drug_features(
    drug_path: pathlib.Path, dict_path: pathlib.Path
) -> tuple[pd.DataFrame, dict]:
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


def _build_drug_design(drug_raw: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    df = drug_raw.copy()
    join_col = "omop_concept_id"
    feature_cols = [c for c in df.columns if c != join_col]

    out_parts = [df[[join_col]]]
    for col in feature_cols:
        if df[col].dtype == object:
            out_parts.append(_cap_and_onehot(df[col], prefix=col))
        elif df[col].dtype == bool:
            out_parts.append(df[[col]].astype(float))
        else:
            out_parts.append(df[[col]])

    return pd.concat(out_parts, axis=1)


def _load_condition_features(cond_path: pathlib.Path, group_path: pathlib.Path) -> pd.DataFrame:
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


# ---- drug_macro_auc machinery (per experiments/METRIC.md) ----


def _per_drug_table(
    ids: pd.Series, y: pd.Series, score: np.ndarray, min_pairs: int = 20
) -> pd.DataFrame:
    """Per-drug AUC / precision@10 / recall@50, restricted to eligible drugs.

    Eligibility: >=min_pairs observed pairs AND both classes present, per drug.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame({"drug": ids.to_numpy(), "y": y.to_numpy(), "score": score})
    rows: list[dict[str, float]] = []
    for drug_id, grp in df.groupby("drug"):
        n = len(grp)
        if n < min_pairs or grp["y"].nunique() < 2:
            continue
        auc = float(roc_auc_score(grp["y"], grp["score"]))
        ranked = grp.sort_values("score", ascending=False)
        top10 = ranked.head(10)
        p10 = float(top10["y"].mean()) if len(top10) else float("nan")
        top50 = ranked.head(50)
        n_pos = grp["y"].sum()
        r50 = float(top50["y"].sum() / n_pos) if n_pos > 0 else float("nan")
        rows.append({"drug": drug_id, "n": n, "auc": auc, "p10": p10, "r50": r50})
    return pd.DataFrame(rows, columns=["drug", "n", "auc", "p10", "r50"])


def _drug_macro_metrics(ids: pd.Series, y: pd.Series, score: np.ndarray) -> dict:
    import numpy as np

    table = _per_drug_table(ids, y, score)
    if table.empty:
        return {
            "drug_macro_auc": float("nan"),
            "drug_macro_p10": float("nan"),
            "drug_macro_r50": float("nan"),
            "n_drugs_scored": 0,
            "table": table,
        }
    return {
        "drug_macro_auc": float(np.mean(table["auc"])),
        "drug_macro_p10": float(np.mean(table["p10"])),
        "drug_macro_r50": float(np.nanmean(table["r50"])),
        "n_drugs_scored": len(table),
        "table": table,
    }


def _bootstrap_increment_ci(
    table_a: pd.DataFrame, table_b: pd.DataFrame, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap CI (over drugs) for mean(auc_b - auc_a) on the drugs eligible in both."""
    import numpy as np

    merged = table_a[["drug", "auc"]].merge(
        table_b[["drug", "auc"]], on="drug", suffixes=("_a", "_b")
    )
    if merged.empty:
        return float("nan"), float("nan"), float("nan")
    diffs = (merged["auc_b"] - merged["auc_a"]).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    point = float(diffs.mean())
    lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    return point, lo, hi


def _fit_score(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    ids_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    ids_val: pd.Series,
    n_folds: int = 3,
) -> dict:
    """Grouped CV inside train scored on drug_macro_auc (held-out CV drugs), then one
    fit on all of train and one validate pass, reporting drug_macro_* plus pooled AP/AUC.
    """
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
    cv_drug_macro_aucs: list[float] = []
    for train_idx, test_idx in gkf.split(x_train, y_train, groups=groups_train):
        model = _make_model()
        model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
        proba = model.predict_proba(x_train.iloc[test_idx])[:, 1]
        fold_metrics = _drug_macro_metrics(ids_train.iloc[test_idx], y_train.iloc[test_idx], proba)
        cv_drug_macro_aucs.append(fold_metrics["drug_macro_auc"])

    final_model = _make_model()
    final_model.fit(x_train, y_train)
    val_proba = final_model.predict_proba(x_val)[:, 1]
    val_ap = float(average_precision_score(y_val, val_proba))
    val_pooled_auc = float(roc_auc_score(y_val, val_proba))
    val_macro = _drug_macro_metrics(ids_val, y_val, val_proba)

    return {
        "model": final_model,
        "cv_drug_macro_aucs": cv_drug_macro_aucs,
        "cv_drug_macro_auc_mean": float(np.nanmean(cv_drug_macro_aucs)),
        "cv_drug_macro_auc_min": float(np.nanmin(cv_drug_macro_aucs)),
        "cv_drug_macro_auc_max": float(np.nanmax(cv_drug_macro_aucs)),
        "val_ap": val_ap,
        "val_pooled_auc": val_pooled_auc,
        "val_drug_macro_auc": val_macro["drug_macro_auc"],
        "val_drug_macro_p10": val_macro["drug_macro_p10"],
        "val_drug_macro_r50": val_macro["drug_macro_r50"],
        "val_n_drugs_scored": val_macro["n_drugs_scored"],
        "val_drug_table": val_macro["table"],
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
    log = logging.getLogger("exp07")

    import numpy as np
    import pandas as pd

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    data_root = pathlib.Path("/data")
    train_path = data_root / "splits" / "train.csv"
    validate_path = data_root / "splits" / "validate.csv"
    drug_path = data_root / "drug" / "ingredient_features.csv"
    cond_path = data_root / "condition" / "condition_features_basic.csv"
    group_path = data_root / "condition" / "condition_group_long.csv"
    dict_path = data_root / "data_dictionary.csv"

    err = _check_preconditions(
        train_path, validate_path, drug_path, cond_path, group_path, dict_path
    )
    if err is not None:
        log.error("precondition check failed: %s", err)
        return {"precondition_failed": 1.0, "precondition_error_message": err}

    log.info("preconditions passed; loading train/validate")
    train = pd.read_csv(train_path)
    validate = pd.read_csv(validate_path)
    cond_basic = pd.read_csv(cond_path)

    fallback_notes: list[str] = []

    # group_key disjointness check underlies the unseen-family question below.
    overlap = set(train["group_key"]) & set(validate["group_key"])
    if overlap:
        fallback_notes.append(
            f"group_key overlap between train/validate is non-empty ({len(overlap)} keys); "
            "unseen-family restriction via group_key would not be trivial."
        )
    unseen_family_is_trivial = len(overlap) == 0

    # Degree + p_c block (bug fix: no faers_case_count/prr/chi_square anywhere).
    degree_train, degree_val, pc_notes = _build_degree_and_pc_features(train, validate, cond_basic)
    fallback_notes.extend(pc_notes)

    log.info("loading drug-intrinsic block")
    drug_raw, drug_diag = _load_drug_features(drug_path, dict_path)
    log.info("drug block diagnostics: %s", drug_diag)
    drug_design = _build_drug_design(drug_raw).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )

    log.info("loading condition-intrinsic block")
    cond_design = _load_condition_features(cond_path, group_path)

    def _assemble(base_df: pd.DataFrame, degree_pc_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        idx = base_df.index
        d = degree_pc_df.set_axis(idx)
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

        return {
            "D+p_c": d.copy(),
            "D+p_c+drug": pd.concat([d, merged_drug], axis=1),
            "D+p_c+cond": pd.concat([d, merged_cond], axis=1),
            "D+p_c+drug+cond": pd.concat([d, merged_drug, merged_cond], axis=1),
        }

    train_designs = _assemble(train, degree_train)
    val_designs = _assemble(validate, degree_val)

    y_train = train["y_faers_signal"]
    y_val = validate["y_faers_signal"]
    groups_train = train["group_key"]
    ids_train = train["ingredient_concept_id"]
    ids_val = validate["ingredient_concept_id"]

    design_order = ["D+p_c", "D+p_c+drug", "D+p_c+cond", "D+p_c+drug+cond"]

    # --- leak audit step 1: blacklist by construction ---
    for name in design_order:
        hits = _audit_columns(list(train_designs[name].columns))
        if hits:
            msg = f"leak audit: design {name} contains blacklisted columns: {hits}"
            log.error(msg)
            return {"precondition_failed": 1.0, "precondition_error_message": msg}
    log.info("leak audit step 1 passed: no blacklisted columns in any design")

    # --- leak audit step 2: per-feature pooled AUC screen over the widest design ---
    screen_cols = train_designs["D+p_c+drug+cond"]
    screen_df = _per_feature_auc_screen(screen_cols, y_train)
    screen_path = out / f"{exp_id}_feature_screen.csv"
    screen_df.to_csv(screen_path, index=False)
    if not screen_df.empty and screen_df["pooled_auc"].max() > 0.75:
        offender = screen_df.iloc[0]
        msg = (
            f"leak audit: feature '{offender['feature']}' has pooled AUC "
            f"{offender['pooled_auc']:.4f} > 0.75"
        )
        log.error(msg)
        results.commit()
        return {"precondition_failed": 1.0, "precondition_error_message": msg}
    log.info(
        "leak audit step 2 passed: max single-feature pooled AUC = %.4f",
        screen_df["pooled_auc"].max() if not screen_df.empty else float("nan"),
    )

    fit_results: dict[str, dict] = {}
    for name in design_order:
        n_folds = 3
        if name == "D+p_c+drug+cond" and train_designs[name].shape[1] > 400:
            n_folds = 2
            fallback_notes.append(
                "Dropped CV to 2 folds for D+p_c+drug+cond due to feature-count budget risk."
            )
        log.info(
            "fitting design=%s n_features=%d n_folds=%d",
            name,
            train_designs[name].shape[1],
            n_folds,
        )
        res = _fit_score(
            train_designs[name],
            y_train,
            groups_train,
            ids_train,
            val_designs[name],
            y_val,
            ids_val,
            n_folds=n_folds,
        )
        fit_results[name] = res
        log.info(
            "design=%s cv_drug_macro_auc_mean=%.4f val_drug_macro_auc=%.4f val_ap=%.4f",
            name,
            res["cv_drug_macro_auc_mean"],
            res["val_drug_macro_auc"],
            res["val_ap"],
        )

        # --- leak audit step 3: train-validate gap on drug_macro_auc ---
        try:
            _check_train_validate_gap(
                res["cv_drug_macro_auc_mean"], res["val_drug_macro_auc"], max_gap=0.15
            )
        except AssertionError as exc:
            log.error("leak audit step 3 failed for design %s: %s", name, exc)
            results.commit()
            return {"precondition_failed": 1.0, "precondition_error_message": str(exc)}
    log.info("leak audit step 3 passed for all four designs")

    # --- floor: p_c-only lookup, no model ---
    floor_metrics = _drug_macro_metrics(ids_val, y_val, val_designs["D+p_c"]["p_c"].to_numpy())
    floor_drug_macro_auc_pc_lookup = floor_metrics["drug_macro_auc"]
    log.info(
        "floor_drug_macro_auc_pc_lookup = %.4f (Round 1 measured 0.5759)",
        floor_drug_macro_auc_pc_lookup,
    )
    if abs(floor_drug_macro_auc_pc_lookup - 0.5759) > 0.01:
        fallback_notes.append(
            f"floor_drug_macro_auc_pc_lookup={floor_drug_macro_auc_pc_lookup:.4f} does not "
            "closely match the Round-1-measured 0.5759 sanity check."
        )

    # --- unseen-family metric ---
    if unseen_family_is_trivial:
        log.info(
            "group_key partitions train/validate disjointly (split is grouped by primary "
            "target gene), so EVERY validate drug already satisfies 'primary-target family "
            "never appears in train' under this split's grouping unit. "
            "drug_macro_auc_unseen_family is therefore identical to the overall validate "
            "drug_macro_auc, not a distinct restricted computation."
        )
        drug_macro_auc_unseen_family = fit_results["D+p_c+drug+cond"]["val_drug_macro_auc"]
        unseen_family_note = (
            "drug_macro_auc_unseen_family == overall validate drug_macro_auc: group_key "
            "(the split's grouping unit) is disjoint between train and validate by "
            "construction, so all validate drugs are 'unseen family' under this scheme; "
            "ingredient_target_long.csv (or a chembl_protein_class_L1-style column) was "
            "not available on the volume to define a finer family unit."
        )
    else:
        drug_macro_auc_unseen_family = fit_results["D+p_c+drug+cond"]["val_drug_macro_auc"]
        unseen_family_note = (
            "group_key overlap detected between train/validate; unseen_family fell back to "
            "the overall validate drug_macro_auc as a conservative approximation."
        )
    log.info(unseen_family_note)

    # --- bootstrap CI over drugs for each increment ---
    bootstrap_rows = []
    pairs = [
        ("D+p_c", "D+p_c+drug"),
        ("D+p_c", "D+p_c+cond"),
        ("D+p_c+drug", "D+p_c+drug+cond"),
        ("D+p_c+cond", "D+p_c+drug+cond"),
        ("D+p_c", "D+p_c+drug+cond"),
    ]
    for a, b in pairs:
        point, lo, hi = _bootstrap_increment_ci(
            fit_results[a]["val_drug_table"], fit_results[b]["val_drug_table"], n_boot=1000
        )
        bootstrap_rows.append(
            {
                "from_design": a,
                "to_design": b,
                "drug_macro_auc_increment": point,
                "ci_lo_2.5": lo,
                "ci_hi_97.5": hi,
                "noise_floor_note": "increment < ~0.01 is noise given across-drug sd 0.128"
                if abs(point) < 0.01
                else "",
            }
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_path = out / f"{exp_id}_bootstrap_increments.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    # --- ablation table ---
    ablation_rows = []
    prev_ap = None
    for name in design_order:
        res = fit_results[name]
        ap_increment = None if prev_ap is None else res["val_ap"] - prev_ap
        ablation_rows.append(
            {
                "design": name,
                "n_features": train_designs[name].shape[1],
                "train_cv_drug_macro_auc_mean": res["cv_drug_macro_auc_mean"],
                "train_cv_drug_macro_auc_min": res["cv_drug_macro_auc_min"],
                "train_cv_drug_macro_auc_max": res["cv_drug_macro_auc_max"],
                "validate_drug_macro_auc": res["val_drug_macro_auc"],
                "validate_drug_macro_p10": res["val_drug_macro_p10"],
                "validate_drug_macro_r50": res["val_drug_macro_r50"],
                "n_drugs_scored": res["val_n_drugs_scored"],
                "validate_ap": res["val_ap"],
                "validate_pooled_auc": res["val_pooled_auc"],
                "ap_increment_over_previous": ap_increment,
            }
        )
        prev_ap = res["val_ap"]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = out / f"{exp_id}_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)

    # --- importances (LightGBM gain proxy, mirrors exp02's approach) ---
    importance_rows = []
    for name in design_order:
        import re

        import lightgbm as lgb

        cols = train_designs[name].columns
        x_lgb = train_designs[name].copy()
        seen: dict[str, int] = {}
        sanitized: list[str] = []
        for c in x_lgb.columns:
            base = re.sub(r"[^0-9A-Za-z_]", "_", str(c))
            cnt = seen.get(base, 0)
            seen[base] = cnt + 1
            sanitized.append(base if cnt == 0 else f"{base}_{cnt}")
        x_lgb.columns = sanitized

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.06, num_leaves=63, random_state=0, verbosity=-1
        )
        lgb_model.fit(x_lgb, y_train)
        importances = lgb_model.booster_.feature_importance(importance_type="gain")
        top_idx = np.argsort(importances)[::-1][:25]
        for rank, i in enumerate(top_idx, start=1):
            importance_rows.append(
                {"design": name, "rank": rank, "feature": cols[i], "gain": float(importances[i])}
            )
    importance_df = pd.DataFrame(importance_rows)
    importance_path = out / f"{exp_id}_importance_top25.csv"
    importance_df.to_csv(importance_path, index=False)

    # --- subgroups: by is_mapped, arm, gene_arm, best_match_tier, has_chembl_match ---
    subgroup_cols_wanted = ["is_mapped", "arm", "gene_arm", "best_match_tier", "has_chembl_match"]
    available_cond_cols = [c for c in subgroup_cols_wanted if c in cond_basic.columns]
    missing_cond_cols = [
        c for c in subgroup_cols_wanted if c not in cond_basic.columns and c != "has_chembl_match"
    ]
    if missing_cond_cols:
        log.info(
            "subgroup columns absent from condition_features_basic.csv, skipping: %s",
            missing_cond_cols,
        )

    cond_meta_cols = ["condition_concept_id"] + [
        c for c in available_cond_cols if c != "has_chembl_match"
    ]
    cond_meta = cond_basic[cond_meta_cols]
    val_meta = validate.merge(cond_meta, on="condition_concept_id", how="left")
    val_meta = val_meta.merge(
        drug_raw[["omop_concept_id", "has_chembl_match"]],
        left_on="ingredient_concept_id",
        right_on="omop_concept_id",
        how="left",
    )
    assert len(val_meta) == len(validate), "subgroup merge must not change row count/order"

    from sklearn.metrics import average_precision_score as _ap

    subgroup_cols_present = [*available_cond_cols, "has_chembl_match"]
    subgroup_rows = []
    for name in design_order:
        proba = fit_results[name]["val_proba"]
        for subgroup_col in subgroup_cols_present:
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

    # --- missingness attribution: D+p_c+drug restricted to mechanism-target ingredients ---
    drug_full = pd.read_csv(drug_path)
    if "target_chembl_ids" in drug_full.columns:
        mechanism_ingredients = set(
            drug_full.loc[drug_full["target_chembl_ids"].notna(), "omop_concept_id"]
        )
    elif "mechanisms_of_action" in drug_full.columns:
        mechanism_ingredients = set(
            drug_full.loc[drug_full["mechanisms_of_action"].notna(), "omop_concept_id"]
        )
    else:
        mechanism_ingredients = set()

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
        x_train_mech = train_designs["D+p_c+drug"].loc[train_mech_mask]
        y_train_mech = y_train.loc[train_mech_mask]
        groups_train_mech = groups_train.loc[train_mech_mask]
        ids_train_mech = ids_train.loc[train_mech_mask]
        x_val_mech = val_designs["D+p_c+drug"].loc[val_mech_mask]
        y_val_mech = y_val.loc[val_mech_mask]
        ids_val_mech = ids_val.loc[val_mech_mask]

        mech_res = _fit_score(
            x_train_mech,
            y_train_mech,
            groups_train_mech,
            ids_train_mech,
            x_val_mech,
            y_val_mech,
            ids_val_mech,
            n_folds=3,
        )
        missingness_rows.append(
            {
                "design": "D+p_c+drug (mechanism subset)",
                "n_train_rows": int(train_mech_mask.sum()),
                "n_validate_rows": int(val_mech_mask.sum()),
                "train_cv_drug_macro_auc_mean": mech_res["cv_drug_macro_auc_mean"],
                "validate_drug_macro_auc": mech_res["val_drug_macro_auc"],
                "validate_ap": mech_res["val_ap"],
                "full_D_p_c_drug_validate_ap": fit_results["D+p_c+drug"]["val_ap"],
                "full_D_p_c_validate_ap": fit_results["D+p_c"]["val_ap"],
            }
        )
    else:
        fallback_notes.append(
            "Mechanism-target subset had zero rows in train or validate; missingness "
            "attribution could not be scored."
        )
    missingness_df = pd.DataFrame(missingness_rows)
    missingness_path = out / f"{exp_id}_missingness_attribution.csv"
    missingness_df.to_csv(missingness_path, index=False)

    results.commit()

    union = fit_results["D+p_c+drug+cond"]
    baseline = fit_results["D+p_c"]
    metrics = {
        "drug_macro_auc": union["val_drug_macro_auc"],
        "drug_macro_auc_unseen_family": drug_macro_auc_unseen_family,
        "drug_macro_p10": union["val_drug_macro_p10"],
        "drug_macro_r50": union["val_drug_macro_r50"],
        "n_drugs_scored": union["val_n_drugs_scored"],
        "floor_drug_macro_auc_pc_lookup": floor_drug_macro_auc_pc_lookup,
        "ap_drug_only_increment": fit_results["D+p_c+drug"]["val_ap"] - baseline["val_ap"],
        "ap_condition_only_increment": fit_results["D+p_c+cond"]["val_ap"] - baseline["val_ap"],
        "pooled_average_precision": union["val_ap"],
        "validate_average_precision": union["val_ap"],
    }

    log.info("fallback notes: %s", fallback_notes)
    log.info("final metrics: %s", metrics)

    extras_path = out / f"{exp_id}_run_extras.csv"
    pd.DataFrame(
        [
            {
                "baseline_degree_pc_drug_macro_auc": baseline["val_drug_macro_auc"],
                "drug_increment_drug_macro_auc": fit_results["D+p_c+drug"]["val_drug_macro_auc"]
                - baseline["val_drug_macro_auc"],
                "cond_increment_drug_macro_auc": fit_results["D+p_c+cond"]["val_drug_macro_auc"]
                - baseline["val_drug_macro_auc"],
                "unseen_family_note": unseen_family_note,
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
        agent="exp07",
        title="Honest nested intrinsic ablation on drug_macro_auc, with a leak audit",
        hypothesis=(
            "With the faers_case_count label term removed and p_c carried as a baseline "
            "column, drug-intrinsic features add >=0.02 drug_macro_auc over the 0.576 "
            "floor while condition-intrinsic features add less."
        ),
        approach=(
            "Re-run of exp02's four nested designs on a clean degree block; grouped 3-fold "
            "CV scored on drug_macro_auc; bootstrap CIs over drugs; mechanism-subset "
            "refit; reusable leak_audit.py with blacklist, per-feature AUC screen and a "
            "train-validate gap assertion."
        ),
        label="y_faers_signal",
        features=["degree", "p_c", "drug_intrinsic", "condition_intrinsic"],
        split="train/validate, grouped by primary target gene",
    )
    metrics = run.remote(exp)
    print(metrics)

    if metrics.get("precondition_failed"):
        ln.complete(
            exp,
            metrics={"precondition_failed": 1.0},
            findings=f"Precondition/leak-audit check failed: {metrics.get('precondition_error_message')}",
            failed=True,
        )
        return

    findings = (
        "Corrected baseline: this run supersedes exp02's baseline_degree_only_ap=0.3431, "
        "which was inflated by a label leak (the degree block included "
        "log1p(faers_case_count), one of the three clauses of y_faers_signal itself). "
        f"floor_drug_macro_auc_pc_lookup={metrics['floor_drug_macro_auc_pc_lookup']:.4f} "
        "(Round-1-measured floor 0.5759). "
        f"Headline drug_macro_auc (D+p_c+drug+cond)={metrics['drug_macro_auc']:.4f}, "
        f"drug_macro_p10={metrics['drug_macro_p10']:.4f}, "
        f"drug_macro_r50={metrics['drug_macro_r50']:.4f}, "
        f"n_drugs_scored={metrics['n_drugs_scored']}. "
        f"pooled/validate AP (continuity with Round 1)={metrics['pooled_average_precision']:.4f}. "
        f"ap_drug_only_increment={metrics['ap_drug_only_increment']:+.4f}, "
        f"ap_condition_only_increment={metrics['ap_condition_only_increment']:+.4f} "
        "(pooled-AP increments; see <exp_id>_bootstrap_increments.csv for the primary "
        "drug_macro_auc increments with 1,000-resample bootstrap CIs over drugs -- an "
        "increment under ~0.01 is noise given across-drug sd 0.128). "
        "<exp_id>_missingness_attribution.csv separates pharmacology from the ChEMBL-"
        "missingness indicator for the drug-side increment. "
        f"drug_macro_auc_unseen_family={metrics['drug_macro_auc_unseen_family']:.4f}: "
        "the split's grouping unit (group_key, primary target gene) is disjoint between "
        "train and validate by construction, so this number is identical to the overall "
        "validate drug_macro_auc rather than a distinct restricted computation -- "
        "ingredient_target_long.csv / a finer chembl_protein_class_L1-style family unit "
        "was not available on the volume to compute a non-trivial restriction."
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"results/{exp}/{exp}_ablation.csv",
            f"results/{exp}/{exp}_feature_screen.csv",
            f"results/{exp}/{exp}_bootstrap_increments.csv",
            f"results/{exp}/{exp}_missingness_attribution.csv",
            f"results/{exp}/{exp}_importance_top25.csv",
            f"results/{exp}/{exp}_subgroups.csv",
        ],
        next_steps=(
            "Pull results/<exp_id>/ from the bridge-results volume. Use this run's numbers "
            "(not exp02's) as the reference floor/increments for exp08-exp10, and import "
            "experiments/leak_audit.py in each of those before fitting."
        ),
        supersedes="exp_20260922_f6645b",
        failed=False,
    )
