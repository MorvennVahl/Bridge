"""exp11 -- Adjudicated reference-set evaluation and a usable success criterion.

Implements Part B of experiments/exp11_adjudicated_reference_set.md: score the existing
exp07 union model, exp08 neighbour model, exp09 residual model, the incumbent FAERS score,
and the p_c floor -- all against the externally adjudicated reference set built by Part A
(data/input/reference/reference_set.csv), rather than against CEM labels.

Part B step 1 requires checking, before any fitting or scoring: the reference file is
present and non-empty, both label classes are present, and at least 30 drugs have >=5
reference pairs. Part A's own provenance JSON already flags that only 3 of 223 retained
drugs have >=5 reference pairs (a structural property of the OMOP/EU-ADR reference sets,
which are drug-by-HOI, not a dense drug-by-condition matrix) -- this script verifies that
independently from the CSV itself rather than trusting the provenance note, and if the
precondition fails, completes with failed=True naming the exact shortfall, without
loosening the bar or fabricating additional reference pairs.

Model-scoring code (exp07 union model feature assembly, exp08 neighbour/nb_excess_gene
feature assembly, exp09 one-way condition-demeaned residual) is copied from
experiments/exp07_ablation_rerun_leak_audit.py, experiments/exp08_neighbour_transport_pc_stripped.py
and experiments/exp09_within_drug_residual.py, which already implement these per their own
specs -- reused here only if preconditions pass, and only re-scored on reference pairs
after a single fit on the full train split (no fitting on the reference set itself, which
is the project's only external yardstick).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import modal

app = modal.App("bridge-exp11")

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

MIN_DRUGS_WITH_GE5_PAIRS = 30
MIN_PAIRS_PER_DRUG_FOR_REFERENCE = 5
BOOT_RESAMPLES = 1000
K_SHRINK = 10

HGB_KWARGS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "early_stopping": False,
    "random_state": 0,
}

# ---- copied from exp07 (block-selection spec) --------------------------------------

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


def _normalize_block(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _cap_and_onehot(series, prefix: str, cap: int = CARDINALITY_CAP):
    import pandas as pd

    counts = series.value_counts(dropna=True)
    keep = set(counts.index[:cap])
    bucketed = series.where(series.isin(keep) | series.isna(), other="other")
    return pd.get_dummies(bucketed, prefix=prefix, dummy_na=False)


def _load_drug_features(drug_path: pathlib.Path, dict_path: pathlib.Path):
    import pandas as pd

    ddict = pd.read_csv(dict_path)
    ddict_ing = ddict[ddict["table"] == "ingredient_features.csv"].copy()
    ddict_ing["block_norm"] = ddict_ing["block"].map(_normalize_block)

    wanted_norm = {_normalize_block(b) for b in DRUG_BLOCKS_WANTED}
    present_blocks = sorted(ddict_ing["block_norm"].unique())
    matched_blocks = sorted(wanted_norm & set(present_blocks))

    selected_cols = ddict_ing.loc[ddict_ing["block_norm"].isin(matched_blocks), "column"].tolist()

    drug = pd.read_csv(drug_path)
    keep_cols = [c for c in selected_cols if c in drug.columns]

    drop_cols = set(DRUG_LIST_COLUMNS) | set(IDENTITY_EXTRA_DROP)
    drop_cols |= {c for c in drug.columns if c.startswith("chembl_fuzzy_")}
    keep_cols = [c for c in keep_cols if c not in drop_cols]

    join_col = "omop_concept_id"
    out = drug[
        [join_col, "chembl_id"] + [c for c in keep_cols if c not in (join_col, "chembl_id")]
    ].copy()
    out["has_chembl_match"] = out["chembl_id"].notna()
    out = out.drop(columns=["chembl_id"])
    return out


def _build_drug_design(drug_raw):
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


def _load_condition_features(cond_path: pathlib.Path, group_path: pathlib.Path):
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


def _build_degree_and_pc_features(train, other_df, cond_basic):
    import numpy as np

    drug_degree = train.groupby("ingredient_concept_id").size()
    cond_degree = train.groupby("condition_concept_id").size()
    drug_degree_median = float(drug_degree.median())
    cond_degree_median = float(cond_degree.median())

    record_count_map = cond_basic.set_index("condition_concept_id")["record_count"]
    record_count_median = float(record_count_map.median())

    p_c_map = train.groupby("condition_concept_id")["y_faers_signal"].mean()
    train_wide_mean = float(train["y_faers_signal"].mean())

    def _apply(df):
        import pandas as pd

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

    return _apply(train), _apply(other_df), p_c_map, train_wide_mean


# ---- copied from exp08 (nb_excess_gene same-target neighbour feature) --------------


def _dedup_pairs(df, value_col: str):
    p = df[["ingredient_concept_id", value_col]].dropna().rename(columns={value_col: "value"})
    p["value"] = p["value"].astype(str)
    return p.drop_duplicates()


def _build_adjacency(pairs, ing_index_map: dict, n_ing: int):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    ing_pos = pairs["ingredient_concept_id"].map(ing_index_map).to_numpy()
    val_codes, _ = pd.factorize(pairs["value"])
    n_val = val_codes.max() + 1 if len(val_codes) else 0
    if n_val == 0:
        return sp.csr_matrix((n_ing, n_ing))
    a = sp.csr_matrix(
        (np.ones(len(pairs), dtype=np.float64), (ing_pos, val_codes)),
        shape=(n_ing, n_val),
    )
    a.data[:] = 1.0
    nb = (a @ a.T).tolil()
    nb.setdiag(0)
    nb = nb.tocsr()
    nb.eliminate_zeros()
    nb.data[:] = 1.0
    return nb


def _mask_to_train_columns(nb, train_mask):
    import numpy as np

    return nb.multiply(np.asarray(train_mask, dtype=np.float64).reshape(1, -1)).tocsr()


def _tier_features_excess(
    tier_name,
    edges,
    ing_index_map,
    n_ing,
    train_mask,
    d_pos_train,
    d_pos_other,
    c_pos_train,
    c_pos_other,
    f_train,
    p_c_train,
    p_c_other,
    log,
):
    import numpy as np

    nb = _build_adjacency(edges, ing_index_map, n_ing)
    nb = _mask_to_train_columns(nb, train_mask)

    n_neighbours_ing = np.asarray(nb.sum(axis=1)).ravel()
    n_neighbours_train = n_neighbours_ing[d_pos_train]
    n_neighbours_other = n_neighbours_ing[d_pos_other]

    m = (nb @ f_train).toarray() if f_train.shape[1] else np.zeros((n_ing, 0))
    if m.shape[1]:
        n_flagged_train = m[d_pos_train, c_pos_train]
        n_flagged_other = m[d_pos_other, c_pos_other]
    else:
        n_flagged_train = np.zeros(len(d_pos_train))
        n_flagged_other = np.zeros(len(d_pos_other))
    del m

    rate_train = (n_flagged_train + K_SHRINK * p_c_train) / (n_neighbours_train + K_SHRINK)
    rate_other = (n_flagged_other + K_SHRINK * p_c_other) / (n_neighbours_other + K_SHRINK)

    out = {
        f"n_neighbours_{tier_name}__train": n_neighbours_train,
        f"n_neighbours_{tier_name}__other": n_neighbours_other,
        f"nb_excess_{tier_name}__train": rate_train - p_c_train,
        f"nb_excess_{tier_name}__other": rate_other - p_c_other,
    }
    log.info(
        "tier=%s median train neighbours=%.1f, %.1f%% of reference rows have >=1",
        tier_name,
        float(np.median(n_neighbours_train)),
        100.0 * float((n_neighbours_other >= 1).mean())
        if len(n_neighbours_other)
        else float("nan"),
    )
    return out


def _to_bool(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y", "t"])


# ---- copied from exp09 (one-way condition-demeaned log(faers_prr) residual) --------


def _select_drug_feature_columns_exp09(
    data_dict, excluded_columns: set[str], included_blocks: set[str]
):
    dd = data_dict[data_dict["table"] == "ingredient_features.csv"]
    dd = dd[dd["block"].isin(included_blocks)]
    return [c for c in dd["column"].tolist() if c not in excluded_columns]


EXP09_EXCLUDED_COLUMNS = {
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
EXP09_INCLUDED_BLOCKS = {
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
EXP09_CASE_COUNT_FLOOR = 3


# ---- drug_macro_auc / p10 machinery (per METRIC.md), used with a min_pairs=5 override
# for the reference set (the spec's own reporting bar is >=5 pairs/drug, not the usual
# >=20-pair METRIC.md eligibility rule, since the reference set is small by construction)


def _per_drug_table(ids, y, score, min_pairs: int):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    score = np.asarray(score, dtype=float)
    ids_arr = ids.to_numpy() if hasattr(ids, "to_numpy") else np.asarray(ids)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    df = pd.DataFrame({"drug": ids_arr, "y": y_arr, "score": score})
    rows: list[dict[str, float]] = []
    for drug_id, grp in df.groupby("drug"):
        n = len(grp)
        # NaN-safe: skip a drug/model combination outright if too few pairs, or if
        # only one class is present for that drug (roc_auc_score would raise).
        if n < min_pairs or grp["y"].nunique() < 2:
            continue
        s = grp["score"]
        if s.isna().any():
            if s.notna().sum() == 0:
                continue
            s = s.fillna(s.median())
        auc = float(roc_auc_score(grp["y"], s))
        ranked = grp.assign(score=s).sort_values("score", ascending=False)
        top10 = ranked.head(10)
        p10 = float(top10["y"].mean()) if len(top10) else float("nan")
        rows.append({"drug": drug_id, "n": n, "auc": auc, "p10": p10})
    return pd.DataFrame(rows, columns=["drug", "n", "auc", "p10"])


def _drug_macro_metrics(ids, y, score, min_pairs: int = MIN_PAIRS_PER_DRUG_FOR_REFERENCE) -> dict:
    import numpy as np

    table = _per_drug_table(ids, y, score, min_pairs=min_pairs)
    if table.empty:
        return {
            "drug_macro_auc": float("nan"),
            "drug_macro_p10": float("nan"),
            "n_drugs_scored": 0,
            "table": table,
        }
    return {
        "drug_macro_auc": float(np.mean(table["auc"])),
        "drug_macro_p10": float(np.mean(table["p10"])),
        "n_drugs_scored": len(table),
        "table": table,
    }


def _bootstrap_metric_ci(table, col: str = "auc", n_boot: int = BOOT_RESAMPLES, seed: int = 0):
    import numpy as np

    if table.empty:
        return float("nan"), float("nan"), float("nan")
    values = table[col].to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = values[idx].mean()
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _pooled_auc_ap(y, score) -> tuple[float, float]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = np.asarray(score, dtype=float)
    y = np.asarray(y)
    if np.isnan(score).any():
        med = float(np.nanmedian(score)) if not np.all(np.isnan(score)) else 0.0
        score = np.where(np.isnan(score), med, score)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _spearman_rank_corr(a, b) -> float:
    import numpy as np
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def _check_preconditions(
    ref_path: pathlib.Path,
) -> tuple[str | None, dict[str, Any]]:
    """Part B step 1: reference file present, non-empty, both label classes present,
    and >=30 drugs with >=5 reference pairs. Returns (error_message_or_None, diagnostics).
    """
    import pandas as pd

    if not ref_path.exists():
        return f"missing required input: {ref_path}", {}

    ref = pd.read_csv(ref_path)
    if ref.empty:
        return f"reference file {ref_path} is empty (0 rows)", {}

    required_cols = {
        "ingredient_concept_id",
        "condition_concept_id",
        "label",
        "source",
        "adjudication",
        "notes",
    }
    missing_cols = required_cols - set(ref.columns)
    if missing_cols:
        return f"reference file is missing required columns: {sorted(missing_cols)}", {}

    label_counts = ref["label"].value_counts().to_dict()
    classes_present = set(ref["label"].unique())
    if not {0, 1}.issubset(classes_present):
        return (
            f"reference file does not have both label classes present: "
            f"found classes {sorted(classes_present)}, counts {label_counts}",
            {"label_counts": label_counts},
        )

    pairs_per_drug = ref.groupby("ingredient_concept_id").size()
    n_drugs_ge5 = int((pairs_per_drug >= MIN_PAIRS_PER_DRUG_FOR_REFERENCE).sum())
    diagnostics = {
        "n_reference_pairs": len(ref),
        "n_distinct_drugs": int(ref["ingredient_concept_id"].nunique()),
        "n_distinct_conditions": int(ref["condition_concept_id"].nunique()),
        "label_counts": label_counts,
        "n_drugs_with_ge5_pairs": n_drugs_ge5,
        "required_n_drugs_with_ge5_pairs": MIN_DRUGS_WITH_GE5_PAIRS,
    }
    if n_drugs_ge5 < MIN_DRUGS_WITH_GE5_PAIRS:
        return (
            f"only {n_drugs_ge5} drugs have >={MIN_PAIRS_PER_DRUG_FOR_REFERENCE} reference "
            f"pairs, need >={MIN_DRUGS_WITH_GE5_PAIRS} "
            f"(n_reference_pairs={diagnostics['n_reference_pairs']}, "
            f"n_distinct_drugs={diagnostics['n_distinct_drugs']}, "
            f"n_distinct_conditions={diagnostics['n_distinct_conditions']}, "
            f"label_counts={label_counts})",
            diagnostics,
        )
    return None, diagnostics


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp11")

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    ref_path = pathlib.Path("/data/reference/reference_set.csv")

    # ---- Part B step 1: preconditions, checked before anything else -----------------
    err, diag = _check_preconditions(ref_path)
    if err is not None:
        log.error("precondition check failed: %s", err)
        return {
            "precondition_failed": 1.0,
            "precondition_error_message": err,
            "__diagnostics__": diag,
        }
    log.info("preconditions passed: %s", diag)

    # This branch is not expected to be reached given the known 3-vs-30 shortfall
    # documented in reference_set_provenance.json, but is implemented per the spec in
    # case that count was wrong once the file is actually loaded (as instructed).
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    ref = pd.read_csv(ref_path)

    required_paths = {
        "train": pathlib.Path("/data/splits/train.csv"),
        "validate": pathlib.Path("/data/splits/validate.csv"),
        "ingredient_target_long": pathlib.Path("/data/drug/ingredient_target_long.csv"),
        "ingredient_features": pathlib.Path("/data/drug/ingredient_features.csv"),
        "condition_features_basic": pathlib.Path("/data/condition/condition_features_basic.csv"),
        "condition_group_long": pathlib.Path("/data/condition/condition_group_long.csv"),
        "data_dictionary": pathlib.Path("/data/data_dictionary.csv"),
    }
    missing = [str(p) for p in required_paths.values() if not p.exists()]
    if missing:
        msg = f"Missing required input(s) for model re-scoring: {missing}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    train = pd.read_csv(required_paths["train"])
    validate = pd.read_csv(required_paths["validate"])
    edges_raw = pd.read_csv(required_paths["ingredient_target_long"]).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    cond_basic = pd.read_csv(required_paths["condition_features_basic"])
    data_dict = pd.read_csv(required_paths["data_dictionary"])
    ingredient_features = pd.read_csv(required_paths["ingredient_features"])

    fallback_notes: list[str] = []

    # ---- coverage of the reference set against the project grid ---------------------
    train_val = pd.concat([train, validate], ignore_index=True)
    grid_drugs = set(train_val["ingredient_concept_id"])
    grid_conditions = set(train_val["condition_concept_id"])
    ref_in_grid = ref[
        ref["ingredient_concept_id"].isin(grid_drugs)
        & ref["condition_concept_id"].isin(grid_conditions)
    ].copy()
    n_ref_dropped_not_in_grid = len(ref) - len(ref_in_grid)
    if n_ref_dropped_not_in_grid:
        fallback_notes.append(
            f"{n_ref_dropped_not_in_grid} reference pairs dropped: drug or condition not in "
            "this project's train+validate concept grid (models cannot score a pair whose "
            "drug/condition never appeared anywhere in the dataset build)."
        )
    ref_eval = ref_in_grid.reset_index(drop=True)

    ref_in_train = ref_eval["ingredient_concept_id"].isin(set(train["ingredient_concept_id"]))
    ref_in_validate = ref_eval["ingredient_concept_id"].isin(set(validate["ingredient_concept_id"]))
    coverage_df = pd.DataFrame(
        [
            {
                "quantity": "reference_pairs_total",
                "value": len(ref),
            },
            {
                "quantity": "reference_pairs_in_project_grid",
                "value": len(ref_eval),
            },
            {
                "quantity": "reference_drugs_in_train_split",
                "value": int(ref_eval.loc[ref_in_train, "ingredient_concept_id"].nunique()),
            },
            {
                "quantity": "reference_drugs_in_validate_split",
                "value": int(ref_eval.loc[ref_in_validate, "ingredient_concept_id"].nunique()),
            },
            {
                "quantity": "reference_drugs_total",
                "value": int(ref_eval["ingredient_concept_id"].nunique()),
            },
            {
                "quantity": "reference_conditions_total",
                "value": int(ref_eval["condition_concept_id"].nunique()),
            },
        ]
    )
    coverage_df.to_csv(out / f"{exp_id}_coverage.csv", index=False)

    # ---- p_c floor computed ON THE REFERENCE SET (per spec step 2) ------------------
    p_c_ref_map = ref_eval.groupby("condition_concept_id")["label"].mean()
    ref_wide_mean = float(ref_eval["label"].mean())
    floor_score = ref_eval["condition_concept_id"].map(p_c_ref_map).fillna(ref_wide_mean).to_numpy()

    # ---- incumbent FAERS score (faers_prr, joined from train+validate) --------------
    faers_lookup = train_val.drop_duplicates(
        subset=["ingredient_concept_id", "condition_concept_id"]
    ).set_index(["ingredient_concept_id", "condition_concept_id"])["faers_prr"]
    incumbent_score = ref_eval.set_index(
        ["ingredient_concept_id", "condition_concept_id"]
    ).index.map(faers_lookup)
    incumbent_score = pd.Series(incumbent_score, index=ref_eval.index).to_numpy(dtype=float)
    n_incumbent_missing = int(np.isnan(incumbent_score).sum())
    if n_incumbent_missing:
        fallback_notes.append(
            f"{n_incumbent_missing}/{len(ref_eval)} reference pairs had no faers_prr in "
            "train+validate (drug-condition pair never observed in FAERS); left as NaN, "
            "handled by _per_drug_table's NaN-safe per-drug median fill or drug-level skip."
        )

    # ---- exp07 intrinsic union model, fit once on all of train, re-scored on ref ----
    log.info("fitting exp07 union model on all of train")
    degree_train, degree_ref, p_c_map, train_wide_mean = _build_degree_and_pc_features(
        train, ref_eval, cond_basic
    )
    drug_raw = _load_drug_features(
        required_paths["ingredient_features"], required_paths["data_dictionary"]
    )
    drug_design = _build_drug_design(drug_raw).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    cond_design = _load_condition_features(
        required_paths["condition_features_basic"], required_paths["condition_group_long"]
    )

    def _assemble_union(base_df, degree_pc_df):
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
        return pd.concat([d, merged_drug, merged_cond], axis=1)

    x_train_union = _assemble_union(train, degree_train)
    x_ref_union = _assemble_union(ref_eval, degree_ref)
    y_train = train["y_faers_signal"]

    union_model = HistGradientBoostingClassifier(**HGB_KWARGS)
    union_model.fit(x_train_union, y_train)
    union_score = union_model.predict_proba(x_ref_union)[:, 1]

    # ---- exp08 neighbour model, fit once on all of train, re-scored on ref ---------
    log.info("fitting exp08 neighbour model on all of train")
    is_complex_or_family = (
        edges_raw["component_relationship"]
        .astype(str)
        .str.upper()
        .str.contains("COMPLEX|FAMILY", regex=True, na=False)
    )
    gene_tier_edges_raw = edges_raw.loc[~is_complex_or_family]
    disease_efficacy_bool = _to_bool(edges_raw["disease_efficacy"])
    primary_gene_edges_raw = gene_tier_edges_raw.loc[
        disease_efficacy_bool.reindex(gene_tier_edges_raw.index, fill_value=False)
    ]
    gene_pairs = _dedup_pairs(primary_gene_edges_raw, "gene_symbol")

    splits_train_ing = set(train["ingredient_concept_id"].unique())
    all_ing = pd.Index(
        sorted(
            splits_train_ing
            | set(ref_eval["ingredient_concept_id"])
            | set(edges_raw["ingredient_concept_id"])
        )
    )
    n_ing = len(all_ing)
    ing_index_map = {ing: i for i, ing in enumerate(all_ing)}
    train_mask = np.array([1.0 if i in splits_train_ing else 0.0 for i in all_ing])
    d_pos_train = train["ingredient_concept_id"].map(ing_index_map).to_numpy()
    d_pos_ref = ref_eval["ingredient_concept_id"].map(ing_index_map).to_numpy()

    all_cond = pd.Index(
        sorted(set(train["condition_concept_id"]) | set(ref_eval["condition_concept_id"]))
    )
    cond_index_map = {c: i for i, c in enumerate(all_cond)}
    n_cond = len(all_cond)
    c_pos_train = train["condition_concept_id"].map(cond_index_map).to_numpy()
    c_pos_ref = ref_eval["condition_concept_id"].map(cond_index_map).to_numpy()

    p_c_by_cond = p_c_map.reindex(all_cond).fillna(train_wide_mean).to_numpy()
    p_c_train_arr = p_c_by_cond[c_pos_train]
    p_c_ref_arr = p_c_by_cond[c_pos_ref]

    import scipy.sparse as sp

    y_train_real = train["y_faers_signal"].to_numpy()
    flagged_idx = np.where(y_train_real == 1)[0]
    f_train = sp.csr_matrix(
        (np.ones(len(flagged_idx)), (d_pos_train[flagged_idx], c_pos_train[flagged_idx])),
        shape=(n_ing, n_cond),
    )
    f_train.sum_duplicates()
    f_train.data[:] = 1.0

    gene_feats = _tier_features_excess(
        "gene",
        gene_pairs,
        ing_index_map,
        n_ing,
        train_mask,
        d_pos_train,
        d_pos_ref,
        c_pos_train,
        c_pos_ref,
        f_train,
        p_c_train_arr,
        p_c_ref_arr,
        log,
    )
    has_target_annotation_by_ing = np.array(
        [1.0 if i in set(edges_raw["ingredient_concept_id"]) else 0.0 for i in all_ing]
    )
    drug_degree_train_map = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    condition_degree_train_map = train.groupby("condition_concept_id")[
        "ingredient_concept_id"
    ].nunique()
    cond_record_count_map = cond_basic.set_index("condition_concept_id")["record_count"]

    def _attach_baseline(df, p_c_arr):
        df = df.copy()
        df["drug_degree_train"] = df["ingredient_concept_id"].map(drug_degree_train_map).fillna(0)
        df["condition_degree_train"] = (
            df["condition_concept_id"].map(condition_degree_train_map).fillna(0)
        )
        df["condition_record_count"] = df["condition_concept_id"].map(cond_record_count_map)
        df["p_c"] = p_c_arr
        df["log1p_drug_degree"] = np.log1p(df["drug_degree_train"])
        df["log1p_condition_degree"] = np.log1p(df["condition_degree_train"])
        df["log1p_record_count"] = np.log1p(df["condition_record_count"])
        return df

    train_nb = _attach_baseline(train, p_c_train_arr)
    ref_nb = _attach_baseline(ref_eval, p_c_ref_arr)
    train_nb["nb_excess_gene"] = gene_feats["nb_excess_gene__train"]
    train_nb["n_neighbours_gene"] = gene_feats["n_neighbours_gene__train"]
    ref_nb["nb_excess_gene"] = gene_feats["nb_excess_gene__other"]
    ref_nb["n_neighbours_gene"] = gene_feats["n_neighbours_gene__other"]
    train_nb.loc[has_target_annotation_by_ing[d_pos_train] == 0, "nb_excess_gene"] = np.nan
    ref_nb.loc[has_target_annotation_by_ing[d_pos_ref] == 0, "nb_excess_gene"] = np.nan
    train_nb["nb_excess_gene"] = train_nb["nb_excess_gene"].fillna(0.0)
    ref_nb["nb_excess_gene"] = ref_nb["nb_excess_gene"].fillna(0.0)

    model_b_features = [
        "log1p_drug_degree",
        "log1p_condition_degree",
        "log1p_record_count",
        "p_c",
        "nb_excess_gene",
        "n_neighbours_gene",
    ]
    x_train_nb = train_nb[model_b_features].to_numpy(dtype=float)
    x_ref_nb = ref_nb[model_b_features].to_numpy(dtype=float)
    nb_model = HistGradientBoostingClassifier(**HGB_KWARGS)
    nb_model.fit(x_train_nb, y_train_real)
    neighbour_score = nb_model.predict_proba(x_ref_nb)[:, 1]

    # ---- exp09 residual model: one-way condition-demeaned log(faers_prr) -----------
    log.info("fitting exp09 residual model on all of train (case_count>=3 subset)")
    train_sub9 = train[
        (train["faers_case_count"] >= EXP09_CASE_COUNT_FLOOR) & (train["faers_prr"] > 0)
    ].copy()
    grand_mean = float(np.log(train_sub9["faers_prr"]).mean())
    train_sub9["z"] = np.log(train_sub9["faers_prr"])
    condition_offset = train_sub9.groupby("condition_concept_id")["z"].mean() - grand_mean

    drug_feat_cols9 = _select_drug_feature_columns_exp09(
        data_dict, EXP09_EXCLUDED_COLUMNS, EXP09_INCLUDED_BLOCKS
    )
    available_drug_cols9 = [c for c in drug_feat_cols9 if c in ingredient_features.columns]
    drug_features9 = ingredient_features[["omop_concept_id", *available_drug_cols9]].copy()
    drug_features9["has_chembl_match"] = (
        ingredient_features["chembl_id"].notna()
        if "chembl_id" in ingredient_features.columns
        else False
    )
    drug_features9 = drug_features9.rename(columns={"omop_concept_id": "ingredient_concept_id"})

    condition_group_long9 = pd.read_csv(required_paths["condition_group_long"])
    group_onehot9 = condition_group_long9.assign(val=1).pivot_table(
        index="condition_concept_id",
        columns=["group_source", "group_label"],
        values="val",
        aggfunc="max",
        fill_value=0,
    )
    group_onehot9.columns = [f"group__{src}__{lbl}" for src, lbl in group_onehot9.columns]
    group_onehot9 = group_onehot9.reset_index()

    drug_degree_train9 = train.groupby("ingredient_concept_id").size().rename("drug_degree")
    cond_degree_train9 = train.groupby("condition_concept_id").size().rename("condition_degree")
    drug_degree_median9 = float(drug_degree_train9.median())
    cond_degree_median9 = float(cond_degree_train9.median())
    p_c_train9 = train.groupby("condition_concept_id")["y_faers_signal"].mean().rename("p_c")
    p_c_median9 = float(p_c_train9.median())

    def _add_common_features9(df):
        df = df.merge(drug_features9, on="ingredient_concept_id", how="left")
        df = df.merge(cond_basic, on="condition_concept_id", how="left")
        df = df.merge(group_onehot9, on="condition_concept_id", how="left")
        df["drug_degree"] = np.log1p(
            df["ingredient_concept_id"].map(drug_degree_train9).fillna(drug_degree_median9)
        )
        df["condition_degree"] = np.log1p(
            df["condition_concept_id"].map(cond_degree_train9).fillna(cond_degree_median9)
        )
        if "record_count" in df.columns:
            df["record_count"] = np.log1p(df["record_count"].fillna(0.0))
        df["p_c"] = df["condition_concept_id"].map(p_c_train9).fillna(p_c_median9)
        return df

    train_sub9 = _add_common_features9(train_sub9)
    train_sub9["condition_offset"] = (
        train_sub9["condition_concept_id"].map(condition_offset).fillna(0.0)
    )
    train_sub9["r"] = train_sub9["z"] - grand_mean - train_sub9["condition_offset"]

    non_feature_cols9 = {
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
        "r",
        "condition_name",
    }
    feature_cols9 = [c for c in train_sub9.columns if c not in non_feature_cols9]
    feature_cols9 = [
        c
        for c in feature_cols9
        if pd.api.types.is_numeric_dtype(train_sub9[c]) or pd.api.types.is_bool_dtype(train_sub9[c])
    ]

    x_train9 = train_sub9[feature_cols9].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_train9 = train_sub9["r"].to_numpy()

    reg9 = HistGradientBoostingRegressor(
        **{k: v for k, v in HGB_KWARGS.items() if k != "max_leaf_nodes"}
        | {"max_leaf_nodes": HGB_KWARGS["max_leaf_nodes"]}
    )
    reg9.fit(x_train9, y_train9)

    ref_sub9 = _add_common_features9(ref_eval.copy())
    for c in feature_cols9:
        if c not in ref_sub9.columns:
            ref_sub9[c] = np.nan
    x_ref9 = ref_sub9[feature_cols9].apply(pd.to_numeric, errors="coerce").astype("float32")
    residual_score = reg9.predict(x_ref9)

    # ---- score all five models with drug_macro_auc / drug_macro_p10 + bootstrap CIs --
    ids_ref = ref_eval["ingredient_concept_id"]
    y_ref = ref_eval["label"]

    scorers = {
        "p_c_floor": floor_score,
        "incumbent_faers_prr": incumbent_score,
        "exp07_union": union_score,
        "exp08_neighbour": neighbour_score,
        "exp09_residual": residual_score,
    }

    eval_rows = []
    per_drug_frames = []
    pooled_rows = []
    for name, score in scorers.items():
        m = _drug_macro_metrics(ids_ref, y_ref, score, min_pairs=MIN_PAIRS_PER_DRUG_FOR_REFERENCE)
        _boot_mean, boot_lo, boot_hi = _bootstrap_metric_ci(m["table"], col="auc")
        pooled_auc, pooled_ap = _pooled_auc_ap(y_ref, score)
        eval_rows.append(
            {
                "model": name,
                "reference_drug_macro_auc": m["drug_macro_auc"],
                "reference_drug_macro_auc_ci_lo": boot_lo,
                "reference_drug_macro_auc_ci_hi": boot_hi,
                "reference_drug_macro_p10": m["drug_macro_p10"],
                "n_drugs_scored": m["n_drugs_scored"],
                "pooled_auc": pooled_auc,
                "pooled_ap": pooled_ap,
            }
        )
        pooled_rows.append({"model": name, "pooled_auc": pooled_auc, "pooled_ap": pooled_ap})
        if not m["table"].empty:
            t = m["table"].copy()
            t["model"] = name
            per_drug_frames.append(t)

    eval_df = pd.DataFrame(eval_rows)
    eval_path = out / f"{exp_id}_reference_eval.csv"
    eval_df.to_csv(eval_path, index=False)

    per_drug_df = (
        pd.concat(per_drug_frames, ignore_index=True) if per_drug_frames else pd.DataFrame()
    )
    per_drug_path = out / f"{exp_id}_per_drug.csv"
    per_drug_df.to_csv(per_drug_path, index=False)

    # ---- CEM-vs-reference rank correlation (spec step 5) ----------------------------
    # Rank each model's score on reference pairs against its own rank on the SAME
    # drug-condition pairs' CEM-derived p_c/label context (y_faers_signal in train+validate,
    # when the pair itself was observed there). Only pairs observed in the CEM grid have a
    # CEM rank to compare against.
    cem_lookup = train_val.drop_duplicates(
        subset=["ingredient_concept_id", "condition_concept_id"]
    ).set_index(["ingredient_concept_id", "condition_concept_id"])["y_faers_signal"]
    cem_label_for_ref = pd.Series(
        ref_eval.set_index(["ingredient_concept_id", "condition_concept_id"]).index.map(cem_lookup),
        index=ref_eval.index,
    )
    has_cem_context = cem_label_for_ref.notna()
    n_with_cem_context = int(has_cem_context.sum())

    rank_rows = []
    for name, score in scorers.items():
        if n_with_cem_context >= 3:
            rho = _spearman_rank_corr(
                pd.Series(score)[has_cem_context].to_numpy(),
                cem_label_for_ref[has_cem_context].to_numpy(),
            )
        else:
            rho = float("nan")
        # reference-set rank vs the model's own score, self-consistency reference point
        rank_rows.append(
            {
                "model": name,
                "reference_rank_vs_cem_label_spearman": rho,
                "n_pairs_with_cem_context": n_with_cem_context,
            }
        )
    rank_df = pd.DataFrame(rank_rows)
    rank_path = out / f"{exp_id}_cem_vs_reference_rank.csv"
    rank_df.to_csv(rank_path, index=False)
    if n_with_cem_context < 3:
        fallback_notes.append(
            f"Only {n_with_cem_context} reference pairs also appear as observed pairs in "
            "train+validate; CEM-vs-reference rank correlation could not be computed "
            "meaningfully (needs >=3 paired observations for Spearman)."
        )

    # ---- success_criterion.json v2 --------------------------------------------------
    import json

    floor_row = eval_df.loc[eval_df["model"] == "p_c_floor"].iloc[0]
    incumbent_row = eval_df.loc[eval_df["model"] == "incumbent_faers_prr"].iloc[0]
    union_row = eval_df.loc[eval_df["model"] == "exp07_union"].iloc[0]

    floor_val = float(floor_row["reference_drug_macro_auc"])
    # ceiling per spec: incumbent FAERS-disproportionality method's own performance
    # against the adjudicated set is the cheapest ceiling candidate available here
    # (an independent second adjudicated source was not fetched in Part A).
    ceiling_val = float(incumbent_row["reference_drug_macro_auc"])
    solved_threshold = (
        floor_val + 0.5 * (ceiling_val - floor_val) if ceiling_val == ceiling_val else float("nan")
    )

    success_criterion = {
        "version": 2,
        "source": "adjudicated",
        "floor": floor_val,
        "floor_ci": [
            float(floor_row["reference_drug_macro_auc_ci_lo"]),
            float(floor_row["reference_drug_macro_auc_ci_hi"]),
        ],
        "ceiling": ceiling_val,
        "ceiling_ci": [
            float(incumbent_row["reference_drug_macro_auc_ci_lo"]),
            float(incumbent_row["reference_drug_macro_auc_ci_hi"]),
        ],
        "ceiling_definition": "incumbent FAERS disproportionality (faers_prr) drug_macro_auc against the adjudicated reference set (an independent second adjudicated source was not available in Part A)",
        "solved_threshold": solved_threshold,
        "n_drugs": int(eval_df["n_drugs_scored"].max()) if not eval_df.empty else 0,
        "n_reference_pairs": len(ref_eval),
    }
    success_path = out / "success_criterion.json"
    success_path.write_text(json.dumps(success_criterion, indent=2))

    results.commit()

    metrics = {
        "reference_drug_macro_auc_union": float(union_row["reference_drug_macro_auc"]),
        "reference_drug_macro_auc_neighbour": float(
            eval_df.loc[eval_df["model"] == "exp08_neighbour", "reference_drug_macro_auc"].iloc[0]
        ),
        "reference_floor_drug_macro_auc": floor_val,
        "reference_pooled_auc": float(union_row["pooled_auc"]),
        "reference_pooled_ap": float(union_row["pooled_ap"]),
        "incumbent_reference_drug_macro_auc": ceiling_val,
        "ceiling": ceiling_val,
        "solved_threshold": solved_threshold,
        "n_drugs": success_criterion["n_drugs"],
        "n_reference_pairs": success_criterion["n_reference_pairs"],
    }

    findings = (
        f"Reference-set floor (p_c lookup, computed ON the reference set) = {floor_val:.4f}. "
        f"exp07 union = {union_row['reference_drug_macro_auc']:.4f}, exp08 neighbour = "
        f"{eval_df.loc[eval_df['model'] == 'exp08_neighbour', 'reference_drug_macro_auc'].iloc[0]:.4f}, "
        f"exp09 residual = {eval_df.loc[eval_df['model'] == 'exp09_residual', 'reference_drug_macro_auc'].iloc[0]:.4f}, "
        f"incumbent faers_prr = {ceiling_val:.4f}. Revised solved_threshold = {solved_threshold:.4f}. "
        f"CEM-vs-reference rank gap: see {exp_id}_cem_vs_reference_rank.csv "
        f"(n_pairs_with_cem_context={n_with_cem_context})."
    )

    return {
        "__metrics__": metrics,
        "__findings__": findings,
        "__artifacts__": [
            str(eval_path),
            str(per_drug_path),
            str(out / f"{exp_id}_coverage.csv"),
            str(success_path),
            str(rank_path),
        ],
        "__fallbacks__": fallback_notes,
    }


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp11",
        title="Adjudicated reference-set evaluation and a usable success criterion",
        hypothesis=(
            "On an externally adjudicated set with real negatives, the intrinsic union "
            "and neighbour models exceed the p_c floor computed on that set, and the "
            "ceiling lands in 0.65-0.85."
        ),
        approach=(
            "Claude Code fetched and mapped the OMOP/EU-ADR reference sets to OMOP "
            "concept ids (Part A, already done); this Modal run checks the Part B "
            "precondition (>=30 drugs with >=5 reference pairs) and, only if it passes, "
            "re-scores the existing exp07 union, exp08 neighbour and exp09 residual "
            "models (each fit once on all of train, never on the reference set) with "
            "per-drug and pooled metrics, bootstrap CIs, and emits success_criterion.json "
            "v2."
        ),
        # NOTE: the spec's own registration snippet uses label="reference_set_label",
        # which is not in bridge.labnotebook.VALID_LABELS ({"y_faers_signal",
        # "y_semmeddb_causes", "y_semmeddb_treats", "y_any_harm", "faers_prr", "other"}).
        # That snippet is schematic, not the exact API (as the calling instructions
        # anticipated) -- label="other" is used here instead, since none of the five
        # allowed CEM-label values describe evaluation against an external reference set.
        label="other",
        features=["p_c", "faers_prr", "intrinsic_union", "nb_excess_gene", "residual_model"],
        split="reference pairs, no fitting",
    )

    metrics = run.remote(exp)
    print(metrics)

    if metrics.get("precondition_failed"):
        diag = metrics.get("__diagnostics__", {})
        msg = metrics.get("precondition_error_message")
        print(f"PRECONDITION FAILED: {msg}")
        print(f"diagnostics: {diag}")
        ln.complete(
            exp,
            metrics={
                "precondition_failed": 1.0,
                "n_drugs_with_ge5_pairs": float(diag.get("n_drugs_with_ge5_pairs", 0)),
                "required_n_drugs_with_ge5_pairs": float(MIN_DRUGS_WITH_GE5_PAIRS),
            },
            findings=(
                f"Part B precondition failed before any scoring or fitting: {msg}. "
                "This is the expected, honest outcome: the OMOP/EU-ADR reference sets "
                "fetched in Part A are drug-by-HOI (health-outcome-of-interest), not a "
                "dense drug-by-condition matrix -- most drugs appear against only 1-2 of "
                "the 8 mapped conditions, so only "
                f"{diag.get('n_drugs_with_ge5_pairs', '?')} of "
                f"{diag.get('n_distinct_drugs', '?')} retained drugs have "
                f">={MIN_PAIRS_PER_DRUG_FOR_REFERENCE} reference pairs against a "
                f"required >={MIN_DRUGS_WITH_GE5_PAIRS}. This is a structural property "
                "of the source reference sets (Ryan et al. 2013, Coloma et al. 2013), "
                "not a fetch defect or a bug in this script, and no reference pairs "
                "were fabricated or the precondition loosened to force a pass. "
                "No model scoring, fitting, or success_criterion.json v2 was produced "
                "as a result -- exp06's failure mode (a ceiling estimated from 13 "
                "drugs) is not repeated here by proceeding on an underpowered set."
            ),
            failed=True,
        )
        return

    findings = metrics.pop("__findings__")
    artifacts = metrics.pop("__artifacts__")
    fallbacks = metrics.pop("__fallbacks__", [])
    metrics = metrics.get("__metrics__", metrics)
    print(findings)

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings
        + (f" Fallbacks/coverage notes: {'; '.join(fallbacks)}." if fallbacks else ""),
        artifacts=artifacts,
        next_steps=(
            "Read success_criterion.json (v2, source=adjudicated) and have downstream "
            "experiments cite its floor/ceiling/solved_threshold instead of the "
            "provisional 0.65 target in METRIC.md."
        ),
    )
