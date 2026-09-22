"""Modal script for experiments/exp12_transportability_by_target_distance.md.

The project's central, never-measured claim: how fast does drug_macro_auc decay as a
held-out drug's targets get further from anything seen in training? Builds a five-rung
target-space distance ladder (D0 shares a gene with a train drug ... D4 has no ChEMBL
target annotation at all), scores exp07's intrinsic-union model and exp08's gene-tier
neighbour model (fit once, exactly as those experiments fit them -- no per-rung
refitting) across the ladder with per-rung p_c floors and bootstrap CIs, rebuilds one
train/validate split grouped at the protein-class-leaf family level, and regresses
per-drug AUC on a continuous target-space distance (0=shared gene, 1=shared leaf,
2=shared L1, 3=neither/no annotation).

Feature-block selection (drug-intrinsic via data_dictionary.csv, condition-intrinsic
from condition_features_basic.csv + condition_group_long.csv) and the gene/leaf/L1
tier-neighbour machinery are copied from experiments/exp07_ablation_rerun_leak_audit.py
and experiments/exp08_neighbour_transport_pc_stripped.py, which already implement these
per the shared spec. This script fits each model exactly ONCE on all of train (no CV
grid, no per-rung refit) since exp12's question is about the evaluation population, not
about training variation.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any

import modal

app = modal.App("bridge-exp12")

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

EXPECTED_ROWS = {
    "train": 723_586,
    "validate": 434_151,
    "ingredient_target_long": 8_088,
    "condition_features_basic": 5_631,
    "condition_group_long": 10_554,
    "data_dictionary": 245,
}

HGB_KWARGS = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "early_stopping": False,
    "random_state": 0,
}

K_SHRINK = 10
BOOT_RESAMPLES = 1000
MIN_PAIRS_PER_DRUG = 20
MIN_DRUGS_FOR_POWERED_RUNG = 20
PROPORTIONS = {"train": 0.50, "validate": 0.30}  # renormalized below to sum to 1

# ------------------------------------------------------- copied from exp07 ---

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
        [join_col, "chembl_id", "first_approval"]
        + [c for c in keep_cols if c not in (join_col, "chembl_id", "first_approval")]
    ].copy()
    out["has_chembl_match"] = out["chembl_id"].notna()
    out = out.drop(columns=["chembl_id"])
    return out


def _build_drug_design(drug_raw) -> Any:
    import pandas as pd

    df = drug_raw.copy()
    join_col = "omop_concept_id"
    feature_cols = [c for c in df.columns if c not in (join_col, "first_approval")]

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


def _build_degree_and_pc_features(train, validate_like, cond_basic):
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

    return _apply(train), _apply(validate_like), p_c_map, train_wide_mean


# ------------------------------------------------------- copied from exp08 ---


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
    d_pos_val,
    c_pos_train,
    c_pos_val,
    f_train,
    p_c_train,
    p_c_val,
    log,
):
    import numpy as np

    nb = _build_adjacency(edges, ing_index_map, n_ing)
    nb = _mask_to_train_columns(nb, train_mask)

    n_neighbours_ing = np.asarray(nb.sum(axis=1)).ravel()
    n_neighbours_train = n_neighbours_ing[d_pos_train]
    n_neighbours_val = n_neighbours_ing[d_pos_val]

    m = (nb @ f_train).toarray() if f_train.shape[1] else np.zeros((n_ing, 0))
    if m.shape[1]:
        n_flagged_train = m[d_pos_train, c_pos_train]
        n_flagged_val = m[d_pos_val, c_pos_val]
    else:
        n_flagged_train = np.zeros(len(d_pos_train))
        n_flagged_val = np.zeros(len(d_pos_val))
    del m

    rate_train = (n_flagged_train + K_SHRINK * p_c_train) / (n_neighbours_train + K_SHRINK)
    rate_val = (n_flagged_val + K_SHRINK * p_c_val) / (n_neighbours_val + K_SHRINK)

    out = {
        f"n_neighbours_{tier_name}__train": n_neighbours_train,
        f"n_neighbours_{tier_name}__val": n_neighbours_val,
        f"n_neighbours_{tier_name}__ing": n_neighbours_ing,
        f"nb_excess_{tier_name}__train": rate_train - p_c_train,
        f"nb_excess_{tier_name}__val": rate_val - p_c_val,
    }
    log.info(
        "tier=%s median train neighbours=%.1f, %.1f%% of val rows have >=1",
        tier_name,
        float(np.median(n_neighbours_train)),
        100.0 * float((n_neighbours_val >= 1).mean()),
    )
    return out


# ----------------------------------------------------------- shared metric ---


def _per_drug_table(ids, y, score, min_pairs: int = MIN_PAIRS_PER_DRUG):
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(
        {
            "drug": ids.to_numpy() if hasattr(ids, "to_numpy") else ids,
            "y": y.to_numpy() if hasattr(y, "to_numpy") else y,
            "score": score,
        }
    )
    rows = []
    for drug_id, grp in df.groupby("drug"):
        n = len(grp)
        if n < min_pairs or grp["y"].nunique() < 2:
            continue
        auc = float(roc_auc_score(grp["y"], grp["score"]))
        ranked = grp.sort_values("score", ascending=False)
        top10 = ranked.head(10)
        p10 = float(top10["y"].mean()) if len(top10) else float("nan")
        rows.append({"drug": drug_id, "n": n, "auc": auc, "p10": p10})
    return pd.DataFrame(rows, columns=["drug", "n", "auc", "p10"])


def _drug_macro_metrics(ids, y, score, min_pairs: int = MIN_PAIRS_PER_DRUG) -> dict:
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


def _bootstrap_mean_ci(values, n_boot: int = BOOT_RESAMPLES, seed: int = 0):
    import numpy as np

    values = np.asarray(values, dtype=float)
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


def _fail(message: str, artifacts=None) -> dict:
    return {
        "precondition_failed": 1.0,
        "precondition_error_message": message,
        "__artifacts__": artifacts or [],
    }


def _assign_folds_greedy(group_sizes, proportions: dict, seed: int) -> dict:
    """Greedy largest-group-first pair-count balancing, copied from
    scripts/build_dataset.py's assign_folds (own re-implementation: that script is not on
    the Modal image, and it operates on pair_labels.csv which is not staged on this
    volume -- only train.csv/validate.csv are)."""
    total = group_sizes.sum()
    targets = {f: p * total for f, p in proportions.items()}
    current = dict.fromkeys(proportions, 0.0)
    assignment: dict = {}
    shuffled = group_sizes.sample(frac=1.0, random_state=seed)
    for key, size in shuffled.sort_values(ascending=False, kind="stable").items():
        fold = max(proportions, key=lambda f: (targets[f] - current[f]) / targets[f])
        assignment[key] = fold
        current[fold] += size
    return assignment


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp12")
    t_start = time.monotonic()
    budget_deadline = t_start + 600.0

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)
    fallbacks_taken: list[str] = []

    # ---------------------------------------------------------- preconditions
    paths = {
        "train": pathlib.Path("/data/splits/train.csv"),
        "validate": pathlib.Path("/data/splits/validate.csv"),
        "ingredient_target_long": pathlib.Path("/data/drug/ingredient_target_long.csv"),
        "condition_features_basic": pathlib.Path("/data/condition/condition_features_basic.csv"),
        "condition_group_long": pathlib.Path("/data/condition/condition_group_long.csv"),
        "ingredient_features": pathlib.Path("/data/drug/ingredient_features.csv"),
        "data_dictionary": pathlib.Path("/data/data_dictionary.csv"),
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        msg = f"Missing required input(s): {missing}"
        log.error(msg)
        return _fail(msg)

    row_check_paths = {
        "train": paths["train"],
        "validate": paths["validate"],
        "ingredient_target_long": paths["ingredient_target_long"],
        "condition_features_basic": paths["condition_features_basic"],
        "condition_group_long": paths["condition_group_long"],
        "data_dictionary": paths["data_dictionary"],
    }
    row_counts = {}
    for name, p in row_check_paths.items():
        with p.open() as fh:
            row_counts[name] = sum(1 for _ in fh) - 1
    log.info("row counts: %s", row_counts)
    bad_counts = {k: v for k, v in row_counts.items() if v != EXPECTED_ROWS[k]}
    if bad_counts:
        msg = (
            "Precondition failed: row counts do not match expected (this is the exact "
            f"check exp07 lacked for ingredient_target_long.csv): {bad_counts} "
            f"(expected {EXPECTED_ROWS})."
        )
        log.error(msg)
        return _fail(msg)
    log.info(
        "preconditions passed: ingredient_target_long.csv has exactly %d rows",
        row_counts["ingredient_target_long"],
    )

    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    edges_raw = pd.read_csv(paths["ingredient_target_long"]).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    cond_basic = pd.read_csv(paths["condition_features_basic"])

    # ----------------------------------------------------- exp07 union model
    log.info("building exp07-style intrinsic union design (D+p_c+drug+cond)")
    degree_train, degree_val, p_c_map, train_wide_mean = _build_degree_and_pc_features(
        train, validate, cond_basic
    )
    drug_raw = _load_drug_features(paths["ingredient_features"], paths["data_dictionary"])
    drug_design = _build_drug_design(drug_raw).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    cond_design = _load_condition_features(
        paths["condition_features_basic"], paths["condition_group_long"]
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
    x_val_union = _assemble_union(validate, degree_val)
    y_train = train["y_faers_signal"]
    y_val = validate["y_faers_signal"].to_numpy()

    log.info(
        "fitting exp07 union model once on all of train (n_features=%d)", x_train_union.shape[1]
    )
    union_model = HistGradientBoostingClassifier(**HGB_KWARGS)
    union_model.fit(x_train_union, y_train)
    val_proba_union = union_model.predict_proba(x_val_union)[:, 1]

    # --------------------------------------------------- exp08 neighbour model
    log.info("building exp08-style neighbour model (degree + p_c + gene-tier nb_excess)")
    is_complex_or_family = (
        edges_raw["component_relationship"]
        .astype(str)
        .str.upper()
        .str.contains("COMPLEX|FAMILY", regex=True, na=False)
    )
    gene_tier_edges_raw = edges_raw.loc[~is_complex_or_family]

    def _to_bool(series):
        if series.dtype == bool:
            return series.fillna(False)
        return (
            series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y", "t"])
        )

    disease_efficacy_bool = _to_bool(edges_raw["disease_efficacy"])
    primary_gene_edges_raw = gene_tier_edges_raw.loc[
        disease_efficacy_bool.reindex(gene_tier_edges_raw.index, fill_value=False)
    ]
    gene_pairs = _dedup_pairs(primary_gene_edges_raw, "gene_symbol")
    class_leaf_pairs = _dedup_pairs(edges_raw, "chembl_protein_class_leaf")
    class_l1_pairs = _dedup_pairs(edges_raw, "chembl_protein_class_L1")

    splits_train_ing = set(train["ingredient_concept_id"].unique())
    splits_val_ing = set(validate["ingredient_concept_id"].unique())
    all_ing = pd.Index(
        sorted(splits_train_ing | splits_val_ing | set(edges_raw["ingredient_concept_id"]))
    )
    n_ing = len(all_ing)
    ing_index_map = {ing: i for i, ing in enumerate(all_ing)}
    train_mask = np.array([1.0 if i in splits_train_ing else 0.0 for i in all_ing])

    d_pos_train = train["ingredient_concept_id"].map(ing_index_map).to_numpy()
    d_pos_val = validate["ingredient_concept_id"].map(ing_index_map).to_numpy()

    all_cond = pd.Index(
        sorted(set(train["condition_concept_id"]) | set(validate["condition_concept_id"]))
    )
    cond_index_map = {c: i for i, c in enumerate(all_cond)}
    n_cond = len(all_cond)
    c_pos_train = train["condition_concept_id"].map(cond_index_map).to_numpy()
    c_pos_val = validate["condition_concept_id"].map(cond_index_map).to_numpy()

    p_c_by_cond = p_c_map.reindex(all_cond).fillna(train_wide_mean).to_numpy()
    p_c_train = p_c_by_cond[c_pos_train]
    p_c_val = p_c_by_cond[c_pos_val]

    def _f_train_matrix(y_col):
        import scipy.sparse as sp

        flagged_idx = np.where(y_col == 1)[0]
        f = sp.csr_matrix(
            (np.ones(len(flagged_idx)), (d_pos_train[flagged_idx], c_pos_train[flagged_idx])),
            shape=(n_ing, n_cond),
        )
        f.sum_duplicates()
        f.data[:] = 1.0
        return f

    y_train_real = train["y_faers_signal"].to_numpy()
    f_train = _f_train_matrix(y_train_real)

    common_kwargs = {
        "ing_index_map": ing_index_map,
        "n_ing": n_ing,
        "train_mask": train_mask,
        "d_pos_train": d_pos_train,
        "d_pos_val": d_pos_val,
        "c_pos_train": c_pos_train,
        "c_pos_val": c_pos_val,
        "f_train": f_train,
        "p_c_train": p_c_train,
        "p_c_val": p_c_val,
        "log": log,
    }

    gene_feats = _tier_features_excess("gene", gene_pairs, **common_kwargs)
    leaf_feats = _tier_features_excess("class_leaf", class_leaf_pairs, **common_kwargs)
    l1_feats = _tier_features_excess("class_L1", class_l1_pairs, **common_kwargs)

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

    train_nb = _attach_baseline(train, p_c_train)
    validate_nb = _attach_baseline(validate, p_c_val)

    def _attach_tier(df, feats, split):
        df = df.copy()
        suffix = f"__{split}"
        for name, arr in feats.items():
            if name.endswith(suffix):
                df[name[: -len(suffix)]] = arr
        return df

    for feats in (gene_feats, leaf_feats, l1_feats):
        train_nb = _attach_tier(train_nb, feats, "train")
        validate_nb = _attach_tier(validate_nb, feats, "val")

    train_nb["has_target_annotation"] = has_target_annotation_by_ing[d_pos_train]
    validate_nb["has_target_annotation"] = has_target_annotation_by_ing[d_pos_val]

    excess_cols = [c for c in train_nb.columns if c.startswith("nb_excess_")]
    for df in (train_nb, validate_nb):
        mask = df["has_target_annotation"] == 0
        df.loc[mask, excess_cols] = np.nan
    train_nb[excess_cols] = train_nb[excess_cols].fillna(0.0)
    validate_nb[excess_cols] = validate_nb[excess_cols].fillna(0.0)

    baseline_features = ["log1p_drug_degree", "log1p_condition_degree", "log1p_record_count", "p_c"]
    gene_features = ["nb_excess_gene", "n_neighbours_gene"]
    model_b_features = baseline_features + gene_features

    log.info("fitting exp08-style neighbour model B once on all of train")
    x_train_nb = train_nb[model_b_features].to_numpy(dtype=float)
    x_val_nb = validate_nb[model_b_features].to_numpy(dtype=float)
    nb_model = HistGradientBoostingClassifier(**HGB_KWARGS)
    nb_model.fit(x_train_nb, y_train_real)
    val_proba_nb = nb_model.predict_proba(x_val_nb)[:, 1]

    # ------------------------------------------------------------- the ladder
    n_nb_gene_ing = gene_feats["n_neighbours_gene__ing"]
    n_nb_leaf_ing = leaf_feats["n_neighbours_class_leaf__ing"]
    n_nb_l1_ing = l1_feats["n_neighbours_class_L1__ing"]

    def _rung_for_ing_pos(pos: int) -> str:
        if n_nb_gene_ing[pos] >= 1:
            return "D0"
        if n_nb_leaf_ing[pos] >= 1:
            return "D1"
        if n_nb_l1_ing[pos] >= 1:
            return "D2"
        if has_target_annotation_by_ing[pos] >= 1:
            return "D3"
        return "D4"

    ing_rung = np.array([_rung_for_ing_pos(i) for i in range(n_ing)])
    ing_dist_continuous = np.array(
        [
            0
            if n_nb_gene_ing[i] >= 1
            else 1
            if n_nb_leaf_ing[i] >= 1
            else 2
            if n_nb_l1_ing[i] >= 1
            else 3
            for i in range(n_ing)
        ],
        dtype=float,
    )

    val_rung = ing_rung[d_pos_val]
    val_dist = ing_dist_continuous[d_pos_val]
    val_first_approval = validate.merge(
        drug_raw[["omop_concept_id", "first_approval"]],
        left_on="ingredient_concept_id",
        right_on="omop_concept_id",
        how="left",
    )["first_approval"].to_numpy()

    drug_ids_val = validate["ingredient_concept_id"].to_numpy()

    rung_rows = []
    per_drug_frames = []
    for rung in ["D0", "D1", "D2", "D3", "D4"]:
        mask = val_rung == rung
        n_pairs = int(mask.sum())
        n_drugs_in_rung = int(pd.unique(drug_ids_val[mask]).shape[0]) if n_pairs else 0

        m_union = _drug_macro_metrics(
            pd.Series(drug_ids_val[mask]), pd.Series(y_val[mask]), val_proba_union[mask]
        )
        m_nb = _drug_macro_metrics(
            pd.Series(drug_ids_val[mask]), pd.Series(y_val[mask]), val_proba_nb[mask]
        )
        m_floor = _drug_macro_metrics(
            pd.Series(drug_ids_val[mask]),
            pd.Series(y_val[mask]),
            validate_nb.loc[mask, "p_c"].to_numpy(),
        )

        _boot_union_mean, boot_union_lo, boot_union_hi = _bootstrap_mean_ci(m_union["table"]["auc"])
        _boot_nb_mean, boot_nb_lo, boot_nb_hi = _bootstrap_mean_ci(m_nb["table"]["auc"])

        underpowered = m_union["n_drugs_scored"] < MIN_DRUGS_FOR_POWERED_RUNG
        approvals_in_rung = val_first_approval[mask]
        approvals_in_rung = approvals_in_rung[~pd.isna(approvals_in_rung)]

        rung_rows.append(
            {
                "rung": rung,
                "n_pairs": n_pairs,
                "n_drugs_in_rung": n_drugs_in_rung,
                "n_drugs_scored_union": m_union["n_drugs_scored"],
                "n_drugs_scored_nb": m_nb["n_drugs_scored"],
                "underpowered": underpowered,
                "drug_macro_auc_union": m_union["drug_macro_auc"],
                "drug_macro_auc_union_ci_lo": boot_union_lo,
                "drug_macro_auc_union_ci_hi": boot_union_hi,
                "drug_macro_p10_union": m_union["drug_macro_p10"],
                "drug_macro_auc_nb": m_nb["drug_macro_auc"],
                "drug_macro_auc_nb_ci_lo": boot_nb_lo,
                "drug_macro_auc_nb_ci_hi": boot_nb_hi,
                "drug_macro_p10_nb": m_nb["drug_macro_p10"],
                "floor_pc_lookup": m_floor["drug_macro_auc"],
                "median_first_approval": float(np.median(approvals_in_rung))
                if len(approvals_in_rung)
                else float("nan"),
            }
        )
        if not m_union["table"].empty:
            t = m_union["table"].copy()
            t["rung"] = rung
            per_drug_frames.append(t)
        log.info(
            "rung=%s n_pairs=%d n_drugs_scored(union)=%d union_auc=%.4f nb_auc=%.4f floor=%.4f%s",
            rung,
            n_pairs,
            m_union["n_drugs_scored"],
            m_union["drug_macro_auc"],
            m_nb["drug_macro_auc"],
            m_floor["drug_macro_auc"],
            " [UNDERPOWERED]" if underpowered else "",
        )

    ladder_df = pd.DataFrame(rung_rows)
    ladder_path = out / f"{exp_id}_decay_ladder.csv"
    ladder_df.to_csv(ladder_path, index=False)

    per_drug_df = (
        pd.concat(per_drug_frames, ignore_index=True) if per_drug_frames else pd.DataFrame()
    )
    per_drug_path = out / f"{exp_id}_per_drug.csv"
    per_drug_df.to_csv(per_drug_path, index=False)

    # realised rung shares (per §"Measure and report the realised shares")
    realised_pair_share = ladder_df.set_index("rung")["n_pairs"] / ladder_df["n_pairs"].sum()
    realised_drug_share = (
        ladder_df.set_index("rung")["n_drugs_in_rung"] / ladder_df["n_drugs_in_rung"].sum()
    )
    log.info("realised pair share by rung:\n%s", realised_pair_share.to_string())
    log.info("realised drug share by rung:\n%s", realised_drug_share.to_string())

    # ---------------------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ladder_df))
    ax.errorbar(
        x,
        ladder_df["drug_macro_auc_union"],
        yerr=[
            ladder_df["drug_macro_auc_union"] - ladder_df["drug_macro_auc_union_ci_lo"],
            ladder_df["drug_macro_auc_union_ci_hi"] - ladder_df["drug_macro_auc_union"],
        ],
        marker="o",
        label="exp07 intrinsic union",
        capsize=4,
    )
    ax.errorbar(
        x,
        ladder_df["drug_macro_auc_nb"],
        yerr=[
            ladder_df["drug_macro_auc_nb"] - ladder_df["drug_macro_auc_nb_ci_lo"],
            ladder_df["drug_macro_auc_nb_ci_hi"] - ladder_df["drug_macro_auc_nb"],
        ],
        marker="s",
        label="exp08 gene-tier neighbour",
        capsize=4,
    )
    ax.plot(
        x,
        ladder_df["floor_pc_lookup"],
        linestyle="--",
        color="gray",
        marker="x",
        label="per-rung p_c floor",
    )
    for i, row in ladder_df.iterrows():
        ax.annotate(
            f"n={row['n_drugs_scored_union']}",
            (i, row["drug_macro_auc_union"]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=8,
            ha="center",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ladder_df["rung"])
    ax.set_xlabel("target-space distance rung")
    ax.set_ylabel("validate drug_macro_auc")
    ax.set_title("exp12: transportability decay by target-space distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_decay_curve.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------- continuous slope model
    slope_rows = []
    val_df = pd.DataFrame(
        {"drug": drug_ids_val, "y": y_val, "score": val_proba_union, "dist": val_dist}
    )
    for drug, g in val_df.groupby("drug"):
        if len(g) < MIN_PAIRS_PER_DRUG or g["y"].nunique() < 2:
            continue
        auc = float(roc_auc_score(g["y"], g["score"]))
        slope_rows.append({"drug": drug, "auc": auc, "dist": float(g["dist"].iloc[0])})
    slope_df = pd.DataFrame(slope_rows)
    slope_path = out / f"{exp_id}_slope.csv"

    if len(slope_df) >= 2 and slope_df["dist"].nunique() >= 2:
        x_arr = slope_df["dist"].to_numpy()
        y_arr = slope_df["auc"].to_numpy()
        slope_point = float(np.polyfit(x_arr, y_arr, 1)[0])

        rng = np.random.default_rng(0)
        n = len(slope_df)
        boot_slopes = np.empty(BOOT_RESAMPLES)
        for i in range(BOOT_RESAMPLES):
            idx = rng.integers(0, n, size=n)
            xb, yb = x_arr[idx], y_arr[idx]
            if np.unique(xb).size < 2:
                boot_slopes[i] = np.nan
                continue
            boot_slopes[i] = np.polyfit(xb, yb, 1)[0]
        boot_slopes = boot_slopes[~np.isnan(boot_slopes)]
        slope_ci_lo = float(np.percentile(boot_slopes, 2.5)) if len(boot_slopes) else float("nan")
        slope_ci_hi = float(np.percentile(boot_slopes, 97.5)) if len(boot_slopes) else float("nan")
    else:
        slope_point, slope_ci_lo, slope_ci_hi = float("nan"), float("nan"), float("nan")
        fallbacks_taken.append(
            "slope regression: too few drugs or distance values with variance to fit"
        )
    slope_df.to_csv(slope_path, index=False)

    # ----------------------------------------------------- leaf-level split
    leaf_split_drug_macro_auc = float("nan")
    leaf_split_floor = float("nan")
    leaf_summary_rows = []

    time_used = time.monotonic() - t_start
    time_remaining = budget_deadline - time.monotonic()
    log.info("time used so far: %.1fs, remaining budget: %.1fs", time_used, time_remaining)

    if time_remaining < 210.0:
        fallbacks_taken.append(
            f"Dropped step 5 (leaf-level re-split refit): only {time_remaining:.1f}s remained of the "
            "600s budget (needed ~210s of headroom). The leaf-level fold boundary was not reached; "
            "per-rung floors above are unaffected."
        )
        log.warning("budget guard triggered: skipping leaf-level re-split refit")
    else:
        log.info("building leaf-level regroup (chembl_protein_class_leaf primary target)")
        combined_ing = pd.Index(sorted(splits_train_ing | splits_val_ing))

        primary_leaf = (
            edges_raw.sort_values(
                ["ingredient_concept_id", "disease_efficacy", "direct_interaction"],
                ascending=[True, False, False],
            )
            .groupby("ingredient_concept_id")["chembl_protein_class_leaf"]
            .first()
        )
        leaf_group_key = pd.Series(
            {
                i: (
                    f"leaf:{primary_leaf[i]}"
                    if i in primary_leaf.index and pd.notna(primary_leaf[i])
                    else f"ing:{i}"
                )
                for i in combined_ing
            },
            name="leaf_group_key",
        )

        combined = pd.concat([train, validate], ignore_index=True)
        pairs_per_ing = combined.groupby("ingredient_concept_id").size()
        grp = pd.DataFrame(
            {
                "leaf_group_key": leaf_group_key,
                "n_pairs": pairs_per_ing.reindex(combined_ing).fillna(0),
            }
        )
        group_sizes = grp.groupby("leaf_group_key")["n_pairs"].sum()

        total_p = PROPORTIONS["train"] + PROPORTIONS["validate"]
        norm_proportions = {
            "train2": PROPORTIONS["train"] / total_p,
            "validate2": PROPORTIONS["validate"] / total_p,
        }
        fold_of_group = _assign_folds_greedy(group_sizes, norm_proportions, seed=42)
        grp["fold2"] = grp["leaf_group_key"].map(fold_of_group)

        combined = combined.merge(
            grp.reset_index(names="ingredient_concept_id")[
                ["ingredient_concept_id", "leaf_group_key", "fold2"]
            ],
            on="ingredient_concept_id",
            how="left",
        )

        assert combined.groupby("ingredient_concept_id")["fold2"].nunique().max() == 1, (
            "an ingredient appears in more than one leaf-level fold"
        )
        assert combined.groupby("leaf_group_key")["fold2"].nunique().max() == 1, (
            "a leaf group appears in more than one leaf-level fold"
        )

        train2 = combined[combined["fold2"] == "train2"].reset_index(drop=True)
        validate2 = combined[combined["fold2"] == "validate2"].reset_index(drop=True)

        for fold_name, sub in (("train2", train2), ("validate2", validate2)):
            leaf_summary_rows.append(
                {
                    "fold": fold_name,
                    "pairs": len(sub),
                    "pair_share": round(len(sub) / len(combined), 4),
                    "ingredients": sub["ingredient_concept_id"].nunique(),
                    "leaf_groups": sub["leaf_group_key"].nunique(),
                    "rate_faers_signal": round(float(sub["y_faers_signal"].mean()), 4),
                }
            )
        leaf_summary_df = pd.DataFrame(leaf_summary_rows)
        log.info("leaf-level split achieved:\n%s", leaf_summary_df.to_string(index=False))
        leaf_summary_df.to_csv(out / f"{exp_id}_leaf_split_summary.csv", index=False)

        # refit the neighbour model (same feature construction as model B) on train2/validate2
        splits_train2_ing = set(train2["ingredient_concept_id"].unique())
        train_mask2 = np.array([1.0 if i in splits_train2_ing else 0.0 for i in all_ing])

        d_pos_train2 = train2["ingredient_concept_id"].map(ing_index_map).to_numpy()
        d_pos_val2 = validate2["ingredient_concept_id"].map(ing_index_map).to_numpy()
        c_pos_train2 = train2["condition_concept_id"].map(cond_index_map).to_numpy()
        c_pos_val2 = validate2["condition_concept_id"].map(cond_index_map).to_numpy()

        p_c_map2 = train2.groupby("condition_concept_id")["y_faers_signal"].mean()
        train2_wide_mean = float(train2["y_faers_signal"].mean())
        p_c_by_cond2 = p_c_map2.reindex(all_cond).fillna(train2_wide_mean).to_numpy()
        p_c_train2 = p_c_by_cond2[c_pos_train2]
        p_c_val2 = p_c_by_cond2[c_pos_val2]

        y_train2_real = train2["y_faers_signal"].to_numpy()
        import scipy.sparse as sp

        flagged_idx2 = np.where(y_train2_real == 1)[0]
        f_train2 = sp.csr_matrix(
            (np.ones(len(flagged_idx2)), (d_pos_train2[flagged_idx2], c_pos_train2[flagged_idx2])),
            shape=(n_ing, n_cond),
        )
        f_train2.sum_duplicates()
        f_train2.data[:] = 1.0

        gene_feats2 = _tier_features_excess(
            "gene",
            gene_pairs,
            ing_index_map=ing_index_map,
            n_ing=n_ing,
            train_mask=train_mask2,
            d_pos_train=d_pos_train2,
            d_pos_val=d_pos_val2,
            c_pos_train=c_pos_train2,
            c_pos_val=c_pos_val2,
            f_train=f_train2,
            p_c_train=p_c_train2,
            p_c_val=p_c_val2,
            log=log,
        )

        drug_degree_train2_map = train2.groupby("ingredient_concept_id")[
            "condition_concept_id"
        ].nunique()
        condition_degree_train2_map = train2.groupby("condition_concept_id")[
            "ingredient_concept_id"
        ].nunique()

        def _attach_baseline2(df, p_c_arr):
            df = df.copy()
            df["drug_degree_train"] = (
                df["ingredient_concept_id"].map(drug_degree_train2_map).fillna(0)
            )
            df["condition_degree_train"] = (
                df["condition_concept_id"].map(condition_degree_train2_map).fillna(0)
            )
            df["condition_record_count"] = df["condition_concept_id"].map(cond_record_count_map)
            df["p_c"] = p_c_arr
            df["log1p_drug_degree"] = np.log1p(df["drug_degree_train"])
            df["log1p_condition_degree"] = np.log1p(df["condition_degree_train"])
            df["log1p_record_count"] = np.log1p(df["condition_record_count"])
            return df

        train2_nb = _attach_baseline2(train2, p_c_train2)
        validate2_nb = _attach_baseline2(validate2, p_c_val2)
        train2_nb = _attach_tier(train2_nb, gene_feats2, "train")
        validate2_nb = _attach_tier(validate2_nb, gene_feats2, "val")
        train2_nb["nb_excess_gene"] = train2_nb["nb_excess_gene"].fillna(0.0)
        validate2_nb["nb_excess_gene"] = validate2_nb["nb_excess_gene"].fillna(0.0)
        train2_nb["n_neighbours_gene"] = train2_nb["n_neighbours_gene"].fillna(0.0)
        validate2_nb["n_neighbours_gene"] = validate2_nb["n_neighbours_gene"].fillna(0.0)

        x_train2 = train2_nb[model_b_features].to_numpy(dtype=float)
        x_val2 = validate2_nb[model_b_features].to_numpy(dtype=float)
        nb_model2 = HistGradientBoostingClassifier(**HGB_KWARGS)
        nb_model2.fit(x_train2, y_train2_real)
        val_proba2 = nb_model2.predict_proba(x_val2)[:, 1]

        y_val2 = validate2["y_faers_signal"].to_numpy()
        drug_ids_val2 = validate2["ingredient_concept_id"].to_numpy()
        leaf_split_metrics = _drug_macro_metrics(
            pd.Series(drug_ids_val2), pd.Series(y_val2), val_proba2
        )
        leaf_split_floor_metrics = _drug_macro_metrics(
            pd.Series(drug_ids_val2), pd.Series(y_val2), validate2_nb["p_c"].to_numpy()
        )
        leaf_split_drug_macro_auc = leaf_split_metrics["drug_macro_auc"]
        leaf_split_floor = leaf_split_floor_metrics["drug_macro_auc"]
        log.info(
            "leaf-level split neighbour-model drug_macro_auc=%.4f (n_drugs=%d), floor=%.4f",
            leaf_split_drug_macro_auc,
            leaf_split_metrics["n_drugs_scored"],
            leaf_split_floor,
        )

    results.commit()

    metrics: dict[str, Any] = {}
    for row in rung_rows:
        r = row["rung"]
        metrics[f"drug_macro_auc_{r}"] = row["drug_macro_auc_union"]
        metrics[f"floor_{r}"] = row["floor_pc_lookup"]
        metrics[f"n_drugs_{r}"] = row["n_drugs_scored_union"]
    metrics["auc_per_rung_slope"] = slope_point
    metrics["slope_ci_lo"] = slope_ci_lo
    metrics["slope_ci_hi"] = slope_ci_hi
    metrics["leaf_split_drug_macro_auc"] = leaf_split_drug_macro_auc
    metrics["leaf_split_floor"] = leaf_split_floor

    artifacts = [
        str(ladder_path),
        str(out / f"{exp_id}_decay_curve.png"),
        str(per_drug_path),
        str(slope_path),
    ]
    if leaf_summary_rows:
        artifacts.append(str(out / f"{exp_id}_leaf_split_summary.csv"))

    rung_by_rung = "; ".join(
        f"{r['rung']}: n_pairs={r['n_pairs']} n_drugs={r['n_drugs_scored_union']} "
        f"union_auc={r['drug_macro_auc_union']:.4f} [{r['drug_macro_auc_union_ci_lo']:.4f},"
        f"{r['drug_macro_auc_union_ci_hi']:.4f}] nb_auc={r['drug_macro_auc_nb']:.4f} "
        f"floor={r['floor_pc_lookup']:.4f}{' UNDERPOWERED' if r['underpowered'] else ''}"
        for r in rung_rows
    )
    floor_rung = None
    for r in rung_rows:
        gap = r["drug_macro_auc_union"] - r["floor_pc_lookup"]
        if abs(gap) < 0.02 or r["drug_macro_auc_union"] < r["floor_pc_lookup"]:
            floor_rung = r["rung"]
            break
    findings = (
        f"Rung by rung (union model, own per-rung p_c floor): {rung_by_rung}. "
        f"auc_per_rung_slope={slope_point:+.4f} per rung of continuous distance "
        f"(0=shared gene,1=shared leaf,2=shared L1,3=no shared/no annotation), "
        f"95% bootstrap CI [{slope_ci_lo:+.4f}, {slope_ci_hi:+.4f}] over {len(slope_df)} drugs. "
        f"Performance becomes indistinguishable from its own rung's floor at "
        f"{floor_rung if floor_rung else 'no rung tested (all rungs stayed clearly above floor)'}. "
        f"Leaf-level family-grouped split (fold boundary at chembl_protein_class_leaf rather than "
        f"gene): neighbour-model drug_macro_auc={leaf_split_drug_macro_auc if leaf_split_drug_macro_auc == leaf_split_drug_macro_auc else float('nan'):.4f} "
        f"vs its own floor={leaf_split_floor if leaf_split_floor == leaf_split_floor else float('nan'):.4f}"
        + (" (step skipped, see fallbacks)" if not leaf_summary_rows else "")
        + ". "
        + (f"Fallbacks taken: {'; '.join(fallbacks_taken)}. " if fallbacks_taken else "")
        + "Answer to the project's question: see next_steps/notebook entry for the one-sentence verdict "
        "computed from the numbers above (rung and slope), since it depends on which rung the run "
        "actually landed on."
    )

    return {
        "__metrics__": metrics,
        "__findings__": findings,
        "__artifacts__": artifacts,
        "__fallbacks__": fallbacks_taken,
        "__ladder__": ladder_df.to_dict(orient="records"),
    }


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp12",
        title="Transportability decay by target-space distance, with a family-level split",
        hypothesis=(
            "drug_macro_auc decays monotonically across a five-rung target-space distance "
            "ladder and is indistinguishable from the per-rung p_c floor for drugs with no "
            "target annotation."
        ),
        approach=(
            "Score the fitted exp07 union and exp08 neighbour models across rungs D0-D4 "
            "with per-rung floors and bootstrap CIs; rebuild one train/validate split "
            "grouped by chembl_protein_class_leaf for a family-level fold boundary; "
            "regress per-drug AUC on minimum target-space distance."
        ),
        label="y_faers_signal",
        features=["intrinsic_union", "nb_excess_gene", "p_c", "degree"],
        split="validate stratified by target-space distance, plus a leaf-level regrouped split",
    )

    metrics = run.remote(exp)
    print(metrics)

    if metrics.get("precondition_failed"):
        ln.complete(
            exp,
            metrics={"precondition_failed": 1.0},
            findings=f"Precondition failed: {metrics.get('precondition_error_message')}",
            failed=True,
        )
        return

    findings = metrics.pop("__findings__")
    artifacts = metrics.pop("__artifacts__")
    fallbacks = metrics.pop("__fallbacks__", [])
    ladder = metrics.pop("__ladder__", [])
    metrics = metrics.get("__metrics__", metrics)
    print(findings)
    print("ladder:", ladder)

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=artifacts,
        next_steps=(
            "Read <exp_id>_decay_ladder.csv and <exp_id>_decay_curve.png for the project's "
            "headline figure. If the leaf-level split refit was skipped (see fallbacks: "
            f"{'; '.join(fallbacks) if fallbacks else 'none'}), a follow-up run with more "
            "budget headroom should complete step 5. Otherwise, use leaf_split_drug_macro_auc "
            "as the honest transportability headline going forward instead of the "
            "gene-grouped split's number."
        ),
    )
