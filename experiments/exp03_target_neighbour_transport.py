"""Modal script for experiments/exp03_target_neighbour_transport.md.

Does target similarity transport across the primary-target-gene split boundary?
Three neighbour-similarity tiers (gene, protein-class leaf, protein-class L1),
train-only leave-one-drug-out shrunk neighbour rates, HistGBM models A/B/C,
and the neighbour-count decay curve that is the round's central figure.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import modal

app = modal.App("bridge-exp03")  # stable name, no random suffix

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
}

# Coverage table from the spec: fraction of validate PAIRS with >=1 / >=3 train
# neighbours, and fraction of validate INGREDIENTS with zero neighbours, per tier.
# Measured on the raw (unfiltered) ingredient_target_long edges, before the
# subunit-expansion exclusion this script applies for modeling features.
EXPECTED_COVERAGE = {
    "gene_symbol": {"ge1": 0.336, "ge3": 0.225, "zero_ing": 0.819},
    "chembl_protein_class_leaf": {"ge1": 0.555, "ge3": 0.527, "zero_ing": 0.707},
    "chembl_protein_class_L1": {"ge1": 0.753, "ge3": 0.747, "zero_ing": 0.626},
    "target_chembl_id": {"ge1": 0.258, "ge3": 0.137, "zero_ing": 0.849},
}
COVERAGE_TOLERANCE = 0.005

HGB_KWARGS = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "early_stopping": False,
    "random_state": 0,
}


def _build_adjacency(pairs, ing_index_map: dict, n_ing: int):
    """Binary ingredient-by-ingredient 'shares >=1 tier value' adjacency.

    `pairs` is a DataFrame with columns ['ingredient_concept_id', 'value']
    (already deduplicated per ingredient x value). Self-loops are zeroed, so
    this matrix already encodes leave-one-out: an ingredient is never its own
    neighbour. Vectorized via one sparse matrix multiply, no per-row loop.
    """
    import numpy as np
    import scipy.sparse as sp

    ing_pos = pairs["ingredient_concept_id"].map(ing_index_map).to_numpy()
    val_codes, _ = pandas_factorize(pairs["value"])
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


def pandas_factorize(series):
    import pandas as pd

    return pd.factorize(series)


def _mask_to_train_columns(nb, train_mask):
    import numpy as np

    return nb.multiply(np.asarray(train_mask, dtype=np.float64).reshape(1, -1)).tocsr()


def _tier_features(
    tier_name: str,
    edges,
    ing_index_map: dict,
    n_ing: int,
    train_mask,
    d_pos_train,
    d_pos_val,
    c_pos_train,
    c_pos_val,
    f_train,
    p_c_train,
    p_c_val,
    k_values: list[int],
    log: logging.Logger,
):
    """Build n_neighbours_t / n_neighbours_flagged_t / nb_rate_t(k) for train + validate."""
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

    out = {
        f"n_neighbours_{tier_name}__train": n_neighbours_train,
        f"n_neighbours_{tier_name}__val": n_neighbours_val,
    }
    for k in k_values:
        rate_train = (n_flagged_train + k * p_c_train) / (n_neighbours_train + k)
        rate_val = (n_flagged_val + k * p_c_val) / (n_neighbours_val + k)
        suffix = "" if k == 10 else f"_k{k}"
        out[f"nb_rate_{tier_name}{suffix}__train"] = rate_train
        out[f"nb_rate_{tier_name}{suffix}__val"] = rate_val
        out[f"nb_rate_{tier_name}{suffix}_minus_p_c__train"] = rate_train - p_c_train
        out[f"nb_rate_{tier_name}{suffix}_minus_p_c__val"] = rate_val - p_c_val
    log.info(
        "tier=%s median train neighbours=%.1f, %.1f%% of train rows have >=1",
        tier_name,
        float(np.median(n_neighbours_train)),
        100.0 * float((n_neighbours_train >= 1).mean()),
    )
    return out


def _coverage_row(
    tier_col: str,
    edges,
    splits_train_ing: set,
    splits_val_ing: set,
    train_pair_lookup: dict,
    log: logging.Logger,
) -> dict:
    """Reproduce one row of the spec's coverage table on RAW (unfiltered) edges."""
    import numpy as np

    edges = edges.dropna(subset=[tier_col])
    train_edges = edges[edges["ingredient_concept_id"].isin(splits_train_ing)]
    val_edges = edges[edges["ingredient_concept_id"].isin(splits_val_ing)]

    # value -> set of train ingredients carrying it
    train_val_to_ings: dict = {}
    for ing, val in zip(train_edges["ingredient_concept_id"], train_edges[tier_col], strict=True):
        train_val_to_ings.setdefault(val, set()).add(ing)

    val_ing_to_vals: dict = {}
    for ing, val in zip(val_edges["ingredient_concept_id"], val_edges[tier_col], strict=True):
        val_ing_to_vals.setdefault(ing, set()).add(val)

    n_neighbours_by_ing = {}
    for ing, vals in val_ing_to_vals.items():
        neighbours = set()
        for v in vals:
            neighbours |= train_val_to_ings.get(v, set())
        neighbours.discard(ing)
        n_neighbours_by_ing[ing] = len(neighbours)

    all_val_ings = splits_val_ing
    counts = np.array([n_neighbours_by_ing.get(i, 0) for i in all_val_ings])
    zero_ing_frac = float((counts == 0).mean()) if len(counts) else float("nan")

    # pair-level: iterate validate PAIRS via the lookup of (ingredient) -> neighbour count
    # a pair's neighbour count only depends on the ingredient, not the condition, for
    # this coverage measurement (matches the feature definition below).
    pair_counts = np.array(
        [n_neighbours_by_ing.get(i, 0) for i in train_pair_lookup["val_ing_per_row"]]
    )
    ge1 = float((pair_counts >= 1).mean()) if len(pair_counts) else float("nan")
    ge3 = float((pair_counts >= 3).mean()) if len(pair_counts) else float("nan")
    return {"tier": tier_col, "ge1": ge1, "ge3": ge3, "zero_ing": zero_ing_frac}


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
    import scipy.sparse as sp
    from sklearn.calibration import calibration_curve
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp03")

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    fallbacks_taken: list[str] = []

    # ---------------------------------------------------------------- step 1
    # Precondition check: paths exist, row counts as expected.
    paths = {
        "train": pathlib.Path("/data/splits/train.csv"),
        "validate": pathlib.Path("/data/splits/validate.csv"),
        "ingredient_target_long": pathlib.Path("/data/drug/ingredient_target_long.csv"),
        "condition_features_basic": pathlib.Path("/data/condition/condition_features_basic.csv"),
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        msg = f"Missing required input(s): {missing}"
        log.error(msg)
        return _fail(exp_id, msg)

    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    edges_raw = pd.read_csv(paths["ingredient_target_long"])
    cond_feats = pd.read_csv(paths["condition_features_basic"])

    row_counts = {
        "train": len(train),
        "validate": len(validate),
        "ingredient_target_long": len(edges_raw),
        "condition_features_basic": len(cond_feats),
    }
    log.info("row counts: %s", row_counts)
    bad_counts = {k: v for k, v in row_counts.items() if v != EXPECTED_ROWS[k]}
    if bad_counts:
        msg = (
            f"Row counts do not match spec's expected inputs: {bad_counts} "
            f"(expected {EXPECTED_ROWS}). Splits or target table may have been rebuilt."
        )
        log.error(msg)
        return _fail(exp_id, msg)

    edges_raw = edges_raw.rename(columns={"omop_concept_id": "ingredient_concept_id"})

    # -------------------------------------------------------- coverage check
    splits_train_ing = set(train["ingredient_concept_id"].unique())
    splits_val_ing = set(validate["ingredient_concept_id"].unique())
    lookup = {"val_ing_per_row": validate["ingredient_concept_id"].to_numpy()}

    coverage_rows = []
    for tier_col in [
        "gene_symbol",
        "chembl_protein_class_leaf",
        "chembl_protein_class_L1",
        "target_chembl_id",
    ]:
        coverage_rows.append(
            _coverage_row(tier_col, edges_raw, splits_train_ing, splits_val_ing, lookup, log)
        )
    coverage_df = pd.DataFrame(coverage_rows)
    log.info("reproduced coverage table:\n%s", coverage_df.to_string(index=False))

    tier_col_to_key = {
        "gene_symbol": "gene_symbol",
        "chembl_protein_class_leaf": "chembl_protein_class_leaf",
        "chembl_protein_class_L1": "chembl_protein_class_L1",
        "target_chembl_id": "target_chembl_id",
    }
    coverage_bad = []
    for _, row in coverage_df.iterrows():
        expected = EXPECTED_COVERAGE[tier_col_to_key[row["tier"]]]
        for metric in ("ge1", "ge3", "zero_ing"):
            if abs(row[metric] - expected[metric]) > COVERAGE_TOLERANCE:
                coverage_bad.append(
                    f"{row['tier']}.{metric}: got {row[metric]:.4f}, expected "
                    f"{expected[metric]:.4f} (tol {COVERAGE_TOLERANCE})"
                )
    coverage_df["coverage_ge1_column_expected"] = coverage_df["tier"].map(
        lambda t: EXPECTED_COVERAGE[tier_col_to_key[t]]["ge1"]
    )
    coverage_df.to_csv(out / f"{exp_id}_coverage.csv", index=False)

    if coverage_bad:
        msg = (
            "Coverage table did not reproduce within tolerance; splits or "
            "ingredient_target_long appear to have changed:\n" + "\n".join(coverage_bad)
        )
        log.error(msg)
        return _fail(exp_id, msg, artifacts=[str(out / f"{exp_id}_coverage.csv")])

    # ------------------------------------------------------- step 3 traps note
    # Detect subunit-expansion rows (family/complex targets) from the ACTUAL
    # distinct values of component_relationship, then exclude them from the
    # gene tier only (per spec traps section).
    distinct_component_rel = sorted(edges_raw["component_relationship"].dropna().unique().tolist())
    log.info("distinct component_relationship values: %s", distinct_component_rel)
    is_complex_or_family = (
        edges_raw["component_relationship"]
        .astype(str)
        .str.upper()
        .str.contains("COMPLEX|FAMILY", regex=True, na=False)
    )
    edges_raw["has_family_or_complex_target"] = is_complex_or_family
    n_excluded_rows = int(is_complex_or_family.sum())
    n_excluded_ingredients = int(
        edges_raw.loc[is_complex_or_family, "ingredient_concept_id"].nunique()
    )
    log.info(
        "gene-tier subunit exclusion: excluding %d/%d edge rows (%d ingredients) "
        "flagged via component_relationship values %s",
        n_excluded_rows,
        len(edges_raw),
        n_excluded_ingredients,
        [v for v in distinct_component_rel if "COMPLEX" in v.upper() or "FAMILY" in v.upper()],
    )
    gene_tier_edges_raw = edges_raw.loc[~is_complex_or_family]

    # disease_efficacy -> boolean, used for the "primary gene" refinement below.
    distinct_disease_efficacy = sorted(
        edges_raw["disease_efficacy"].dropna().astype(str).unique().tolist()
    )
    log.info("distinct disease_efficacy raw values: %s", distinct_disease_efficacy)

    def _to_bool(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series.fillna(False)
        return (
            series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y", "t"])
        )

    disease_efficacy_bool = _to_bool(edges_raw["disease_efficacy"])
    action_type_notna = edges_raw["action_type"].notna()

    # ---------------------------------------------------------------- step 2
    # Shared ingredient universe and positional index across train + validate +
    # anything with a target-table row, so all tier matrices share one axis.
    all_ing = pd.Index(
        sorted(
            set(train["ingredient_concept_id"])
            | set(validate["ingredient_concept_id"])
            | set(edges_raw["ingredient_concept_id"])
        )
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

    p_c_series = train.groupby("condition_concept_id")["y_faers_signal"].mean()
    p_c_by_cond = p_c_series.reindex(all_cond).to_numpy()  # NaN where cond unseen in train
    p_c_train = p_c_by_cond[c_pos_train]
    p_c_val = p_c_by_cond[c_pos_val]

    flagged_train = train.loc[train["y_faers_signal"] == 1]
    f_train = sp.csr_matrix(
        (
            np.ones(len(flagged_train)),
            (
                flagged_train["ingredient_concept_id"].map(ing_index_map).to_numpy(),
                flagged_train["condition_concept_id"].map(cond_index_map).to_numpy(),
            ),
        ),
        shape=(n_ing, n_cond),
    )
    f_train.sum_duplicates()
    f_train.data[:] = 1.0

    has_target_annotation_by_ing = np.array(
        [1.0 if i in set(edges_raw["ingredient_concept_id"]) else 0.0 for i in all_ing]
    )
    has_target_annotation_train = has_target_annotation_by_ing[d_pos_train]
    has_target_annotation_val = has_target_annotation_by_ing[d_pos_val]

    k_values = [10, 5, 20]
    budget_row_estimate = n_ing * n_cond
    if budget_row_estimate > 5e8:  # generous guard, unlikely to trigger at these sizes
        k_values = [10]
        fallbacks_taken.append(
            "dropped k in {5,20} sensitivity: ingredient x condition matrix too large for budget"
        )
        log.warning("dropping k-sensitivity pass: n_ing*n_cond=%d", budget_row_estimate)

    def _dedup_pairs(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        p = df[["ingredient_concept_id", value_col]].dropna().rename(columns={value_col: "value"})
        p["value"] = p["value"].astype(str)
        return p.drop_duplicates()

    # gene tier (subunit-excluded)
    gene_pairs = _dedup_pairs(gene_tier_edges_raw, "gene_symbol")
    # primary-gene refinement: gene-tier edges where ChEMBL marks the target as the
    # one believed to drive efficacy (disease_efficacy == True). This is the
    # dataset's own notion of "primary" target, not the split's grouping variable
    # (using group_key would be circular/degenerate: split groups partition
    # entirely into train xor validate, so a validate row would always show 0
    # group_key-neighbours by construction).
    primary_gene_edges = gene_tier_edges_raw.loc[
        disease_efficacy_bool.reindex(gene_tier_edges_raw.index, fill_value=False)
    ]
    primary_gene_pairs = _dedup_pairs(primary_gene_edges, "gene_symbol")
    log.info(
        "primary-gene refinement: %d/%d gene-tier edges kept (disease_efficacy=True), "
        "%d distinct ingredients",
        len(primary_gene_edges),
        len(gene_tier_edges_raw),
        primary_gene_edges["ingredient_concept_id"].nunique(),
    )

    # action-type-matched refinement: composite (gene, action_type) value so that
    # sharing the value means both ingredients act via the same action_type at the
    # same gene.
    action_edges = gene_tier_edges_raw.loc[
        action_type_notna.reindex(gene_tier_edges_raw.index, fill_value=False)
    ].copy()
    action_edges["gene_action"] = (
        action_edges["gene_symbol"].astype(str) + "||" + action_edges["action_type"].astype(str)
    )
    action_pairs = _dedup_pairs(action_edges, "gene_action")
    log.info(
        "action-type refinement: %d/%d gene-tier edges have a non-null action_type",
        len(action_edges),
        len(gene_tier_edges_raw),
    )

    class_leaf_pairs = _dedup_pairs(edges_raw, "chembl_protein_class_leaf")
    class_l1_pairs = _dedup_pairs(edges_raw, "chembl_protein_class_L1")

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
        "k_values": k_values,
        "log": log,
    }

    gene_feats = _tier_features("gene", gene_pairs, **common_kwargs)
    primary_gene_feats = _tier_features("primary_gene", primary_gene_pairs, **common_kwargs)
    action_feats = _tier_features("action_matched", action_pairs, **common_kwargs)
    leaf_feats = _tier_features("class_leaf", class_leaf_pairs, **common_kwargs)
    l1_feats = _tier_features("class_L1", class_l1_pairs, **common_kwargs)

    # ---------------------------------------------------------------- step 2b
    # Degree features (exp01's three), train-only, applied to validate.
    drug_degree_train_map = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    condition_degree_train_map = train.groupby("condition_concept_id")[
        "ingredient_concept_id"
    ].nunique()
    cond_record_count_map = cond_feats.set_index("condition_concept_id")["record_count"]

    def _attach_degree(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["drug_degree_train"] = df["ingredient_concept_id"].map(drug_degree_train_map).fillna(0)
        df["condition_degree_train"] = (
            df["condition_concept_id"].map(condition_degree_train_map).fillna(0)
        )
        df["condition_record_count"] = df["condition_concept_id"].map(cond_record_count_map)
        return df

    train = _attach_degree(train)
    validate = _attach_degree(validate)

    def _attach_tier(df: pd.DataFrame, feats: dict, split: str) -> pd.DataFrame:
        df = df.copy()
        suffix = f"__{split}"
        for name, arr in feats.items():
            if name.endswith(suffix):
                df[name[: -len(suffix)]] = arr
        return df

    train = _attach_tier(train, gene_feats, "train")
    validate = _attach_tier(validate, gene_feats, "val")
    train = _attach_tier(train, primary_gene_feats, "train")
    validate = _attach_tier(validate, primary_gene_feats, "val")
    train = _attach_tier(train, action_feats, "train")
    validate = _attach_tier(validate, action_feats, "val")
    train = _attach_tier(train, leaf_feats, "train")
    validate = _attach_tier(validate, leaf_feats, "val")
    train = _attach_tier(train, l1_feats, "train")
    validate = _attach_tier(validate, l1_feats, "val")

    train["has_target_annotation"] = has_target_annotation_train
    validate["has_target_annotation"] = has_target_annotation_val

    # Features undefined for drugs with no target annotation at all -> NaN
    # (HistGBM handles missing values natively; this is deliberate, not a bug).
    nb_rate_cols = [c for c in train.columns if c.startswith("nb_rate_")]
    for df in (train, validate):
        mask = df["has_target_annotation"] == 0
        df.loc[mask, nb_rate_cols] = np.nan

    # ---------------------------------------------------------------- step 3
    degree_features = ["drug_degree_train", "condition_degree_train", "condition_record_count"]
    gene_tier_features = [
        "n_neighbours_gene",
        "nb_rate_gene",
        "nb_rate_gene_minus_p_c",
        "nb_rate_primary_gene",
        "nb_rate_primary_gene_minus_p_c",
        "nb_rate_action_matched",
        "nb_rate_action_matched_minus_p_c",
        "has_target_annotation",
    ]
    class_tier_features = [
        "nb_rate_class_leaf",
        "nb_rate_class_leaf_minus_p_c",
        "nb_rate_class_L1",
        "nb_rate_class_L1_minus_p_c",
    ]

    model_features = {
        "A": degree_features,
        "B": degree_features + gene_tier_features,
        "C": degree_features + gene_tier_features + class_tier_features,
    }

    x_train_full = {m: train[cols].to_numpy(dtype=float) for m, cols in model_features.items()}
    x_val_full = {m: validate[cols].to_numpy(dtype=float) for m, cols in model_features.items()}
    y_train = train["y_faers_signal"].to_numpy()
    y_val = validate["y_faers_signal"].to_numpy()
    groups = train["group_key"].to_numpy()

    gene_ge1_val = validate["n_neighbours_gene"].fillna(0).to_numpy() >= 1
    coverage_ge1_nb = float(gene_ge1_val.mean())

    gkf = GroupKFold(n_splits=3)
    cv_ap = {}
    val_preds = {}
    for m in ("A", "B", "C"):
        fold_aps = []
        for tr_idx, te_idx in gkf.split(x_train_full[m], y_train, groups):
            clf = HistGradientBoostingClassifier(**HGB_KWARGS)
            clf.fit(x_train_full[m][tr_idx], y_train[tr_idx])
            p = clf.predict_proba(x_train_full[m][te_idx])[:, 1]
            fold_aps.append(average_precision_score(y_train[te_idx], p))
        cv_ap[m] = (float(np.mean(fold_aps)), float(np.std(fold_aps)))
        log.info("model %s train CV AP: mean=%.4f std=%.4f (%s)", m, *cv_ap[m], fold_aps)

        clf_full = HistGradientBoostingClassifier(**HGB_KWARGS)
        clf_full.fit(x_train_full[m], y_train)
        val_preds[m] = clf_full.predict_proba(x_val_full[m])[:, 1]

    def _ap_auc(y_true, y_score) -> tuple[float, float]:
        if len(np.unique(y_true)) < 2:
            return float("nan"), float("nan")
        return average_precision_score(y_true, y_score), roc_auc_score(y_true, y_score)

    models_rows = []
    for m in ("A", "B", "C"):
        ap_u, auc_u = _ap_auc(y_val, val_preds[m])
        ap_c, auc_c = _ap_auc(y_val[gene_ge1_val], val_preds[m][gene_ge1_val])
        cv_mean, cv_std = cv_ap[m]
        models_rows.append(
            {
                "model": m,
                "features": ",".join(model_features[m]),
                "train_cv_ap_mean": cv_mean,
                "train_cv_ap_std": cv_std,
                "validate_ap_unconditional": ap_u,
                "validate_roc_auc_unconditional": auc_u,
                "validate_prevalence_unconditional": float(y_val.mean()),
                "validate_ap_conditional_ge1_gene_nb": ap_c,
                "validate_roc_auc_conditional_ge1_gene_nb": auc_c,
                "validate_prevalence_conditional_ge1_gene_nb": float(y_val[gene_ge1_val].mean()),
                "n_validate_conditional": int(gene_ge1_val.sum()),
            }
        )
    models_df = pd.DataFrame(models_rows)
    models_df.to_csv(out / f"{exp_id}_models.csv", index=False)

    # ---------------------------------------------------------------- step 5
    bucket_edges = [(-0.5, 0.5, "0"), (0.5, 2.5, "1_2"), (2.5, 5.5, "3_5"), (5.5, np.inf, "6plus")]
    n_nb_gene_val = validate["n_neighbours_gene"].fillna(0).to_numpy()
    decay_rows = []
    ap_by_bucket = {"A": {}, "B": {}}
    for lo, hi, label in bucket_edges:
        bucket_mask = (n_nb_gene_val > lo) & (n_nb_gene_val <= hi)
        n_pairs = int(bucket_mask.sum())
        row = {"bucket": label, "n_pairs": n_pairs}
        for m in ("A", "B"):
            if n_pairs > 0 and len(np.unique(y_val[bucket_mask])) > 1:
                ap = average_precision_score(y_val[bucket_mask], val_preds[m][bucket_mask])
            else:
                ap = float("nan")
            row[f"ap_model_{m}"] = ap
            ap_by_bucket[m][label] = ap
        decay_rows.append(row)
    decay_df = pd.DataFrame(decay_rows)
    decay_df.to_csv(out / f"{exp_id}_decay_curve.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(decay_df))
    width = 0.35
    ax.bar(x - width / 2, decay_df["ap_model_A"], width, label="A: degree only")
    ax.bar(x + width / 2, decay_df["ap_model_B"], width, label="B: degree + gene tier")
    for i, row in decay_df.iterrows():
        ax.text(
            i,
            max(row["ap_model_A"], row["ap_model_B"], 0) + 0.01,
            f"n={row['n_pairs']}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(decay_df["bucket"])
    ax.set_xlabel("n_neighbours_gene bucket")
    ax.set_ylabel("Validate average precision")
    ax.set_title("exp03: AP by gene-tier neighbour count (A vs B)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_decay_curve.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- step 6
    # "only tier available" is judged on the tier's own neighbour COUNT, not
    # just whether nb_rate is non-null (nb_rate is defined even at count 0).
    n_nb_leaf_arr = (
        validate["n_neighbours_class_leaf"].fillna(0).to_numpy()
        if "n_neighbours_class_leaf" in validate.columns
        else np.zeros(len(validate))
    )
    n_nb_l1_arr = (
        validate["n_neighbours_class_L1"].fillna(0).to_numpy()
        if "n_neighbours_class_L1" in validate.columns
        else np.zeros(len(validate))
    )

    only_leaf_mask = (n_nb_gene_val == 0) & (n_nb_leaf_arr >= 1)
    only_l1_mask = (n_nb_gene_val == 0) & (n_nb_leaf_arr == 0) & (n_nb_l1_arr >= 1)

    tier_decay_rows = []
    for name, mask in (("only_class_leaf", only_leaf_mask), ("only_class_L1", only_l1_mask)):
        n_pairs = int(mask.sum())
        if n_pairs > 0 and len(np.unique(y_val[mask])) > 1:
            ap = average_precision_score(y_val[mask], val_preds["B"][mask])
            auc = roc_auc_score(y_val[mask], val_preds["B"][mask])
        else:
            ap, auc = float("nan"), float("nan")
        tier_decay_rows.append(
            {
                "subset": name,
                "n_pairs": n_pairs,
                "model_B_ap": ap,
                "model_B_roc_auc": auc,
                "prevalence": float(y_val[mask].mean()) if n_pairs else float("nan"),
            }
        )
    tier_decay_df = pd.DataFrame(tier_decay_rows)
    tier_decay_df.to_csv(out / f"{exp_id}_tier_decay.csv", index=False)

    # ---------------------------------------------------------- calibration
    cal_mask = gene_ge1_val
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    if cal_mask.sum() > 0 and len(np.unique(y_val[cal_mask])) > 1:
        frac_pos, mean_pred = calibration_curve(
            y_val[cal_mask], val_preds["B"][cal_mask], n_bins=10
        )
        ax2.plot(mean_pred, frac_pos, marker="o", label="model B, >=1 gene neighbour")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax2.set_xlabel("mean predicted probability")
    ax2.set_ylabel("observed frequency")
    ax2.set_title("exp03: model B calibration (validate, >=1 gene neighbour)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out / f"{exp_id}_calibration.png", dpi=150)
    plt.close(fig2)

    # ---------------------------------------------------------------- sanity
    b_row = models_df.loc[models_df["model"] == "B"].iloc[0]
    train_cv_b = b_row["train_cv_ap_mean"]
    val_cond_b = b_row["validate_ap_conditional_ge1_gene_nb"]
    leakage_gap = train_cv_b - val_cond_b
    log.info(
        "leakage sanity check: model B train CV AP=%.4f vs validate (>=1 nb) AP=%.4f, gap=%.4f",
        train_cv_b,
        val_cond_b,
        leakage_gap,
    )
    if leakage_gap > 0.15:
        fallbacks_taken.append(
            f"WARNING: model B train CV AP exceeds validate conditional AP by {leakage_gap:.3f}, "
            "larger than expected -- inspect leave-one-drug-out for a leak"
        )

    a_row = models_df.loc[models_df["model"] == "A"].iloc[0]
    metrics = {
        "validate_average_precision": float(val_cond_b),
        "validate_average_precision_unconditional": float(b_row["validate_ap_unconditional"]),
        "train_cv_average_precision": float(train_cv_b),
        "baseline_degree_only_ap": float(a_row["validate_ap_unconditional"]),
        "ap_by_nb_bucket_0": float(ap_by_bucket["B"].get("0", float("nan"))),
        "ap_by_nb_bucket_1_2": float(ap_by_bucket["B"].get("1_2", float("nan"))),
        "ap_by_nb_bucket_3_5": float(ap_by_bucket["B"].get("3_5", float("nan"))),
        "ap_by_nb_bucket_6plus": float(ap_by_bucket["B"].get("6plus", float("nan"))),
        "coverage_ge1_nb": coverage_ge1_nb,
    }

    decay_shape = ", ".join(
        f"{r['bucket']}: A={r['ap_model_A']:.3f} B={r['ap_model_B']:.3f} (n={r['n_pairs']})"
        for _, r in decay_df.iterrows()
    )
    leaf_ap = tier_decay_df.loc[tier_decay_df["subset"] == "only_class_leaf", "model_B_ap"].iloc[0]
    l1_ap = tier_decay_df.loc[tier_decay_df["subset"] == "only_class_L1", "model_B_ap"].iloc[0]
    gain_b_over_a_cond = val_cond_b - a_row["validate_ap_conditional_ge1_gene_nb"]

    findings = (
        f"+{gain_b_over_a_cond:.3f} AP (model B vs degree-only A) on the "
        f"{coverage_ge1_nb * 100:.1f}% of validate pairs with >=1 gene-level train "
        f"neighbour (conditional AP: A={a_row['validate_ap_conditional_ge1_gene_nb']:.4f}, "
        f"B={val_cond_b:.4f}); unconditional AP is diluted by the "
        f"{(1 - coverage_ge1_nb) * 100:.1f}% of pairs with no gene neighbour "
        f"(A={a_row['validate_ap_unconditional']:.4f}, B={b_row['validate_ap_unconditional']:.4f}). "
        f"Decay curve by n_neighbours_gene bucket: {decay_shape}. "
        f"Tier decay: model B AP on pairs where only class_leaf similarity is "
        f"available = {leaf_ap:.4f}; only class_L1 available = {l1_ap:.4f} "
        f"(vs conditional gene-tier AP {val_cond_b:.4f}) -- "
        + (
            "gain shrinks as the tier loosens from gene to class-level, consistent with "
            "target-specific transport rather than a broad class effect."
            if (
                not np.isnan(leaf_ap)
                and not np.isnan(l1_ap)
                and leaf_ap <= val_cond_b
                and l1_ap <= leaf_ap
            )
            else "gain does not monotonically shrink as the tier loosens; inspect "
            "whether this reflects a class-level effect rather than target-specific "
            "pharmacology per the falsification criterion in the spec."
        )
        + f" Leave-one-drug-out sanity check: train CV AP ({train_cv_b:.4f}) vs validate "
        f"conditional AP ({val_cond_b:.4f}), gap={leakage_gap:.4f} "
        f"({'no sign of self-leakage' if leakage_gap <= 0.15 else 'gap larger than expected, see fallbacks/notes'})."
        + (f" Fallbacks taken: {'; '.join(fallbacks_taken)}." if fallbacks_taken else "")
        + (
            " Design note: the gene-tier 'primary gene' refinement uses ChEMBL's own "
            "disease_efficacy flag to mean 'primary target', not the split's group_key "
            "(group_key-based sharing would always be 0 on validate by construction of "
            "the grouped split, which would be a degenerate, not a meaningful, feature)."
        )
        + " Coverage table reproduced on unfiltered edges per the spec's stated "
        "measurement method; the gene-tier subunit-expansion exclusion (component_"
        "relationship COMPLEX/FAMILY rows) is applied only to the modeling features, "
        "not to the coverage reproduction, since the spec describes the coverage "
        "numbers as measured before that fix."
    )

    artifacts = [
        str(out / f"{exp_id}_models.csv"),
        str(out / f"{exp_id}_decay_curve.csv"),
        str(out / f"{exp_id}_decay_curve.png"),
        str(out / f"{exp_id}_tier_decay.csv"),
        str(out / f"{exp_id}_coverage.csv"),
        str(out / f"{exp_id}_calibration.png"),
    ]

    results.commit()
    log.info("metrics: %s", metrics)
    return {"__metrics__": metrics, "__findings__": findings, "__artifacts__": artifacts}


def _fail(exp_id: str, message: str, artifacts: list[str] | None = None) -> dict[str, Any]:
    return {
        "__metrics__": {"failed": 1.0},
        "__findings__": message,
        "__artifacts__": artifacts or [],
        "__failed__": True,
    }


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp03",
        title="Same-target neighbour transport across the primary-gene group boundary",
        hypothesis=(
            "A train-only shrunk neighbour flag-rate beats the degree floor on validate "
            "pairs with >=1 neighbour, and the gain decays as similarity loosens from "
            "shared gene to shared protein class."
        ),
        approach=(
            "Three similarity tiers (gene, protein-class leaf, protein-class L1), "
            "empirical-Bayes shrinkage k=10 toward the condition's train rate, "
            "leave-one-drug-out on train rows, HistGBM, AP by neighbour-count bucket."
        ),
        label="y_faers_signal",
        features=[
            "degree",
            "nb_rate_gene",
            "nb_rate_class_leaf",
            "nb_rate_class_L1",
            "nb_rate_primary_gene",
            "nb_rate_action_matched",
        ],
        split="train/validate, grouped by primary target gene",
    )

    result = run.remote(exp)

    if result.get("__failed__"):
        ln.complete(
            exp,
            metrics=result["__metrics__"],
            findings=result["__findings__"],
            artifacts=result.get("__artifacts__", []),
            failed=True,
        )
        print("FAILED:", result["__findings__"])
        return

    metrics = result["__metrics__"]
    findings = result["__findings__"]
    artifacts = result["__artifacts__"]
    print(metrics)
    print(findings)

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=artifacts,
        next_steps=(
            "If the gene-tier gain holds and decays as class-level tiers loosen, "
            "proceed to Round 2 pathway-overlap features restricted to target-specific "
            "biology. If gain is flat/increasing with looser tiers, redirect toward "
            "class-level priors instead."
        ),
    )
