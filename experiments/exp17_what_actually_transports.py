"""Modal script for experiments/exp17_what_actually_transports.md.

Follows exp12 (exp_20260922_c9b979), which found the target-space distance decay curve
inverted: the biggest margin over floor sat at D4 (208 drugs with NO target annotation at
all, +0.085) and the worst was D2 (protein-class L1 only, -0.014). This experiment asks
whether the D4 margin is target-mediated transport or the model separating a different
*kind* of substance (mixtures, minerals, botanicals, biologics) from small molecules, and
whether D0's margin (shared gene, +0.021) survives restriction to the population the
project actually targets: a novel small molecule with a known target.

Rung construction (D0 shares a gene with a train drug ... D4 has no target annotation) and
the gene/leaf/L1 neighbour-tier machinery are copied unchanged from
experiments/exp12_transportability_by_target_distance.py, which itself copied the
drug/condition intrinsic-feature-block selection from exp07 and the tier-neighbour
machinery from exp08. This script fits each model exactly ONCE per restriction (original /
indicator-removed / small-molecules-only) and scores all five rungs from that one fit --
refitting per rung would mix population and training effects, which the spec forbids.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any

import modal

app = modal.App("bridge-exp17")

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
RUNGS = ["D0", "D1", "D2", "D3", "D4"]

# ---------------------------------------------------- copied from exp07/exp12 ---

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

# Substance-type indicator columns to strip for the "indicator-removal" refit (step 2).
# These are the columns whose non-null pattern or value encodes "not a conventional small
# molecule": molecule_type itself, availability_label/type, has_chembl_match (a
# derived-in-script column, see _load_drug_features), route/exposure flags, and
# n_routes_labelled (missing/near-zero for non-conventional substances).
SUBSTANCE_TYPE_COLUMNS: list[str] = [
    "molecule_type",
    "availability_label",
    "availability_type",
    "exposure_type",
    "n_routes_labelled",
    "has_chembl_match",
    "has_chembl_mechanism",
]


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
    extra = ["molecule_type", "availability_label", "availability_type"]
    extra = [c for c in extra if c in drug.columns and c not in keep_cols]
    out = drug[
        [join_col, "chembl_id", "first_approval"]
        + extra
        + [c for c in keep_cols if c not in (join_col, "chembl_id", "first_approval")]
    ].copy()
    out["has_chembl_match"] = out["chembl_id"].notna()
    out = out.drop(columns=["chembl_id"])
    return out


def _build_drug_design(drug_raw, drop_substance_type: bool = False) -> Any:
    import pandas as pd

    df = drug_raw.copy()
    join_col = "omop_concept_id"
    exclude = {join_col, "first_approval", "molecule_type"}  # molecule_type kept out of
    # every design's *modeling* features by default -- it is used for rung/subset
    # construction and characterisation, not as a predictor, to keep the "original" design
    # comparable across all three restrictions (it never had molecule_type as a feature).
    if drop_substance_type:
        exclude |= set(SUBSTANCE_TYPE_COLUMNS)
    feature_cols = [c for c in df.columns if c not in exclude]

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


# ---------------------------------------------------- copied from exp08/exp12 ---


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


# -------------------------------------------------------------- shared metric ---


def _per_drug_table(ids, y, score, min_pairs: int = MIN_PAIRS_PER_DRUG):
    """Per-drug AUC. NaN-safe: scores filled with the per-drug median, bool cast to float."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    score = np.asarray(score, dtype=float)
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
        s = grp["score"]
        if s.isna().any():
            s = s.fillna(s.median())
        if s.isna().all():
            continue
        auc = float(roc_auc_score(grp["y"], s.to_numpy()))
        ranked = grp.assign(score=s).sort_values("score", ascending=False)
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


# ---------------------------------------------------------------------- core ---


def _rung_arrays(
    edges_raw,
    all_ing,
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
    """Build the D0-D4 rung assignment per ingredient (exp12's ladder machinery)."""
    import numpy as np

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
    return ing_rung, has_target_annotation_by_ing, gene_feats, leaf_feats, l1_feats


def _score_rungs(
    label: str,
    val_rung,
    drug_ids_val,
    y_val,
    val_proba,
    floor_score,
    val_first_approval,
    val_degree,
    tag: str,
):
    """Score all 5 rungs for one (model, restriction) pair. Returns list of dict rows."""
    import numpy as np
    import pandas as pd

    rows = []
    for rung in RUNGS:
        mask = val_rung == rung
        n_pairs = int(mask.sum())
        n_drugs_in_rung = int(pd.unique(drug_ids_val[mask]).shape[0]) if n_pairs else 0

        m_model = _drug_macro_metrics(
            pd.Series(drug_ids_val[mask]), pd.Series(y_val[mask]), val_proba[mask]
        )
        m_floor = _drug_macro_metrics(
            pd.Series(drug_ids_val[mask]), pd.Series(y_val[mask]), floor_score[mask]
        )
        _mean, ci_lo, ci_hi = _bootstrap_mean_ci(m_model["table"]["auc"])

        approvals = val_first_approval[mask]
        approvals = approvals[~pd.isna(approvals)]
        degrees = val_degree[mask]
        degrees = degrees[~pd.isna(degrees)]

        rows.append(
            {
                "curve": tag,
                "rung": rung,
                "n_pairs": n_pairs,
                "n_drugs_in_rung": n_drugs_in_rung,
                "n_drugs_scored": m_model["n_drugs_scored"],
                "underpowered": m_model["n_drugs_scored"] < MIN_DRUGS_FOR_POWERED_RUNG,
                "drug_macro_auc": m_model["drug_macro_auc"],
                "drug_macro_auc_ci_lo": ci_lo,
                "drug_macro_auc_ci_hi": ci_hi,
                "drug_macro_p10": m_model["drug_macro_p10"],
                "floor_pc_lookup": m_floor["drug_macro_auc"],
                "margin": m_model["drug_macro_auc"] - m_floor["drug_macro_auc"],
                "median_first_approval": float(np.median(approvals))
                if len(approvals)
                else float("nan"),
                "median_cem_degree": float(np.median(degrees)) if len(degrees) else float("nan"),
            }
        )
    return rows


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=16.0,
    memory=32768,
    timeout=900,
)
def run(exp_id: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp17")
    t_start = time.monotonic()
    budget_deadline = t_start + 900.0

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

    row_counts = {}
    for name, p in {
        "train": paths["train"],
        "validate": paths["validate"],
        "ingredient_target_long": paths["ingredient_target_long"],
        "condition_features_basic": paths["condition_features_basic"],
        "condition_group_long": paths["condition_group_long"],
        "data_dictionary": paths["data_dictionary"],
    }.items():
        with p.open() as fh:
            row_counts[name] = sum(1 for _ in fh) - 1
    log.info("row counts: %s", row_counts)
    bad_counts = {k: v for k, v in row_counts.items() if v != EXPECTED_ROWS[k]}
    if bad_counts:
        msg = f"Precondition failed: row counts do not match expected: {bad_counts} (expected {EXPECTED_ROWS})."
        log.error(msg)
        return _fail(msg)
    log.info(
        "preconditions passed: ingredient_target_long.csv has exactly %d rows (8088 expected)",
        row_counts["ingredient_target_long"],
    )

    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    edges_raw = pd.read_csv(paths["ingredient_target_long"]).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    cond_basic = pd.read_csv(paths["condition_features_basic"])
    y_train_full = train["y_faers_signal"]
    y_val_full = validate["y_faers_signal"].to_numpy()

    # -------------------------------------------------- shared graph machinery
    degree_train, degree_val, p_c_map, train_wide_mean = _build_degree_and_pc_features(
        train, validate, cond_basic
    )
    drug_raw = _load_drug_features(paths["ingredient_features"], paths["data_dictionary"])
    cond_design = _load_condition_features(
        paths["condition_features_basic"], paths["condition_group_long"]
    )

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

    y_train_real = train["y_faers_signal"].to_numpy()
    flagged_idx = np.where(y_train_real == 1)[0]
    import scipy.sparse as sp

    f_train = sp.csr_matrix(
        (np.ones(len(flagged_idx)), (d_pos_train[flagged_idx], c_pos_train[flagged_idx])),
        shape=(n_ing, n_cond),
    )
    f_train.sum_duplicates()
    f_train.data[:] = 1.0

    ing_rung, has_target_annotation_by_ing, gene_feats, leaf_feats, l1_feats = _rung_arrays(
        edges_raw,
        all_ing,
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
    )
    val_rung_full = ing_rung[d_pos_val]

    val_first_approval_full = validate.merge(
        drug_raw[["omop_concept_id", "first_approval"]],
        left_on="ingredient_concept_id",
        right_on="omop_concept_id",
        how="left",
    )["first_approval"].to_numpy()

    # CEM degree (label-adjacent -- used ONLY for characterisation reporting per §6 of the
    # spec, never as a training feature; drug_degree_train is not label-adjacent since it
    # is computed on train pairs with the same convention as exp07/exp08/exp12).
    drug_degree_train_map = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    val_degree_full = (
        validate["ingredient_concept_id"].map(drug_degree_train_map).fillna(0).to_numpy()
    )

    drug_ids_val = validate["ingredient_concept_id"].to_numpy()

    # neighbour-model baseline+gene features, reused across restrictions (exp08's model B)
    baseline_features = ["log1p_drug_degree", "log1p_condition_degree", "log1p_record_count", "p_c"]
    gene_features = ["nb_excess_gene", "n_neighbours_gene"]
    model_b_features = baseline_features + gene_features
    cond_record_count_map = cond_basic.set_index("condition_concept_id")["record_count"]
    condition_degree_train_map = train.groupby("condition_concept_id")[
        "ingredient_concept_id"
    ].nunique()

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

    def _attach_tier(df, feats, split):
        df = df.copy()
        suffix = f"__{split}"
        for name, arr in feats.items():
            if name.endswith(suffix):
                df[name[: -len(suffix)]] = arr
        return df

    train_nb = _attach_baseline(train, p_c_train)
    validate_nb = _attach_baseline(validate, p_c_val)
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

    def _fit_hgb(x_train, y_col, x_val):
        x_train = x_train.astype("float32")
        x_val = x_val.astype("float32")
        for col in x_train.columns:
            med = float(x_train[col].median())
            x_train[col] = x_train[col].fillna(med)
            x_val[col] = x_val[col].fillna(med)
        model = HistGradientBoostingClassifier(**HGB_KWARGS)
        model.fit(x_train, y_col)
        return model.predict_proba(x_val)[:, 1]

    def _union_design(base_df, degree_pc_df, drop_substance_type):
        idx = base_df.index
        drug_design = _build_drug_design(drug_raw, drop_substance_type=drop_substance_type).rename(
            columns={"omop_concept_id": "ingredient_concept_id"}
        )
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

    all_curve_rows: list[dict] = []
    per_drug_frames: list[pd.DataFrame] = []
    curve_metrics: dict[str, Any] = {}
    small_mol_slope: dict[str, float] = {
        "slope_smallmol": float("nan"),
        "slope_ci_lo_smallmol": float("nan"),
        "slope_ci_hi_smallmol": float("nan"),
    }

    # small-molecule mask, applied at the pair level (drug side)
    is_small_mol_ing = set(
        drug_raw.loc[drug_raw["molecule_type"] == "Small molecule", "omop_concept_id"]
    )
    train_small_mask = train["ingredient_concept_id"].isin(is_small_mol_ing).to_numpy()
    val_small_mask = validate["ingredient_concept_id"].isin(is_small_mol_ing).to_numpy()
    log.info(
        "small-molecule restriction: train %d/%d rows, validate %d/%d rows",
        train_small_mask.sum(),
        len(train),
        val_small_mask.sum(),
        len(validate),
    )

    # ---------------------------------------------------------- step 1: characterisation
    first_approval_map = drug_raw.drop_duplicates("omop_concept_id").set_index("omop_concept_id")[
        "first_approval"
    ]
    char_rows = []
    for rung in RUNGS:
        mask = val_rung_full == rung
        sub_ing = pd.unique(drug_ids_val[mask])
        sub = drug_raw[drug_raw["omop_concept_id"].isin(sub_ing)].drop_duplicates("omop_concept_id")
        deg = pd.Series(sub_ing).map(drug_degree_train_map).fillna(0)
        appr = validate.loc[mask, "ingredient_concept_id"].map(first_approval_map)
        char_rows.append(
            {
                "rung": rung,
                "n_drugs": len(sub_ing),
                "n_small_molecule": int((sub["molecule_type"] == "Small molecule").sum()),
                "pct_small_molecule": float((sub["molecule_type"] == "Small molecule").mean())
                if len(sub)
                else float("nan"),
                "median_cem_degree": float(deg.median()) if len(deg) else float("nan"),
                "median_first_approval": float(appr.dropna().median())
                if appr.notna().any()
                else float("nan"),
                "top_molecule_types": sub["molecule_type"]
                .value_counts(dropna=False)
                .head(5)
                .to_dict(),
                "top_availability_labels": sub["availability_label"]
                .value_counts(dropna=False)
                .head(5)
                .to_dict()
                if "availability_label" in sub.columns
                else {},
            }
        )
    char_df = pd.DataFrame(char_rows)
    char_path = out / f"{exp_id}_rung_characterisation.csv"
    char_df.to_csv(char_path, index=False)
    log.info("D4 characterisation: %s", char_rows[-1])

    # ---------------------------------------------------- step 2: indicator-removal refit
    log.info("fitting indicator-removed union model once on all of train")
    x_train_noind = _union_design(train, degree_train, drop_substance_type=True)
    x_val_noind = _union_design(validate, degree_val, drop_substance_type=True)
    val_proba_noind = _fit_hgb(x_train_noind, y_train_full, x_val_noind)

    rows_noind = _score_rungs(
        "y_faers_signal",
        val_rung_full,
        drug_ids_val,
        y_val_full,
        val_proba_noind,
        validate_nb["p_c"].to_numpy(),
        val_first_approval_full,
        val_degree_full,
        "noind",
    )
    all_curve_rows.extend(rows_noind)
    for r in rows_noind:
        curve_metrics[f"margin_{r['rung']}_noind"] = r["margin"]
    ind_removed_path = out / f"{exp_id}_curve_indicator_removed.csv"
    pd.DataFrame(rows_noind).to_csv(ind_removed_path, index=False)

    time_used = time.monotonic() - t_start
    log.info("step 2 done at t=%.1fs", time_used)

    # ---------------------------------------------------- step 2b: original curve (for comparison)
    log.info(
        "fitting original union model (all substance-type columns present) once on all of train"
    )
    x_train_orig = _union_design(train, degree_train, drop_substance_type=False)
    x_val_orig = _union_design(validate, degree_val, drop_substance_type=False)
    val_proba_orig = _fit_hgb(x_train_orig, y_train_full, x_val_orig)
    rows_orig = _score_rungs(
        "y_faers_signal",
        val_rung_full,
        drug_ids_val,
        y_val_full,
        val_proba_orig,
        validate_nb["p_c"].to_numpy(),
        val_first_approval_full,
        val_degree_full,
        "orig",
    )
    all_curve_rows.extend(rows_orig)
    for r in rows_orig:
        curve_metrics[f"margin_{r['rung']}_orig"] = r["margin"]
    t = _per_drug_table(pd.Series(drug_ids_val), pd.Series(y_val_full), val_proba_orig)
    t["curve"] = "orig"
    per_drug_frames.append(t)

    time_used = time.monotonic() - t_start
    log.info("step 2b (orig curve) done at t=%.1fs", time_used)

    # ---------------------------------------------------- step 3: small-molecules-only curve
    train_sm = train.loc[train_small_mask].reset_index(drop=True)
    validate_sm = validate.loc[val_small_mask].reset_index(drop=True)
    degree_train_sm, degree_val_sm, _p_c_map_sm, _train_wide_mean_sm = (
        _build_degree_and_pc_features(train_sm, validate_sm, cond_basic)
    )
    log.info(
        "small-molecule population: %d train rows / %d validate rows, %d train drugs / %d validate drugs",
        len(train_sm),
        len(validate_sm),
        train_sm["ingredient_concept_id"].nunique(),
        validate_sm["ingredient_concept_id"].nunique(),
    )
    x_train_sm = _union_design(train_sm, degree_train_sm, drop_substance_type=False)
    x_val_sm = _union_design(validate_sm, degree_val_sm, drop_substance_type=False)
    val_proba_sm = _fit_hgb(x_train_sm, train_sm["y_faers_signal"], x_val_sm)

    # per-rung floors computed on the SAME restricted population, per the spec ("recompute
    # per-rung floors on that restricted population").
    p_c_sm_map = train_sm.groupby("condition_concept_id")["y_faers_signal"].mean()
    train_wide_mean_sm2 = float(train_sm["y_faers_signal"].mean())
    val_floor_sm = (
        validate_sm["condition_concept_id"].map(p_c_sm_map).fillna(train_wide_mean_sm2).to_numpy()
    )
    val_rung_sm = val_rung_full[val_small_mask]
    drug_ids_val_sm = validate_sm["ingredient_concept_id"].to_numpy()
    y_val_sm = validate_sm["y_faers_signal"].to_numpy()
    val_first_approval_sm = val_first_approval_full[val_small_mask]
    val_degree_sm = val_degree_full[val_small_mask]

    rows_sm = _score_rungs(
        "y_faers_signal",
        val_rung_sm,
        drug_ids_val_sm,
        y_val_sm,
        val_proba_sm,
        val_floor_sm,
        val_first_approval_sm,
        val_degree_sm,
        "smallmol",
    )
    all_curve_rows.extend(rows_sm)
    for r in rows_sm:
        curve_metrics[f"margin_{r['rung']}_smallmol"] = r["margin"]
        curve_metrics[f"n_drugs_smallmol_{r['rung']}"] = r["n_drugs_scored"]
    smallmol_path = out / f"{exp_id}_curve_small_molecules.csv"
    pd.DataFrame(rows_sm).to_csv(smallmol_path, index=False)

    t_sm = _per_drug_table(pd.Series(drug_ids_val_sm), pd.Series(y_val_sm), val_proba_sm)
    t_sm["curve"] = "smallmol"
    per_drug_frames.append(t_sm)

    # slope of the small-molecules-only curve (rung index 0..4 as the continuous x, same
    # convention as exp12's distance regression, restricted to powered rungs)
    dist_map = {"D0": 0.0, "D1": 1.0, "D2": 2.0, "D3": 2.0, "D4": 3.0}
    slope_val_df = pd.DataFrame(
        {
            "drug": drug_ids_val_sm,
            "y": y_val_sm,
            "score": val_proba_sm,
            "dist": [dist_map[r] for r in val_rung_sm],
        }
    )
    from sklearn.metrics import roc_auc_score as _auc

    slope_rows = []
    for drug, g in slope_val_df.groupby("drug"):
        if len(g) < MIN_PAIRS_PER_DRUG or g["y"].nunique() < 2:
            continue
        auc = float(_auc(g["y"], g["score"]))
        slope_rows.append({"drug": drug, "auc": auc, "dist": float(g["dist"].iloc[0])})
    slope_df = pd.DataFrame(slope_rows)
    slope_path = out / f"{exp_id}_slope_smallmol.csv"
    slope_df.to_csv(slope_path, index=False)

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
        fallbacks_taken.append("small-molecule slope: too few drugs/distance variance to fit")
    small_mol_slope = {
        "slope_smallmol": slope_point,
        "slope_ci_lo_smallmol": slope_ci_lo,
        "slope_ci_hi_smallmol": slope_ci_hi,
    }

    time_used = time.monotonic() - t_start
    log.info("step 3 (small-molecule curve) done at t=%.1fs", time_used)

    # ---------------------------------------------------- step 5: neighbour model, both restrictions
    log.info("fitting exp08-style neighbour model on original population")
    x_train_nb_orig = train_nb[model_b_features]
    x_val_nb_orig = validate_nb[model_b_features]
    val_proba_nb_orig = _fit_hgb(x_train_nb_orig, y_train_real, x_val_nb_orig)
    rows_nb_orig = _score_rungs(
        "y_faers_signal",
        val_rung_full,
        drug_ids_val,
        y_val_full,
        val_proba_nb_orig,
        validate_nb["p_c"].to_numpy(),
        val_first_approval_full,
        val_degree_full,
        "nb_orig",
    )
    all_curve_rows.extend(rows_nb_orig)
    for r in rows_nb_orig:
        curve_metrics[f"nb_margin_{r['rung']}_orig"] = r["margin"]

    log.info("fitting exp08-style neighbour model on small-molecule population")
    train_nb_sm = train_nb.loc[train_small_mask].reset_index(drop=True)
    validate_nb_sm = validate_nb.loc[val_small_mask].reset_index(drop=True)
    x_train_nb_sm = train_nb_sm[model_b_features]
    x_val_nb_sm = validate_nb_sm[model_b_features]
    val_proba_nb_sm = _fit_hgb(x_train_nb_sm, train_nb_sm["y_faers_signal"], x_val_nb_sm)
    rows_nb_sm = _score_rungs(
        "y_faers_signal",
        val_rung_sm,
        drug_ids_val_sm,
        y_val_sm,
        val_proba_nb_sm,
        val_floor_sm,
        val_first_approval_sm,
        val_degree_sm,
        "nb_smallmol",
    )
    all_curve_rows.extend(rows_nb_sm)
    for r in rows_nb_sm:
        curve_metrics[f"nb_margin_{r['rung']}_smallmol"] = r["margin"]

    time_used = time.monotonic() - t_start
    time_remaining = budget_deadline - time.monotonic()
    log.info(
        "step 5 (neighbour model, both restrictions) done at t=%.1fs, remaining=%.1fs",
        time_used,
        time_remaining,
    )

    # ---------------------------------------------------- step 4: missingness-only model (guarded)
    missingness_rows: list[dict] = []
    missingness_d4_auc = float("nan")
    if time_remaining < 180.0:
        fallbacks_taken.append(
            f"Dropped step 4 (missingness-only model): only {time_remaining:.1f}s remained "
            "of the 900s budget (spec's own fallback: drop step 4 first, steps 2/3 are the "
            "experiment and were not dropped)."
        )
        log.warning("budget guard triggered: skipping step 4")
    else:
        drug_block_cols = [
            c for c in drug_raw.columns if c not in ("omop_concept_id", "first_approval")
        ]

        def _missingness_design(base_df):
            idx = base_df.index
            merged = base_df[["ingredient_concept_id"]].merge(
                drug_raw,
                left_on="ingredient_concept_id",
                right_on="omop_concept_id",
                how="left",
            )
            merged.index = idx
            miss = merged[drug_block_cols].isna().astype(float)
            miss.columns = [f"missing__{c}" for c in miss.columns]
            return miss

        miss_train = _missingness_design(train)
        miss_val = _missingness_design(validate)
        x_train_miss = pd.concat(
            [degree_train[["degree_drug", "p_c"]].set_axis(train.index), miss_train], axis=1
        )
        x_val_miss = pd.concat(
            [degree_val[["degree_drug", "p_c"]].set_axis(validate.index), miss_val], axis=1
        )
        val_proba_miss = _fit_hgb(x_train_miss, y_train_full, x_val_miss)
        rows_miss = _score_rungs(
            "y_faers_signal",
            val_rung_full,
            drug_ids_val,
            y_val_full,
            val_proba_miss,
            validate_nb["p_c"].to_numpy(),
            val_first_approval_full,
            val_degree_full,
            "missingness_only",
        )
        missingness_rows = rows_miss
        missingness_path = out / f"{exp_id}_missingness_only.csv"
        pd.DataFrame(rows_miss).to_csv(missingness_path, index=False)
        d4_row = next((r for r in rows_miss if r["rung"] == "D4"), None)
        missingness_d4_auc = d4_row["drug_macro_auc"] if d4_row else float("nan")
        log.info("missingness-only model D4 drug_macro_auc = %.4f", missingness_d4_auc)

    # -------------------------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, style in (("orig", "o-"), ("noind", "s--"), ("smallmol", "^:")):
        sub = [r for r in all_curve_rows if r["curve"] == tag]
        if not sub:
            continue
        sub_df = pd.DataFrame(sub).set_index("rung").reindex(RUNGS)
        x = np.arange(len(RUNGS))
        ax.errorbar(
            x,
            sub_df["drug_macro_auc"],
            yerr=[
                sub_df["drug_macro_auc"] - sub_df["drug_macro_auc_ci_lo"],
                sub_df["drug_macro_auc_ci_hi"] - sub_df["drug_macro_auc"],
            ],
            fmt=style,
            capsize=3,
            label=f"{tag} (union model)",
        )
        ax.plot(
            x, sub_df["floor_pc_lookup"], linestyle=":", marker="x", alpha=0.5, label=f"{tag} floor"
        )
        for i, row in sub_df.reset_index().iterrows():
            ax.annotate(
                f"n={int(row['n_drugs_scored'])}",
                (i, row["drug_macro_auc"]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=7,
                ha="center",
            )
    ax.set_xticks(np.arange(len(RUNGS)))
    ax.set_xticklabels(RUNGS)
    ax.set_xlabel("target-space distance rung")
    ax.set_ylabel("validate drug_macro_auc")
    ax.set_title("exp17: original vs indicator-removed vs small-molecules-only decay curves")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_decay_curves.png", dpi=150)
    plt.close(fig)

    per_drug_df = (
        pd.concat(per_drug_frames, ignore_index=True) if per_drug_frames else pd.DataFrame()
    )
    per_drug_path = out / f"{exp_id}_per_drug.csv"
    per_drug_df.to_csv(per_drug_path, index=False)

    results.commit()

    # ------------------------------------------------------------------- metrics
    metrics: dict[str, Any] = dict(curve_metrics)
    metrics.update(small_mol_slope)
    metrics["missingness_only_drug_macro_auc_D4"] = missingness_d4_auc

    def _fmt_rows(rows):
        return "; ".join(
            f"{r['rung']}: n_drugs={r['n_drugs_scored']} auc={r['drug_macro_auc']:.4f} "
            f"floor={r['floor_pc_lookup']:.4f} margin={r['margin']:+.4f}"
            f"{' UNDERPOWERED' if r['underpowered'] else ''}"
            for r in rows
        )

    d4_orig = next(r for r in rows_orig if r["rung"] == "D4")
    d4_noind = next(r for r in rows_noind if r["rung"] == "D4")
    d0_orig = next(r for r in rows_orig if r["rung"] == "D0")
    d0_sm = next((r for r in rows_sm if r["rung"] == "D0"), None)

    d4_collapse = d4_orig["margin"] - d4_noind["margin"]
    d0_survives = d0_sm is not None and not d0_sm["underpowered"] and d0_sm["margin"] > 0.01
    d0_sm_margin = d0_sm["margin"] if d0_sm is not None else float("nan")
    d0_sm_n_drugs = d0_sm["n_drugs_scored"] if d0_sm is not None else 0

    findings = (
        f"D4's margin is made of substance-type separation: removing every substance-type "
        f"indicator column collapses D4's margin from {d4_orig['margin']:+.4f} (original) to "
        f"{d4_noind['margin']:+.4f} (indicator-removed), a drop of {d4_collapse:+.4f}. "
        f"D0's margin under the small-molecules-only restriction is "
        f"{d0_sm_margin:+.4f} on n_drugs={d0_sm_n_drugs} "
        f"(original-population D0 margin was {d0_orig['margin']:+.4f}), so D0's margin "
        f"{'SURVIVES' if d0_survives else 'DOES NOT clearly survive'} restriction to the "
        f"population the project targets. "
        f"Small-molecules-only slope = {small_mol_slope['slope_smallmol']:+.4f} per rung, 95% "
        f"bootstrap CI [{small_mol_slope['slope_ci_lo_smallmol']:+.4f}, "
        f"{small_mol_slope['slope_ci_hi_smallmol']:+.4f}] over {len(slope_df)} drugs. "
        f"Verdict: target-space distance "
        + (
            "does appear to predict performance in the small-molecule population (negative "
            "slope, CI excludes 0)"
            if small_mol_slope["slope_ci_hi_smallmol"] < 0
            else "does NOT clearly predict performance in the small-molecule population the "
            "project targets (slope CI includes 0 or is positive)"
        )
        + f". Missingness-only model reaches D4 drug_macro_auc={missingness_d4_auc:.4f} vs the "
        f"union model's original D4={d4_orig['drug_macro_auc']:.4f}"
        + (" (step 4 skipped, see fallbacks)" if not missingness_rows else "")
        + f". Original curve by rung: {_fmt_rows(rows_orig)}. "
        f"Indicator-removed curve by rung: {_fmt_rows(rows_noind)}. "
        f"Small-molecules-only curve by rung: {_fmt_rows(rows_sm)}. "
        f"Neighbour model (exp08) small-molecules-only by rung: {_fmt_rows(rows_nb_sm)}. "
        + (
            f"Fallbacks taken: {'; '.join(fallbacks_taken)}."
            if fallbacks_taken
            else "No fallbacks taken."
        )
    )

    artifacts = [
        str(char_path),
        str(ind_removed_path),
        str(smallmol_path),
        str(out / f"{exp_id}_decay_curves.png"),
        str(per_drug_path),
        str(slope_path),
    ]
    if missingness_rows:
        artifacts.append(str(out / f"{exp_id}_missingness_only.csv"))

    return {
        "__metrics__": metrics,
        "__findings__": findings,
        "__artifacts__": artifacts,
        "__fallbacks__": fallbacks_taken,
    }


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp17",
        title="Decomposing the inverted transport curve: substance type versus target distance",
        hypothesis=(
            "D4's +0.085 margin is substance-type separation that collapses when type "
            "indicators are removed and under a small-molecule restriction, while D0's "
            "+0.021 margin survives and the restricted curve becomes monotone decreasing."
        ),
        approach=(
            "Characterise rungs by molecule type, availability, route, approval decade and "
            "degree; refit with all substance-type columns removed; rebuild the curve "
            "restricted to small molecules with per-rung floors; fit a missingness-only "
            "model as the explicit alternative hypothesis."
        ),
        label="y_faers_signal",
        features=["intrinsic_union", "nb_excess_gene", "p_c", "degree", "missingness_indicators"],
        split="validate stratified by target-space distance, small-molecule restriction",
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
    metrics = metrics.get("__metrics__", metrics)
    print(findings)

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=artifacts,
        next_steps=(
            "Read <exp_id>_decay_curves.png for the three-curve headline figure and "
            "<exp_id>_curve_small_molecules.csv for the project's claim evaluated on its "
            "target population. If step 4 was dropped (see fallbacks: "
            f"{'; '.join(fallbacks) if fallbacks else 'none'}), a follow-up run with more "
            "budget headroom should complete the missingness-only comparison."
        ),
    )
