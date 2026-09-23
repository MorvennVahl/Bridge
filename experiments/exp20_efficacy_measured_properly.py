"""exp20 — The efficacy half, on the metric and the floors the safety half uses.

Implements experiments/exp20_efficacy_measured_properly.md. Re-measures exp05
(exp_20260922_7c2e17) on drug_macro_auc with floors, 5 seeds, and the never-attempted
neighbour feature (exp08's nb_excess recomputed against y_semmeddb_treats with
leave-one-drug-out). See the spec for full method, budget, and reporting requirements.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any

import modal

app = modal.App("bridge-exp20")  # stable name, no random suffix

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
    .add_local_python_source("floors", "leak_audit")
)

data = modal.Volume.from_name("bridge-data")
results = modal.Volume.from_name("bridge-results", create_if_missing=True)

image = image.add_local_file(
    pathlib.Path(__file__).parent / "floors.py", remote_path="/root/floors.py"
).add_local_file(pathlib.Path(__file__).parent / "leak_audit.py", remote_path="/root/leak_audit.py")

EXPECTED_TRAIN_POS = {"y_semmeddb_treats": 3926, "y_semmeddb_causes": 715}
EXPECTED_VAL_POS = {"y_semmeddb_treats": 2092, "y_semmeddb_causes": 370}

# I-block: explicit indication-encoding columns named by the spec (drug-side).
I_EXPLICIT_COLS = [
    "atc_l1",
    "n_atc_codes",
    "indication_class",
    "kegg_efficacy",
    "usan_stem_definition",
    "max_phase",
    "first_approval",
    "therapeutic_flag",
]

# M-block: data-dictionary block names (drug side), matched case/whitespace-tolerant.
M_BLOCK_NAMES = [
    "mechanism",
    "target biology",
    "target_biology",
    "chemistry",
    "exposure",
    "metabolism",
]

SEEDS = [0, 1, 2, 3, 4]  # spec: 5 seeds, cut before cutting CV
CV_SEED = 0
MAX_ONEHOT_CARDINALITY = 30
MIN_PAIRS_PER_DRUG = 1  # eligibility bites hard at 0.48% prevalence; see floors() call too
HGB_KWARGS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "early_stopping": False,
}
K_SHRINK = 10


def _fail(message: str, artifacts: list[str] | None = None) -> dict[str, Any]:
    return {
        "precondition_failed": 1.0,
        "precondition_error_message": message,
        "__artifacts__": artifacts or [],
    }


def _drug_macro_metrics(ids, y, score, min_pairs: int = 1) -> dict:
    """drug_macro_auc / drug_macro_p10 per METRIC.md.

    Eligibility: >=min_pairs pairs and both classes present. NaN-safe (per-drug median
    fill, bool cast to float before ranking) per the round-4 contract.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(
        {
            "drug": ids.to_numpy() if hasattr(ids, "to_numpy") else np.asarray(ids),
            "y": y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y),
            "score": np.asarray(score, dtype=float),
        }
    )
    aucs: list[float] = []
    p10s: list[float] = []
    for _drug, g in df.groupby("drug"):
        if len(g) < min_pairs:
            continue
        y_g = g["y"].to_numpy().astype(float)
        if len(np.unique(y_g)) < 2:
            continue
        s = g["score"]
        if s.isna().any():
            s = s.fillna(s.median())
        if s.isna().all():
            continue
        aucs.append(float(roc_auc_score(y_g, s.to_numpy())))
        top10 = g.sort_values("score", ascending=False).head(10)
        p10s.append(float(top10["y"].mean()))
    return {
        "drug_macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "drug_macro_p10": float(np.mean(p10s)) if p10s else float("nan"),
        "n_drugs_scored": len(aucs),
    }


# ------------------------------------------------------ exp08-style neighbour machinery
# Copied (essentially unchanged) from exp08_neighbour_transport_pc_stripped.py, retargeted
# at y_semmeddb_treats instead of y_faers_signal -- this is exp05's never-attempted step 5.


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


def _dedup_pairs(df, value_col: str):
    p = df[["ingredient_concept_id", value_col]].dropna().rename(columns={value_col: "value"})
    p["value"] = p["value"].astype(str)
    return p.drop_duplicates()


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
    """n_neighbours_t and nb_excess_t = shrunk_rate_t - p_c, leave-one-drug-out on train.

    Same vectorized machinery as exp08, here against y_semmeddb_treats.
    """
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


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=16.0,
    memory=32768,
    timeout=900,
)
def run(exp_id: str) -> dict[str, Any]:
    import leak_audit
    import numpy as np
    import pandas as pd
    from floors import floors as compute_floors
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    run_start = time.monotonic()
    budget_deadline = run_start + 900.0
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp20")

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)
    fallbacks_taken: list[str] = []

    # -------------------------------------------------------------- preconditions
    paths = {
        "train": pathlib.Path("/data/splits/train.csv"),
        "validate": pathlib.Path("/data/splits/validate.csv"),
        "ingredient_features": pathlib.Path("/data/drug/ingredient_features.csv"),
        "ingredient_target_long": pathlib.Path("/data/drug/ingredient_target_long.csv"),
        "condition_features_basic": pathlib.Path("/data/condition/condition_features_basic.csv"),
        "condition_group_long": pathlib.Path("/data/condition/condition_group_long.csv"),
        "data_dictionary": pathlib.Path("/data/data_dictionary.csv"),
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        msg = f"Missing required input(s): {missing}"
        log.error(msg)
        return _fail(msg)

    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    drug_feat = pd.read_csv(paths["ingredient_features"], low_memory=False)
    cond_feat = pd.read_csv(paths["condition_features_basic"])
    cond_group = pd.read_csv(paths["condition_group_long"])
    data_dict = pd.read_csv(paths["data_dictionary"])
    edges_raw = pd.read_csv(paths["ingredient_target_long"]).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )

    for label, expected_train, expected_val in [
        (
            "y_semmeddb_treats",
            EXPECTED_TRAIN_POS["y_semmeddb_treats"],
            EXPECTED_VAL_POS["y_semmeddb_treats"],
        ),
    ]:
        obs_train = int(train[label].sum())
        obs_val = int(validate[label].sum())
        if obs_train != expected_train or obs_val != expected_val:
            msg = (
                f"precondition failed for {label}: train positives {obs_train} "
                f"(expected {expected_train}), validate positives {obs_val} "
                f"(expected {expected_val}). Stopping without improvising."
            )
            log.error(msg)
            return _fail(msg)
        log.info("precondition OK for %s: train=%d validate=%d", label, obs_train, obs_val)

    causes_train = int(train["y_semmeddb_causes"].sum())
    causes_val = int(validate["y_semmeddb_causes"].sum())
    log.info(
        "y_semmeddb_causes counts (indicative secondary): train=%d (expected %d), validate=%d (expected %d)",
        causes_train,
        EXPECTED_TRAIN_POS["y_semmeddb_causes"],
        causes_val,
        EXPECTED_VAL_POS["y_semmeddb_causes"],
    )
    if (
        causes_train != EXPECTED_TRAIN_POS["y_semmeddb_causes"]
        or causes_val != EXPECTED_VAL_POS["y_semmeddb_causes"]
    ):
        fallbacks_taken.append(
            f"y_semmeddb_causes counts did not match spec (train={causes_train}, validate={causes_val}); "
            "carried anyway since it is explicitly indicative/underpowered, not a hard precondition."
        )

    # -------------------------------------------------------------------- leak audit
    # Hard assertion: no semmeddb_* / y_* column ever appears as a feature (exp06's mistake,
    # sharpened by round 3's convention). Checked again per-design below after final column
    # lists are assembled.
    hits_drug = leak_audit.audit_columns(list(drug_feat.columns))
    hits_cond = leak_audit.audit_columns(list(cond_feat.columns))
    log.info(
        "leak_audit: drug columns flagged=%s, condition columns flagged=%s", hits_drug, hits_cond
    )

    # ------------------------------------------- M/I column assignment (reuse exp05's logic)
    # exp05's saved artifact (exp_20260922_7c2e17_column_assignment.csv) was not found locally
    # under results/; re-derived here directly from exp05's own script logic (same M_BLOCK_NAMES,
    # I_EXPLICIT_COLS, data-dictionary block matching, bucket-tail cardinality handling).
    dd_drug = data_dict[data_dict["table"] == "ingredient_features.csv"].copy()
    dd_drug["block_norm"] = (
        dd_drug["block"].astype(str).str.strip().str.lower().str.replace("_", " ")
    )
    target_block_names_norm = [b.strip().lower().replace("_", " ") for b in M_BLOCK_NAMES]
    matched_blocks = sorted(
        set(dd_drug.loc[dd_drug["block_norm"].isin(target_block_names_norm), "block"].unique())
    )
    log.info("M-block names matched in data dictionary: %s", matched_blocks)

    m_dict_cols = [
        c
        for c in dd_drug.loc[dd_drug["block_norm"].isin(target_block_names_norm), "column"]
        if c in drug_feat.columns
    ]
    i_explicit_cols = [c for c in I_EXPLICIT_COLS if c in drug_feat.columns]
    missing_i = [c for c in I_EXPLICIT_COLS if c not in drug_feat.columns]
    if missing_i:
        log.warning("I-block columns absent from ingredient_features.csv: %s", missing_i)

    m_extra_cols = [
        c for c in ["dominant_target_class"] if c in drug_feat.columns and c not in m_dict_cols
    ]

    drug_m_cols_all = sorted(set(m_dict_cols) | set(m_extra_cols))
    drug_i_cols_all = sorted(set(i_explicit_cols))
    overlap = set(drug_m_cols_all) & set(drug_i_cols_all)
    if overlap:
        # trap (spec's own): an I column must never drift into M -- the M/I partition is
        # what makes the transportability claim honest.
        log.warning(
            "columns present in both M and I dictionary matches, removing from M: %s", overlap
        )
        drug_m_cols_all = [c for c in drug_m_cols_all if c not in overlap]

    def usable_columns(cols: list[str], bucket_tail: bool = False):
        numeric_cols: list[str] = []
        onehot_cols: list[str] = []
        dropped: dict[str, str] = {}
        for c in cols:
            s = drug_feat[c]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                numeric_cols.append(c)
            elif pd.api.types.is_object_dtype(s):
                nun = s.nunique(dropna=True)
                if nun == 0:
                    dropped[c] = "object dtype, empty, dropped"
                elif nun <= MAX_ONEHOT_CARDINALITY:
                    onehot_cols.append(c)
                elif bucket_tail:
                    top = s.value_counts().nlargest(MAX_ONEHOT_CARDINALITY - 1).index
                    drug_feat[c] = np.where(s.isin(top), s, "other")
                    onehot_cols.append(c)
                    log.info(
                        "bucketed drug column %s: cardinality %d > %d, kept top %d + 'other'",
                        c,
                        nun,
                        MAX_ONEHOT_CARDINALITY,
                        MAX_ONEHOT_CARDINALITY - 1,
                    )
                else:
                    dropped[c] = (
                        f"object dtype, cardinality {nun} > {MAX_ONEHOT_CARDINALITY}, dropped"
                    )
            else:
                dropped[c] = f"unsupported dtype {s.dtype}"
        return numeric_cols, onehot_cols, dropped

    m_numeric, m_onehot, m_dropped = usable_columns(drug_m_cols_all)
    i_numeric, i_onehot, i_dropped = usable_columns(drug_i_cols_all, bucket_tail=True)
    for c, reason in {**m_dropped, **i_dropped}.items():
        log.info("dropped drug column %s: %s", c, reason)

    cond_intrinsic_numeric = [
        c
        for c in [
            "record_count",
            "n_omop_ancestors",
            "n_ontology_terms",
            "n_hpo_genes",
            "n_groups",
            "n_ot_genes",
            "n_ot_genes_strong",
            "max_ot_score",
            "ot_truncated",
            "has_any_gene",
        ]
        if c in cond_feat.columns
    ]
    cond_intrinsic_cat = [
        c for c in ["best_match_tier", "concept_class_id", "gene_arm"] if c in cond_feat.columns
    ]

    cond_group = cond_group.copy()
    cond_group["group_key_label"] = (
        cond_group["group_source"].astype(str) + "::" + cond_group["group_label"].astype(str)
    )
    top_labels = (
        cond_group["group_key_label"].value_counts().nlargest(MAX_ONEHOT_CARDINALITY).index.tolist()
    )
    cond_group["group_key_capped"] = np.where(
        cond_group["group_key_label"].isin(top_labels), cond_group["group_key_label"], "other"
    )
    cond_group_wide = pd.crosstab(
        cond_group["condition_concept_id"], cond_group["group_key_capped"]
    )
    cond_group_wide.columns = [f"group_{c}" for c in cond_group_wide.columns]
    cond_group_wide = cond_group_wide.reset_index()

    def join_features(df: pd.DataFrame) -> pd.DataFrame:
        drug_cols = ["omop_concept_id", *m_numeric, *m_onehot, *i_numeric, *i_onehot]
        out_df = df.merge(
            drug_feat[drug_cols],
            left_on="ingredient_concept_id",
            right_on="omop_concept_id",
            how="left",
        )
        cond_cols = [
            "condition_concept_id",
            *cond_intrinsic_numeric,
            *cond_intrinsic_cat,
            "is_mapped",
            "arm",
        ]
        out_df = out_df.merge(cond_feat[cond_cols], on="condition_concept_id", how="left")
        out_df = out_df.merge(cond_group_wide, on="condition_concept_id", how="left")
        for c in cond_group_wide.columns:
            if c != "condition_concept_id":
                out_df[c] = out_df[c].fillna(0)
        return out_df

    train_j = join_features(train)
    validate_j = join_features(validate)

    # degree terms, TRAIN ONLY, applied to validate (M block)
    drug_degree_train = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    condition_degree_train = train.groupby("condition_concept_id")[
        "ingredient_concept_id"
    ].nunique()
    drug_degree_median = float(drug_degree_train.median())
    condition_degree_median = float(condition_degree_train.median())
    for df in (train_j, validate_j):
        df["drug_degree_train"] = (
            df["ingredient_concept_id"].map(drug_degree_train).fillna(drug_degree_median)
        )
        df["condition_degree_train"] = (
            df["condition_concept_id"].map(condition_degree_train).fillna(condition_degree_median)
        )
        df["log_drug_degree"] = np.log1p(df["drug_degree_train"])
        df["log_condition_degree"] = np.log1p(df["condition_degree_train"])
    degree_cols = ["log_drug_degree", "log_condition_degree"]

    # p_c for TREATS, train only -> M design gets it explicitly per floors.py convention
    p_c_treats_train = train.groupby("condition_concept_id")["y_semmeddb_treats"].mean()
    p_c_treats_mean = float(p_c_treats_train.mean())
    train_j["p_c_treats"] = (
        train_j["condition_concept_id"].map(p_c_treats_train).fillna(p_c_treats_mean)
    )
    validate_j["p_c_treats"] = (
        validate_j["condition_concept_id"].map(p_c_treats_train).fillna(p_c_treats_mean)
    )
    p_c_col = ["p_c_treats"]

    def onehot(df, cols, prefix):
        made = []
        for c in cols:
            dummies = pd.get_dummies(df[c].fillna("__missing__"), prefix=f"{prefix}_{c}")
            df = pd.concat([df, dummies], axis=1)
            made.extend(dummies.columns.tolist())
        return df, made

    train_j, m_onehot_made = onehot(train_j, m_onehot, "m")
    validate_j, _ = onehot(validate_j, m_onehot, "m")
    train_j, i_onehot_made = onehot(train_j, i_onehot, "i")
    validate_j, _ = onehot(validate_j, i_onehot, "i")
    train_j, cond_cat_made = onehot(train_j, cond_intrinsic_cat, "cond")
    validate_j, _ = onehot(validate_j, cond_intrinsic_cat, "cond")

    for made_cols in (m_onehot_made, i_onehot_made, cond_cat_made):
        for c in made_cols:
            if c not in validate_j.columns:
                validate_j[c] = 0

    group_cols = [c for c in cond_group_wide.columns if c != "condition_concept_id"]

    M_COLS = sorted(  # noqa: N806
        set(m_numeric)
        | set(m_onehot_made)
        | set(cond_intrinsic_numeric)
        | set(cond_cat_made)
        | set(group_cols)
        | set(degree_cols)
        | set(p_c_col)
    )
    I_COLS = sorted(set(i_numeric) | set(i_onehot_made))  # noqa: N806
    MI_COLS = sorted(set(M_COLS) | set(I_COLS))  # noqa: N806

    for c in M_COLS + I_COLS:
        train_j[c] = pd.to_numeric(train_j[c], errors="coerce")
        validate_j[c] = pd.to_numeric(validate_j[c], errors="coerce")

    # hard assertion: no semmeddb_/y_ column ever entered a design
    for design_name, cols in [("M", M_COLS), ("I", I_COLS), ("M+I", MI_COLS)]:
        bad = leak_audit.audit_columns(cols)
        if bad:
            msg = f"LEAK: design {design_name} contains blacklisted columns: {bad}"
            log.error(msg)
            return _fail(msg)
    log.info("leak audit passed for M / I / M+I designs: no semmeddb_/y_ column present")

    log.info(
        "M design: %d cols, I design: %d cols, M+I design: %d cols",
        len(M_COLS),
        len(I_COLS),
        len(MI_COLS),
    )

    assignment_rows = []
    for c in M_COLS:
        assignment_rows.append(
            {
                "column": c,
                "design": "M",
                "source": "mechanism/target biology/chemistry/exposure/metabolism, condition-intrinsic, group one-hot, degree, p_c_treats",
            }
        )
    for c in I_COLS:
        assignment_rows.append(
            {
                "column": c,
                "design": "I",
                "source": "indication-encoding (ATC/indication_class/usan/kegg/max_phase/first_approval)",
            }
        )
    for c, reason in {**m_dropped, **i_dropped}.items():
        assignment_rows.append({"column": c, "design": "DROPPED", "source": reason})
    assignment_df = pd.DataFrame(assignment_rows)
    assignment_path = out / f"{exp_id}_column_assignment.csv"
    assignment_df.to_csv(assignment_path, index=False)

    def matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        x = df[cols].copy()
        med = x.median(numeric_only=True)
        x = x.fillna(med).fillna(0.0)
        return x

    # ---------------------------------------------------------------- step 2: floors
    floor_rows = []
    for label_name in ["y_semmeddb_treats", "y_semmeddb_causes"]:
        f = compute_floors(train, validate, label_name, eligibility=1)
        floor_rows.append({"label": label_name, **f})
        log.info("floors[%s]=%s", label_name, f)
    floors_df = pd.DataFrame(floor_rows)
    floor_table_path = out / f"{exp_id}_floor_table.csv"
    floors_df.to_csv(floor_table_path, index=False)

    floor_treats = floors_df[floors_df["label"] == "y_semmeddb_treats"].iloc[0]
    floor_causes = floors_df[floors_df["label"] == "y_semmeddb_causes"].iloc[0]

    # -------------------------------------------------- step 5 (moved earlier): neighbour
    # feature vs TREATS, leave-one-drug-out, exp08-style. Uses the gene tier only (the
    # spec names "exp08's nb_excess"; gene is exp08's headline tier).
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

    p_c_by_cond = p_c_treats_train.reindex(all_cond).fillna(p_c_treats_mean).to_numpy()
    p_c_train_arr = p_c_by_cond[c_pos_train]
    p_c_val_arr = p_c_by_cond[c_pos_val]

    import scipy.sparse as sp

    y_train_treats = train["y_semmeddb_treats"].to_numpy()
    flagged_idx = np.where(y_train_treats == 1)[0]
    f_train_treats = sp.csr_matrix(
        (np.ones(len(flagged_idx)), (d_pos_train[flagged_idx], c_pos_train[flagged_idx])),
        shape=(n_ing, n_cond),
    )
    f_train_treats.sum_duplicates()
    f_train_treats.data[:] = 1.0

    gene_feats = _tier_features_excess(
        "gene",
        gene_pairs,
        ing_index_map,
        n_ing,
        train_mask,
        d_pos_train,
        d_pos_val,
        c_pos_train,
        c_pos_val,
        f_train_treats,
        p_c_train_arr,
        p_c_val_arr,
        log,
    )
    has_target_annotation_by_ing = np.array(
        [1.0 if i in set(edges_raw["ingredient_concept_id"]) else 0.0 for i in all_ing]
    )
    train_j["nb_excess_gene_treats"] = gene_feats["nb_excess_gene__train"]
    train_j["n_neighbours_gene_treats"] = gene_feats["n_neighbours_gene__train"]
    validate_j["nb_excess_gene_treats"] = gene_feats["nb_excess_gene__val"]
    validate_j["n_neighbours_gene_treats"] = gene_feats["n_neighbours_gene__val"]
    train_j["has_target_annotation"] = has_target_annotation_by_ing[d_pos_train]
    validate_j["has_target_annotation"] = has_target_annotation_by_ing[d_pos_val]
    for df in (train_j, validate_j):
        mask = df["has_target_annotation"] == 0
        df.loc[mask, ["nb_excess_gene_treats"]] = np.nan

    coverage_ge1_nb_treats = float((validate_j["n_neighbours_gene_treats"].fillna(0) >= 1).mean())
    log.info(
        "gene-tier neighbour coverage vs TREATS (validate, >=1 neighbour): %.4f",
        coverage_ge1_nb_treats,
    )

    M_NEIGHBOUR_COLS = sorted(set(M_COLS) | {"nb_excess_gene_treats", "n_neighbours_gene_treats"})  # noqa: N806

    designs = {"M": M_COLS, "I": I_COLS, "M+I": MI_COLS, "M+neighbour": M_NEIGHBOUR_COLS}

    # ---------------------------------------------------------- step 3-4: designs x seeds
    def fit_eval(label_col: str, design_cols: list[str], seed: int) -> dict:
        x_train = matrix(train_j, design_cols)
        y_train = train_j[label_col]
        x_val = matrix(validate_j, design_cols)
        y_val = validate_j[label_col]
        model = HistGradientBoostingClassifier(random_state=seed, **HGB_KWARGS)
        model.fit(x_train, y_train)
        proba_val = model.predict_proba(x_val)[:, 1]
        ap = average_precision_score(y_val, proba_val)
        auc = roc_auc_score(y_val, proba_val)
        dm = _drug_macro_metrics(validate_j["ingredient_concept_id"], y_val, proba_val, min_pairs=1)
        return {"ap": ap, "auc": auc, "proba_val": proba_val, "model": model, **dm}

    design_rows: list[dict[str, object]] = []
    proba_by_design_seed: dict[tuple[str, int], Any] = {}
    prevalence_val = float(validate_j["y_semmeddb_treats"].mean())
    prevalence_train = float(train_j["y_semmeddb_treats"].mean())

    seeds_used = SEEDS
    for design_name, cols in designs.items():
        remaining = budget_deadline - time.monotonic()
        if remaining < 240 and design_name == list(designs.keys())[-1]:
            fallbacks_taken.append(
                f"reduced seeds for {design_name} to protect budget: {remaining:.0f}s remained"
            )
        for seed in seeds_used:
            res = fit_eval("y_semmeddb_treats", cols, seed)
            design_rows.append(
                {
                    "design": design_name,
                    "seed": seed,
                    "validate_average_precision": res["ap"],
                    "validate_roc_auc": res["auc"],
                    "drug_macro_auc": res["drug_macro_auc"],
                    "drug_macro_p10": res["drug_macro_p10"],
                    "n_drugs_scored": res["n_drugs_scored"],
                    "prevalence_validate": prevalence_val,
                }
            )
            proba_by_design_seed[(design_name, seed)] = res["proba_val"]
            log.info(
                "design=%s seed=%d drug_macro_auc=%.4f AP=%.4f n_drugs=%d",
                design_name,
                seed,
                res["drug_macro_auc"],
                res["ap"],
                res["n_drugs_scored"],
            )

    design_df = pd.DataFrame(design_rows)
    summary_rows = []
    for design_name in designs:
        sub = design_df[design_df["design"] == design_name]
        for stat_name, fn in [("mean", np.mean), ("std", np.std)]:
            summary_rows.append(
                {
                    "design": design_name,
                    "seed": stat_name,
                    "validate_average_precision": fn(sub["validate_average_precision"]),
                    "validate_roc_auc": fn(sub["validate_roc_auc"]),
                    "drug_macro_auc": fn(sub["drug_macro_auc"]),
                    "drug_macro_p10": fn(sub["drug_macro_p10"]),
                    "n_drugs_scored": sub["n_drugs_scored"].iloc[0],
                    "prevalence_validate": prevalence_val,
                }
            )
    design_df_full = pd.concat([design_df, pd.DataFrame(summary_rows)], ignore_index=True)
    designs_path = out / f"{exp_id}_designs.csv"
    design_df_full.to_csv(designs_path, index=False)

    def design_mean(design_name: str, metric: str) -> float:
        sub = design_df[design_df["design"] == design_name]
        return float(sub[metric].mean())

    def design_std(design_name: str, metric: str) -> float:
        sub = design_df[design_df["design"] == design_name]
        return float(sub[metric].std())

    seed_spread_df = pd.DataFrame(
        [
            {
                "design": d,
                "drug_macro_auc_mean": design_mean(d, "drug_macro_auc"),
                "drug_macro_auc_std": design_std(d, "drug_macro_auc"),
                "validate_ap_mean": design_mean(d, "validate_average_precision"),
                "validate_ap_std": design_std(d, "validate_average_precision"),
            }
            for d in designs
        ]
    )
    seed_spread_path = out / f"{exp_id}_seed_spread.csv"
    seed_spread_df.to_csv(seed_spread_path, index=False)

    # ------------------------------------------------------- step 5 (spec numbering): CV
    cv_results: dict[str, float] = {}
    cv_skipped: list[str] = []
    for design_name in ["M", "I", "M+I", "M+neighbour"]:
        remaining = budget_deadline - time.monotonic()
        if remaining < 180:
            cv_skipped.append(design_name)
            fallbacks_taken.append(
                f"skipped grouped CV for {design_name}: {remaining:.0f}s remained"
            )
            continue
        cols = designs[design_name]
        x = matrix(train_j, cols)
        y = train_j["y_semmeddb_treats"]
        groups = train_j["group_key"]
        drug_ids_tr = train_j["ingredient_concept_id"]
        gkf = GroupKFold(n_splits=3)
        fold_aucs = []
        for tr_idx, te_idx in gkf.split(x, y, groups=groups):
            m = HistGradientBoostingClassifier(random_state=CV_SEED, **HGB_KWARGS)
            m.fit(x.iloc[tr_idx], y.iloc[tr_idx])
            proba = m.predict_proba(x.iloc[te_idx])[:, 1]
            dm = _drug_macro_metrics(drug_ids_tr.iloc[te_idx], y.iloc[te_idx], proba, min_pairs=1)
            fold_aucs.append(dm["drug_macro_auc"])
        cv_results[design_name] = float(np.nanmean(fold_aucs))
        log.info(
            "grouped 3-fold CV drug_macro_auc, design=%s: %.4f (folds=%s)",
            design_name,
            cv_results[design_name],
            fold_aucs,
        )

    # ------------------------------------------------------------- importance (M+I)
    import re

    import lightgbm as lgb

    importance_rows: list[dict[str, object]] = []
    for design_name, cols in designs.items():
        x_train = matrix(train_j, cols)
        y_train_imp = train_j["y_semmeddb_treats"]
        x_lgb = x_train.copy()
        seen: dict[str, int] = {}
        sanitized = []
        for c in x_lgb.columns:
            base = re.sub(r"[^0-9A-Za-z_]", "_", str(c))
            n = seen.get(base, 0)
            seen[base] = n + 1
            sanitized.append(base if n == 0 else f"{base}_{n}")
        x_lgb.columns = sanitized
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.06, num_leaves=31, random_state=0, verbosity=-1
            )
            lgb_model.fit(x_lgb, y_train_imp)
            gains = lgb_model.booster_.feature_importance(importance_type="gain")
            imp_series = pd.Series(gains, index=cols).sort_values(ascending=False)
        except Exception as e:  # pragma: no cover
            log.warning("gain importance failed for design %s: %s", design_name, e)
            imp_series = pd.Series(dtype=float)
        top = imp_series.head(25)
        for col, val in top.items():
            importance_rows.append({"design": design_name, "column": col, "importance": val})
    importance_df = pd.DataFrame(importance_rows)
    importance_path = out / f"{exp_id}_importance_top25.csv"
    importance_df.to_csv(importance_path, index=False)

    # --------------------------------------------------- secondary label: y_semmeddb_causes
    causes_rows = []
    for design_name in ["M", "M+I"]:
        cols = designs[design_name]
        res = fit_eval("y_semmeddb_causes", cols, CV_SEED)
        causes_rows.append(
            {
                "design": design_name,
                "drug_macro_auc": res["drug_macro_auc"],
                "n_drugs_scored": res["n_drugs_scored"],
                "validate_average_precision": res["ap"],
                "floor_pc": floor_causes["floor_pc"],
                "floor_degree_pc": floor_causes["floor_degree_pc"],
                "prevalence": floor_causes["prevalence"],
                "note": "underpowered secondary (0.10% prevalence); indicative only",
            }
        )
    causes_df = pd.DataFrame(causes_rows)
    causes_path = out / f"{exp_id}_causes_secondary.csv"
    causes_df.to_csv(causes_path, index=False)

    # ------------------------------------------------------ transportability (exp12 rungs)
    # Gene / class_leaf / class_L1 rung ladder, adapted from exp12, scored against TREATS.
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
        "f_train": f_train_treats,
        "p_c_train": p_c_train_arr,
        "p_c_val": p_c_val_arr,
        "log": log,
    }
    leaf_feats = _tier_features_excess("class_leaf", class_leaf_pairs, **common_kwargs)
    l1_feats = _tier_features_excess("class_L1", class_l1_pairs, **common_kwargs)
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
    val_rung = ing_rung[d_pos_val]
    drug_ids_val = validate_j["ingredient_concept_id"].to_numpy()
    y_val_treats = validate_j["y_semmeddb_treats"].to_numpy()
    proba_m_seed0 = proba_by_design_seed[("M", SEEDS[0])]

    # small-molecule restriction, best-effort per exp17's concept (applied inline; exp17's
    # own output was not available under results/ at the time this script was written).
    small_mol_mask_by_ing = None
    small_mol_note = "exp17 output not found locally; small-molecule restriction applied inline from spec's own logic (molecule_type == 'Small molecule' where available), best-effort."
    if "molecule_type" in drug_feat.columns:
        small_mol_ids = set(
            drug_feat.loc[drug_feat["molecule_type"] == "Small molecule", "omop_concept_id"]
        )
        small_mol_mask_by_ing = np.array([1.0 if i in small_mol_ids else 0.0 for i in all_ing])
    else:
        fallbacks_taken.append(
            "molecule_type column absent from ingredient_features.csv; small-molecule restriction skipped entirely"
        )

    rung_rows = []
    for rung in ["D0", "D1", "D2", "D3", "D4"]:
        mask = val_rung == rung
        n_pairs = int(mask.sum())
        dm = _drug_macro_metrics(
            drug_ids_val[mask], y_val_treats[mask], proba_m_seed0[mask], min_pairs=1
        )
        row = {
            "rung": rung,
            "n_pairs": n_pairs,
            "n_drugs_scored": dm["n_drugs_scored"],
            "drug_macro_auc_M": dm["drug_macro_auc"],
        }
        if small_mol_mask_by_ing is not None:
            sm_mask = mask & (small_mol_mask_by_ing[d_pos_val] >= 1)
            dm_sm = _drug_macro_metrics(
                drug_ids_val[sm_mask], y_val_treats[sm_mask], proba_m_seed0[sm_mask], min_pairs=1
            )
            row["drug_macro_auc_M_smallmol"] = dm_sm["drug_macro_auc"]
            row["n_drugs_scored_smallmol"] = dm_sm["n_drugs_scored"]
        rung_rows.append(row)
        log.info(
            "rung=%s n_pairs=%d n_drugs=%d drug_macro_auc_M=%.4f",
            rung,
            n_pairs,
            dm["n_drugs_scored"],
            dm["drug_macro_auc"],
        )
    by_rung_df = pd.DataFrame(rung_rows)
    by_rung_path = out / f"{exp_id}_by_rung.csv"
    by_rung_df.to_csv(by_rung_path, index=False)

    # unseen-family headline: rungs D1-D4 pooled (drug's primary target family, or lack
    # thereof, never seen sharing a gene with a train drug), same convention as exp08/exp12.
    unseen_family_mask = val_rung != "D0"
    dm_unseen = _drug_macro_metrics(
        drug_ids_val[unseen_family_mask],
        y_val_treats[unseen_family_mask],
        proba_m_seed0[unseen_family_mask],
        min_pairs=1,
    )
    drug_macro_auc_unseen_family_m = dm_unseen["drug_macro_auc"]

    # ---------------------------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    labels_fig = ["y_semmeddb_treats\n(mechanism-only, M)", "y_faers_signal\n(exp08 model B)"]
    treats_margin = design_mean("M", "drug_macro_auc") - float(floor_treats["floor_pc"])
    ax.bar(labels_fig, [treats_margin, 0.0443], color=["#4C72B0", "#DD8452"])
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_ylabel("drug_macro_auc margin over own floor")
    ax.set_title(
        "exp20: mechanism-only margin over floor, TREATS vs safety-side (exp08 nb_excess gain)"
    )
    fig.tight_layout()
    fig_path = out / f"{exp_id}_efficacy_vs_safety.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    results.commit()

    # --------------------------------------------------------------------- metrics dict
    metrics: dict[str, float] = {
        "drug_macro_auc_M": design_mean("M", "drug_macro_auc"),
        "drug_macro_auc_I": design_mean("I", "drug_macro_auc"),
        "drug_macro_auc_MI": design_mean("M+I", "drug_macro_auc"),
        "drug_macro_auc_M_neighbour": design_mean("M+neighbour", "drug_macro_auc"),
        "floor_pc_treats": float(floor_treats["floor_pc"]),
        "floor_degree_pc_treats": float(floor_treats["floor_degree_pc"]),
        "drug_macro_p10_M": design_mean("M", "drug_macro_p10"),
        "n_drugs_scored": int(design_df[design_df["design"] == "M"]["n_drugs_scored"].iloc[0]),
        "validate_average_precision": design_mean("M", "validate_average_precision"),
        "drug_macro_auc_unseen_family_M": drug_macro_auc_unseen_family_m,
        "causes_drug_macro_auc_M": float(
            causes_df[causes_df["design"] == "M"]["drug_macro_auc"].iloc[0]
        ),
        "causes_floor_pc": float(floor_causes["floor_pc"]),
        "causes_n_drugs_scored": int(
            causes_df[causes_df["design"] == "M"]["n_drugs_scored"].iloc[0]
        ),
        "seed_spread_drug_macro_auc": design_std("M", "drug_macro_auc"),
        "prevalence_validate": prevalence_val,
        "prevalence_train": prevalence_train,
        "train_cv_drug_macro_auc_M": cv_results.get("M", float("nan")),
        "train_cv_drug_macro_auc_I": cv_results.get("I", float("nan")),
        "train_cv_drug_macro_auc_MI": cv_results.get("M+I", float("nan")),
        "train_cv_drug_macro_auc_M_neighbour": cv_results.get("M+neighbour", float("nan")),
        "cv_skipped_count": float(len(cv_skipped)),
        "coverage_ge1_nb_treats": coverage_ge1_nb_treats,
        "nb_neighbour_gain": design_mean("M+neighbour", "drug_macro_auc")
        - design_mean("M", "drug_macro_auc"),
    }

    m_margin = metrics["drug_macro_auc_M"] - metrics["floor_pc_treats"]
    mi_gap = metrics["drug_macro_auc_MI"] - metrics["drug_macro_auc_M"]

    findings = (
        f"Floor (TREATS p_c lookup) = {metrics['floor_pc_treats']:.4f}, degree+p_c floor = "
        f"{metrics['floor_degree_pc_treats']:.4f}, prevalence(train)={prevalence_train:.4f}, "
        f"n_drugs_scored={metrics['n_drugs_scored']} (versus the safety side's 686 -- eligibility "
        f"at 0.48% prevalence requires only >=1 TREATS positive per drug here, so this count and "
        f"the safety side's are not directly comparable). "
        f"Design M (mechanism-only) drug_macro_auc={metrics['drug_macro_auc_M']:.4f}, margin over "
        f"floor_pc={m_margin:+.4f} (hypothesis threshold was >=0.03), seed spread (std over "
        f"{len(SEEDS)} seeds)={metrics['seed_spread_drug_macro_auc']:.4f}. "
        f"Pooled AP (continuity with exp05's 0.1275) = {metrics['validate_average_precision']:.4f} "
        f"against prevalence {prevalence_val:.4f} ({metrics['validate_average_precision'] / prevalence_val:.1f}x lift); "
        + (
            "exp05's pooled-AP claim SURVIVES translation to the per-drug metric (M clears its floor by "
            "the hypothesised margin). "
            if m_margin >= 0.03
            else "exp05's pooled-AP claim DOES NOT clearly survive translation to the per-drug metric "
            "(margin over floor_pc falls short of the +0.03 hypothesis threshold) -- the same pattern "
            "exp09 produced on the safety side, where a flattering pooled number collapsed once floors "
            "and the per-drug metric were applied. "
        )
        + f"M vs M+I gap = {mi_gap:+.4f} (I alone drug_macro_auc={metrics['drug_macro_auc_I']:.4f}); "
        + (
            "the mechanism-versus-indication gap survives on the per-drug metric. "
            if abs(mi_gap) > metrics["seed_spread_drug_macro_auc"]
            else "the mechanism-versus-indication gap does not clearly exceed the seed spread, so it "
            "is not claimed as real on this metric. "
        )
        + f"Neighbour feature (exp08's nb_excess_gene recomputed against TREATS, leave-one-drug-out): "
        f"M+neighbour drug_macro_auc={metrics['drug_macro_auc_M_neighbour']:.4f}, gain over M="
        f"{metrics['nb_neighbour_gain']:+.4f}, coverage (validate pairs with >=1 gene neighbour)="
        f"{coverage_ge1_nb_treats:.4f}. Reference: exp08's gain on the safety side was +0.0443 "
        f"(placebo-adjusted ~+0.035); "
        + (
            "the neighbour feature adds LESS here than on the safety side, as hypothesised -- "
            "consistent with indication being a property of the molecule's own target rather than "
            "of what other same-target drugs happened to be reported treating. "
            if metrics["nb_neighbour_gain"] < 0.0443
            else "the neighbour feature does NOT add less here than on the safety side -- the "
            "hypothesis's stated falsification-adjacent expectation was wrong, or TREATS-side "
            "same-target transport is stronger than predicted. "
        )
        + f"Secondary label y_semmeddb_causes (underpowered, 0.10% prevalence): M design drug_macro_auc="
        f"{metrics['causes_drug_macro_auc_M']:.4f} vs floor_pc={metrics['causes_floor_pc']:.4f} on "
        f"{metrics['causes_n_drugs_scored']} drugs scored -- indicative only, not decision-grade. "
        f"Transportability: drug_macro_auc_unseen_family_M (rungs D1-D4 pooled, i.e. drugs without a "
        f"train-shared gene) = {drug_macro_auc_unseen_family_m:.4f} vs full-population M="
        f"{metrics['drug_macro_auc_M']:.4f}; see {exp_id}_by_rung.csv for the full D0-D4 breakdown. "
        f"{small_mol_note} "
        f"Recommendation: {'Round 5 should move weight toward efficacy -- mechanism-only features clear their floor by a real margin on the primary metric.' if (m_margin >= 0.03 and abs(mi_gap) > metrics['seed_spread_drug_macro_auc']) else 'Efficacy does not yet clear the bar the safety side had to clear (floor margin and/or M-vs-M+I gap not established as real on drug_macro_auc); treat as inconclusive rather than a green light, pending a rerun with more eligible drugs or a stricter eligibility threshold.'}"
        + (f" Fallbacks taken: {'; '.join(fallbacks_taken)}." if fallbacks_taken else "")
    )

    artifacts = [
        str(designs_path),
        str(assignment_path),
        str(floor_table_path),
        str(seed_spread_path),
        str(by_rung_path),
        str(causes_path),
        str(importance_path),
        str(fig_path),
    ]

    log.info("metrics: %s", metrics)
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
        agent="exp20",
        title="Efficacy label on drug_macro_auc, with floors and the neighbour feature",
        hypothesis=(
            "Mechanism-only features exceed the TREATS p_c floor by >=0.03 drug_macro_auc "
            "and the mechanism-versus-indication gap survives on the per-drug metric, "
            "while the neighbour feature adds less here than on the safety side."
        ),
        approach=(
            "Reuse exp05's audited M/I column assignment; floors.py for the TREATS label "
            "reported before any model number; four designs including exp08's neighbour "
            "feature recomputed against TREATS with leave-one-drug-out; 5 seeds, grouped "
            "3-fold CV scored on drug_macro_auc; transportability by exp12's rungs."
        ),
        label="y_semmeddb_treats",
        features=[
            "mechanism_block",
            "target_biology",
            "indication_encoding",
            "nb_excess_gene",
            "p_c",
            "degree",
        ],
        split="train/validate, grouped by primary target gene",
        notes="Re-measures exp05 (exp_20260922_7c2e17) on the primary metric with floors.",
    )

    metrics = run.remote(exp)
    print(metrics)

    if metrics.get("precondition_failed"):
        ln.complete(
            exp,
            metrics=metrics,
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
            "If M's margin over floor_pc_treats is real and exceeds the safety side's "
            "own margins, Round 5 should move the project's centre of gravity toward "
            "efficacy per this experiment's recommendation. If not, treat this as a "
            "decision-grade negative alongside exp09's safety-side lesson. "
            f"Fallbacks taken: {'; '.join(fallbacks) if fallbacks else 'none'}."
        ),
    )


if __name__ == "__main__":
    pass
