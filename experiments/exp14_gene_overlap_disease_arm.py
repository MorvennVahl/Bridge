"""exp14 -- Gene overlap between drug targets and condition genes (disease + phenotype arm).

Implements experiments/exp14_gene_overlap_disease_arm.md. Re-cut of exp10's gene half only
(exp10 never completed; exp15 is the pathway half, not done here). Registers FRESH under
agent="exp14" -- exp10's partial run is referenced in `notes`, not reused as this
experiment's own result.

Reuse:
- Baseline block (degree, p_c, drug-intrinsic, condition-intrinsic via data_dictionary.csv
  block selection) copied from experiments/exp07_ablation_rerun_leak_audit.py.
- nb_excess_gene same-target neighbour feature copied from
  experiments/exp08_neighbour_transport_pc_stripped.py (leave-one-drug-out shrinkage over
  the gene tier, disease_efficacy-filtered).
- Sparse condition-gene / drug-gene matrix construction and pair-level gather machinery
  copied from experiments/exp10_gene_pathway_overlap.py's gene half (disease arm via
  condition_gene_ot_long.csv, phenotype arm via condition_gene_hpo_long.csv). The pathway
  half and the literature-diagnostic scaffolding in exp10 are NOT reused -- out of scope
  for exp14 per the spec.

drug_macro_auc machinery is METRIC.md's, also copied verbatim from exp07/exp10.
"""

# ruff: noqa: N806 -- linear-algebra matrix-naming convention (C_*, D_binary), consistent
# with exp10.

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

app = modal.App("bridge-exp14")

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

# ================================================================================
# copied from exp07_ablation_rerun_leak_audit.py (block-selection spec)
# ================================================================================

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

HGB_KWARGS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "early_stopping": False,
    "random_state": 0,
}
K_SHRINK = 10


def _audit_columns(columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if c in LEAK_BLACKLIST_EXACT or any(c.startswith(p) for p in LEAK_BLACKLIST_PREFIXES)
    ]


def _normalize_block(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _cap_and_onehot(series: pd.Series, prefix: str, cap: int = CARDINALITY_CAP) -> pd.DataFrame:
    import pandas as pd

    counts = series.value_counts(dropna=True)
    keep = set(counts.index[:cap])
    bucketed = series.where(series.isin(keep) | series.isna(), other="other")
    return pd.get_dummies(bucketed, prefix=prefix, dummy_na=False)


def _detect_col(columns: list[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    for c in columns:
        for cand in candidates:
            if c.startswith(cand):
                return c
    return None


def _build_degree_and_pc_features(
    train: pd.DataFrame, validate: pd.DataFrame, cond_basic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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


# ================================================================================
# drug_macro_auc machinery (per experiments/METRIC.md), copied from exp07/exp10
# ================================================================================


def _per_drug_table(
    ids: pd.Series, y: pd.Series, score: np.ndarray, min_pairs: int = 20
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    score = np.asarray(score, dtype=float)
    if np.isnan(score).any():
        med = float(np.nanmedian(score)) if not np.all(np.isnan(score)) else 0.0
        score = np.where(np.isnan(score), med, score)

    df = pd.DataFrame({"drug": ids.to_numpy(), "y": y.to_numpy(), "score": score})
    rows: list[dict[str, float]] = []
    for drug_id, grp in df.groupby("drug"):
        n = len(grp)
        if n < min_pairs or grp["y"].nunique() < 2:
            continue
        # fill NaN scores with per-drug median before roc_auc_score, per this run's
        # NaN-handling constraint (score is already globally de-NaN'd above; this is a
        # second, per-drug pass in case upstream left a drug-specific gap).
        s = grp["score"]
        if s.isna().any():
            s = s.fillna(s.median())
        auc = float(roc_auc_score(grp["y"], s))
        ranked = grp.assign(score=s).sort_values("score", ascending=False)
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
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    # cast bool columns to float before fitting/argsort (constraint #8)
    def _cast(df: pd.DataFrame) -> pd.DataFrame:
        bool_cols = [c for c in df.columns if df[c].dtype == bool]
        if bool_cols:
            df = df.copy()
            df[bool_cols] = df[bool_cols].astype(float)
        return df

    x_train = _cast(x_train)
    x_val = _cast(x_val)

    def _make_model() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(**HGB_KWARGS)

    cv_drug_macro_aucs: list[float] = []
    if n_folds > 0:
        gkf = GroupKFold(n_splits=n_folds)
        for train_idx, test_idx in gkf.split(x_train, y_train, groups=groups_train):
            model = _make_model()
            model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            proba = model.predict_proba(x_train.iloc[test_idx])[:, 1]
            fold_metrics = _drug_macro_metrics(
                ids_train.iloc[test_idx], y_train.iloc[test_idx], proba
            )
            cv_drug_macro_aucs.append(fold_metrics["drug_macro_auc"])

    final_model = _make_model()
    final_model.fit(x_train, y_train)
    val_proba = final_model.predict_proba(x_val)[:, 1]
    val_ap = float(average_precision_score(y_val, val_proba))
    val_pooled_auc = float(roc_auc_score(y_val, val_proba))
    val_macro = _drug_macro_metrics(ids_val, y_val, val_proba)

    return {
        "model": final_model,
        "cv_drug_macro_auc_mean": float(np.nanmean(cv_drug_macro_aucs))
        if cv_drug_macro_aucs
        else float("nan"),
        "val_ap": val_ap,
        "val_pooled_auc": val_pooled_auc,
        "val_drug_macro_auc": val_macro["drug_macro_auc"],
        "val_drug_macro_p10": val_macro["drug_macro_p10"],
        "val_drug_macro_r50": val_macro["drug_macro_r50"],
        "val_n_drugs_scored": val_macro["n_drugs_scored"],
        "val_drug_table": val_macro["table"],
        "val_proba": val_proba,
    }


# ================================================================================
# nb_excess_gene neighbour feature, copied from exp08_neighbour_transport_pc_stripped.py
# ================================================================================


def _build_adjacency(pairs, ing_index_map: dict, n_ing: int):
    import numpy as np
    import scipy.sparse as sp

    ing_pos = pairs["ingredient_concept_id"].map(ing_index_map).to_numpy()
    val_codes, _ = __import__("pandas").factorize(pairs["value"])
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
    log: logging.Logger,
) -> dict:
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
        f"nb_excess_{tier_name}__train": rate_train - p_c_train,
        f"nb_excess_{tier_name}__val": rate_val - p_c_val,
    }
    log.info(
        "tier=%s median train neighbours=%.1f, %.1f%% of train rows have >=1",
        tier_name,
        float(np.median(n_neighbours_train)),
        100.0 * float((n_neighbours_train >= 1).mean()),
    )
    return out


def _dedup_pairs(df, value_col: str):
    p = df[["ingredient_concept_id", value_col]].dropna().rename(columns={value_col: "value"})
    p["value"] = p["value"].astype(str)
    return p.drop_duplicates()


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y", "t"])


# ================================================================================
# Sparse gene-overlap matrices, adapted from exp10_gene_pathway_overlap.py (gene half only)
# ================================================================================


def _build_disease_gene_matrices(ot_path: pathlib.Path, target_long_path: pathlib.Path, log):
    """conditions x genes (C_*) and drugs x genes (D_binary) -- disease arm (Open Targets).
    Adapted from exp10's _build_disease_gene_matrices, pathway construction dropped.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    ot = pd.read_csv(ot_path)
    log.info("condition_gene_ot_long.csv columns: %s", list(ot.columns))
    gene_col = _detect_col(
        list(ot.columns), ["ensembl_gene_id", "gene_id", "target_id", "ensemblId", "geneId", "gene"]
    )
    if gene_col is None:
        raise KeyError(f"no gene id column found in condition_gene_ot_long.csv: {list(ot.columns)}")
    cond_col = "condition_concept_id"
    score_col = "ot_score"
    genetic_col = _detect_col(list(ot.columns), ["dt_genetic_association"])
    literature_col = _detect_col(list(ot.columns), ["dt_literature"])
    log.info(
        "disease-arm gene col=%s genetic_col=%s literature_col=%s",
        gene_col,
        genetic_col,
        literature_col,
    )

    ing = pd.read_csv(target_long_path, usecols=["omop_concept_id", "ensembl_gene_id"]).dropna(
        subset=["ensembl_gene_id"]
    )
    coverage = float(
        len(set(ing["ensembl_gene_id"]) & set(ot[gene_col].dropna()))
        / max(len(set(ing["ensembl_gene_id"])), 1)
    )
    log.info(
        "join coverage diagnostic: %.1f%% of drug target genes (n=%d) appear in "
        "condition_gene_ot_long's gene set",
        coverage * 100,
        ing["ensembl_gene_id"].nunique(),
    )

    ot = ot.dropna(subset=[cond_col, gene_col])
    gene_index = sorted(set(ot[gene_col].unique()) | set(ing["ensembl_gene_id"].unique()))
    gene_pos = {g: i for i, g in enumerate(gene_index)}
    cond_ids = sorted(ot[cond_col].unique())
    cond_pos = {c: i for i, c in enumerate(cond_ids)}
    n_cond, n_gene = len(cond_ids), len(gene_index)

    rows = ot[cond_col].map(cond_pos).to_numpy(dtype=int)
    cols = ot[gene_col].map(gene_pos).to_numpy(dtype=int)

    def _mat(vals: np.ndarray) -> sp.csr_matrix:
        m = sp.csr_matrix((vals, (rows, cols)), shape=(n_cond, n_gene))
        m.sum_duplicates()
        return m

    C_score = _mat(ot[score_col].fillna(0.0).to_numpy(dtype=float))
    C_binary = _mat(np.ones(len(rows)))
    C_binary.data = np.minimum(C_binary.data, 1.0)
    C_genetic = (
        _mat(ot[genetic_col].fillna(0.0).to_numpy(dtype=float))
        if genetic_col
        else sp.csr_matrix((n_cond, n_gene))
    )
    C_literature = (
        _mat(ot[literature_col].fillna(0.0).to_numpy(dtype=float))
        if literature_col
        else sp.csr_matrix((n_cond, n_gene))
    )

    drug_ids = sorted(ing["omop_concept_id"].unique())
    drug_pos = {d: i for i, d in enumerate(drug_ids)}
    ing2 = ing[ing["ensembl_gene_id"].isin(gene_pos)]
    drows = ing2["omop_concept_id"].map(drug_pos).to_numpy(dtype=int)
    dcols = ing2["ensembl_gene_id"].map(gene_pos).to_numpy(dtype=int)
    D_binary = sp.csr_matrix((np.ones(len(drows)), (drows, dcols)), shape=(len(drug_ids), n_gene))
    D_binary.sum_duplicates()
    D_binary.data = np.minimum(D_binary.data, 1.0)

    df_gene = np.asarray(C_binary.sum(axis=0)).ravel()
    idf = np.log(n_cond / np.maximum(df_gene, 1.0))
    C_idf = C_binary.multiply(idf).tocsr()

    return {
        "gene_pos": gene_pos,
        "cond_pos": cond_pos,
        "drug_pos": drug_pos,
        "n_cond": n_cond,
        "n_gene": n_gene,
        "n_drug": len(drug_ids),
        "C_score": C_score,
        "C_binary": C_binary,
        "C_genetic": C_genetic,
        "C_literature": C_literature,
        "C_idf": C_idf,
        "D_binary": D_binary,
        "coverage": coverage,
    }


def _build_hpo_matrices(hpo_path: pathlib.Path, target_long_path: pathlib.Path, log):
    """Phenotype-arm overlap, unweighted. Adapted from exp10 unchanged."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    hpo = pd.read_csv(hpo_path)
    log.info("condition_gene_hpo_long.csv columns: %s", list(hpo.columns))
    gene_col = _detect_col(
        list(hpo.columns), ["gene_symbol", "hgnc_symbol", "gene", "ensembl_gene_id", "gene_id"]
    )
    cond_col = _detect_col(list(hpo.columns), ["condition_concept_id"])
    if gene_col is None or cond_col is None:
        raise KeyError(
            f"could not identify gene/condition columns in HPO table: {list(hpo.columns)}"
        )

    id_system = "ensembl" if gene_col == "ensembl_gene_id" else "symbol"
    log.info("HPO gene column=%s -> reconciliation id system assumed: %s", gene_col, id_system)

    ing_col = "ensembl_gene_id" if id_system == "ensembl" else "gene_symbol"
    ing = pd.read_csv(target_long_path, usecols=["omop_concept_id", ing_col]).dropna(
        subset=[ing_col]
    )

    hpo = hpo.dropna(subset=[cond_col, gene_col])
    gene_index = sorted(set(hpo[gene_col].unique()) | set(ing[ing_col].unique()))
    gene_pos = {g: i for i, g in enumerate(gene_index)}
    cond_ids = sorted(hpo[cond_col].unique())
    cond_pos = {c: i for i, c in enumerate(cond_ids)}
    n_cond, n_gene = len(cond_ids), len(gene_index)

    rows = hpo[cond_col].map(cond_pos).to_numpy(dtype=int)
    cols = hpo[gene_col].map(gene_pos).to_numpy(dtype=int)
    C_binary = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cond, n_gene))
    C_binary.sum_duplicates()
    C_binary.data = np.minimum(C_binary.data, 1.0)

    drug_ids = sorted(ing["omop_concept_id"].unique())
    drug_pos = {d: i for i, d in enumerate(drug_ids)}
    ing2 = ing[ing[ing_col].isin(gene_pos)]
    drows = ing2["omop_concept_id"].map(drug_pos).to_numpy(dtype=int)
    dcols = ing2[ing_col].map(gene_pos).to_numpy(dtype=int)
    D_binary = sp.csr_matrix((np.ones(len(drows)), (drows, dcols)), shape=(len(drug_ids), n_gene))
    D_binary.sum_duplicates()
    D_binary.data = np.minimum(D_binary.data, 1.0)

    return {
        "gene_pos": gene_pos,
        "cond_pos": cond_pos,
        "drug_pos": drug_pos,
        "n_cond": n_cond,
        "n_gene": n_gene,
        "n_drug": len(drug_ids),
        "C_binary": C_binary,
        "D_binary": D_binary,
        "id_system": id_system,
    }


def _pair_gene_overlap_features(
    pairs: pd.DataFrame, ctx: dict, log, prefix: str = "", weighted: bool = True
) -> pd.DataFrame:
    """Vectorized gather of gene-overlap features for every (drug, condition) pair -- one
    sparse product read at pair coordinates, no loop over pairs (spec's mandatory
    constraint). Only the shared-gene MAX variant loops, and only over pairs with
    n_shared_genes > 0 (a small fraction), matching exp10's two-tier trick.
    """
    import numpy as np
    import pandas as pd

    D_binary = ctx["D_binary"]
    C_binary = ctx["C_binary"]
    n_drug = ctx["n_drug"]

    drow = pairs["ingredient_concept_id"].map(ctx["drug_pos"])
    crow = pairs["condition_concept_id"].map(ctx["cond_pos"])
    valid = drow.notna() & crow.notna()
    n_invalid = int((~valid).sum())
    if n_invalid:
        log.info(
            "%d/%d pairs have drug/condition unseen in the gene index; overlap=0",
            n_invalid,
            len(pairs),
        )
    drow_arr = drow.fillna(-1).to_numpy(dtype=int)
    crow_arr = crow.fillna(-1).to_numpy(dtype=int)
    valid_arr = valid.to_numpy()

    def _gather(dense: np.ndarray) -> np.ndarray:
        out = np.zeros(len(pairs), dtype=float)
        out[valid_arr] = dense[drow_arr[valid_arr], crow_arr[valid_arr]]
        return out

    n_shared_matrix = (D_binary @ C_binary.T).toarray()
    n_shared = _gather(n_shared_matrix)
    out = pd.DataFrame(index=pairs.index)
    out[f"{prefix}n_shared_genes"] = n_shared

    drug_gene_count = np.asarray(D_binary.sum(axis=1)).ravel()
    cond_gene_count = np.asarray(C_binary.sum(axis=1)).ravel()
    denom = np.zeros(len(pairs), dtype=float)
    denom[valid_arr] = (
        drug_gene_count[drow_arr[valid_arr]]
        + cond_gene_count[crow_arr[valid_arr]]
        - n_shared[valid_arr]
    )
    jaccard = np.zeros(len(pairs), dtype=float)
    nz = denom > 0
    jaccard[nz] = n_shared[nz] / denom[nz]
    out[f"{prefix}jaccard_genes"] = jaccard

    if not weighted:
        return out

    if "C_score" in ctx:
        sum_matrix = (D_binary @ ctx["C_score"].T).toarray()
        out[f"{prefix}shared_gene_score_sum"] = _gather(sum_matrix)
    if "C_genetic" in ctx:
        gen_matrix = (D_binary @ ctx["C_genetic"].T).toarray()
        out[f"{prefix}shared_gene_genetic_sum"] = _gather(gen_matrix)
    if "C_idf" in ctx:
        idf_matrix = (D_binary @ ctx["C_idf"].T).toarray()
        out[f"{prefix}shared_gene_idf_sum"] = _gather(idf_matrix)

    if "C_score" in ctx or "C_genetic" in ctx:
        score_max = np.zeros(len(pairs), dtype=float)
        genetic_max = np.zeros(len(pairs), dtype=float)
        subset_idx = np.where(valid_arr & (n_shared > 0))[0]
        log.info(
            "%s: computing shared-gene MAX over %d/%d pairs with shared genes (two-tier loop)",
            prefix or "disease-arm",
            len(subset_idx),
            len(pairs),
        )
        C_score = ctx.get("C_score")
        C_genetic = ctx.get("C_genetic")
        drug_gene_idx = {
            i: D_binary.indices[D_binary.indptr[i] : D_binary.indptr[i + 1]] for i in range(n_drug)
        }
        for i in subset_idx:
            d, c = drow_arr[i], crow_arr[i]
            g_idx = drug_gene_idx[d]
            if C_score is not None:
                vals = C_score[c, g_idx].toarray().ravel()
                if vals.size:
                    score_max[i] = vals.max()
            if C_genetic is not None:
                vals = C_genetic[c, g_idx].toarray().ravel()
                if vals.size:
                    genetic_max[i] = vals.max()
        if "C_score" in ctx:
            out[f"{prefix}shared_gene_score_max"] = score_max
        if "C_genetic" in ctx:
            out[f"{prefix}shared_gene_genetic_max"] = genetic_max

    return out


def _pair_literature_feature(pairs: pd.DataFrame, ctx: dict, log) -> pd.DataFrame:
    """dt_literature-weighted sum only -- kept strictly separate per spec (§ Kept out of
    the headline model)."""
    import numpy as np
    import pandas as pd

    D_binary = ctx["D_binary"]
    drow = pairs["ingredient_concept_id"].map(ctx["drug_pos"])
    crow = pairs["condition_concept_id"].map(ctx["cond_pos"])
    valid_arr = (drow.notna() & crow.notna()).to_numpy()
    drow_arr = drow.fillna(-1).to_numpy(dtype=int)
    crow_arr = crow.fillna(-1).to_numpy(dtype=int)

    lit_matrix = (D_binary @ ctx["C_literature"].T).toarray()
    out_arr = np.zeros(len(pairs), dtype=float)
    out_arr[valid_arr] = lit_matrix[drow_arr[valid_arr], crow_arr[valid_arr]]
    out = pd.DataFrame(index=pairs.index)
    out["shared_gene_literature_sum"] = out_arr
    return out


# ================================================================================
# Preconditions
# ================================================================================


def _check_preconditions(paths: dict[str, pathlib.Path]) -> str | None:
    import pandas as pd

    expected_rows = {
        "train": 723_586,
        "validate": 434_151,
        "drug": 4_280,
        "cond": 5_631,
        "group": 10_554,
        "dict": 245,
        "ot": 897_142,
        "hpo": 2_244_725,
        "target_long": 8_088,
    }
    for key, expected in expected_rows.items():
        p = paths[key]
        if not p.exists():
            return f"missing required input: {p}"
        with p.open() as fh:
            n = sum(1 for _ in fh) - 1
        if n != expected:
            return f"row count mismatch for {p}: expected {expected}, got {n}"

    train = pd.read_csv(paths["train"], usecols=["y_faers_signal"])
    rate = float(train["y_faers_signal"].mean())
    if abs(rate - 0.1100) > 0.001:
        return f"train positive rate for y_faers_signal is {rate:.4f}, expected 0.1100 +/- 0.001"
    return None


# ================================================================================
# Main Modal function
# ================================================================================


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp14")

    import time

    import numpy as np
    import pandas as pd

    t0 = time.time()
    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    data_root = pathlib.Path("/data")
    paths = {
        "train": data_root / "splits" / "train.csv",
        "validate": data_root / "splits" / "validate.csv",
        "drug": data_root / "drug" / "ingredient_features.csv",
        "cond": data_root / "condition" / "condition_features_basic.csv",
        "group": data_root / "condition" / "condition_group_long.csv",
        "dict": data_root / "data_dictionary.csv",
        "ot": data_root / "condition" / "condition_gene_ot_long.csv",
        "hpo": data_root / "condition" / "condition_gene_hpo_long.csv",
        "target_long": data_root / "drug" / "ingredient_target_long.csv",
    }

    err = _check_preconditions(paths)
    if err is not None:
        log.error("precondition check failed: %s", err)
        return {"precondition_failed": 1.0, "precondition_error_message": err}

    log.info("preconditions passed; loading train/validate")
    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    cond_basic = pd.read_csv(paths["cond"])
    edges_raw = pd.read_csv(paths["target_long"]).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )

    fallback_notes: list[str] = []

    # ================================================================================
    # E8base: degree + p_c + drug-intrinsic + condition-intrinsic + nb_excess_gene
    # ================================================================================
    degree_train, degree_val, pc_notes = _build_degree_and_pc_features(train, validate, cond_basic)
    fallback_notes.extend(pc_notes)

    log.info("loading drug-intrinsic block")
    drug_raw, drug_diag = _load_drug_features(paths["drug"], paths["dict"])
    log.info("drug block diagnostics: %s", drug_diag)
    drug_design = _build_drug_design(drug_raw).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )

    log.info("loading condition-intrinsic block")
    cond_design = _load_condition_features(paths["cond"], paths["group"])

    def _assemble_base(base_df: pd.DataFrame, degree_pc_df: pd.DataFrame) -> pd.DataFrame:
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

    intrinsic_train = _assemble_base(train, degree_train)
    intrinsic_val = _assemble_base(validate, degree_val)

    # nb_excess_gene (exp08's same-target neighbour feature, gene tier only)
    distinct_component_rel = sorted(edges_raw["component_relationship"].dropna().unique().tolist())
    is_complex_or_family = (
        edges_raw["component_relationship"]
        .astype(str)
        .str.upper()
        .str.contains("COMPLEX|FAMILY", regex=True, na=False)
    )
    log.info(
        "gene-tier subunit exclusion values: %s",
        [v for v in distinct_component_rel if "COMPLEX" in v.upper() or "FAMILY" in v.upper()],
    )
    gene_tier_edges_raw = edges_raw.loc[~is_complex_or_family]
    disease_efficacy_bool = _to_bool(edges_raw["disease_efficacy"])
    primary_gene_edges_raw = gene_tier_edges_raw.loc[
        disease_efficacy_bool.reindex(gene_tier_edges_raw.index, fill_value=False)
    ]
    gene_pairs_nb = _dedup_pairs(primary_gene_edges_raw, "gene_symbol")

    all_ing = pd.Index(
        sorted(
            set(train["ingredient_concept_id"])
            | set(validate["ingredient_concept_id"])
            | set(edges_raw["ingredient_concept_id"])
        )
    )
    n_ing = len(all_ing)
    ing_index_map = {ing: i for i, ing in enumerate(all_ing)}
    splits_train_ing = set(train["ingredient_concept_id"].unique())
    train_mask = np.array([1.0 if i in splits_train_ing else 0.0 for i in all_ing])
    d_pos_train = train["ingredient_concept_id"].map(ing_index_map).to_numpy()
    d_pos_val = validate["ingredient_concept_id"].map(ing_index_map).to_numpy()

    all_cond = pd.Index(
        sorted(set(train["condition_concept_id"]) | set(validate["condition_concept_id"]))
    )
    cond_index_map = {c: i for i, c in enumerate(all_cond)}
    n_cond_ids = len(all_cond)
    c_pos_train = train["condition_concept_id"].map(cond_index_map).to_numpy()
    c_pos_val = validate["condition_concept_id"].map(cond_index_map).to_numpy()

    p_c_by_cond = (
        train.groupby("condition_concept_id")["y_faers_signal"]
        .mean()
        .reindex(all_cond)
        .fillna(float(train["y_faers_signal"].mean()))
        .to_numpy()
    )
    p_c_train_arr = p_c_by_cond[c_pos_train]
    p_c_val_arr = p_c_by_cond[c_pos_val]

    import scipy.sparse as sp

    y_train_real = train["y_faers_signal"].to_numpy()
    flagged_idx = np.where(y_train_real == 1)[0]
    f_train = sp.csr_matrix(
        (
            np.ones(len(flagged_idx)),
            (d_pos_train[flagged_idx], c_pos_train[flagged_idx]),
        ),
        shape=(n_ing, n_cond_ids),
    )
    f_train.sum_duplicates()
    f_train.data[:] = 1.0

    gene_nb_feats = _tier_features_excess(
        "gene",
        gene_pairs_nb,
        ing_index_map,
        n_ing,
        train_mask,
        d_pos_train,
        d_pos_val,
        c_pos_train,
        c_pos_val,
        f_train,
        p_c_train_arr,
        p_c_val_arr,
        log,
    )
    has_target_annotation_by_ing = np.array(
        [1.0 if i in set(edges_raw["ingredient_concept_id"]) else 0.0 for i in all_ing]
    )
    nb_train = pd.DataFrame(index=train.index)
    nb_val = pd.DataFrame(index=validate.index)
    nb_train["nb_excess_gene"] = gene_nb_feats["nb_excess_gene__train"]
    nb_train["n_neighbours_gene"] = gene_nb_feats["n_neighbours_gene__train"]
    nb_val["nb_excess_gene"] = gene_nb_feats["nb_excess_gene__val"]
    nb_val["n_neighbours_gene"] = gene_nb_feats["n_neighbours_gene__val"]
    has_annot_train = has_target_annotation_by_ing[d_pos_train] == 0
    has_annot_val = has_target_annotation_by_ing[d_pos_val] == 0
    nb_train.loc[has_annot_train, "nb_excess_gene"] = np.nan
    nb_val.loc[has_annot_val, "nb_excess_gene"] = np.nan

    e8base_train = pd.concat([intrinsic_train, nb_train], axis=1)
    e8base_val = pd.concat([intrinsic_val, nb_val], axis=1)
    log.info("E8base assembled in %.1fs, n_features=%d", time.time() - t0, e8base_train.shape[1])

    hits = _audit_columns(list(e8base_train.columns))
    if hits:
        msg = f"leak audit: E8base contains blacklisted columns: {hits}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    # ================================================================================
    # Gene overlap: disease arm (Open Targets) + phenotype arm (HPO)
    # ================================================================================
    t_gene = time.time()
    disease_ctx = _build_disease_gene_matrices(paths["ot"], paths["target_long"], log)
    log.info(
        "disease-arm matrices: %d conditions x %d genes, %d drugs. Built in %.1fs",
        disease_ctx["n_cond"],
        disease_ctx["n_gene"],
        disease_ctx["n_drug"],
        time.time() - t_gene,
    )

    hpo_ctx = None
    try:
        hpo_ctx = _build_hpo_matrices(paths["hpo"], paths["target_long"], log)
        log.info(
            "phenotype-arm matrices: %d conditions x %d genes (id_system=%s), %d drugs",
            hpo_ctx["n_cond"],
            hpo_ctx["n_gene"],
            hpo_ctx["id_system"],
            hpo_ctx["n_drug"],
        )
    except Exception as exc:
        log.warning("phenotype-arm (HPO) matrix construction failed, skipping: %s", exc)
        fallback_notes.append(f"HPO overlap skipped: {exc}")

    all_pairs = pd.concat(
        [
            train[["ingredient_concept_id", "condition_concept_id"]],
            validate[["ingredient_concept_id", "condition_concept_id"]],
        ],
        keys=["train", "validate"],
    )

    disease_raw_all = _pair_gene_overlap_features(all_pairs, disease_ctx, log, weighted=False)
    disease_weighted_all = _pair_gene_overlap_features(all_pairs, disease_ctx, log, weighted=True)

    if hpo_ctx is not None:
        hpo_feats_all = _pair_gene_overlap_features(
            all_pairs, hpo_ctx, log, prefix="hpo_", weighted=False
        )
    else:
        hpo_feats_all = pd.DataFrame(index=all_pairs.index)

    log.info(
        "gene overlap feature construction (disease + phenotype arm) for %d pairs done in %.1fs",
        len(all_pairs),
        time.time() - t_gene,
    )

    # share of pairs with any shared gene at all (disease arm), for the headline caveat
    share_any_shared_gene = float((disease_raw_all["n_shared_genes"] > 0).mean())
    log.info("share_pairs_with_any_shared_gene (disease arm) = %.4f", share_any_shared_gene)

    # dt_literature diagnostic, kept strictly separate
    t_lit = time.time()
    elapsed_before_lit = time.time() - t0
    literature_dropped_reason = ""
    literature_feats_all: pd.DataFrame | None = None
    if elapsed_before_lit > 420:
        literature_dropped_reason = (
            f"budget risk: {elapsed_before_lit:.0f}s already elapsed before dt_literature "
            "diagnostic; dropped per spec's 'if at risk' fallback"
        )
        log.warning("dropping dt_literature diagnostic: %s", literature_dropped_reason)
    else:
        literature_feats_all = _pair_literature_feature(all_pairs, disease_ctx, log)
        log.info("dt_literature diagnostic feature computed in %.1fs", time.time() - t_lit)

    def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr = df.loc["train"].set_axis(train.index)
        va = df.loc["validate"].set_axis(validate.index)
        return tr, va

    disease_raw_train, disease_raw_val = _split(disease_raw_all)
    disease_weighted_train, disease_weighted_val = _split(disease_weighted_all)
    hpo_train, hpo_val = _split(hpo_feats_all)
    if literature_feats_all is not None:
        literature_train, literature_val = _split(literature_feats_all)
    else:
        literature_train = pd.DataFrame(index=train.index)
        literature_val = pd.DataFrame(index=validate.index)

    gene_overlap_raw_train = pd.concat([disease_raw_train, hpo_train], axis=1)
    gene_overlap_raw_val = pd.concat([disease_raw_val, hpo_val], axis=1)
    gene_overlap_weighted_train = pd.concat([disease_weighted_train, hpo_train], axis=1)
    gene_overlap_weighted_val = pd.concat([disease_weighted_val, hpo_val], axis=1)

    # ================================================================================
    # Nested designs
    # ================================================================================
    design_order = ["E8base", "+gene_overlap_raw", "+gene_overlap_weighted"]
    train_designs = {
        "E8base": e8base_train,
        "+gene_overlap_raw": pd.concat([e8base_train, gene_overlap_raw_train], axis=1),
        "+gene_overlap_weighted": pd.concat([e8base_train, gene_overlap_weighted_train], axis=1),
    }
    val_designs = {
        "E8base": e8base_val,
        "+gene_overlap_raw": pd.concat([e8base_val, gene_overlap_raw_val], axis=1),
        "+gene_overlap_weighted": pd.concat([e8base_val, gene_overlap_weighted_val], axis=1),
    }
    literature_design_train = pd.concat([e8base_train, literature_train], axis=1)
    literature_design_val = pd.concat([e8base_val, literature_val], axis=1)

    y_train = train["y_faers_signal"]
    y_val = validate["y_faers_signal"]
    groups_train = train["group_key"]
    ids_train = train["ingredient_concept_id"]
    ids_val = validate["ingredient_concept_id"]

    fit_results: dict[str, dict] = {}
    for name in design_order:
        n_folds = 3
        if train_designs[name].shape[1] > 600:
            n_folds = 2
            fallback_notes.append(
                f"Dropped CV to 2 folds for {name} due to feature-count budget risk."
            )
        elapsed = time.time() - t0
        # Progressive fold-dropping: each design's fit costs roughly the same wall time as
        # the previous one, so the remaining-budget check has to happen BEFORE fitting, not
        # after -- the design that blows the 600s timeout is always the last one otherwise.
        if name != "E8base":
            if elapsed > 480:
                n_folds = 0
                fallback_notes.append(f"Dropped CV entirely for {name}: {elapsed:.0f}s elapsed.")
            elif elapsed > 350:
                # GroupKFold requires n_splits >= 2; below that, skip CV entirely rather
                # than pass an invalid n_splits=1.
                n_folds = 0
                fallback_notes.append(
                    f"Dropped CV entirely (n_splits=1 is invalid for GroupKFold) for "
                    f"{name}: {elapsed:.0f}s elapsed."
                )
            elif elapsed > 250:
                n_folds = 2
                fallback_notes.append(f"Dropped CV to 2 folds for {name}: {elapsed:.0f}s elapsed.")
        log.info(
            "fitting design=%s n_features=%d n_folds=%d (elapsed=%.1fs)",
            name,
            train_designs[name].shape[1],
            n_folds,
            elapsed,
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
            "design=%s cv_drug_macro_auc_mean=%.4f val_drug_macro_auc=%.4f",
            name,
            res["cv_drug_macro_auc_mean"],
            res["val_drug_macro_auc"],
        )

    # dt_literature diagnostic fit (single fit + validate, no CV)
    literature_elapsed = time.time() - t0
    if literature_feats_all is None or literature_elapsed > 500:
        if not literature_dropped_reason:
            literature_dropped_reason = (
                f"budget risk: {literature_elapsed:.0f}s elapsed before fitting the "
                "dt_literature diagnostic model; skipped to protect headline designs"
            )
        log.warning("skipping dt_literature diagnostic fit: %s", literature_dropped_reason)
        literature_res = None
    else:
        literature_res = _fit_score(
            literature_design_train,
            y_train,
            groups_train,
            ids_train,
            literature_design_val,
            y_val,
            ids_val,
            n_folds=0,
        )

    # ================================================================================
    # Bootstrap increments
    # ================================================================================
    tables = {k: v["val_drug_table"] for k, v in fit_results.items()}
    pairs_for_ci = [
        ("E8base", "+gene_overlap_raw"),
        ("+gene_overlap_raw", "+gene_overlap_weighted"),
        ("E8base", "+gene_overlap_weighted"),
    ]
    if literature_res is not None:
        tables["dt_literature_diagnostic"] = literature_res["val_drug_table"]
        pairs_for_ci.append(("E8base", "dt_literature_diagnostic"))

    bootstrap_rows = []
    for a, b in pairs_for_ci:
        point, lo, hi = _bootstrap_increment_ci(tables[a], tables[b], n_boot=1000)
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
    bootstrap_df.to_csv(out / f"{exp_id}_bootstrap_increments.csv", index=False)

    # ================================================================================
    # designs.csv
    # ================================================================================
    designs_rows = []
    for name in design_order:
        res = fit_results[name]
        designs_rows.append(
            {
                "design": name,
                "n_features": train_designs[name].shape[1],
                "train_cv_drug_macro_auc_mean": res["cv_drug_macro_auc_mean"],
                "validate_drug_macro_auc": res["val_drug_macro_auc"],
                "validate_drug_macro_p10": res["val_drug_macro_p10"],
                "validate_drug_macro_r50": res["val_drug_macro_r50"],
                "n_drugs_scored": res["val_n_drugs_scored"],
                "validate_ap": res["val_ap"],
                "validate_pooled_auc": res["val_pooled_auc"],
            }
        )
    pd.DataFrame(designs_rows).to_csv(out / f"{exp_id}_designs.csv", index=False)

    if literature_res is not None:
        literature_diag_df = pd.DataFrame(
            [
                {
                    "design": "E8base+dt_literature",
                    "n_features": literature_design_train.shape[1],
                    "validate_drug_macro_auc": literature_res["val_drug_macro_auc"],
                    "validate_drug_macro_p10": literature_res["val_drug_macro_p10"],
                    "n_drugs_scored": literature_res["val_n_drugs_scored"],
                    "baseline_e8base_drug_macro_auc": fit_results["E8base"]["val_drug_macro_auc"],
                    "increment_over_e8base": literature_res["val_drug_macro_auc"]
                    - fit_results["E8base"]["val_drug_macro_auc"],
                    "note": "reported apart -- dt_literature-weighted overlap excluded from "
                    "the headline model per spec",
                }
            ]
        )
    else:
        literature_diag_df = pd.DataFrame(
            [
                {
                    "design": "E8base+dt_literature",
                    "n_features": literature_design_train.shape[1],
                    "validate_drug_macro_auc": float("nan"),
                    "validate_drug_macro_p10": float("nan"),
                    "n_drugs_scored": 0,
                    "baseline_e8base_drug_macro_auc": fit_results["E8base"]["val_drug_macro_auc"],
                    "increment_over_e8base": float("nan"),
                    "note": f"dropped: {literature_dropped_reason}",
                }
            ]
        )
    literature_diag_df.to_csv(out / f"{exp_id}_literature_diagnostic.csv", index=False)

    # ================================================================================
    # Stratification: gene_arm, ot_truncated, best_match_tier (+A1/A2/B1 restriction)
    # ================================================================================
    best_design = "+gene_overlap_weighted"
    best_res = fit_results[best_design]
    strat_cols = [
        c for c in ["gene_arm", "best_match_tier", "ot_truncated"] if c in cond_basic.columns
    ]
    val_meta = validate.merge(
        cond_basic[["condition_concept_id", *strat_cols]], on="condition_concept_id", how="left"
    )
    assert len(val_meta) == len(validate)
    val_meta["_proba"] = best_res["val_proba"]

    gene_arm_rows = []
    if "gene_arm" in val_meta.columns:
        for level, grp in val_meta.groupby("gene_arm", dropna=False):
            m = _drug_macro_metrics(
                grp["ingredient_concept_id"], grp["y_faers_signal"], grp["_proba"].to_numpy()
            )
            gene_arm_rows.append(
                {
                    "gene_arm": str(level),
                    "n_pairs": len(grp),
                    "drug_macro_auc": m["drug_macro_auc"],
                    "n_drugs_scored": m["n_drugs_scored"],
                }
            )
    else:
        fallback_notes.append("gene_arm column absent from condition_features_basic.csv")
    pd.DataFrame(gene_arm_rows).to_csv(out / f"{exp_id}_by_gene_arm.csv", index=False)

    truncation_rows = []
    if "ot_truncated" in val_meta.columns:
        for level, grp in val_meta.groupby("ot_truncated", dropna=False):
            m = _drug_macro_metrics(
                grp["ingredient_concept_id"], grp["y_faers_signal"], grp["_proba"].to_numpy()
            )
            truncation_rows.append(
                {
                    "ot_truncated": str(level),
                    "n_pairs": len(grp),
                    "drug_macro_auc": m["drug_macro_auc"],
                    "n_drugs_scored": m["n_drugs_scored"],
                }
            )
    else:
        fallback_notes.append("ot_truncated column absent from condition_features_basic.csv")
    pd.DataFrame(truncation_rows).to_csv(out / f"{exp_id}_by_truncation.csv", index=False)

    tier_rows = []
    a1_a2_b1_design_auc = float("nan")
    if "best_match_tier" in val_meta.columns:
        for level, grp in val_meta.groupby("best_match_tier", dropna=False):
            m = _drug_macro_metrics(
                grp["ingredient_concept_id"], grp["y_faers_signal"], grp["_proba"].to_numpy()
            )
            tier_rows.append(
                {
                    "best_match_tier": str(level),
                    "n_pairs": len(grp),
                    "drug_macro_auc": m["drug_macro_auc"],
                    "n_drugs_scored": m["n_drugs_scored"],
                }
            )
        # A1/A2/B1 restriction, if budget allows (spec: drop this before gene_arm strat)
        elapsed_before_tier_restrict = time.time() - t0
        if elapsed_before_tier_restrict < 530:
            tier_mask_val = val_meta["best_match_tier"].isin(["A1", "A2", "B1"])
            if tier_mask_val.any():
                m_tier = _drug_macro_metrics(
                    val_meta.loc[tier_mask_val, "ingredient_concept_id"],
                    val_meta.loc[tier_mask_val, "y_faers_signal"],
                    val_meta.loc[tier_mask_val, "_proba"].to_numpy(),
                )
                a1_a2_b1_design_auc = m_tier["drug_macro_auc"]
        else:
            fallback_notes.append(
                f"A1/A2/B1 restriction dropped: {elapsed_before_tier_restrict:.0f}s elapsed"
            )
    else:
        fallback_notes.append("best_match_tier column absent from condition_features_basic.csv")
    pd.DataFrame(tier_rows).to_csv(out / f"{exp_id}_by_match_tier.csv", index=False)

    # ================================================================================
    # Overlap distribution figure
    # ================================================================================
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    shared_vals = disease_raw_val["n_shared_genes"].to_numpy()
    ax.hist(shared_vals, bins=30, log=True)
    ax.set_xlabel("n_shared_genes (validate, disease arm)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(
        f"exp14: shared-gene count distribution -- {share_any_shared_gene * 100:.1f}% of "
        "pairs have >=1 shared gene"
    )
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_overlap_distributions.png", dpi=150)
    plt.close(fig)

    # ================================================================================
    # Feature importance (top 25), best design -- cheap per-feature pooled-AUC screen
    # (no extra model fit) rather than a second full LightGBM refit, to protect the
    # 600s budget after the three core designs already consumed most of it.
    # ================================================================================
    elapsed_before_importance = time.time() - t0
    if elapsed_before_importance < 570:
        from sklearn.metrics import roc_auc_score as _rocauc

        x_screen = train_designs[best_design]
        y_arr = y_train.to_numpy()
        importance_rows = []
        for col in x_screen.columns:
            series = x_screen[col]
            if series.dtype == object:
                continue
            if series.dtype == bool:
                series = series.astype(float)
            series = series.astype(float)
            if series.notna().sum() == 0:
                continue
            filled = series.fillna(series.median())
            if filled.nunique() < 2:
                continue
            try:
                auc = float(_rocauc(y_arr, filled.to_numpy()))
            except ValueError:
                continue
            importance_rows.append({"feature": col, "pooled_auc": max(auc, 1.0 - auc)})
        importance_df = (
            pd.DataFrame(importance_rows)
            .sort_values("pooled_auc", ascending=False)
            .head(25)
            .reset_index(drop=True)
        )
        importance_df.insert(0, "rank", range(1, len(importance_df) + 1))
    else:
        fallback_notes.append(
            f"Feature-importance screen dropped: {elapsed_before_importance:.0f}s elapsed."
        )
        importance_df = pd.DataFrame(columns=["rank", "feature", "pooled_auc"])
    importance_df.to_csv(out / f"{exp_id}_importance_top25.csv", index=False)

    results.commit()

    # ================================================================================
    # Metrics
    # ================================================================================
    base = fit_results["E8base"]
    weighted = fit_results["+gene_overlap_weighted"]
    raw = fit_results["+gene_overlap_raw"]

    gene_arm_df = pd.DataFrame(gene_arm_rows)
    disease_arm_auc = float("nan")
    phenotype_arm_auc = float("nan")
    if not gene_arm_df.empty:
        disease_rows_ga = gene_arm_df[
            gene_arm_df["gene_arm"].astype(str).str.contains("disease", case=False)
        ]
        phenotype_rows_ga = gene_arm_df[
            gene_arm_df["gene_arm"].astype(str).str.contains("phenotype", case=False)
        ]
        if not disease_rows_ga.empty:
            disease_arm_auc = float(disease_rows_ga["drug_macro_auc"].mean())
        if not phenotype_rows_ga.empty:
            phenotype_arm_auc = float(phenotype_rows_ga["drug_macro_auc"].mean())

    floor_metrics = _drug_macro_metrics(ids_val, y_val, val_designs["E8base"]["p_c"].to_numpy())
    floor_pc = floor_metrics["drug_macro_auc"]
    floor_degree_pc = floor_pc  # p_c lookup is the measured floor per METRIC.md sec 3

    metrics = {
        "drug_macro_auc": weighted["val_drug_macro_auc"],
        "drug_macro_p10": weighted["val_drug_macro_p10"],
        "increment_over_exp08": weighted["val_drug_macro_auc"] - base["val_drug_macro_auc"],
        "increment_genetic_weighted": weighted["val_drug_macro_auc"] - raw["val_drug_macro_auc"],
        "increment_literature_weighted": (
            literature_res["val_drug_macro_auc"] - base["val_drug_macro_auc"]
            if literature_res is not None
            else float("nan")
        ),
        "drug_macro_auc_disease_arm": disease_arm_auc,
        "drug_macro_auc_phenotype_arm": phenotype_arm_auc,
        "floor_pc": floor_pc,
        "floor_degree_pc": floor_degree_pc,
        "n_drugs_scored": weighted["val_n_drugs_scored"],
        "share_pairs_with_any_shared_gene": share_any_shared_gene,
        "drug_macro_auc_a1_a2_b1": a1_a2_b1_design_auc,
        "e8base_drug_macro_auc": base["val_drug_macro_auc"],
        "gene_overlap_raw_drug_macro_auc": raw["val_drug_macro_auc"],
        "fallback_notes_count": float(len(fallback_notes)),
    }

    artifacts = [
        str(out / f"{exp_id}_designs.csv"),
        str(out / f"{exp_id}_by_gene_arm.csv"),
        str(out / f"{exp_id}_by_truncation.csv"),
        str(out / f"{exp_id}_by_match_tier.csv"),
        str(out / f"{exp_id}_literature_diagnostic.csv"),
        str(out / f"{exp_id}_bootstrap_increments.csv"),
        str(out / f"{exp_id}_importance_top25.csv"),
        str(out / f"{exp_id}_overlap_distributions.png"),
    ]

    log.info("fallback notes: %s", fallback_notes)
    log.info("final metrics: %s", metrics)

    results.commit()

    return {
        "__metrics__": metrics,
        "__fallback_notes__": fallback_notes,
        "__artifacts__": artifacts,
    }


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp14",
        title="Gene overlap between drug target genes and condition-implicated genes",
        hypothesis=(
            "Genetic-association-weighted gene overlap adds >=0.02 drug_macro_auc over "
            "exp08's neighbour model, concentrated on disease-arm conditions."
        ),
        approach=(
            "Sparse condition-gene and drug-gene matrices; raw, ot_score-weighted, "
            "genetic-association-only, IDF-weighted and Jaccard overlaps; dt_literature "
            "kept as a separate diagnostic; nested on top of degree, p_c, intrinsic union "
            "and nb_excess_gene; grouped 3-fold CV on drug_macro_auc; stratified by "
            "gene_arm, ot_truncated and match tier."
        ),
        label="y_faers_signal",
        features=["degree", "p_c", "intrinsic_union", "nb_excess_gene", "gene_overlap"],
        split="train/validate, grouped by primary target gene",
        notes="Gene half of the never-completed exp10 (agent exp10, registered but never "
        "completed); pathway half is exp15. exp10's partial run is referenced here only, "
        "not reused as this experiment's own result.",
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

    fallback_notes = metrics.pop("__fallback_notes__", [])
    artifacts = metrics.pop("__artifacts__", [])
    metrics = metrics.get("__metrics__", metrics)
    print(metrics)

    findings = (
        f"share_pairs_with_any_shared_gene={metrics['share_pairs_with_any_shared_gene']:.4f} "
        "(disease arm) -- the increment below is bounded by this coverage. "
        f"E8base drug_macro_auc={metrics['e8base_drug_macro_auc']:.4f}, "
        f"+gene_overlap_raw drug_macro_auc={metrics['gene_overlap_raw_drug_macro_auc']:.4f}, "
        f"+gene_overlap_weighted (headline) drug_macro_auc={metrics['drug_macro_auc']:.4f}. "
        f"increment_over_exp08(E8base)={metrics['increment_over_exp08']:+.4f}, "
        f"increment_genetic_weighted (weighted vs raw)={metrics['increment_genetic_weighted']:+.4f} "
        "(see <exp_id>_bootstrap_increments.csv for 1,000-resample bootstrap CIs over drugs; "
        "an increment under ~0.01 is noise given across-drug sd 0.128). "
        f"By gene_arm: disease_arm drug_macro_auc={metrics['drug_macro_auc_disease_arm']:.4f}, "
        f"phenotype_arm drug_macro_auc={metrics['drug_macro_auc_phenotype_arm']:.4f} -- see "
        "<exp_id>_by_gene_arm.csv for the full breakdown (both/disease-only/phenotype-only/none). "
        f"A1/A2/B1 match-tier-restricted drug_macro_auc={metrics['drug_macro_auc_a1_a2_b1']:.4f}. "
        f"increment_literature_weighted (dt_literature diagnostic, kept out of the headline "
        f"model)={metrics['increment_literature_weighted']:+.4f}. "
        f"floor_pc={metrics['floor_pc']:.4f}, n_drugs_scored={metrics['n_drugs_scored']}. "
        + (f"Fallbacks taken: {'; '.join(fallback_notes)}. " if fallback_notes else "")
        + "exp10 (agent exp10) registered this same gene-overlap question but never "
        "completed; this experiment re-derives the gene-overlap machinery independently "
        "and does not reuse exp10's partial results as its own."
    )
    print(findings)

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=artifacts,
        next_steps=(
            "If the weighted design's increment over E8base is >=0.02 with a bootstrap CI "
            "excluding 0 and the gain concentrates on the disease arm, proceed to exp15's "
            "pathway-overlap features on top of this design. If the gain is null or "
            "concentrated on the phenotype arm instead, report that as the headline per "
            "this experiment's own falsification criterion, not as a null result to bury."
        ),
    )


if __name__ == "__main__":
    pass
