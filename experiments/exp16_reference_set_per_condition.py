"""exp16 -- Adjudicated reference-set evaluation with a per-condition metric.

Supersedes exp11's failed precondition (exp_20260922_250b1f) -- not by loosening it, but by
transposing the metric. exp11 fetched the reference sets correctly (472 pairs, 223 drugs,
8 conditions, 198 positives / 274 negatives; data/input/reference/reference_set.csv) and then
could not use them: only 3 of 223 drugs had >=5 reference pairs against the 30 the per-DRUG
metric (drug_macro_auc, rank a drug's conditions) required. Adjudicated pharmacovigilance
reference sets are built drug-by-health-outcome-of-interest -- many drugs against a handful
of carefully adjudicated outcomes -- so the metric this experiment needs is the transpose:
`condition_macro_auc = mean over outcomes c of ROC-AUC(y[:, c], score[:, c])`, restricted to
outcomes with >=50 pairs and both classes present. Four outcomes qualify (4026032, 4329847,
192671, 197320), carrying 437 pairs and 183 positives.

The floor for this population is NOT p_c (a condition's own flag rate is identical for every
drug within one outcome, so it cannot discriminate) -- it is p_d, the DRUG's own train
`y_faers_signal` flag rate, plus drug degree. `floors.py`'s `floors()` is built for the
drug_macro_auc direction (p_c lookup, grouped by drug); it is not modified (exp13 shipped and
finalized it) -- this script implements the transposed p_d / degree+p_d floor locally,
following the same pattern (train-only statistics, HistGBM on <=4 features, no CV, single
fit scored once).

Model-scoring code (exp07 union model feature assembly, exp08 neighbour/nb_excess_gene
feature assembly, exp14 gene-overlap disease+phenotype arm feature assembly) is copied from
experiments/exp07_ablation_rerun_leak_audit.py, experiments/exp08_neighbour_transport_pc_stripped.py
and experiments/exp14_gene_overlap_disease_arm.py (also the pattern exp11 used for exp07/
exp08) -- reused here only after preconditions pass, and only re-scored on reference pairs
after a single fit on the full train split. No fitting happens on the reference set itself,
per AGENT.md's test-set-analogue rule for the project's only external yardstick: hyperparameters
and model shape are exactly those experiments' own choices, not tuned here.

Pooled AUC/AP are legitimate here and nowhere else in the project (real adjudicated negatives,
not "absence is not a negative" CEM pairs).
"""

# ruff: noqa: N806 -- linear-algebra matrix-naming convention (C_*, D_binary), consistent
# with exp14, from which this gene-overlap machinery is copied.

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

app = modal.App("bridge-exp16")

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

MIN_PAIRS_PER_OUTCOME = 50
MIN_QUALIFYING_OUTCOMES = 4
BOOT_RESAMPLES = 1000
K_SHRINK = 10

HGB_KWARGS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "early_stopping": False,
    "random_state": 0,
}

# ---- copied from exp07 / exp14 (block-selection spec) ------------------------------

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


# ---- copied from exp14 (gene-overlap disease-arm / phenotype-arm sparse machinery) --


def _build_disease_gene_matrices(ot_path: pathlib.Path, target_long_path: pathlib.Path, log):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    ot = pd.read_csv(ot_path)
    gene_col = _detect_col(
        list(ot.columns), ["ensembl_gene_id", "gene_id", "target_id", "ensemblId", "geneId", "gene"]
    )
    if gene_col is None:
        raise KeyError(f"no gene id column found in condition_gene_ot_long.csv: {list(ot.columns)}")
    cond_col = "condition_concept_id"
    score_col = "ot_score"
    genetic_col = _detect_col(list(ot.columns), ["dt_genetic_association"])

    ing = pd.read_csv(target_long_path, usecols=["omop_concept_id", "ensembl_gene_id"]).dropna(
        subset=["ensembl_gene_id"]
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

    drug_ids = sorted(ing["omop_concept_id"].unique())
    drug_pos = {d: i for i, d in enumerate(drug_ids)}
    ing2 = ing[ing["ensembl_gene_id"].isin(gene_pos)]
    drows = ing2["omop_concept_id"].map(drug_pos).to_numpy(dtype=int)
    dcols = ing2["ensembl_gene_id"].map(gene_pos).to_numpy(dtype=int)
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
        "C_score": C_score,
        "C_binary": C_binary,
        "C_genetic": C_genetic,
        "D_binary": D_binary,
    }


def _build_hpo_matrices(hpo_path: pathlib.Path, target_long_path: pathlib.Path, log):
    import pandas as pd
    import scipy.sparse as sp

    hpo = pd.read_csv(hpo_path)
    gene_col = _detect_col(
        list(hpo.columns), ["gene_symbol", "hgnc_symbol", "gene", "ensembl_gene_id", "gene_id"]
    )
    cond_col = _detect_col(list(hpo.columns), ["condition_concept_id"])
    if gene_col is None or cond_col is None:
        raise KeyError(
            f"could not identify gene/condition columns in HPO table: {list(hpo.columns)}"
        )

    id_system = "ensembl" if gene_col == "ensembl_gene_id" else "symbol"
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

    import numpy as np

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


def _pair_gene_overlap_features(pairs, ctx: dict, log, prefix: str = "", weighted: bool = True):
    import numpy as np
    import pandas as pd

    D_binary = ctx["D_binary"]
    C_binary = ctx["C_binary"]
    n_drug = ctx["n_drug"]

    drow = pairs["ingredient_concept_id"].map(ctx["drug_pos"])
    crow = pairs["condition_concept_id"].map(ctx["cond_pos"])
    valid = drow.notna() & crow.notna()
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

    if "C_score" in ctx or "C_genetic" in ctx:
        score_max = np.zeros(len(pairs), dtype=float)
        genetic_max = np.zeros(len(pairs), dtype=float)
        subset_idx = np.where(valid_arr & (n_shared > 0))[0]
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


# ---- exp16-local transposed floors (p_d / degree+p_d), per floors.py's pattern -----


def _condition_macro_auc(
    ids: pd.Series, y, score, outcomes: list, min_pairs: int
) -> tuple[float, pd.DataFrame]:
    """Transpose of floors.py's _drug_macro_auc: macro-average ROC-AUC of score vs
    binary y within each OUTCOME (condition) restricted to `outcomes`, with >=min_pairs
    rows and both classes present. NaN-safe: bool cast to float, NaN scores filled with
    the per-outcome median before roc_auc_score.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(
        {
            "condition": np.asarray(ids),
            "y": np.asarray(y, dtype=float),
            "score": np.asarray(score, dtype=float),
        }
    )
    rows: list[dict[str, float]] = []
    for cond_id in outcomes:
        grp = df.loc[df["condition"] == cond_id]
        if len(grp) < min_pairs:
            continue
        y_grp = grp["y"].to_numpy()
        if len(np.unique(y_grp)) < 2:
            continue
        s = grp["score"]
        if s.isna().any():
            if s.notna().sum() == 0:
                continue
            s = s.fillna(s.median())
        auc = float(roc_auc_score(y_grp, s.to_numpy()))
        rows.append({"condition": cond_id, "n": len(grp), "auc": auc})
    table = pd.DataFrame(rows, columns=["condition", "n", "auc"])
    if table.empty:
        return float("nan"), table
    return float(table["auc"].mean()), table


def _transposed_floors(
    train: pd.DataFrame, ref: pd.DataFrame, outcomes: list, min_pairs: int
) -> dict[str, Any]:
    """p_d and degree+p_d floors -- the transpose of floors.py's floors(): within a single
    outcome every drug's p_c is identical (it is the outcome's own prevalence), so p_c
    cannot discriminate drugs; the trivial baseline is the DRUG's own train flag rate p_d
    (its `y_faers_signal` rate across all its train conditions) plus drug degree. Train-only
    statistics, single HistGBM fit, no CV -- exactly floors.py's pattern, transposed.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    p_d_train = train.groupby("ingredient_concept_id")["y_faers_signal"].mean()
    p_d_mean = float(p_d_train.mean())
    ref_p_d = ref["ingredient_concept_id"].map(p_d_train).astype(float).fillna(p_d_mean)

    floor_pd, table_pd = _condition_macro_auc(
        ref["condition_concept_id"], ref["label"], ref_p_d.to_numpy(), outcomes, min_pairs
    )

    drug_degree_train = train.groupby("ingredient_concept_id").size()
    drug_degree_median = float(drug_degree_train.median())

    def _feats(df: pd.DataFrame, p_d_series: pd.Series) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["drug_degree"] = np.log1p(
            df["ingredient_concept_id"]
            .map(drug_degree_train)
            .astype(float)
            .fillna(drug_degree_median)
        )
        f["p_d"] = p_d_series.to_numpy()
        return f

    train_p_d = train["ingredient_concept_id"].map(p_d_train).astype(float).fillna(p_d_mean)
    x_train = _feats(train, train_p_d).astype("float32")
    x_ref = _feats(ref, ref_p_d).astype("float32")
    for col in x_train.columns:
        med = float(x_train[col].median())
        x_train[col] = x_train[col].fillna(med)
        x_ref[col] = x_ref[col].fillna(med)

    y_train = train["y_faers_signal"].astype(float).to_numpy()
    if len(np.unique(y_train)) < 2:
        floor_degree_pd, table_degree_pd = float("nan"), pd.DataFrame()
    else:
        model = HistGradientBoostingClassifier(**HGB_KWARGS)
        model.fit(x_train, y_train)
        eval_proba = model.predict_proba(x_ref)[:, 1]
        floor_degree_pd, table_degree_pd = _condition_macro_auc(
            ref["condition_concept_id"], ref["label"], eval_proba, outcomes, min_pairs
        )

    return {
        "floor_pd": floor_pd,
        "floor_degree_pd": floor_degree_pd,
        "table_pd": table_pd,
        "table_degree_pd": table_degree_pd,
        "score_pd": ref_p_d.to_numpy(),
    }


def _bootstrap_condition_macro_auc_ci(
    ref: pd.DataFrame,
    score,
    outcomes: list,
    min_pairs: int,
    n_boot: int = BOOT_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample DRUGS (not pairs) within each qualifying outcome, recompute
    condition_macro_auc, and take percentiles across n_boot resamples -- per the spec's
    'resample drugs' instruction and this session's constraint #9."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(
        {
            "drug": ref["ingredient_concept_id"].to_numpy(),
            "condition": ref["condition_concept_id"].to_numpy(),
            "y": ref["label"].to_numpy(dtype=float),
            "score": np.asarray(score, dtype=float),
        }
    )
    by_outcome = {c: df.loc[df["condition"] == c].reset_index(drop=True) for c in outcomes}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        outcome_aucs = []
        for _c, grp in by_outcome.items():
            n = len(grp)
            if n < min_pairs:
                continue
            idx = rng.integers(0, n, size=n)
            samp = grp.iloc[idx]
            y_s = samp["y"].to_numpy()
            if len(np.unique(y_s)) < 2:
                continue
            s = samp["score"]
            if s.isna().any():
                if s.notna().sum() == 0:
                    continue
                s = s.fillna(s.median())
            outcome_aucs.append(roc_auc_score(y_s, s.to_numpy()))
        boot_means[b] = np.mean(outcome_aucs) if outcome_aucs else np.nan
    boot_means = boot_means[~np.isnan(boot_means)]
    if len(boot_means) == 0:
        return float("nan"), float("nan"), float("nan")
    point, _ = _condition_macro_auc(df["condition"], df["y"], df["score"], outcomes, min_pairs)
    return point, float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _pooled_auc_ap(y, score) -> tuple[float, float]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=float)
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


def _check_preconditions(ref_path: pathlib.Path) -> tuple[str | None, dict[str, Any]]:
    """Precondition per exp16.md step 1: reference file present; >=4 outcomes with
    >=50 pairs and both classes present; both labels present overall."""
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
            f"reference file does not have both label classes present overall: "
            f"found classes {sorted(classes_present)}, counts {label_counts}",
            {"label_counts": label_counts},
        )

    per_outcome = ref.groupby("condition_concept_id").agg(
        n=("label", "size"), n_pos=("label", "sum")
    )
    per_outcome["n_neg"] = per_outcome["n"] - per_outcome["n_pos"]
    qualifying = per_outcome[
        (per_outcome["n"] >= MIN_PAIRS_PER_OUTCOME)
        & (per_outcome["n_pos"] > 0)
        & (per_outcome["n_neg"] > 0)
    ]
    diagnostics = {
        "n_reference_pairs": len(ref),
        "n_distinct_drugs": int(ref["ingredient_concept_id"].nunique()),
        "n_distinct_conditions": int(ref["condition_concept_id"].nunique()),
        "label_counts": label_counts,
        "n_qualifying_outcomes": len(qualifying),
        "qualifying_outcomes": qualifying.index.tolist(),
        "per_outcome_counts": per_outcome.reset_index().to_dict(orient="records"),
    }
    if len(qualifying) < MIN_QUALIFYING_OUTCOMES:
        return (
            f"only {len(qualifying)} outcomes have >={MIN_PAIRS_PER_OUTCOME} pairs and both "
            f"classes present, need >={MIN_QUALIFYING_OUTCOMES} "
            f"(n_reference_pairs={diagnostics['n_reference_pairs']}, "
            f"n_distinct_drugs={diagnostics['n_distinct_drugs']}, "
            f"label_counts={label_counts})",
            diagnostics,
        )
    return None, diagnostics


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=16.0,
    memory=32768,
    timeout=900,
)
def run(exp_id: str) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp16")

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    ref_path = pathlib.Path("/data/reference/reference_set.csv")

    # ---- step 1: preconditions, checked before anything else ------------------------
    err, diag = _check_preconditions(ref_path)
    if err is not None:
        log.error("precondition check failed: %s", err)
        return {
            "precondition_failed": 1.0,
            "precondition_error_message": err,
            "__diagnostics__": diag,
        }
    log.info("preconditions passed: n_qualifying_outcomes=%d", diag["n_qualifying_outcomes"])

    import time

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    fallback_notes: list[str] = []

    required_paths = {
        "train": pathlib.Path("/data/splits/train.csv"),
        "validate": pathlib.Path("/data/splits/validate.csv"),
        "ingredient_target_long": pathlib.Path("/data/drug/ingredient_target_long.csv"),
        "ingredient_features": pathlib.Path("/data/drug/ingredient_features.csv"),
        "condition_features_basic": pathlib.Path("/data/condition/condition_features_basic.csv"),
        "condition_group_long": pathlib.Path("/data/condition/condition_group_long.csv"),
        "data_dictionary": pathlib.Path("/data/data_dictionary.csv"),
        "condition_gene_ot_long": pathlib.Path("/data/condition/condition_gene_ot_long.csv"),
        "condition_gene_hpo_long": pathlib.Path("/data/condition/condition_gene_hpo_long.csv"),
    }
    missing = [str(p) for p in required_paths.values() if not p.exists()]
    if missing:
        msg = f"Missing required input(s) for model re-scoring: {missing}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    ref = pd.read_csv(ref_path)
    train = pd.read_csv(required_paths["train"])
    validate = pd.read_csv(required_paths["validate"])
    edges_raw = pd.read_csv(required_paths["ingredient_target_long"]).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    cond_basic = pd.read_csv(required_paths["condition_features_basic"])

    outcomes = diag["qualifying_outcomes"]
    log.info("qualifying outcomes (>=50 pairs, both classes): %s", outcomes)

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
            "this project's train+validate concept grid."
        )
    ref_eval = ref_in_grid.reset_index(drop=True)
    # re-derive qualifying outcomes on the in-grid subset actually being scored
    per_outcome_grid = ref_eval.groupby("condition_concept_id").agg(
        n=("label", "size"), n_pos=("label", "sum")
    )
    per_outcome_grid["n_neg"] = per_outcome_grid["n"] - per_outcome_grid["n_pos"]
    outcomes_scored = per_outcome_grid[
        (per_outcome_grid["n"] >= MIN_PAIRS_PER_OUTCOME)
        & (per_outcome_grid["n_pos"] > 0)
        & (per_outcome_grid["n_neg"] > 0)
    ].index.tolist()
    if sorted(outcomes_scored) != sorted(outcomes):
        fallback_notes.append(
            f"in-grid filtering changed the qualifying-outcome set from {outcomes} to "
            f"{outcomes_scored}."
        )
    outcomes = outcomes_scored

    n_first_approval_note = ""
    # first_approval distribution of reference-set drugs vs the full ingredient grid,
    # for the spec's "drugs skew old and widely used" trap -- reported if the column exists.
    ingredient_features_df = pd.read_csv(required_paths["ingredient_features"])
    if "first_approval" in ingredient_features_df.columns:
        fa_ref = ingredient_features_df.loc[
            ingredient_features_df["omop_concept_id"].isin(ref_eval["ingredient_concept_id"]),
            "first_approval",
        ].dropna()
        fa_all = ingredient_features_df["first_approval"].dropna()
        if len(fa_ref) and len(fa_all):
            n_first_approval_note = (
                f"reference-set drugs first_approval median={fa_ref.median():.0f} "
                f"(n={len(fa_ref)}) vs full ingredient grid median={fa_all.median():.0f} "
                f"(n={len(fa_all)})."
            )

    coverage_df = pd.DataFrame(
        [
            {"quantity": "reference_pairs_total", "value": len(ref)},
            {"quantity": "reference_pairs_in_project_grid", "value": len(ref_eval)},
            {"quantity": "n_qualifying_outcomes_scored", "value": len(outcomes)},
            {"quantity": "qualifying_outcomes", "value": str(outcomes)},
            {
                "quantity": "reference_drugs_total",
                "value": int(ref_eval["ingredient_concept_id"].nunique()),
            },
            {"quantity": "first_approval_note", "value": n_first_approval_note},
            {
                "quantity": "negative_control_definition",
                "value": (
                    "Ryan et al. 2013 / Coloma et al. 2013: negative controls are drug-outcome "
                    "pairs BELIEVED, on current evidence, to have no causal relationship -- not "
                    "pairs verified safe. Adjudication can be revised by later evidence."
                ),
            },
        ]
    )
    coverage_df.to_csv(out / f"{exp_id}_coverage.csv", index=False)

    # ---- transposed floors: p_d, degree+p_d, computed on the reference set ----------
    log.info("computing transposed floors (p_d, degree+p_d)")
    floor_result = _transposed_floors(train, ref_eval, outcomes, MIN_PAIRS_PER_OUTCOME)

    # ---- incumbent FAERS quantities (log(faers_prr), faers_chi_square, Evans flag) --
    faers_cols = ["faers_prr", "faers_chi_square", "y_faers_signal"]
    faers_lookup = train_val.drop_duplicates(
        subset=["ingredient_concept_id", "condition_concept_id"]
    ).set_index(["ingredient_concept_id", "condition_concept_id"])[faers_cols]
    ref_idx = ref_eval.set_index(["ingredient_concept_id", "condition_concept_id"]).index
    faers_joined = faers_lookup.reindex(ref_idx).reset_index(drop=True)

    log_faers_prr = np.log(faers_joined["faers_prr"].where(faers_joined["faers_prr"] > 0))
    faers_chi_square = faers_joined["faers_chi_square"].astype(float)
    evans_flag = faers_joined["y_faers_signal"].astype(float)
    n_incumbent_missing = int(log_faers_prr.isna().sum())
    if n_incumbent_missing:
        fallback_notes.append(
            f"{n_incumbent_missing}/{len(ref_eval)} reference pairs had no observed "
            "faers_prr in train+validate (pair never in FAERS); left NaN, handled by "
            "per-outcome NaN-safe median fill."
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
    log.info(
        "exp07 union model fit+scored in %.1fs (elapsed=%.1fs)", time.time() - t0, time.time() - t0
    )

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
    log.info("exp08 neighbour model fit+scored (elapsed=%.1fs)", time.time() - t0)

    # ---- exp14 gene-overlap model (E8base + disease-arm weighted + HPO-arm) --------
    log.info("building exp14 gene-overlap features (disease + phenotype arm)")
    e8base_train = pd.concat(
        [
            _assemble_union(train, degree_train),
            train_nb[["nb_excess_gene", "n_neighbours_gene"]],
        ],
        axis=1,
    )
    e8base_ref = pd.concat(
        [
            _assemble_union(ref_eval, degree_ref),
            ref_nb[["nb_excess_gene", "n_neighbours_gene"]],
        ],
        axis=1,
    )

    disease_ctx = _build_disease_gene_matrices(
        required_paths["condition_gene_ot_long"], required_paths["ingredient_target_long"], log
    )
    log.info(
        "disease-arm matrices: %d conditions x %d genes, %d drugs (elapsed=%.1fs)",
        disease_ctx["n_cond"],
        disease_ctx["n_gene"],
        disease_ctx["n_drug"],
        time.time() - t0,
    )
    hpo_ctx = None
    try:
        hpo_ctx = _build_hpo_matrices(
            required_paths["condition_gene_hpo_long"], required_paths["ingredient_target_long"], log
        )
        log.info(
            "phenotype-arm matrices: %d conditions x %d genes, %d drugs (elapsed=%.1fs)",
            hpo_ctx["n_cond"],
            hpo_ctx["n_gene"],
            hpo_ctx["n_drug"],
            time.time() - t0,
        )
    except Exception as exc:
        log.warning("phenotype-arm (HPO) matrix construction failed, skipping: %s", exc)
        fallback_notes.append(f"HPO overlap skipped: {exc}")

    all_pairs_gene = pd.concat(
        [
            train[["ingredient_concept_id", "condition_concept_id"]],
            ref_eval[["ingredient_concept_id", "condition_concept_id"]],
        ],
        keys=["train", "ref"],
    )
    disease_weighted_all = _pair_gene_overlap_features(
        all_pairs_gene, disease_ctx, log, weighted=True
    )
    if hpo_ctx is not None:
        hpo_feats_all = _pair_gene_overlap_features(
            all_pairs_gene, hpo_ctx, log, prefix="hpo_", weighted=False
        )
    else:
        hpo_feats_all = pd.DataFrame(index=all_pairs_gene.index)

    def _split_gene(df):
        tr = df.loc["train"].set_axis(train.index)
        rf = df.loc["ref"].set_axis(ref_eval.index)
        return tr, rf

    disease_train, disease_ref = _split_gene(disease_weighted_all)
    hpo_train, hpo_ref = _split_gene(hpo_feats_all)
    gene_overlap_weighted_train = pd.concat([disease_train, hpo_train], axis=1)
    gene_overlap_weighted_ref = pd.concat([disease_ref, hpo_ref], axis=1)

    x_train_overlap = pd.concat([e8base_train, gene_overlap_weighted_train], axis=1)
    x_ref_overlap = pd.concat([e8base_ref, gene_overlap_weighted_ref], axis=1)

    overlap_model = HistGradientBoostingClassifier(**HGB_KWARGS)
    overlap_model.fit(x_train_overlap, y_train)
    overlap_score = overlap_model.predict_proba(x_ref_overlap)[:, 1]
    log.info("exp14 gene-overlap model fit+scored in %.1fs total elapsed", time.time() - t0)

    # ---- score all scorers with condition_macro_auc / pooled AUC/AP + bootstrap CIs -
    y_ref = ref_eval["label"]
    cond_ids_ref = ref_eval["condition_concept_id"]

    scorers = {
        "floor_pd": floor_result["score_pd"],
        "incumbent_log_faers_prr": log_faers_prr.to_numpy(),
        "incumbent_faers_chi_square": faers_chi_square.to_numpy(),
        "incumbent_evans_flag": evans_flag.to_numpy(),
        "intrinsic_union": union_score,
        "neighbour": neighbour_score,
        "gene_overlap": overlap_score,
    }

    eval_rows = []
    per_outcome_frames = []
    for name, score in scorers.items():
        cma, table = _condition_macro_auc(
            cond_ids_ref, y_ref, score, outcomes, MIN_PAIRS_PER_OUTCOME
        )
        _point, boot_lo, boot_hi = _bootstrap_condition_macro_auc_ci(
            ref_eval, score, outcomes, MIN_PAIRS_PER_OUTCOME
        )
        pooled_auc, pooled_ap = _pooled_auc_ap(y_ref, score)
        eval_rows.append(
            {
                "model": name,
                "condition_macro_auc": cma,
                "condition_macro_auc_ci_lo": boot_lo,
                "condition_macro_auc_ci_hi": boot_hi,
                "n_outcomes_scored": len(table),
                "pooled_auc": pooled_auc,
                "pooled_ap": pooled_ap,
            }
        )
        if not table.empty:
            t = table.copy()
            t["model"] = name
            per_outcome_frames.append(t)
    # degree+p_d floor: already computed with its own fitted scorer in _transposed_floors
    eval_rows.append(
        {
            "model": "floor_degree_pd",
            "condition_macro_auc": floor_result["floor_degree_pd"],
            "condition_macro_auc_ci_lo": float("nan"),
            "condition_macro_auc_ci_hi": float("nan"),
            "n_outcomes_scored": len(floor_result["table_degree_pd"]),
            "pooled_auc": float("nan"),
            "pooled_ap": float("nan"),
        }
    )
    if not floor_result["table_degree_pd"].empty:
        t = floor_result["table_degree_pd"].copy()
        t["model"] = "floor_degree_pd"
        per_outcome_frames.append(t)

    eval_df = pd.DataFrame(eval_rows)
    eval_path = out / f"{exp_id}_reference_eval.csv"
    eval_df.to_csv(eval_path, index=False)

    per_outcome_df = (
        pd.concat(per_outcome_frames, ignore_index=True) if per_outcome_frames else pd.DataFrame()
    )
    per_outcome_path = out / f"{exp_id}_per_outcome.csv"
    per_outcome_df.to_csv(per_outcome_path, index=False)

    # ---- CEM-vs-reference rank correlation (spec step 6) -----------------------------
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
    for name, score in {**scorers, "floor_degree_pd": None}.items():
        if score is None:
            continue
        if n_with_cem_context >= 3:
            rho = _spearman_rank_corr(
                pd.Series(score)[has_cem_context].to_numpy(),
                cem_label_for_ref[has_cem_context].to_numpy(),
            )
        else:
            rho = float("nan")
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

    # ---- success_criterion.json v2 ----------------------------------------------------
    import json

    floor_row = eval_df.loc[eval_df["model"] == "floor_pd"].iloc[0]
    incumbent_row = eval_df.loc[eval_df["model"] == "incumbent_log_faers_prr"].iloc[0]
    union_row = eval_df.loc[eval_df["model"] == "intrinsic_union"].iloc[0]
    neighbour_row = eval_df.loc[eval_df["model"] == "neighbour"].iloc[0]
    overlap_row = eval_df.loc[eval_df["model"] == "gene_overlap"].iloc[0]

    floor_val = float(floor_row["condition_macro_auc"])
    ceiling_val = float(incumbent_row["condition_macro_auc"])
    solved_threshold = (
        floor_val + 0.5 * (ceiling_val - floor_val) if ceiling_val == ceiling_val else float("nan")
    )

    success_criterion = {
        "version": 2,
        "source": "adjudicated_per_condition",
        "floor_pd": floor_val,
        "floor_pd_ci": [
            float(floor_row["condition_macro_auc_ci_lo"]),
            float(floor_row["condition_macro_auc_ci_hi"]),
        ],
        "floor_degree_pd": float(floor_result["floor_degree_pd"]),
        "ceiling_incumbent": ceiling_val,
        "ceiling_ci": [
            float(incumbent_row["condition_macro_auc_ci_lo"]),
            float(incumbent_row["condition_macro_auc_ci_hi"]),
        ],
        "ceiling_definition": (
            "incumbent FAERS disproportionality (log faers_prr) condition_macro_auc against "
            "the adjudicated reference set -- what a reviewer gets today from RWD for these "
            "four outcomes."
        ),
        "solved_threshold": solved_threshold,
        "n_outcomes": len(outcomes),
        "n_reference_pairs": len(ref_eval),
        "matches_incumbent": {
            "intrinsic_union": bool(union_row["condition_macro_auc"] >= ceiling_val),
            "neighbour": bool(neighbour_row["condition_macro_auc"] >= ceiling_val),
            "gene_overlap": bool(overlap_row["condition_macro_auc"] >= ceiling_val),
        },
    }
    success_path = out / "success_criterion.json"
    success_path.write_text(json.dumps(success_criterion, indent=2))

    results.commit()

    metrics = {
        "condition_macro_auc_union": float(union_row["condition_macro_auc"]),
        "condition_macro_auc_neighbour": float(neighbour_row["condition_macro_auc"]),
        "condition_macro_auc_overlap": float(overlap_row["condition_macro_auc"]),
        "incumbent_condition_macro_auc": ceiling_val,
        "floor_pd": floor_val,
        "floor_degree_pd": float(floor_result["floor_degree_pd"]),
        "pooled_auc_union": float(union_row["pooled_auc"]),
        "pooled_ap_union": float(union_row["pooled_ap"]),
        "n_outcomes": len(outcomes),
        "n_pairs": len(ref_eval),
        "ceiling_incumbent": ceiling_val,
        "solved_threshold": solved_threshold,
    }

    findings = (
        f"Floor on this set (p_d lookup, transpose of METRIC.md's p_c) = {floor_val:.4f}; "
        f"degree+p_d floor = {floor_result['floor_degree_pd']:.4f}. "
        f"intrinsic_union = {union_row['condition_macro_auc']:.4f}, "
        f"neighbour = {neighbour_row['condition_macro_auc']:.4f}, "
        f"gene_overlap = {overlap_row['condition_macro_auc']:.4f}, "
        f"incumbent log(faers_prr) = {ceiling_val:.4f} (chi_square="
        f"{eval_df.loc[eval_df['model'] == 'incumbent_faers_chi_square', 'condition_macro_auc'].iloc[0]:.4f}, "
        f"evans_flag={eval_df.loc[eval_df['model'] == 'incumbent_evans_flag', 'condition_macro_auc'].iloc[0]:.4f}). "
        f"Revised solved_threshold = {solved_threshold:.4f}. n_outcomes={len(outcomes)} "
        f"({outcomes}), n_pairs={len(ref_eval)}. Pooled AUC/AP ARE legitimate metrics here "
        "(the negatives are adjudicated, not merely unobserved) -- unlike everywhere else "
        f"in this project: pooled_auc(union)={union_row['pooled_auc']:.4f}, "
        f"pooled_ap(union)={union_row['pooled_ap']:.4f}. "
        f"CEM-vs-reference rank gap: see {exp_id}_cem_vs_reference_rank.csv "
        f"(n_pairs_with_cem_context={n_with_cem_context}). Per-outcome spread in "
        f"{exp_id}_per_outcome.csv. None of our scorers match the incumbent's "
        f"condition_macro_auc on this set."
        if not any(success_criterion["matches_incumbent"].values())
        else (
            "At least one scorer matches or exceeds the incumbent's condition_macro_auc on "
            "this set -- see success_criterion.json's matches_incumbent."
        )
    )

    return {
        "__metrics__": metrics,
        "__findings__": findings,
        "__artifacts__": [
            str(eval_path),
            str(per_outcome_path),
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
        agent="exp16",
        title="Adjudicated reference-set evaluation with a per-condition metric",
        hypothesis=(
            "On four adjudicated outcomes with real negatives, our scorers exceed the p_d "
            "floor computed on that set, and the incumbent FAERS score's "
            "condition_macro_auc gives the operational ceiling for the success criterion."
        ),
        approach=(
            "Transpose the metric to condition_macro_auc (rank drugs within an outcome) to "
            "match the drug-by-HOI shape of Ryan 2013 / Coloma 2013; score existing fitted "
            "models (exp07 union, exp08 neighbour, exp14 gene-overlap, each refit once on "
            "all of train, never on the reference set) plus p_d and degree+p_d floors; "
            "pooled AUC/AP legitimate here because negatives are adjudicated; emit "
            "success_criterion.json v2."
        ),
        # exp16.md's own registration snippet uses label="reference_set_label", which is
        # not in bridge.labnotebook.VALID_LABELS ({"y_faers_signal", "y_semmeddb_causes",
        # "y_semmeddb_treats", "y_any_harm", "faers_prr", "other"}) -- that snippet is
        # schematic (exp11 hit and documented the same mismatch); label="other" used here.
        label="other",
        features=["p_d", "faers_prr", "intrinsic_union", "nb_excess_gene", "gene_overlap"],
        split="reference pairs, no fitting on reference pairs",
        notes=(
            "Supersedes exp11 (exp_20260922_250b1f)'s failed precondition -- not by "
            "loosening the bar, but by transposing the metric to condition_macro_auc "
            "(rank drugs within an outcome) to match the reference sets' drug-by-HOI shape."
        ),
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
                "n_qualifying_outcomes": float(diag.get("n_qualifying_outcomes", 0)),
                "required_n_qualifying_outcomes": float(MIN_QUALIFYING_OUTCOMES),
            },
            findings=f"Precondition failed before any scoring or fitting: {msg}.",
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
            "Read success_criterion.json (v2, source=adjudicated_per_condition) and have "
            "downstream experiments cite its floor_pd/ceiling_incumbent/solved_threshold "
            "instead of the provisional per-drug 0.65 target in METRIC.md, which this "
            "experiment's population cannot speak to directly."
        ),
    )
