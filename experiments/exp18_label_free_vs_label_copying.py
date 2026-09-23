"""exp18 -- Label-free biology versus label-copying, with a constructed cold-condition split.

Implements experiments/exp18_label_free_vs_label_copying.md -- Round 4's central experiment.

Three strictly partitioned designs, fit on the standard split AND on a constructed
cold-condition split (20% of conditions held out entirely so p_c is undefined by
construction for every cold-validate row):

  L   -- label-derived only: p_c, drug degree, condition degree, record_count, exp08's
         nb_excess at three tiers (gene/leaf/L1), n_neighbours. Nothing else.
  B   -- label-free biology only: gene overlap (disease + phenotype arm), pathway overlap
         (OT target parquet, joined on approvedSymbol/ensembl id per exp15's fix),
         drug-intrinsic mechanism/target-biology (exp07's blocks minus substance-type
         columns), condition-intrinsic biology (n_ot_genes, max_ot_score, organ system,
         therapeutic area), off-target panel affinities and the syndrome-interaction
         feature, when the underlying files are present.
  L+B -- both.

Reuse, per the task's instructions:
- degree/p_c block: experiments/exp07_ablation_rerun_leak_audit.py
- nb_excess/n_neighbours tier features: experiments/exp08_neighbour_transport_pc_stripped.py
- gene-overlap sparse-matrix machinery (disease + phenotype arm):
  experiments/exp14_gene_overlap_disease_arm.py
- pathway overlap join (ensembl_gene_id, OT target parquet 'pathways' column) and its
  >=60% coverage assertion: experiments/exp15_pathway_overlap_and_valid_placebos.py
- floors.py (p_c-lookup and degree+p_c floors) for every population
- leak_audit.py's blacklist logic, extended here with a by-construction whitelist check
  for B: every B column must come from one of a small set of named, code-reviewed
  construction blocks (gene overlap, pathway overlap, drug-intrinsic minus substance-type,
  condition-intrinsic biology subset, off-target panel, syndrome interaction) -- never from
  scanning the full feature space by name pattern alone.
"""

# ruff: noqa: N806 -- linear-algebra matrix-naming convention (C_score, D_binary, GP,
# Drug_pathway, Cond_pathway), consistent with exp10/exp14/exp15's use of the same
# convention in this repo.

from __future__ import annotations

import logging
import pathlib
import time
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

app = modal.App("bridge-exp18")

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
    .add_local_python_source("floors")
)
image = image.add_local_file(
    pathlib.Path(__file__).parent / "floors.py", remote_path="/root/floors.py"
)

data = modal.Volume.from_name("bridge-data")
results = modal.Volume.from_name("bridge-results", create_if_missing=True)

# ================================================================================
# shared constants (copied from exp07/exp08/exp14/exp15)
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

# exp17 is investigating these as confounds (substance-type separation, not biology).
# Excluded from B here per this spec's instruction.
SUBSTANCE_TYPE_EXCLUDE: set[str] = {
    "molecule_type",
    "availability_label",
    "has_chembl_match",
    "oral",
    "parenteral",
    "topical",
    "n_routes_labelled",
    "exposure_type",
}

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

# Columns that identify a label-derived (L) feature. B must never contain any of these
# names or a column starting with one of these prefixes -- the "back door" trap named in
# the spec (target encodings / shrinkage priors / anything aggregated over train labels).
L_ONLY_BLACKLIST_EXACT: set[str] = {"p_c", "drug_degree_train", "condition_degree_train"}
L_ONLY_BLACKLIST_PREFIXES: tuple[str, ...] = (
    "degree_",
    "nb_excess_",
    "n_neighbours_",
    "log1p_drug_degree",
    "log1p_condition_degree",
    "log1p_record_count",
)

HGB_KWARGS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "early_stopping": False,
    "random_state": 0,
}
K_SHRINK = 10
BOOT_RESAMPLES = 1000
TOP_M_PATHWAY_GENES = 50
COLD_CONDITION_HOLDOUT_FRACTION = 0.20
COLD_SPLIT_SEED = 0


def _fail(message: str) -> dict[str, Any]:
    return {"precondition_failed": 1.0, "precondition_error_message": message}


def _audit_columns(columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if c in LEAK_BLACKLIST_EXACT or any(c.startswith(p) for p in LEAK_BLACKLIST_PREFIXES)
    ]


def _audit_b_columns(columns: list[str]) -> list[str]:
    """B whitelist check: reject any column that is (a) label-blacklisted, or (b) matches
    the L-only feature naming convention. B is assembled from a small, fixed set of named
    construction blocks (see run()); this is a second, independent check on top of that
    by-construction guarantee, not a substitute for it.
    """
    hits = _audit_columns(columns)
    for c in columns:
        if c in L_ONLY_BLACKLIST_EXACT or any(c.startswith(p) for p in L_ONLY_BLACKLIST_PREFIXES):
            hits.append(c)
    return sorted(set(hits))


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


# ================================================================================
# L: degree + p_c + nb_excess/n_neighbours tiers -- computed fresh for ANY (train, val)
# partition, so the same function serves the standard split and the constructed
# cold-condition split.
# ================================================================================


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
        (np.ones(len(pairs), dtype=np.float64), (ing_pos, val_codes)), shape=(n_ing, n_val)
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


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y", "t"])


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

    return {
        f"n_neighbours_{tier_name}__train": n_neighbours_train,
        f"n_neighbours_{tier_name}__val": n_neighbours_val,
        f"nb_excess_{tier_name}__train": rate_train - p_c_train,
        f"nb_excess_{tier_name}__val": rate_val - p_c_val,
    }


def _build_l_design(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cond_basic: pd.DataFrame,
    edges_raw: pd.DataFrame,
    gene_pairs: pd.DataFrame,
    leaf_pairs: pd.DataFrame,
    l1_pairs: pd.DataFrame,
    label: str,
    log: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """L, computed strictly from (train_df, val_df). Reusable for the standard split
    and the constructed cold-condition split -- everything here is fit on `train_df`
    only and mapped onto `val_df`.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    notes: list[str] = []

    drug_degree = train_df.groupby("ingredient_concept_id").size()
    cond_degree = train_df.groupby("condition_concept_id").size()
    drug_degree_median = float(drug_degree.median()) if len(drug_degree) else 0.0
    cond_degree_median = float(cond_degree.median()) if len(cond_degree) else 0.0

    record_count_map = cond_basic.set_index("condition_concept_id")["record_count"]
    record_count_median = float(record_count_map.median())

    p_c_map = train_df.groupby("condition_concept_id")[label].mean()
    train_wide_mean = float(train_df[label].mean())
    n_unseen_val = int((~val_df["condition_concept_id"].isin(p_c_map.index)).sum())
    if n_unseen_val:
        notes.append(
            f"{n_unseen_val} evaluate rows had a condition unseen in this split's train; "
            f"p_c fell back to the train-wide {label} rate ({train_wide_mean:.4f})."
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

    l_train = _apply(train_df)
    l_val = _apply(val_df)

    all_ing = pd.Index(
        sorted(
            set(train_df["ingredient_concept_id"])
            | set(val_df["ingredient_concept_id"])
            | set(edges_raw["ingredient_concept_id"])
        )
    )
    n_ing = len(all_ing)
    ing_index_map = {ing: i for i, ing in enumerate(all_ing)}
    splits_train_ing = set(train_df["ingredient_concept_id"].unique())
    train_mask = np.array([1.0 if i in splits_train_ing else 0.0 for i in all_ing])
    d_pos_train = train_df["ingredient_concept_id"].map(ing_index_map).to_numpy()
    d_pos_val = val_df["ingredient_concept_id"].map(ing_index_map).to_numpy()

    all_cond = pd.Index(
        sorted(set(train_df["condition_concept_id"]) | set(val_df["condition_concept_id"]))
    )
    cond_index_map = {c: i for i, c in enumerate(all_cond)}
    n_cond = len(all_cond)
    c_pos_train = train_df["condition_concept_id"].map(cond_index_map).to_numpy()
    c_pos_val = val_df["condition_concept_id"].map(cond_index_map).to_numpy()

    p_c_by_cond = p_c_map.reindex(all_cond).fillna(train_wide_mean).to_numpy()
    p_c_train_arr = p_c_by_cond[c_pos_train]
    p_c_val_arr = p_c_by_cond[c_pos_val]

    y_train_real = train_df[label].to_numpy()
    flagged_idx = np.where(y_train_real == 1)[0]
    f_train = sp.csr_matrix(
        (np.ones(len(flagged_idx)), (d_pos_train[flagged_idx], c_pos_train[flagged_idx])),
        shape=(n_ing, n_cond),
    )
    f_train.sum_duplicates()
    f_train.data[:] = 1.0

    common_kwargs = {
        "ing_index_map": ing_index_map,
        "n_ing": n_ing,
        "train_mask": train_mask,
        "d_pos_train": d_pos_train,
        "d_pos_val": d_pos_val,
        "c_pos_train": c_pos_train,
        "c_pos_val": c_pos_val,
        "f_train": f_train,
        "p_c_train": p_c_train_arr,
        "p_c_val": p_c_val_arr,
        "log": log,
    }
    gene_feats = _tier_features_excess("gene", gene_pairs, **common_kwargs)
    leaf_feats = _tier_features_excess("class_leaf", leaf_pairs, **common_kwargs)
    l1_feats = _tier_features_excess("class_L1", l1_pairs, **common_kwargs)

    has_target_annotation_by_ing = np.array(
        [1.0 if i in set(edges_raw["ingredient_concept_id"]) else 0.0 for i in all_ing]
    )

    def _attach(df_out: pd.DataFrame, feats: dict, split: str, d_pos) -> pd.DataFrame:
        suffix = f"__{split}"
        for name, arr in feats.items():
            if name.endswith(suffix):
                df_out[name[: -len(suffix)]] = arr
        df_out["has_target_annotation"] = has_target_annotation_by_ing[d_pos]
        return df_out

    for feats in (gene_feats, leaf_feats, l1_feats):
        l_train = _attach(l_train, feats, "train", d_pos_train)
        l_val = _attach(l_val, feats, "val", d_pos_val)

    excess_cols = [c for c in l_train.columns if c.startswith("nb_excess_")]
    for df_out in (l_train, l_val):
        mask = df_out["has_target_annotation"] == 0
        df_out.loc[mask, excess_cols] = float("nan")

    l_train = l_train.drop(columns=["has_target_annotation"])
    l_val = l_val.drop(columns=["has_target_annotation"])
    return l_train, l_val, notes


# ================================================================================
# B: drug-intrinsic (minus substance-type), condition-intrinsic biology subset
# ================================================================================


def _load_drug_features_b(
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

    drop_cols = set(DRUG_LIST_COLUMNS) | set(IDENTITY_EXTRA_DROP) | SUBSTANCE_TYPE_EXCLUDE
    drop_cols |= {c for c in drug.columns if c.startswith("chembl_fuzzy_")}
    keep_cols = [c for c in keep_cols if c not in drop_cols]

    join_col = "omop_concept_id"
    out = drug[[join_col, "chembl_id", *keep_cols]].copy()

    diagnostics = {
        "present_blocks_in_dictionary": present_blocks,
        "matched_blocks": matched_blocks,
        "unmatched_wanted_blocks": unmatched_wanted,
        "n_drug_columns_selected": len(keep_cols),
    }
    return out, diagnostics


def _build_drug_design_b(drug_raw: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    df = drug_raw.copy()
    join_col = "omop_concept_id"
    feature_cols = [c for c in df.columns if c not in (join_col, "chembl_id")]

    out_parts = [df[[join_col]]]
    for col in feature_cols:
        if df[col].dtype == object:
            out_parts.append(_cap_and_onehot(df[col], prefix=f"drugB_{col}"))
        elif df[col].dtype == bool:
            out_parts.append(df[[col]].astype(float).rename(columns={col: f"drugB_{col}"}))
        else:
            out_parts.append(df[[col]].rename(columns={col: f"drugB_{col}"}))

    return pd.concat(out_parts, axis=1)


def _load_condition_biology_b(cond_path: pathlib.Path, group_path: pathlib.Path) -> pd.DataFrame:
    """condition-intrinsic biology subset for B: n_ot_genes, max_ot_score, organ system,
    therapeutic area only -- deliberately narrower than exp07's full condition-intrinsic
    block, which includes record_count (an observation-volume/degree quantity, already in
    L) and best_match_tier/n_hpo_genes (mapping-quality, not biology).
    """
    import pandas as pd

    cond = pd.read_csv(cond_path)
    base = cond[["condition_concept_id", "n_ot_genes", "max_ot_score"]].copy()
    base = base.rename(
        columns={"n_ot_genes": "condB_n_ot_genes", "max_ot_score": "condB_max_ot_score"}
    )
    base["condB_max_ot_score"] = base["condB_max_ot_score"].fillna(0.0)

    group_long = pd.read_csv(group_path)
    group_long = group_long[
        group_long["group_source"].isin(["hpo_organ_system", "ot_therapeutic_area"])
    ]
    group_long["col_label"] = (
        "condB_" + group_long["group_source"] + "__" + group_long["group_label"]
    )
    group_wide = (
        group_long.pivot_table(
            index="condition_concept_id", columns="col_label", values="group_id", aggfunc="count"
        )
        .fillna(0)
        .clip(upper=1)
    )
    group_wide = group_wide.reset_index()

    out = base.merge(group_wide, on="condition_concept_id", how="left")
    group_cols = [c for c in out.columns if c.startswith("condB_") and "__" in c]
    out[group_cols] = out[group_cols].fillna(0.0)
    return out


# ================================================================================
# gene overlap (disease + phenotype arm), adapted from exp14
# ================================================================================


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
    coverage = float(
        len(set(ing["ensembl_gene_id"]) & set(ot[gene_col].dropna()))
        / max(len(set(ing["ensembl_gene_id"])), 1)
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

    df_gene = np.asarray(C_binary.sum(axis=0)).ravel()
    idf = np.log(n_cond / np.maximum(df_gene, 1.0))
    C_idf = C_binary.multiply(idf).tocsr()

    log.info(
        "disease-arm gene join coverage: %.1f%% of drug target genes appear in the disease gene set",
        coverage * 100,
    )
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
        "C_idf": C_idf,
        "D_binary": D_binary,
        "coverage": coverage,
    }


def _build_hpo_matrices(hpo_path: pathlib.Path, target_long_path: pathlib.Path, log):
    import numpy as np
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
    import numpy as np
    import pandas as pd

    D_binary = ctx["D_binary"]
    C_binary = ctx["C_binary"]

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
    if "C_idf" in ctx:
        idf_matrix = (D_binary @ ctx["C_idf"].T).toarray()
        out[f"{prefix}shared_gene_idf_sum"] = _gather(idf_matrix)

    return out


# ================================================================================
# pathway overlap, adapted from exp15 (ensembl_gene_id join, >=60% coverage assert)
# ================================================================================


def _build_gene_pathway_matrix(target_glob: list[pathlib.Path], gene_index: dict, log):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    tgt = pd.concat([pd.read_parquet(p) for p in target_glob], ignore_index=True)
    gene_id_col = _detect_col(
        list(tgt.columns), ["id", "target_id", "ensembl_gene_id", "ensemblId"]
    )
    pathway_col = _detect_col(
        list(tgt.columns),
        ["pathways", "reactome", "reactomeIds", "pathwayIds", "reactome_pathways"],
    )
    assert gene_id_col is not None, (
        f"target parquet has no gene identifier column; columns were {list(tgt.columns)}"
    )
    assert pathway_col is not None, (
        f"target parquet has no pathway-membership column; columns were {list(tgt.columns)}"
    )
    log.info("pathway join: gene id col=%s, pathway col=%s", gene_id_col, pathway_col)

    def _flatten(val: object) -> list[str]:
        if val is None:
            return []
        try:
            items = list(val)
        except TypeError:
            return []
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                pid = item.get("pathwayId") or item.get("id") or item.get("pathway")
                if pid:
                    out.append(str(pid))
            elif item is not None:
                out.append(str(item))
        return out

    tgt["_pathways"] = tgt[pathway_col].apply(_flatten)
    exploded = tgt[[gene_id_col, "_pathways"]].explode("_pathways").dropna(subset=["_pathways"])
    exploded = exploded[exploded[gene_id_col].isin(gene_index)]

    covered_genes = set(exploded[gene_id_col].unique())
    covered_share = len(covered_genes & set(gene_index)) / max(len(gene_index), 1)

    if exploded.empty:
        return None, covered_share

    pathway_ids = sorted(exploded["_pathways"].unique())
    pathway_pos = {p: i for i, p in enumerate(pathway_ids)}
    n_gene = len(gene_index)
    n_pathway = len(pathway_ids)
    grows = exploded[gene_id_col].map(gene_index).to_numpy(dtype=int)
    gcols = exploded["_pathways"].map(pathway_pos).to_numpy(dtype=int)
    GP = sp.csr_matrix((np.ones(len(grows)), (grows, gcols)), shape=(n_gene, n_pathway))
    GP.sum_duplicates()
    GP.data = np.minimum(GP.data, 1.0)
    return {"GP": GP, "pathway_pos": pathway_pos, "n_pathway": n_pathway}, covered_share


# ================================================================================
# off-target panel affinities + syndrome-interaction feature
# ================================================================================


def _build_offtarget_wide(offtarget_path: pathlib.Path, drug_path: pathlib.Path, log):
    """Per-ingredient wide off-target affinity table: max pChEMBL value per off-target,
    joined ingredient_concept_id -> chembl_id (via ingredient_features.csv) ->
    molecule_chembl_id (the off-target file's own key). Purely drug-intrinsic and
    label-free.
    """
    import pandas as pd

    off = pd.read_csv(offtarget_path)
    drug = pd.read_csv(drug_path, usecols=["omop_concept_id", "chembl_id"]).dropna(
        subset=["chembl_id"]
    )
    off = off.merge(drug, left_on="molecule_chembl_id", right_on="chembl_id", how="inner")
    wide = off.pivot_table(
        index="omop_concept_id", columns="offtarget", values="pchembl_value", aggfunc="max"
    )
    wide.columns = [f"offtargetB_pchembl_{c}" for c in wide.columns]
    wide = wide.reset_index().rename(columns={"omop_concept_id": "ingredient_concept_id"})
    n_ingredients = wide["ingredient_concept_id"].nunique()
    log.info(
        "off-target panel: %d ingredients matched via chembl_id, %d off-targets",
        n_ingredients,
        wide.shape[1] - 1,
    )
    return wide, n_ingredients


def _build_syndrome_feature(
    syndrome_map: dict[str, list[int]],
    offtarget_wide: pd.DataFrame,
    pairs: pd.DataFrame,
    log,
) -> pd.DataFrame:
    """For each (drug, condition) pair: the max off-target affinity (pChEMBL) among
    off-targets whose syndrome map lists this pair's condition_concept_id, else 0, plus
    a binary flag. Sparse by construction (most pairs have no match) and entirely
    label-free: syndrome_map and off-target affinities are both drug/condition-intrinsic.
    """
    import numpy as np
    import pandas as pd

    off_cols = [c for c in offtarget_wide.columns if c.startswith("offtargetB_pchembl_")]
    target_names = [c[len("offtargetB_pchembl_") :] for c in off_cols]

    off_by_ing = offtarget_wide.set_index("ingredient_concept_id")

    cond_to_targets: dict[int, list[str]] = {}
    for target_name, cond_ids in syndrome_map.items():
        if target_name not in target_names:
            continue
        for cid in cond_ids:
            cond_to_targets.setdefault(int(cid), []).append(target_name)

    merged = pairs[["ingredient_concept_id", "condition_concept_id"]].merge(
        off_by_ing, on="ingredient_concept_id", how="left"
    )

    max_affinity = np.zeros(len(merged), dtype=float)
    cond_ids_arr = merged["condition_concept_id"].to_numpy()
    for cid, target_list in cond_to_targets.items():
        row_mask = cond_ids_arr == cid
        if not row_mask.any():
            continue
        cols = [f"offtargetB_pchembl_{t}" for t in target_list]
        vals = merged.loc[row_mask, cols].to_numpy(dtype=float)
        max_affinity[row_mask] = np.nanmax(
            np.where(np.isnan(vals), -np.inf, vals), axis=1, initial=-np.inf
        )
    max_affinity = np.where(np.isneginf(max_affinity), 0.0, max_affinity)
    max_affinity = np.nan_to_num(max_affinity, nan=0.0)

    out = pd.DataFrame(index=pairs.index)
    out["syndromeB_max_affinity"] = max_affinity
    out["syndromeB_has_match"] = (max_affinity > 0).astype(float)
    n_matched = int(out["syndromeB_has_match"].sum())
    log.info("syndrome-interaction feature: %d/%d pairs matched", n_matched, len(pairs))
    return out


# ================================================================================
# drug_macro_auc machinery (per METRIC.md), copied from exp14/exp15
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
        s = grp["score"]
        if s.isna().any():
            s = s.fillna(s.median())
        auc = float(roc_auc_score(grp["y"], s.to_numpy(dtype=float)))
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


def _bootstrap_vs_floor_ci(
    table: pd.DataFrame, floor: float, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    import numpy as np

    if table.empty or not np.isfinite(floor):
        return float("nan"), float("nan"), float("nan")
    diffs = (table["auc"] - floor).to_numpy()
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
        "model": final_model,
    }


# ================================================================================
# constructed cold-condition split
# ================================================================================


def _build_constructed_cold_split(
    train: pd.DataFrame, validate: pd.DataFrame, log
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the split holding out 20% of CONDITIONS entirely, from the union of
    train+validate (never test). Drugs may appear on both sides; conditions may not --
    p_c is undefined by construction for every cold_validate row.
    """
    import numpy as np
    import pandas as pd

    pool = pd.concat([train, validate], ignore_index=True)
    all_conditions = np.array(sorted(pool["condition_concept_id"].unique()))
    rng = np.random.default_rng(COLD_SPLIT_SEED)
    shuffled = rng.permutation(all_conditions)
    n_cold = round(len(shuffled) * COLD_CONDITION_HOLDOUT_FRACTION)
    cold_conditions = set(shuffled[:n_cold])

    cold_val_mask = pool["condition_concept_id"].isin(cold_conditions)
    cold_train = pool.loc[~cold_val_mask].reset_index(drop=True)
    cold_val = pool.loc[cold_val_mask].reset_index(drop=True)

    overlap = set(cold_train["condition_concept_id"]) & set(cold_val["condition_concept_id"])
    log.info(
        "constructed cold split: %d/%d conditions held out, cold_train=%d rows, "
        "cold_val=%d rows, condition overlap=%d (must be 0)",
        len(cold_conditions),
        len(all_conditions),
        len(cold_train),
        len(cold_val),
        len(overlap),
    )
    assert not overlap, "constructed cold split leaked conditions across train/validate"
    return cold_train, cold_val


# ================================================================================
# preconditions
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
# main Modal function
# ================================================================================


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=16.0,
    memory=32768,
    timeout=900,
)
def run(exp_id: str) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp18")

    import numpy as np
    import pandas as pd
    from floors import floors

    t0 = time.time()
    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)
    fallback_notes: list[str] = []

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

    offtarget_path = data_root / "drug" / "offtarget_activities.csv"
    syndrome_path = data_root / "offtarget_syndrome_map.json"
    have_offtarget = offtarget_path.exists()
    have_syndrome = syndrome_path.exists()
    log.info("off-target panel present=%s, syndrome map present=%s", have_offtarget, have_syndrome)
    if not have_offtarget:
        fallback_notes.append("offtarget_activities.csv not staged on the volume; panel skipped.")
    if not have_syndrome:
        fallback_notes.append(
            "offtarget_syndrome_map.json not staged on the volume; feature skipped."
        )

    log.info("loading train/validate/reference tables")
    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    cond_basic = pd.read_csv(paths["cond"])
    edges_raw = pd.read_csv(paths["target_long"]).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    label = "y_faers_signal"

    # ---- gene-tier edges for nb_excess (exp08's construction) ----
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
    gene_pairs_nb = _dedup_pairs(primary_gene_edges_raw, "gene_symbol")
    leaf_pairs_nb = _dedup_pairs(edges_raw, "chembl_protein_class_leaf")
    l1_pairs_nb = _dedup_pairs(edges_raw, "chembl_protein_class_L1")

    # ================================================================================
    # L on the standard split
    # ================================================================================
    log.info("building L (standard split)")
    l_train, l_val, l_notes = _build_l_design(
        train,
        validate,
        cond_basic,
        edges_raw,
        gene_pairs_nb,
        leaf_pairs_nb,
        l1_pairs_nb,
        label,
        log,
    )
    fallback_notes.extend(l_notes)

    # ================================================================================
    # B: static per-pair / per-drug / per-condition features, shared by BOTH splits
    # ================================================================================
    log.info("building B (drug-intrinsic, condition-intrinsic biology subset)")
    drug_raw_b, _drug_diag = _load_drug_features_b(paths["drug"], paths["dict"])
    drug_design_b = _build_drug_design_b(drug_raw_b).rename(
        columns={"omop_concept_id": "ingredient_concept_id"}
    )
    cond_design_b = _load_condition_biology_b(paths["cond"], paths["group"])

    log.info("building gene-overlap matrices (disease arm + phenotype arm)")
    disease_ctx = _build_disease_gene_matrices(paths["ot"], paths["target_long"], log)
    try:
        hpo_ctx = _build_hpo_matrices(paths["hpo"], paths["target_long"], log)
    except Exception as exc:
        log.warning("phenotype-arm (HPO) matrix construction failed, skipping: %s", exc)
        hpo_ctx = None
        fallback_notes.append(f"HPO overlap skipped: {exc}")

    all_pairs = pd.concat(
        [
            train[["ingredient_concept_id", "condition_concept_id"]],
            validate[["ingredient_concept_id", "condition_concept_id"]],
        ],
        keys=["train", "validate"],
    )
    disease_raw_all = _pair_gene_overlap_features(
        all_pairs, disease_ctx, log, prefix="geneB_", weighted=False
    )
    disease_weighted_all = _pair_gene_overlap_features(
        all_pairs, disease_ctx, log, prefix="geneB_", weighted=True
    )
    if hpo_ctx is not None:
        hpo_feats_all = _pair_gene_overlap_features(
            all_pairs, hpo_ctx, log, prefix="geneB_hpo_", weighted=False
        )
    else:
        hpo_feats_all = pd.DataFrame(index=all_pairs.index)
    gene_overlap_all = pd.concat([disease_weighted_all, hpo_feats_all], axis=1)
    share_any_shared_gene = float((disease_raw_all["geneB_n_shared_genes"] > 0).mean())
    log.info("share of pairs with any shared gene (disease arm) = %.4f", share_any_shared_gene)

    # ---- pathway overlap (exp15's fix: ensembl_gene_id key, >=60% coverage assert) ----
    log.info("building pathway overlap")
    target_glob = sorted((data_root / "ref" / "ot").glob("target__part-*.parquet"))
    if not target_glob:
        msg = "no target__part-*.parquet files found under /data/ref/ot"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    drug_target_genes = set(edges_raw["ensembl_gene_id"].dropna().unique())
    gene_index = {g: i for i, g in enumerate(sorted(drug_target_genes))}
    pathway_ctx, pathway_coverage = _build_gene_pathway_matrix(target_glob, gene_index, log)
    log.info(
        "gene->pathway map covers %.1f%% of %d drug target genes (need >=60%%)",
        pathway_coverage * 100,
        len(gene_index),
    )
    pathway_coverage_df = pd.DataFrame(
        [{"n_drug_target_genes": len(gene_index), "covered_share": pathway_coverage}]
    )
    pathway_coverage_df.to_csv(out / f"{exp_id}_pathway_join_coverage.csv", index=False)
    if pathway_coverage < 0.60 or pathway_ctx is None:
        msg = (
            f"gene->pathway map covers only {pathway_coverage:.3f} of {len(gene_index)} drug "
            "target genes (need >=0.60)"
        )
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    import scipy.sparse as sp

    n_ing_all = len(set(train["ingredient_concept_id"]) | set(validate["ingredient_concept_id"]))
    all_ing_idx = sorted(
        set(train["ingredient_concept_id"]) | set(validate["ingredient_concept_id"])
    )
    drug_pos_pw = {d: i for i, d in enumerate(all_ing_idx)}
    all_genes_edges = edges_raw.dropna(subset=["ensembl_gene_id"])
    drug_gene_pairs = all_genes_edges[
        ["ingredient_concept_id", "ensembl_gene_id"]
    ].drop_duplicates()
    drug_gene_pairs = drug_gene_pairs[
        drug_gene_pairs["ensembl_gene_id"].isin(gene_index)
        & drug_gene_pairs["ingredient_concept_id"].isin(drug_pos_pw)
    ]
    D_genes = sp.csr_matrix(
        (
            np.ones(len(drug_gene_pairs)),
            (
                drug_gene_pairs["ingredient_concept_id"].map(drug_pos_pw).to_numpy(dtype=int),
                drug_gene_pairs["ensembl_gene_id"].map(gene_index).to_numpy(dtype=int),
            ),
        ),
        shape=(n_ing_all, len(gene_index)),
    )
    D_genes.sum_duplicates()
    D_genes.data = np.minimum(D_genes.data, 1.0)

    all_cond_idx = sorted(
        set(train["condition_concept_id"]) | set(validate["condition_concept_id"])
    )
    cond_pos_pw = {c: i for i, c in enumerate(all_cond_idx)}
    ot = pd.read_csv(paths["ot"], usecols=["condition_concept_id", "ensembl_gene_id", "ot_score"])
    ot = ot.dropna(subset=["condition_concept_id", "ensembl_gene_id"])
    ot = ot[ot["ensembl_gene_id"].isin(gene_index) & ot["condition_concept_id"].isin(cond_pos_pw)]
    ot["ot_score"] = ot["ot_score"].fillna(0.0)
    rows_l, cols_l = [], []
    for cid, g in ot.groupby("condition_concept_id"):
        gg = g.nlargest(TOP_M_PATHWAY_GENES, "ot_score")
        ci = cond_pos_pw[cid]
        for gene_id in gg["ensembl_gene_id"]:
            rows_l.append(ci)
            cols_l.append(gene_index[gene_id])
    C_topm = (
        sp.csr_matrix(
            (np.ones(len(rows_l)), (rows_l, cols_l)), shape=(len(all_cond_idx), len(gene_index))
        )
        if rows_l
        else sp.csr_matrix((len(all_cond_idx), len(gene_index)))
    )
    C_topm.sum_duplicates()
    C_topm.data = np.minimum(C_topm.data, 1.0)

    GP = pathway_ctx["GP"]
    Drug_pathway = (D_genes @ GP).tocsr()
    Drug_pathway.data = np.minimum(Drug_pathway.data, 1.0)
    Cond_pathway = (C_topm @ GP).tocsr()
    Cond_pathway.data = np.minimum(Cond_pathway.data, 1.0)
    n_shared_pathway_matrix = (Drug_pathway @ Cond_pathway.T).toarray()
    d_pw_count = np.asarray(Drug_pathway.sum(axis=1)).ravel()
    c_pw_count = np.asarray(Cond_pathway.sum(axis=1)).ravel()
    denom_pw = d_pw_count[:, None] + c_pw_count[None, :] - n_shared_pathway_matrix
    jaccard_pathway_matrix = np.divide(
        n_shared_pathway_matrix,
        denom_pw,
        out=np.zeros_like(n_shared_pathway_matrix),
        where=denom_pw > 0,
    )

    def _gather_pathway(df: pd.DataFrame) -> pd.DataFrame:
        d_arr = df["ingredient_concept_id"].map(drug_pos_pw).to_numpy()
        c_arr = df["condition_concept_id"].map(cond_pos_pw).to_numpy()
        valid = ~(pd.isna(d_arr) | pd.isna(c_arr))
        d_i = np.where(valid, d_arr, 0).astype(int)
        c_i = np.where(valid, c_arr, 0).astype(int)
        n_shared = np.zeros(len(df))
        n_shared[valid] = n_shared_pathway_matrix[d_i[valid], c_i[valid]]
        jac = np.zeros(len(df))
        jac[valid] = jaccard_pathway_matrix[d_i[valid], c_i[valid]]
        out = pd.DataFrame(index=df.index)
        out["pathwayB_n_shared_pathways"] = n_shared
        out["pathwayB_jaccard_pathways"] = jac
        return out

    pathway_all = _gather_pathway(
        pd.concat(
            [
                train[["ingredient_concept_id", "condition_concept_id"]],
                validate[["ingredient_concept_id", "condition_concept_id"]],
            ],
            keys=["train", "validate"],
        )
    )

    # ---- off-target panel + syndrome interaction (label-free, optional) ----
    offtarget_wide = None
    if have_offtarget:
        offtarget_wide, _n_off_ing = _build_offtarget_wide(offtarget_path, paths["drug"], log)
        offtarget_pair_all = all_pairs[["ingredient_concept_id", "condition_concept_id"]].merge(
            offtarget_wide, on="ingredient_concept_id", how="left"
        )
        off_cols = [c for c in offtarget_wide.columns if c.startswith("offtargetB_pchembl_")]
        offtarget_pair_all[off_cols] = offtarget_pair_all[off_cols].fillna(0.0)
        offtarget_pair_all.index = all_pairs.index
        offtarget_pair_all = offtarget_pair_all[off_cols]
    else:
        offtarget_pair_all = pd.DataFrame(index=all_pairs.index)

    syndrome_all = pd.DataFrame(index=all_pairs.index)
    if have_offtarget and have_syndrome:
        import json

        with syndrome_path.open() as fh:
            syndrome_map = json.load(fh)
        syndrome_all = _build_syndrome_feature(
            syndrome_map,
            offtarget_wide,
            all_pairs[["ingredient_concept_id", "condition_concept_id"]],
            log,
        )
        syndrome_all.index = all_pairs.index

    def _split_by_key(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr = df.loc["train"].set_axis(train.index)
        va = df.loc["validate"].set_axis(validate.index)
        return tr, va

    gene_overlap_train, gene_overlap_val = _split_by_key(gene_overlap_all)
    pathway_train, pathway_val = _split_by_key(pathway_all)
    offtarget_train, offtarget_val = _split_by_key(offtarget_pair_all)
    syndrome_train, syndrome_val = _split_by_key(syndrome_all)

    def _assemble_b_pairlevel(base_df: pd.DataFrame, idx) -> pd.DataFrame:
        merged_drug = base_df[["ingredient_concept_id"]].merge(
            drug_design_b, on="ingredient_concept_id", how="left"
        )
        merged_drug.index = idx
        merged_drug = merged_drug.drop(columns=["ingredient_concept_id"])
        merged_cond = base_df[["condition_concept_id"]].merge(
            cond_design_b, on="condition_concept_id", how="left"
        )
        merged_cond.index = idx
        merged_cond = merged_cond.drop(columns=["condition_concept_id"])
        return merged_drug, merged_cond

    train_drug_b, train_cond_b = _assemble_b_pairlevel(train, train.index)
    val_drug_b, val_cond_b = _assemble_b_pairlevel(validate, validate.index)

    b_train = pd.concat(
        [
            train_drug_b,
            train_cond_b,
            gene_overlap_train,
            pathway_train,
            offtarget_train,
            syndrome_train,
        ],
        axis=1,
    )
    b_val = pd.concat(
        [val_drug_b, val_cond_b, gene_overlap_val, pathway_val, offtarget_val, syndrome_val], axis=1
    )

    # ---- leak audits: L must not carry label blacklist; B must not carry label
    # blacklist OR any L-only feature (the "back door" trap) ----
    l_hits = _audit_columns(list(l_train.columns))
    b_hits = _audit_b_columns(list(b_train.columns))
    if l_hits or b_hits:
        msg = f"leak audit failed: L hits={l_hits}, B hits={b_hits}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}
    log.info(
        "leak audit passed: L has %d cols, B has %d cols, no overlap/blacklist hits",
        len(l_train.columns),
        len(b_train.columns),
    )

    lb_train = pd.concat([l_train, b_train], axis=1)
    lb_val = pd.concat([l_val, b_val], axis=1)

    y_train = train[label]
    y_val = validate[label]
    groups_train = train["group_key"]
    ids_train = train["ingredient_concept_id"]
    ids_val = validate["ingredient_concept_id"]

    # ================================================================================
    # fit L, B, L+B on the standard split
    # ================================================================================
    warm_designs = {"L": (l_train, l_val), "B": (b_train, b_val), "LB": (lb_train, lb_val)}
    warm_results: dict[str, dict] = {}
    for name, (xtr, xva) in warm_designs.items():
        elapsed = time.time() - t0
        n_folds = 3
        if xtr.shape[1] > 500 or elapsed > 300:
            n_folds = 2
            fallback_notes.append(
                f"Dropped CV to 2 folds for warm/{name} (elapsed={elapsed:.0f}s)."
            )
        log.info(
            "fitting warm/%s: n_features=%d n_folds=%d elapsed=%.1fs",
            name,
            xtr.shape[1],
            n_folds,
            elapsed,
        )
        warm_results[name] = _fit_score(
            xtr, y_train, groups_train, ids_train, xva, y_val, ids_val, n_folds=n_folds
        )
        log.info(
            "warm/%s: cv=%.4f val_drug_macro_auc=%.4f",
            name,
            warm_results[name]["cv_drug_macro_auc_mean"],
            warm_results[name]["val_drug_macro_auc"],
        )

    # ---- natural cold conditions: validate conditions absent from (standard) train ----
    train_conditions = set(train["condition_concept_id"].unique())
    natural_cold_mask = ~validate["condition_concept_id"].isin(train_conditions)
    n_natural_cold_pairs = int(natural_cold_mask.sum())
    n_natural_cold_conditions = int(
        validate.loc[natural_cold_mask, "condition_concept_id"].nunique()
    )
    n_natural_cold_drugs = int(validate.loc[natural_cold_mask, "ingredient_concept_id"].nunique())
    log.info(
        "natural cold conditions: %d pairs, %d conditions, %d drugs (direction, not an "
        "estimate, if this is a few hundred rows)",
        n_natural_cold_pairs,
        n_natural_cold_conditions,
        n_natural_cold_drugs,
    )

    natural_cold_metrics = {}
    for name in ("L", "B", "LB"):
        proba_full = (
            warm_results[name]["model"].predict_proba(warm_designs[name][1].loc[natural_cold_mask])[
                :, 1
            ]
            if natural_cold_mask.any()
            else np.array([])
        )
        m = _drug_macro_metrics(
            ids_val.loc[natural_cold_mask], y_val.loc[natural_cold_mask], proba_full
        )
        natural_cold_metrics[name] = m

    natural_cold_floor = (
        floors(train, validate.loc[natural_cold_mask], label, eligibility=5)
        if n_natural_cold_pairs
        else {
            "floor_pc": float("nan"),
            "floor_degree_pc": float("nan"),
            "prevalence": float("nan"),
            "n_drugs_scored": 0,
        }
    )

    # ================================================================================
    # constructed cold-condition split: rebuild L (fresh p_c/degree/nb_excess), reuse
    # B (static per-pair -- just re-sliced), refit L / B / L+B
    # ================================================================================
    log.info("building constructed cold-condition split")
    cold_train, cold_val = _build_constructed_cold_split(train, validate, log)

    cold_l_train, cold_l_val, cold_l_notes = _build_l_design(
        cold_train,
        cold_val,
        cond_basic,
        edges_raw,
        gene_pairs_nb,
        leaf_pairs_nb,
        l1_pairs_nb,
        label,
        log,
    )
    fallback_notes.extend(f"[cold split] {n}" for n in cold_l_notes)

    # B is static per (ingredient, condition) pair -- rebuild it for the cold_train /
    # cold_val row sets by recomputing the same pair-level feature blocks on the union
    # (cold_train/cold_val is a strict re-partition of train+validate's rows).
    cold_pairs = pd.concat(
        [
            cold_train[["ingredient_concept_id", "condition_concept_id"]],
            cold_val[["ingredient_concept_id", "condition_concept_id"]],
        ],
        keys=["train", "validate"],
    )
    cold_disease_weighted = _pair_gene_overlap_features(
        cold_pairs, disease_ctx, log, prefix="geneB_", weighted=True
    )
    if hpo_ctx is not None:
        cold_hpo = _pair_gene_overlap_features(
            cold_pairs, hpo_ctx, log, prefix="geneB_hpo_", weighted=False
        )
    else:
        cold_hpo = pd.DataFrame(index=cold_pairs.index)
    cold_gene_overlap = pd.concat([cold_disease_weighted, cold_hpo], axis=1)
    cold_pathway = _gather_pathway(cold_pairs)

    if have_offtarget:
        cold_offtarget = cold_pairs[["ingredient_concept_id", "condition_concept_id"]].merge(
            offtarget_wide, on="ingredient_concept_id", how="left"
        )
        off_cols = [c for c in offtarget_wide.columns if c.startswith("offtargetB_pchembl_")]
        cold_offtarget[off_cols] = cold_offtarget[off_cols].fillna(0.0)
        cold_offtarget.index = cold_pairs.index
        cold_offtarget = cold_offtarget[off_cols]
    else:
        cold_offtarget = pd.DataFrame(index=cold_pairs.index)

    cold_syndrome = pd.DataFrame(index=cold_pairs.index)
    if have_offtarget and have_syndrome:
        cold_syndrome = _build_syndrome_feature(
            syndrome_map,
            offtarget_wide,
            cold_pairs[["ingredient_concept_id", "condition_concept_id"]],
            log,
        )
        cold_syndrome.index = cold_pairs.index

    def _split_cold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr = df.loc["train"].set_axis(cold_train.index)
        va = df.loc["validate"].set_axis(cold_val.index)
        return tr, va

    cold_gene_tr, cold_gene_va = _split_cold(cold_gene_overlap)
    cold_pw_tr, cold_pw_va = _split_cold(cold_pathway)
    cold_off_tr, cold_off_va = _split_cold(cold_offtarget)
    cold_syn_tr, cold_syn_va = _split_cold(cold_syndrome)

    cold_train_drug_b, cold_train_cond_b = _assemble_b_pairlevel(cold_train, cold_train.index)
    cold_val_drug_b, cold_val_cond_b = _assemble_b_pairlevel(cold_val, cold_val.index)

    cold_b_train = pd.concat(
        [cold_train_drug_b, cold_train_cond_b, cold_gene_tr, cold_pw_tr, cold_off_tr, cold_syn_tr],
        axis=1,
    )
    cold_b_val = pd.concat(
        [cold_val_drug_b, cold_val_cond_b, cold_gene_va, cold_pw_va, cold_off_va, cold_syn_va],
        axis=1,
    )
    cold_lb_train = pd.concat([cold_l_train, cold_b_train], axis=1)
    cold_lb_val = pd.concat([cold_l_val, cold_b_val], axis=1)

    cold_l_hits = _audit_columns(list(cold_l_train.columns))
    cold_b_hits = _audit_b_columns(list(cold_b_train.columns))
    if cold_l_hits or cold_b_hits:
        msg = f"leak audit failed on cold split: L hits={cold_l_hits}, B hits={cold_b_hits}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    cold_y_train = cold_train[label]
    cold_y_val = cold_val[label]
    cold_groups_train = cold_train["group_key"]
    cold_ids_train = cold_train["ingredient_concept_id"]
    cold_ids_val = cold_val["ingredient_concept_id"]

    cold_designs = {
        "L": (cold_l_train, cold_l_val),
        "B": (cold_b_train, cold_b_val),
        "LB": (cold_lb_train, cold_lb_val),
    }
    cold_results: dict[str, dict] = {}
    for name, (xtr, xva) in cold_designs.items():
        elapsed = time.time() - t0
        n_folds = 3
        if elapsed > 780:
            n_folds = 0
            fallback_notes.append(
                f"Dropped CV entirely for cold/{name}: {elapsed:.0f}s elapsed (protecting the "
                "cold-condition full-fit numbers, per the spec's budget-guard rule)."
            )
        elif xtr.shape[1] > 500 or elapsed > 600:
            n_folds = 2
            fallback_notes.append(
                f"Dropped CV to 2 folds for cold/{name} (elapsed={elapsed:.0f}s)."
            )
        log.info(
            "fitting cold/%s: n_features=%d n_folds=%d elapsed=%.1fs",
            name,
            xtr.shape[1],
            n_folds,
            elapsed,
        )
        cold_results[name] = _fit_score(
            xtr,
            cold_y_train,
            cold_groups_train,
            cold_ids_train,
            xva,
            cold_y_val,
            cold_ids_val,
            n_folds=n_folds,
        )
        log.info(
            "cold/%s: cv=%.4f val_drug_macro_auc=%.4f n_drugs=%d",
            name,
            cold_results[name]["cv_drug_macro_auc_mean"],
            cold_results[name]["val_drug_macro_auc"],
            cold_results[name]["val_n_drugs_scored"],
        )

    # ================================================================================
    # floors (warm, natural cold, constructed cold) via floors.py
    # ================================================================================
    floor_warm = floors(train, validate, label, eligibility=20)
    floor_cold_constructed = floors(cold_train, cold_val, label, eligibility=20)
    log.info("floor_warm=%s", floor_warm)
    log.info("floor_cold_constructed=%s", floor_cold_constructed)

    # ================================================================================
    # bootstrap CIs: B-vs-floor and B-vs-L, on cold (constructed) conditions
    # ================================================================================
    cold_b_table = cold_results["B"]["val_drug_table"]
    cold_l_table = cold_results["L"]["val_drug_table"]
    cold_lb_table = cold_results["LB"]["val_drug_table"]

    b_vs_floor_point, b_vs_floor_lo, b_vs_floor_hi = _bootstrap_vs_floor_ci(
        cold_b_table, floor_cold_constructed["floor_degree_pc"]
    )
    b_vs_l_point, b_vs_l_lo, b_vs_l_hi = _bootstrap_increment_ci(cold_l_table, cold_b_table)
    lb_vs_l_point, lb_vs_l_lo, lb_vs_l_hi = _bootstrap_increment_ci(cold_l_table, cold_lb_table)

    bootstrap_df = pd.DataFrame(
        [
            {
                "comparison": "B_vs_floor_degree_pc_cold_constructed",
                "point": b_vs_floor_point,
                "ci_lo_2.5": b_vs_floor_lo,
                "ci_hi_97.5": b_vs_floor_hi,
            },
            {
                "comparison": "B_vs_L_cold_constructed",
                "point": b_vs_l_point,
                "ci_lo_2.5": b_vs_l_lo,
                "ci_hi_97.5": b_vs_l_hi,
            },
            {
                "comparison": "LB_vs_L_cold_constructed",
                "point": lb_vs_l_point,
                "ci_lo_2.5": lb_vs_l_lo,
                "ci_hi_97.5": lb_vs_l_hi,
            },
        ]
    )
    bootstrap_df.to_csv(out / f"{exp_id}_bootstrap.csv", index=False)

    # ================================================================================
    # feature importance for B on cold conditions only
    # ================================================================================
    import re

    import lightgbm as lgb

    x_lgb = cold_b_train.copy()
    seen: dict[str, int] = {}
    sanitized: list[str] = []
    for c in x_lgb.columns:
        base = re.sub(r"[^0-9A-Za-z_]", "_", str(c))
        cnt = seen.get(base, 0)
        seen[base] = cnt + 1
        sanitized.append(base if cnt == 0 else f"{base}_{cnt}")
    x_lgb.columns = sanitized
    bool_cols = [c for c in x_lgb.columns if x_lgb[c].dtype == bool]
    if bool_cols:
        x_lgb[bool_cols] = x_lgb[bool_cols].astype(float)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.06, num_leaves=63, random_state=0, verbosity=-1
    )
    lgb_model.fit(x_lgb, cold_y_train)
    importances = lgb_model.booster_.feature_importance(importance_type="gain")
    top_idx = np.argsort(importances)[::-1][:25]
    importance_rows = [
        {"rank": rank, "feature": cold_b_train.columns[i], "gain": float(importances[i])}
        for rank, i in enumerate(top_idx, start=1)
    ]
    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(out / f"{exp_id}_B_importance_cold.csv", index=False)

    # ================================================================================
    # deliverables
    # ================================================================================
    models_rows = []
    for pop, res_dict, floor_dict in (
        ("warm", warm_results, floor_warm),
        ("cold_constructed", cold_results, floor_cold_constructed),
    ):
        for name in ("L", "B", "LB"):
            r = res_dict[name]
            models_rows.append(
                {
                    "population": pop,
                    "design": name,
                    "train_cv_drug_macro_auc": r["cv_drug_macro_auc_mean"],
                    "validate_drug_macro_auc": r["val_drug_macro_auc"],
                    "validate_drug_macro_p10": r["val_drug_macro_p10"],
                    "n_drugs_scored": r["val_n_drugs_scored"],
                    "floor_pc": floor_dict["floor_pc"],
                    "floor_degree_pc": floor_dict["floor_degree_pc"],
                    "prevalence": floor_dict["prevalence"],
                }
            )
    for name in ("L", "B", "LB"):
        m = natural_cold_metrics[name]
        models_rows.append(
            {
                "population": "natural_cold",
                "design": name,
                "train_cv_drug_macro_auc": float("nan"),
                "validate_drug_macro_auc": m["drug_macro_auc"],
                "validate_drug_macro_p10": m["drug_macro_p10"],
                "n_drugs_scored": m["n_drugs_scored"],
                "floor_pc": natural_cold_floor["floor_pc"],
                "floor_degree_pc": natural_cold_floor["floor_degree_pc"],
                "prevalence": natural_cold_floor["prevalence"],
            }
        )
    models_df = pd.DataFrame(models_rows)
    models_df.to_csv(out / f"{exp_id}_models.csv", index=False)

    cold_split_summary_df = pd.DataFrame(
        [
            {
                "n_conditions_total": len(
                    set(train["condition_concept_id"]) | set(validate["condition_concept_id"])
                ),
                "n_conditions_cold_val": cold_val["condition_concept_id"].nunique(),
                "n_conditions_cold_train": cold_train["condition_concept_id"].nunique(),
                "n_rows_cold_train": len(cold_train),
                "n_rows_cold_val": len(cold_val),
                "n_drugs_cold_val": cold_val["ingredient_concept_id"].nunique(),
                "label_rate_cold_train": float(cold_train[label].mean()),
                "label_rate_cold_val": float(cold_val[label].mean()),
            }
        ]
    )
    cold_split_summary_df.to_csv(out / f"{exp_id}_cold_split_summary.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels_plot = ["warm", "natural_cold", "cold_constructed"]
    for name, color in (("L", "tab:blue"), ("B", "tab:orange"), ("LB", "tab:green")):
        ys = [
            warm_results[name]["val_drug_macro_auc"],
            natural_cold_metrics[name]["drug_macro_auc"],
            cold_results[name]["val_drug_macro_auc"],
        ]
        ax.plot(labels_plot, ys, marker="o", label=name, color=color)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_ylabel("drug_macro_auc")
    ax.set_title("exp18: L / B / L+B across warm and cold-condition populations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_cold_vs_warm.png", dpi=150)
    plt.close(fig)

    elapsed_total = time.time() - t0
    log.info("total elapsed: %.1fs", elapsed_total)

    metrics = {
        "drug_macro_auc_L": warm_results["L"]["val_drug_macro_auc"],
        "drug_macro_auc_B": warm_results["B"]["val_drug_macro_auc"],
        "drug_macro_auc_LB": warm_results["LB"]["val_drug_macro_auc"],
        "drug_macro_auc_B_cold_natural": natural_cold_metrics["B"]["drug_macro_auc"],
        "drug_macro_auc_B_cold_constructed": cold_results["B"]["val_drug_macro_auc"],
        "drug_macro_auc_L_cold_constructed": cold_results["L"]["val_drug_macro_auc"],
        "drug_macro_auc_LB_cold_constructed": cold_results["LB"]["val_drug_macro_auc"],
        "floor_warm": floor_warm["floor_pc"],
        "floor_cold_constructed": floor_cold_constructed["floor_degree_pc"],
        "n_drugs_cold": cold_results["B"]["val_n_drugs_scored"],
        "n_conditions_cold": int(cold_val["condition_concept_id"].nunique()),
        "pathway_gene_coverage": pathway_coverage,
        "n_natural_cold_pairs": n_natural_cold_pairs,
        "n_natural_cold_conditions": n_natural_cold_conditions,
        "n_natural_cold_drugs": n_natural_cold_drugs,
        "B_vs_floor_cold_gain": b_vs_floor_point,
        "B_vs_floor_cold_ci_lo": b_vs_floor_lo,
        "B_vs_floor_cold_ci_hi": b_vs_floor_hi,
        "B_vs_L_cold_gain": b_vs_l_point,
        "B_vs_L_cold_ci_lo": b_vs_l_lo,
        "B_vs_L_cold_ci_hi": b_vs_l_hi,
        "validate_average_precision": warm_results["LB"]["val_ap"],
    }

    verdict_biological = (
        b_vs_floor_lo > 0
        and cold_results["B"]["val_drug_macro_auc"] > floor_cold_constructed["floor_degree_pc"]
    )
    findings = (
        (
            "Biological method: "
            if verdict_biological
            else "Nearest-neighbour method, not a biological method: "
        )
        + f"on the constructed cold-condition split ({cold_split_summary_df.iloc[0]['n_conditions_cold_val']:.0f} "
        f"conditions held out entirely, {cold_split_summary_df.iloc[0]['n_rows_cold_val']:.0f} validate rows, "
        f"{int(metrics['n_drugs_cold'])} drugs scored), label-free B scores "
        f"drug_macro_auc={metrics['drug_macro_auc_B_cold_constructed']:.4f} against its own floor "
        f"(degree+p_c-less degree-only floor)={floor_cold_constructed['floor_degree_pc']:.4f}: "
        f"B-vs-floor gain={b_vs_floor_point:+.4f}, 95% CI over drugs [{b_vs_floor_lo:+.4f}, {b_vs_floor_hi:+.4f}]. "
        f"B-vs-L on cold conditions (L collapses to degree+neighbour rates here, no p_c): gain="
        f"{b_vs_l_point:+.4f}, CI [{b_vs_l_lo:+.4f}, {b_vs_l_hi:+.4f}]. L+B vs L cold gain={lb_vs_l_point:+.4f}, "
        f"CI [{lb_vs_l_lo:+.4f}, {lb_vs_l_hi:+.4f}]. "
        f"Natural cold conditions (exp14's population, a direction not an estimate at this size): "
        f"{n_natural_cold_pairs} pairs / {n_natural_cold_conditions} conditions / {n_natural_cold_drugs} drugs, "
        f"B drug_macro_auc={natural_cold_metrics['B']['drug_macro_auc']:.4f}. "
        f"Warm-split reference: L={metrics['drug_macro_auc_L']:.4f}, B={metrics['drug_macro_auc_B']:.4f}, "
        f"L+B={metrics['drug_macro_auc_LB']:.4f} (floor_pc={floor_warm['floor_pc']:.4f}). "
        f"Pathway join coverage={pathway_coverage:.3f} of drug target genes (>=0.60 required, asserted "
        "before use, per exp15's fix -- joined on ensembl_gene_id against the OT target parquet's 'id' "
        "column). Share of validate pairs with any shared gene (disease arm)="
        f"{share_any_shared_gene:.4f}. "
        + (f"Fallbacks taken: {'; '.join(fallback_notes)}. " if fallback_notes else "")
        + "Off-target panel and syndrome-interaction feature "
        + (
            "were included in B."
            if have_offtarget and have_syndrome
            else "were NOT available and were skipped (see fallbacks)."
        )
    )

    artifacts = [
        str(out / f"{exp_id}_models.csv"),
        str(out / f"{exp_id}_cold_split_summary.csv"),
        str(out / f"{exp_id}_B_importance_cold.csv"),
        str(out / f"{exp_id}_bootstrap.csv"),
        str(out / f"{exp_id}_pathway_join_coverage.csv"),
        str(out / f"{exp_id}_cold_vs_warm.png"),
    ]

    results.commit()
    log.info("metrics: %s", metrics)
    return {"__metrics__": metrics, "__findings__": findings, "__artifacts__": artifacts}


@app.local_entrypoint()
def main() -> None:
    import subprocess

    from bridge import labnotebook as ln

    # Stage the two off-target files onto the volume if they exist locally and aren't
    # already there -- these are new inputs this round, not part of the README's
    # original staging list.
    repo_root = pathlib.Path(__file__).parent.parent
    local_offtarget = repo_root / "data" / "input" / "drug" / "offtarget_activities.csv"
    local_syndrome = repo_root / "offtarget_syndrome_map.json"
    for local_path, remote_path in (
        (local_offtarget, "/drug/offtarget_activities.csv"),
        (local_syndrome, "/offtarget_syndrome_map.json"),
    ):
        if local_path.exists():
            print(f"staging {local_path} -> bridge-data:{remote_path}")
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "modal",
                    "volume",
                    "put",
                    "bridge-data",
                    str(local_path),
                    remote_path,
                    "-f",
                ],
                check=False,
            )
        else:
            print(f"local file not found, skipping staging: {local_path}")

    exp = ln.register(
        agent="exp18",
        title="Label-free biology versus label-copying, with a constructed cold-condition split",
        hypothesis=(
            "A strictly label-free biology model beats chance on all conditions and beats "
            "the label-derived model on cold conditions, where p_c is undefined."
        ),
        approach=(
            "Three strictly partitioned designs L / B / L+B; pathway overlap unblocked via "
            "the OT target parquet 'pathways' column joined on approvedSymbol; a rebuilt "
            "split holding out 20% of conditions entirely so p_c is undefined by "
            "construction; floors for every population; importance for B on cold conditions."
        ),
        label="y_faers_signal",
        features=[
            "label_derived_block",
            "gene_overlap",
            "pathway_overlap",
            "mechanism",
            "offtarget_panel",
            "condition_biology",
        ],
        split="train/validate grouped by primary target gene, plus a drug-and-condition held-out split",
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

    findings = metrics.pop("__findings__")
    artifacts = metrics.pop("__artifacts__")
    metrics = metrics.get("__metrics__", metrics)
    print(findings)

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=artifacts,
        next_steps=(
            "If B is at floor on cold conditions, Round 5 should either commit to the "
            "nearest-neighbour method as the deliverable, or move to exp16's adjudicated "
            "outcomes / exp20's efficacy label, where biology may transport better. If B "
            "beats floor, prioritize densifying the off-target panel (exp19) and re-running "
            "this cold-condition test."
        ),
    )


if __name__ == "__main__":
    pass
