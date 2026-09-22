"""exp10 -- The bridge hypothesis, first direct test.

Implements experiments/exp10_gene_pathway_overlap.md exactly, scoring per
experiments/METRIC.md's drug_macro_auc definition. Baseline block (degree, p_c,
drug-intrinsic, condition-intrinsic block selection via the data dictionary) is copied
from experiments/exp07_ablation_rerun_leak_audit.py, which is itself copied from
experiments/exp02_intrinsic_blocks.py -- see those files for the block-selection spec.

Tightest-budget experiment in the round (~9 of a 10 min hard limit). The sparse-matrix
implementation note in the spec is not optional: every gene/pathway overlap feature for
all ~1.16M observed pairs is computed as one dense-ish matrix product (drugs x conditions)
read at the pairs' coordinates, never a Python loop over pairs.
"""

# ruff: noqa: N806 -- this file follows linear-algebra convention throughout (capital
# letters for matrices: C_score, C_binary, D_binary, etc.), consistent with the sklearn
# X/X_train convention already noqa'd elsewhere in this repo's experiment scripts.

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

app = modal.App("bridge-exp10")

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

# ---- copied from exp07_ablation_rerun_leak_audit.py (exp02's block-selection spec) ----

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

TOP_M_PATHWAY_GENES: int = 50


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


# ================================================================================
# Baseline block (E7union), copied from exp07
# ================================================================================


def _check_preconditions(
    paths: dict[str, pathlib.Path], target_glob: list[pathlib.Path]
) -> str | None:
    import pandas as pd

    expected_rows = {
        "train": 723_586,
        "validate": 434_151,
        "drug": 4_280,
        "cond": 5_631,
        "group": 10_554,
        "dict": 245,
    }
    for key, expected in expected_rows.items():
        p = paths[key]
        if not p.exists():
            return f"missing required input: {p}"
        with p.open() as fh:
            n = sum(1 for _ in fh) - 1
        if n != expected:
            return f"row count mismatch for {p}: expected {expected}, got {n}"

    for key in ("ot", "hpo", "target_long"):
        p = paths[key]
        if not p.exists():
            return f"missing required input: {p}"

    if not target_glob:
        return "no target__part-*.parquet files found under /data/ref/ot"

    train = pd.read_csv(paths["train"], usecols=["y_faers_signal"])
    rate = float(train["y_faers_signal"].mean())
    if abs(rate - 0.1100) > 0.001:
        return f"train positive rate for y_faers_signal is {rate:.4f}, expected 0.1100 +/- 0.001"
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
# drug_macro_auc machinery (per experiments/METRIC.md), copied from exp07
# ================================================================================


def _per_drug_table(
    ids: pd.Series, y: pd.Series, score: np.ndarray, min_pairs: int = 20
) -> pd.DataFrame:
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

    def _make_model() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
        )

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
        "cv_drug_macro_auc_mean": float(np.nanmean(cv_drug_macro_aucs)),
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
# Gene / pathway overlap: sparse-matrix construction (the budget-critical part)
# ================================================================================


def _build_disease_gene_matrices(ot_path: pathlib.Path, target_long_path: pathlib.Path, log):
    """Build C_* (conditions x genes) and D_binary (drugs x genes) over a shared Ensembl
    gene index, per the spec's Implementation note. Returns a context dict."""
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
    """Phenotype-arm overlap, unweighted. HPO likely keys genes by symbol, not Ensembl id,
    so this builds a SEPARATE gene index keyed on whatever id system HPO uses."""
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
    pairs: pd.DataFrame, ctx: dict, log, prefix: str = ""
) -> pd.DataFrame:
    """Vectorized gather of gene-overlap features for every (drug, condition) pair,
    per the spec's 'seconds, not a loop' trick. Only the shared-gene MAX variant loops,
    and only over the subset of pairs with n_shared_genes > 0."""
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

    if "C_score" in ctx:
        sum_matrix = (D_binary @ ctx["C_score"].T).toarray()
        out[f"{prefix}shared_gene_score_sum"] = _gather(sum_matrix)
    if "C_genetic" in ctx:
        gen_matrix = (D_binary @ ctx["C_genetic"].T).toarray()
        out[f"{prefix}shared_gene_genetic_sum"] = _gather(gen_matrix)
    if "C_literature" in ctx:
        lit_matrix = (D_binary @ ctx["C_literature"].T).toarray()
        out[f"{prefix}shared_gene_literature_sum"] = _gather(lit_matrix)
    if "C_idf" in ctx:
        idf_matrix = (D_binary @ ctx["C_idf"].T).toarray()
        out[f"{prefix}shared_gene_idf_sum"] = _gather(idf_matrix)

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

    # Two-tier max: only loop over pairs with n_shared_genes > 0 (a small fraction).
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


def _build_pathway_context(target_glob: list[pathlib.Path], disease_ctx: dict, log) -> dict | None:
    """Gene x pathway overlap from the OT target parquets (NOT the reactome hierarchy
    parquet, which has no gene column). Returns None if the schema can't be reconciled
    in budget -- per spec, dropping this design is an explicit, sanctioned fallback."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    tgt = pd.concat([pd.read_parquet(p) for p in target_glob], ignore_index=True)
    log.info("target parquet columns: %s", list(tgt.columns))

    gene_id_col = _detect_col(
        list(tgt.columns), ["id", "target_id", "ensembl_gene_id", "ensemblId"]
    )
    pathway_col = _detect_col(
        list(tgt.columns),
        ["pathways", "reactome", "reactomeIds", "pathwayIds", "reactome_pathways"],
    )
    if gene_id_col is None or pathway_col is None:
        log.warning(
            "could not identify gene id / pathway columns in target parquet (gene=%s, pathway=%s); "
            "columns were: %s -- skipping pathway overlap",
            gene_id_col,
            pathway_col,
            list(tgt.columns),
        )
        return None

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
    exploded = exploded[exploded[gene_id_col].isin(disease_ctx["gene_pos"])]
    if exploded.empty:
        log.warning("gene x pathway table is empty after restricting to the disease-arm gene index")
        return None

    pathway_ids = sorted(exploded["_pathways"].unique())
    pathway_pos = {p: i for i, p in enumerate(pathway_ids)}
    n_gene, n_pathway = disease_ctx["n_gene"], len(pathway_ids)
    grows = exploded[gene_id_col].map(disease_ctx["gene_pos"]).to_numpy(dtype=int)
    gcols = exploded["_pathways"].map(pathway_pos).to_numpy(dtype=int)
    GP = sp.csr_matrix((np.ones(len(grows)), (grows, gcols)), shape=(n_gene, n_pathway))
    GP.sum_duplicates()
    GP.data = np.minimum(GP.data, 1.0)
    log.info(
        "gene x pathway matrix: %d genes x %d pathways, %d edges", n_gene, n_pathway, len(grows)
    )

    C_score = disease_ctx["C_score"].tocsr()
    n_cond = disease_ctx["n_cond"]
    row_list, col_list = [], []
    for ci in range(n_cond):
        start, end = C_score.indptr[ci], C_score.indptr[ci + 1]
        idx = C_score.indices[start:end]
        vals = C_score.data[start:end]
        if len(idx) > TOP_M_PATHWAY_GENES:
            top = np.argpartition(-vals, TOP_M_PATHWAY_GENES - 1)[:TOP_M_PATHWAY_GENES]
            idx = idx[top]
        row_list.append(np.full(len(idx), ci))
        col_list.append(idx)
    rows_cm = np.concatenate(row_list) if row_list else np.array([], dtype=int)
    cols_cm = np.concatenate(col_list) if col_list else np.array([], dtype=int)
    C_topm_binary = sp.csr_matrix(
        (np.ones(len(rows_cm)), (rows_cm, cols_cm)), shape=(n_cond, disease_ctx["n_gene"])
    )

    Cond_pathway = (C_topm_binary @ GP).tocsr()
    Cond_pathway.data = np.minimum(Cond_pathway.data, 1.0)
    Drug_pathway = (disease_ctx["D_binary"] @ GP).tocsr()
    Drug_pathway.data = np.minimum(Drug_pathway.data, 1.0)

    n_shared_matrix = (Drug_pathway @ Cond_pathway.T).toarray()
    drug_pw_count = np.asarray(Drug_pathway.sum(axis=1)).ravel()
    cond_pw_count = np.asarray(Cond_pathway.sum(axis=1)).ravel()
    denom = drug_pw_count[:, None] + cond_pw_count[None, :] - n_shared_matrix
    jaccard_matrix = np.divide(
        n_shared_matrix, denom, out=np.zeros_like(n_shared_matrix), where=denom > 0
    )

    return {
        "n_shared_matrix": n_shared_matrix,
        "jaccard_matrix": jaccard_matrix,
        "drug_pos": disease_ctx["drug_pos"],
        "cond_pos": disease_ctx["cond_pos"],
        "n_pathway": n_pathway,
    }


def _pair_pathway_features(pairs: pd.DataFrame, pctx: dict) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    drow = pairs["ingredient_concept_id"].map(pctx["drug_pos"])
    crow = pairs["condition_concept_id"].map(pctx["cond_pos"])
    valid = (drow.notna() & crow.notna()).to_numpy()
    drow_arr = drow.fillna(-1).to_numpy(dtype=int)
    crow_arr = crow.fillna(-1).to_numpy(dtype=int)

    n_shared = np.zeros(len(pairs), dtype=float)
    jaccard = np.zeros(len(pairs), dtype=float)
    n_shared[valid] = pctx["n_shared_matrix"][drow_arr[valid], crow_arr[valid]]
    jaccard[valid] = pctx["jaccard_matrix"][drow_arr[valid], crow_arr[valid]]

    out = pd.DataFrame(index=pairs.index)
    out["n_shared_pathways"] = n_shared
    out["jaccard_pathways"] = jaccard
    return out


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
def run(exp_id: str) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp10")

    import json
    import time

    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    target_glob = sorted((data_root / "ref" / "ot").glob("target__part-*.parquet"))
    reactome_glob = sorted((data_root / "ref" / "ot").glob("reactome__part-*.parquet"))

    err = _check_preconditions(paths, target_glob)
    if err is not None:
        log.error("precondition check failed: %s", err)
        return {"precondition_failed": 1.0, "precondition_error_message": err}

    # Confirm the file-identity trap the spec calls out: reactome parquet has no gene col.
    if reactome_glob:
        reactome_sample = pd.read_parquet(reactome_glob[0]).head(3)
        log.info(
            "reactome__part-*.parquet columns (hierarchy, NOT gene membership): %s",
            list(reactome_sample.columns),
        )
    target_sample = pd.read_parquet(target_glob[0]).head(3)
    log.info(
        "target__part-*.parquet columns (gene -> pathway membership lives here): %s",
        list(target_sample.columns),
    )

    log.info("preconditions passed; loading train/validate")
    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    cond_basic = pd.read_csv(paths["cond"])

    n_cond_ot_check = pd.read_csv(paths["ot"], usecols=["condition_concept_id"])[
        "condition_concept_id"
    ].nunique()
    log.info(
        "condition_gene_ot_long.csv covers %d distinct conditions (spec expects 3,337)",
        n_cond_ot_check,
    )

    ot_truncated_available = "ot_truncated" in cond_basic.columns
    log.info(
        "ot_truncated column available on condition_features_basic.csv: %s", ot_truncated_available
    )

    fallback_notes: list[str] = []

    overlap = set(train["group_key"]) & set(validate["group_key"])
    unseen_family_is_trivial = len(overlap) == 0
    if not unseen_family_is_trivial:
        fallback_notes.append(
            f"group_key overlap between train/validate is non-empty ({len(overlap)} keys)"
        )

    # ---- E7union baseline block ----
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

    e7union_train = _assemble_base(train, degree_train)
    e7union_val = _assemble_base(validate, degree_val)
    log.info(
        "E7union baseline assembled in %.1fs, n_features=%d",
        time.time() - t0,
        e7union_train.shape[1],
    )

    hits = _audit_columns(list(e7union_train.columns))
    if hits:
        msg = f"leak audit: E7union baseline contains blacklisted columns: {hits}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    # ================================================================================
    # Gene / pathway overlap features
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
    disease_feats_all = _pair_gene_overlap_features(all_pairs, disease_ctx, log)
    log.info(
        "disease-arm overlap features computed for %d pairs in %.1fs",
        len(all_pairs),
        time.time() - t_gene,
    )

    if hpo_ctx is not None:
        hpo_feats_all = _pair_gene_overlap_features(all_pairs, hpo_ctx, log, prefix="hpo_")
        hpo_feats_all = hpo_feats_all[
            [c for c in hpo_feats_all.columns if "n_shared_genes" in c or "jaccard" in c]
        ]
    else:
        hpo_feats_all = pd.DataFrame(index=all_pairs.index)

    t_pathway = time.time()
    pathway_ctx = None
    pathway_dropped_reason = ""
    elapsed_so_far = time.time() - t0
    if elapsed_so_far > 420:  # 7 min already spent -- honor the spec's explicit drop instruction
        pathway_dropped_reason = (
            f"budget risk: {elapsed_so_far:.0f}s already elapsed before pathway construction"
        )
        log.warning("dropping pathway overlap design: %s", pathway_dropped_reason)
    else:
        try:
            pathway_ctx = _build_pathway_context(target_glob, disease_ctx, log)
        except Exception as exc:
            pathway_dropped_reason = f"pathway construction raised: {exc}"
            log.warning("dropping pathway overlap design: %s", pathway_dropped_reason)
        if pathway_ctx is None and not pathway_dropped_reason:
            pathway_dropped_reason = (
                "gene/pathway schema could not be reconciled from target parquet"
            )

    if pathway_ctx is not None:
        pathway_feats_all = _pair_pathway_features(all_pairs, pathway_ctx)
        log.info("pathway overlap features computed in %.1fs", time.time() - t_pathway)
    else:
        pathway_feats_all = pd.DataFrame(index=all_pairs.index)
        fallback_notes.append(f"pathway overlap design dropped: {pathway_dropped_reason}")

    def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr = df.loc["train"].set_axis(train.index)
        va = df.loc["validate"].set_axis(validate.index)
        return tr, va

    disease_train, disease_val = _split(disease_feats_all)
    hpo_train, hpo_val = _split(hpo_feats_all)
    pathway_train, pathway_val = _split(pathway_feats_all)

    literature_cols = [c for c in disease_train.columns if "literature" in c]
    gene_only_cols = [c for c in disease_train.columns if c not in literature_cols]

    gene_overlap_train = pd.concat([disease_train[gene_only_cols], hpo_train], axis=1)
    gene_overlap_val = pd.concat([disease_val[gene_only_cols], hpo_val], axis=1)
    literature_train = disease_train[literature_cols]
    literature_val = disease_val[literature_cols]

    log.info("total gene/pathway feature construction: %.1fs elapsed", time.time() - t0)

    # ================================================================================
    # Nested designs
    # ================================================================================
    design_order = ["E7union", "+gene_overlap", "+pathway_overlap"]
    train_designs = {
        "E7union": e7union_train,
        "+gene_overlap": pd.concat([e7union_train, gene_overlap_train], axis=1),
        "+pathway_overlap": pd.concat([e7union_train, gene_overlap_train, pathway_train], axis=1),
    }
    val_designs = {
        "E7union": e7union_val,
        "+gene_overlap": pd.concat([e7union_val, gene_overlap_val], axis=1),
        "+pathway_overlap": pd.concat([e7union_val, gene_overlap_val, pathway_val], axis=1),
    }
    literature_design_train = pd.concat([e7union_train, literature_train], axis=1)
    literature_design_val = pd.concat([e7union_val, literature_val], axis=1)

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
        log.info(
            "fitting design=%s n_features=%d n_folds=%d (elapsed=%.1fs)",
            name,
            train_designs[name].shape[1],
            n_folds,
            time.time() - t0,
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

    # Budget guard: this diagnostic is explicitly secondary to the headline designs above,
    # which are already fit and logged by this point. Single fit + validate pass only (no
    # CV) to keep its cost low; skip entirely if the 600s timeout is genuinely at risk.
    literature_elapsed = time.time() - t0
    literature_dropped_reason = ""
    if literature_elapsed > 480:
        literature_dropped_reason = (
            f"budget risk: {literature_elapsed:.0f}s already elapsed before the "
            "dt_literature diagnostic; skipped to protect the headline designs' results"
        )
        log.warning("skipping dt_literature diagnostic: %s", literature_dropped_reason)
        literature_res = None
    else:
        log.info(
            "fitting dt_literature diagnostic design, single fit only (elapsed=%.1fs)",
            literature_elapsed,
        )
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
    bootstrap_rows = []
    pairs_for_ci = [("E7union", "+gene_overlap")]
    if pathway_ctx is not None:
        pairs_for_ci += [("+gene_overlap", "+pathway_overlap"), ("E7union", "+pathway_overlap")]
    tables = {k: v["val_drug_table"] for k, v in fit_results.items()}
    if literature_res is not None:
        pairs_for_ci.append(("E7union", "dt_literature_diagnostic"))
        tables["dt_literature_diagnostic"] = literature_res["val_drug_table"]
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
                    "design": "E7union+dt_literature",
                    "n_features": literature_design_train.shape[1],
                    "validate_drug_macro_auc": literature_res["val_drug_macro_auc"],
                    "validate_drug_macro_p10": literature_res["val_drug_macro_p10"],
                    "n_drugs_scored": literature_res["val_n_drugs_scored"],
                    "baseline_e7union_drug_macro_auc": fit_results["E7union"]["val_drug_macro_auc"],
                    "increment_over_e7union": literature_res["val_drug_macro_auc"]
                    - fit_results["E7union"]["val_drug_macro_auc"],
                    "note": "reported apart -- dt_literature-weighted overlap excluded from the headline model per spec",
                }
            ]
        )
    else:
        literature_diag_df = pd.DataFrame(
            [
                {
                    "design": "E7union+dt_literature",
                    "n_features": literature_design_train.shape[1],
                    "validate_drug_macro_auc": float("nan"),
                    "validate_drug_macro_p10": float("nan"),
                    "n_drugs_scored": 0,
                    "baseline_e7union_drug_macro_auc": fit_results["E7union"]["val_drug_macro_auc"],
                    "increment_over_e7union": float("nan"),
                    "note": f"dropped: {literature_dropped_reason}",
                }
            ]
        )
    literature_diag_df.to_csv(out / f"{exp_id}_literature_diagnostic.csv", index=False)

    # ================================================================================
    # Stratification: gene_arm, best_match_tier, ot_truncated
    # ================================================================================
    best_design = "+pathway_overlap" if pathway_ctx is not None else "+gene_overlap"
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
        fallback_notes.append(
            "gene_arm column not found on condition_features_basic.csv; stratification skipped"
        )
    pd.DataFrame(gene_arm_rows).to_csv(out / f"{exp_id}_by_gene_arm.csv", index=False)

    tier_rows = []
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
        a1a2b1 = val_meta[val_meta["best_match_tier"].isin(["A1", "A2", "B1"])]
        if not a1a2b1.empty:
            m = _drug_macro_metrics(
                a1a2b1["ingredient_concept_id"],
                a1a2b1["y_faers_signal"],
                a1a2b1["_proba"].to_numpy(),
            )
            tier_rows.append(
                {
                    "best_match_tier": "A1/A2/B1 combined",
                    "n_pairs": len(a1a2b1),
                    "drug_macro_auc": m["drug_macro_auc"],
                    "n_drugs_scored": m["n_drugs_scored"],
                }
            )
    else:
        fallback_notes.append("best_match_tier column not found; stratification skipped")
    if "ot_truncated" in val_meta.columns:
        for level, grp in val_meta.groupby("ot_truncated", dropna=False):
            m = _drug_macro_metrics(
                grp["ingredient_concept_id"], grp["y_faers_signal"], grp["_proba"].to_numpy()
            )
            tier_rows.append(
                {
                    "best_match_tier": f"ot_truncated={level}",
                    "n_pairs": len(grp),
                    "drug_macro_auc": m["drug_macro_auc"],
                    "n_drugs_scored": m["n_drugs_scored"],
                }
            )
    pd.DataFrame(tier_rows).to_csv(out / f"{exp_id}_by_match_tier.csv", index=False)

    disease_arm_mask = val_meta.get("gene_arm", pd.Series("none", index=val_meta.index)).isin(
        ["both", "disease_only"]
    )
    phenotype_arm_mask = val_meta.get("gene_arm", pd.Series("none", index=val_meta.index)).isin(
        ["both", "phenotype_only"]
    )
    disease_m = _drug_macro_metrics(
        val_meta.loc[disease_arm_mask, "ingredient_concept_id"],
        val_meta.loc[disease_arm_mask, "y_faers_signal"],
        val_meta.loc[disease_arm_mask, "_proba"].to_numpy(),
    )
    phenotype_m = _drug_macro_metrics(
        val_meta.loc[phenotype_arm_mask, "ingredient_concept_id"],
        val_meta.loc[phenotype_arm_mask, "y_faers_signal"],
        val_meta.loc[phenotype_arm_mask, "_proba"].to_numpy(),
    )

    # ================================================================================
    # per_drug.csv, importance_top25.csv, overlap_distributions.png
    # ================================================================================
    per_drug_df = best_res["val_drug_table"].copy()
    per_drug_df.to_csv(out / f"{exp_id}_per_drug.csv", index=False)

    importance_rows = []
    for name in [*design_order, "dt_literature_diagnostic"]:
        import re

        import lightgbm as lgb

        x_src = train_designs.get(name, literature_design_train)
        cols = x_src.columns
        x_lgb = x_src.copy()
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
    pd.DataFrame(importance_rows).to_csv(out / f"{exp_id}_importance_top25.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    shared = disease_val["n_shared_genes"].to_numpy()
    arm_series = val_meta.get("gene_arm", pd.Series("unknown", index=val_meta.index)).astype(str)
    for level in sorted(arm_series.unique()):
        mask = (arm_series == level).to_numpy()
        vals = shared[mask]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.5, label=f"{level} (n={len(vals)})")
    ax.set_xlabel("n_shared_genes (validate pairs)")
    ax.set_ylabel("count")
    ax.set_title("Shared-gene count distribution by gene_arm (validate)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{exp_id}_overlap_distributions.png", dpi=120)
    plt.close(fig)

    # ================================================================================
    # solved_threshold from exp06
    # ================================================================================
    solved_threshold = float("nan")
    try:
        candidates = sorted(
            pathlib.Path("/results").glob("*/*_success_criterion.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            with candidates[0].open() as f:
                sc = json.load(f)
            solved_threshold = float(sc.get("solved_threshold", float("nan")))
            log.info("read solved_threshold=%.4f from %s", solved_threshold, candidates[0])
        else:
            log.warning(
                "no exp06 success_criterion.json found on results volume; solved_threshold=nan"
            )
            fallback_notes.append(
                "exp06 success_criterion.json not found; solved_threshold reported as nan"
            )
    except Exception as exc:
        log.warning("failed to read solved_threshold: %s", exc)
        fallback_notes.append(f"failed to read exp06 success_criterion.json: {exc}")

    results.commit()

    increment_gene_overlap = (
        fit_results["+gene_overlap"]["val_drug_macro_auc"]
        - fit_results["E7union"]["val_drug_macro_auc"]
    )
    increment_pathway_overlap = (
        fit_results["+pathway_overlap"]["val_drug_macro_auc"]
        - fit_results["+gene_overlap"]["val_drug_macro_auc"]
        if pathway_ctx is not None
        else 0.0
    )
    increment_literature = (
        literature_res["val_drug_macro_auc"] - fit_results["E7union"]["val_drug_macro_auc"]
        if literature_res is not None
        else float("nan")
    )

    unseen_family_note = (
        "group_key (the split's grouping unit) is disjoint between train and validate by construction, "
        "so drug_macro_auc_unseen_family is identical to the overall validate drug_macro_auc for the best design."
        if unseen_family_is_trivial
        else "group_key overlap detected; unseen_family fell back to the overall validate number."
    )
    log.info(unseen_family_note)

    metrics = {
        "drug_macro_auc": best_res["val_drug_macro_auc"],
        "drug_macro_auc_unseen_family": best_res["val_drug_macro_auc"],
        "drug_macro_p10": best_res["val_drug_macro_p10"],
        "increment_gene_overlap": increment_gene_overlap,
        "increment_pathway_overlap": increment_pathway_overlap
        if pathway_ctx is not None
        else float("nan"),
        "increment_literature_weighted": increment_literature,
        "drug_macro_auc_disease_arm": disease_m["drug_macro_auc"],
        "drug_macro_auc_phenotype_arm": phenotype_m["drug_macro_auc"],
        "n_drugs_scored": best_res["val_n_drugs_scored"],
        "baseline_e7_union_drug_macro_auc": fit_results["E7union"]["val_drug_macro_auc"],
        "solved_threshold": solved_threshold,
        "pathway_design_reached": 1.0 if pathway_ctx is not None else 0.0,
    }
    log.info("fallback notes: %s", fallback_notes)
    log.info("total elapsed: %.1fs", time.time() - t0)
    log.info("final metrics: %s", metrics)

    extras = pd.DataFrame(
        [
            {
                "join_coverage_drug_genes_in_ot_table": disease_ctx["coverage"],
                "unseen_family_note": unseen_family_note,
                "fallback_notes": "; ".join(fallback_notes) if fallback_notes else "none",
                "best_design": best_design,
                "n_conditions_ot_table": n_cond_ot_check,
                "total_elapsed_seconds": time.time() - t0,
            }
        ]
    )
    extras.to_csv(out / f"{exp_id}_run_extras.csv", index=False)
    results.commit()

    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp10",
        title="Gene and pathway overlap between drug targets and condition genes",
        hypothesis=(
            "Genetic-evidence-weighted overlap between a drug's target genes and a "
            "condition's implicated genes adds >=0.02 drug_macro_auc over the intrinsic "
            "union, concentrated on disease-arm conditions."
        ),
        approach=(
            "Sparse condition-gene and drug-gene matrices; raw, ot_score-weighted, "
            "genetic-association-only, IDF-weighted and Jaccard overlaps; Reactome pathway "
            "overlap from the OT target parquets; dt_literature weighting kept as a "
            "separate diagnostic; grouped 3-fold CV scored on drug_macro_auc; stratified "
            "by gene_arm and match tier."
        ),
        label="y_faers_signal",
        features=["degree", "p_c", "intrinsic_union", "gene_overlap", "pathway_overlap"],
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

    increment = metrics.get("increment_gene_overlap", float("nan"))
    increment_pathway = metrics.get("increment_pathway_overlap", float("nan"))
    disease_arm = metrics.get("drug_macro_auc_disease_arm", float("nan"))
    phenotype_arm = metrics.get("drug_macro_auc_phenotype_arm", float("nan"))
    solved_threshold = metrics.get("solved_threshold", float("nan"))
    headline = metrics.get("drug_macro_auc", float("nan"))
    baseline = metrics.get("baseline_e7_union_drug_macro_auc", float("nan"))
    pathway_reached = bool(metrics.get("pathway_design_reached", 0.0))
    increment_lit = metrics.get("increment_literature_weighted", float("nan"))

    null_result = not (increment >= 0.02 and disease_arm >= phenotype_arm)
    threshold_note = (
        f"vs solved_threshold={solved_threshold:.4f}"
        if solved_threshold == solved_threshold
        else "solved_threshold unavailable (exp06 success_criterion.json not found on the results volume)"
    )

    findings = (
        f"E7union baseline drug_macro_auc={baseline:.4f}. "
        f"+gene_overlap increment={increment:+.4f} over E7union (see bootstrap_increments.csv for the CI). "
        f"{'+pathway_overlap increment=' + format(increment_pathway, '+.4f') + ' over +gene_overlap' if pathway_reached else 'pathway_overlap design was NOT reached (dropped for budget/schema reasons, see run_extras.csv fallback_notes) -- gene-overlap-only result stands as the headline.'} "
        f"Headline (best design) drug_macro_auc={headline:.4f}, {threshold_note}. "
        f"disease-arm drug_macro_auc={disease_arm:.4f} vs phenotype-arm drug_macro_auc={phenotype_arm:.4f} -- "
        f"hypothesis required the gain to concentrate on the disease arm. "
        f"dt_literature diagnostic (reported apart, not combined with the headline): increment={increment_lit:+.4f}. "
        + (
            "RESULT: the bridge hypothesis PASSED its first direct test at this feature resolution -- "
            "gene overlap adds >=0.02 drug_macro_auc and the gain concentrates on the disease arm."
            if not null_result
            else "RESULT: NULL. Either the gene-overlap increment fell short of 0.02, or the gain did not "
            "concentrate on the disease arm (or both) -- the bridge hypothesis failed its first direct "
            "test at this feature resolution. The path drug -> target -> gene -> condition does not carry "
            "information about the RWD edge beyond drug/condition identity at this resolution. What would "
            "have to change: better condition->gene evidence (denser/curated rather than propagated-up-the-"
            "hierarchy OT associations), a different label (see exp06's ceiling work), or a different "
            "similarity notion than raw/weighted gene-set overlap (e.g. pathway-level or network-diffusion "
            "similarity, which this run attempted only in a simplified one-hop form)."
        )
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"results/{exp}/{exp}_designs.csv",
            f"results/{exp}/{exp}_by_gene_arm.csv",
            f"results/{exp}/{exp}_by_match_tier.csv",
            f"results/{exp}/{exp}_literature_diagnostic.csv",
            f"results/{exp}/{exp}_importance_top25.csv",
            f"results/{exp}/{exp}_bootstrap_increments.csv",
            f"results/{exp}/{exp}_per_drug.csv",
            f"results/{exp}/{exp}_overlap_distributions.png",
            f"results/{exp}/{exp}_run_extras.csv",
        ],
        next_steps=(
            "If null: hand Round 3 to the label (adjudicated reference sets, calibrated OHDSI "
            "estimates) rather than more features, per the spec's falsification clause. "
            "If positive: check whether pathway_overlap (if reached) adds anything over gene "
            "overlap alone before investing further in pathway-hop features; shared_pathway_min_level "
            "was not implemented (simplified to unweighted n_shared_pathways/jaccard_pathways per the "
            "spec's explicit allowance to skip the level-weighting refinement under time constraints)."
        ),
    )
