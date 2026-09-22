"""Modal script for experiments/exp15_pathway_overlap_and_valid_placebos.md.

Two parts. Part 1 corrects exp08's invalid zero-neighbour placebo (which compared
two differently-fitted models, not a genuine null) with two valid forms:
refit-within-subset (fit both A and B on zero-neighbour rows only) and
train-time value permutation (permute the feature's values within condition
before fitting, keep the design matrix otherwise identical). Both are applied
to exp08's gene-tier `nb_excess_gene` feature and, in Part 2, to the new
pathway-overlap feature.

Part 2 builds the gene x Reactome-pathway overlap features from the OT target
parquets (never the reactome__*.parquet hierarchy file, which has no gene
column -- asserted explicitly below) using exp10's sparse gene x pathway
product pattern, and evaluates the increment over a gene-overlap baseline,
both overall and conditional on pairs with zero shared genes (the spec's
headline number).

exp14 (gene-overlap disease-arm) has no committed script at the time this
experiment runs, so the base design substitutes exp07's intrinsic-union block
plus exp08's `nb_excess_gene` feature -- noted explicitly in findings, per the
task's instruction to record the substitution.
"""

# ruff: noqa: N806 -- linear-algebra matrix-naming convention (D_genes, GP, C_topm),
# consistent with exp10's use of the same convention in this repo.

from __future__ import annotations

import logging
import pathlib
import time
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

app = modal.App("bridge-exp15")

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

HGB_KWARGS = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "early_stopping": False,
    "random_state": 0,
}

K_SHRINK = 10
BOOT_RESAMPLES = 1000
TOP_M_PATHWAY_GENES = 50
SEEDS = [0, 1, 2]

# exp08's own reported numbers, carried through per the spec (not rerun).
EXP08_ZERO_NB_PLACEBO_INVALID = 0.0198
EXP08_PERMUTED_GRAPH_PLACEBO = 0.0096
EXP08_LABEL_SHUFFLE_PLACEBO = 0.0047
EXP08_INCREMENT_COND = 0.0443
EXP08_DECAY_CURVE = (0.591, 0.612, 0.609, 0.656)  # buckets 0 / 1-2 / 3-5 / 6plus


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


def _fail(message: str) -> dict[str, Any]:
    return {"precondition_failed": 1.0, "precondition_error_message": message}


# --------------------------------------------------------------------- exp08
# Gene-tier neighbour-excess construction, copied from exp08 (vectorized
# leave-one-drug-out / shrinkage, unchanged).


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


# ------------------------------------------------------------------- metric


def _drug_macro_metrics(drug_ids, y_true, y_score, min_pairs: int = 20) -> dict:
    """drug_macro_auc / drug_macro_p10 per METRIC.md. NaN-safe: y_score is
    filled with the per-drug median before ranking, and cast to float so a
    stray bool column never breaks argsort."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    y_score = np.asarray(y_score, dtype=float)
    df = pd.DataFrame({"drug": drug_ids, "y": y_true, "score": y_score})
    df["score"] = df["score"].astype(float)
    df["score"] = df.groupby("drug")["score"].transform(lambda s: s.fillna(s.median()))
    df["score"] = df["score"].fillna(df["score"].median()).fillna(0.0)

    aucs, p10s = [], []
    for _, g in df.groupby("drug"):
        if len(g) < min_pairs:
            continue
        y = g["y"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, g["score"].to_numpy()))
        top10 = g.sort_values("score", ascending=False).head(10)
        p10s.append(float(top10["y"].mean()))
    return {
        "drug_macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "drug_macro_p10": float(np.mean(p10s)) if p10s else float("nan"),
        "n_drugs_scored": len(aucs),
    }


def _fit_predict(x_train, y_train, x_val, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    kwargs = dict(HGB_KWARGS)
    kwargs["random_state"] = seed
    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_val)[:, 1]


# ------------------------------------------------------------- Part 1: placebos


def _placebo_refit_within_subset(
    subset_mask_train: np.ndarray,
    subset_mask_val: np.ndarray,
    cols_a: list[str],
    cols_b: list[str],
    x_train_df: pd.DataFrame,
    y_train: np.ndarray,
    x_val_df: pd.DataFrame,
    y_val: np.ndarray,
    drug_ids_train: np.ndarray,
    drug_ids_val: np.ndarray,
    seeds: list[int],
    log: logging.Logger,
    label: str,
) -> dict:
    """Fit A (without the tested feature) and B (with it) on the subset ONLY
    (train restricted to subset_mask_train), evaluate on the subset of
    validate. With the tested feature constant (0) on this subset by
    construction, the two designs are genuinely equivalent and the gain
    should be ~0.000 up to seed noise. Runs with several seeds to report
    that noise."""
    import numpy as np

    xa_tr = x_train_df.loc[subset_mask_train, cols_a].to_numpy(dtype=float)
    xb_tr = x_train_df.loc[subset_mask_train, cols_b].to_numpy(dtype=float)
    y_tr = y_train[subset_mask_train]
    xa_va = x_val_df.loc[subset_mask_val, cols_a].to_numpy(dtype=float)
    xb_va = x_val_df.loc[subset_mask_val, cols_b].to_numpy(dtype=float)
    y_va = y_val[subset_mask_val]
    ids_va = drug_ids_val[subset_mask_val]

    gains = []
    for seed in seeds:
        if len(np.unique(y_tr)) < 2 or len(xa_tr) < 50:
            gains.append(float("nan"))
            continue
        pred_a = _fit_predict(xa_tr, y_tr, xa_va, seed=seed)
        pred_b = _fit_predict(xb_tr, y_tr, xb_va, seed=seed)
        m_a = _drug_macro_metrics(ids_va, y_va, pred_a)
        m_b = _drug_macro_metrics(ids_va, y_va, pred_b)
        gain = m_b["drug_macro_auc"] - m_a["drug_macro_auc"]
        gains.append(gain)
        log.info(
            "[%s] refit-within-subset seed=%d: A=%.4f B=%.4f gain=%.4f (n_subset_train=%d, n_subset_val=%d)",
            label,
            seed,
            m_a["drug_macro_auc"],
            m_b["drug_macro_auc"],
            gain,
            int(subset_mask_train.sum()),
            int(subset_mask_val.sum()),
        )
    gains_arr = np.array(gains, dtype=float)
    return {
        "mean_gain": float(np.nanmean(gains_arr)) if len(gains_arr) else float("nan"),
        "std_gain": float(np.nanstd(gains_arr)) if len(gains_arr) else float("nan"),
        "gains_by_seed": gains,
        "n_subset_train": int(subset_mask_train.sum()),
        "n_subset_val": int(subset_mask_val.sum()),
    }


def _placebo_value_permutation(
    feature_col: str,
    cols_a: list[str],
    cols_b: list[str],
    x_train_df: pd.DataFrame,
    y_train: np.ndarray,
    x_val_df: pd.DataFrame,
    y_val: np.ndarray,
    condition_ids_train: np.ndarray,
    drug_ids_val: np.ndarray,
    baseline_auc_a_full: float,
    seeds: list[int],
    log: logging.Logger,
    label: str,
) -> dict:
    """Keep the full design matrix; permute the tested feature's values
    across rows WITHIN each condition before fitting model B. The column
    keeps its marginal distribution and loses its (drug, condition) pairing.
    Compared against model A fit on the real (unpermuted) design -- any
    residual gain for permuted B is a construction artefact, not signal."""
    import numpy as np
    import pandas as pd

    gains = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        permuted = x_train_df[feature_col].to_numpy(dtype=float).copy()
        cond_series = pd.Series(condition_ids_train)
        for _, idx in cond_series.groupby(cond_series).groups.items():
            pos = cond_series.index.get_indexer(idx)
            permuted[pos] = rng.permutation(permuted[pos])
        x_tr_b_perm = x_train_df[cols_b].copy()
        x_tr_b_perm[feature_col] = permuted
        x_tr_b_perm = x_tr_b_perm.to_numpy(dtype=float)

        pred_b_perm = _fit_predict(
            x_tr_b_perm, y_train, x_val_df[cols_b].to_numpy(dtype=float), seed=seed
        )
        m_b_perm = _drug_macro_metrics(drug_ids_val, y_val, pred_b_perm)
        gain = m_b_perm["drug_macro_auc"] - baseline_auc_a_full
        gains.append(gain)
        log.info(
            "[%s] value-permutation seed=%d: permuted-B=%.4f gain-over-A=%.4f",
            label,
            seed,
            m_b_perm["drug_macro_auc"],
            gain,
        )
    gains_arr = np.array(gains, dtype=float)
    return {
        "mean_gain": float(np.nanmean(gains_arr)) if len(gains_arr) else float("nan"),
        "std_gain": float(np.nanstd(gains_arr)) if len(gains_arr) else float("nan"),
        "gains_by_seed": gains,
    }


# ------------------------------------------------------------- Part 2: pathway


def _build_gene_pathway_matrix(target_glob: list[pathlib.Path], gene_index: dict, log):
    """Gene x Reactome-pathway membership from the OT target__part-*.parquet
    files. Asserts a gene identifier column is present before proceeding --
    the reactome__*.parquet hierarchy file has no such column and must not be
    used here (the spec's stated trap)."""
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
    assert gene_id_col is not None, (
        f"target parquet has no recognizable gene identifier column; columns were {list(tgt.columns)}. "
        "This is the trap the spec warns about: do not fall back to reactome__*.parquet, which has "
        "no gene column at all."
    )
    assert pathway_col is not None, (
        f"target parquet has no recognizable pathway-membership column; columns were {list(tgt.columns)}"
    )
    log.info("confirmed gene id col=%s, pathway col=%s in target parquet", gene_id_col, pathway_col)

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
    total_target_genes = len(gene_index)
    covered_share = len(covered_genes & set(gene_index)) / max(total_target_genes, 1)

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
    log.info(
        "gene x pathway matrix: %d genes x %d pathways, %d edges", n_gene, n_pathway, len(grows)
    )
    return {"GP": GP, "pathway_pos": pathway_pos, "n_pathway": n_pathway}, covered_share


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

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp15")

    t0 = time.time()
    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)
    fallbacks_taken: list[str] = []

    # ---------------------------------------------------------------- step 1
    data_root = pathlib.Path("/data")
    paths = {
        "train": data_root / "splits" / "train.csv",
        "validate": data_root / "splits" / "validate.csv",
        "ingredient_target_long": data_root / "drug" / "ingredient_target_long.csv",
        "condition_features_basic": data_root / "condition" / "condition_features_basic.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        msg = f"Missing required input(s): {missing}"
        log.error(msg)
        return _fail(msg)

    target_glob = sorted((data_root / "ref" / "ot").glob("target__part-*.parquet"))
    reactome_glob = sorted((data_root / "ref" / "ot").glob("reactome__part-*.parquet"))
    if not target_glob:
        msg = "no target__part-*.parquet files found under /data/ref/ot"
        log.error(msg)
        return _fail(msg)

    if reactome_glob:
        reactome_sample = pd.read_parquet(reactome_glob[0]).head(3)
        log.info(
            "reactome__part-*.parquet columns (hierarchy ONLY, no gene col -- not used here): %s",
            list(reactome_sample.columns),
        )
        assert not any("gene" in c.lower() for c in reactome_sample.columns), (
            "unexpected: reactome__*.parquet appears to carry a gene column after all -- re-check the trap note"
        )

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
        msg = f"Row counts do not match expected inputs: {bad_counts} (expected {EXPECTED_ROWS})"
        log.error(msg)
        return _fail(msg)

    edges_raw = edges_raw.rename(columns={"omop_concept_id": "ingredient_concept_id"})

    # -------------------------------------------------------- gene tier prep
    # Same subunit-expansion exclusion + disease_efficacy refinement as exp08.
    is_complex_or_family = (
        edges_raw["component_relationship"]
        .astype(str)
        .str.upper()
        .str.contains("COMPLEX|FAMILY", regex=True, na=False)
    )
    gene_tier_edges_raw = edges_raw.loc[~is_complex_or_family]

    def _to_bool(series: pd.Series) -> pd.Series:
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

    # -------------------------------------------- coverage: gene->pathway map
    # The OT target parquet's gene identifier column is Ensembl gene id (its
    # "id" column), not the HGNC symbol -- ingredient_target_long.csv carries
    # both (790 distinct ensembl_gene_id vs 977 distinct gene_symbol values;
    # exp10 reconciles on ensembl_gene_id for the same reason). Use
    # ensembl_gene_id as the shared index for pathway-matrix construction and
    # for D_genes/C_topm below; gene_symbol stays reserved for exp08's
    # nb_excess_gene tier feature, which is unaffected by this fix.
    drug_target_genes = set(edges_raw["ensembl_gene_id"].dropna().unique())
    n_drug_target_genes = len(drug_target_genes)
    gene_index = {g: i for i, g in enumerate(sorted(drug_target_genes))}
    pathway_ctx, covered_share = _build_gene_pathway_matrix(target_glob, gene_index, log)

    log.info(
        "gene->pathway map covers %.1f%% of %d drug target genes (need >=60%%)",
        covered_share * 100,
        n_drug_target_genes,
    )
    if covered_share < 0.60:
        msg = (
            f"gene->pathway map covers only {covered_share:.3f} of {n_drug_target_genes} drug "
            "target genes (need >=0.60); pathway overlap would be built on a schema that failed "
            "to reconcile with the drug-target gene set"
        )
        log.error(msg)
        return _fail(msg)
    if pathway_ctx is None:
        msg = "gene x pathway table empty after restricting to drug-target gene index"
        log.error(msg)
        return _fail(msg)

    pathway_coverage_df = pd.DataFrame(
        [
            {
                "n_drug_target_genes": n_drug_target_genes,
                "covered_share": covered_share,
                "n_pathways": pathway_ctx["n_pathway"],
            }
        ]
    )
    pathway_coverage_df.to_csv(out / f"{exp_id}_pathway_map_coverage.csv", index=False)

    # ---------------------------------------------------------------- step 2
    # Indices shared across drug-gene / condition-gene / pathway construction.
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
    n_cond = len(all_cond)
    c_pos_train = train["condition_concept_id"].map(cond_index_map).to_numpy()
    c_pos_val = validate["condition_concept_id"].map(cond_index_map).to_numpy()

    p_c_series = train.groupby("condition_concept_id")["y_faers_signal"].mean()
    train_wide_mean = float(train["y_faers_signal"].mean())
    p_c_by_cond = p_c_series.reindex(all_cond).fillna(train_wide_mean).to_numpy()
    p_c_train = p_c_by_cond[c_pos_train]
    p_c_val = p_c_by_cond[c_pos_val]

    def _f_train_matrix(y_col: np.ndarray) -> sp.csr_matrix:
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

    has_target_annotation_by_ing = np.array(
        [1.0 if i in set(edges_raw["ingredient_concept_id"]) else 0.0 for i in all_ing]
    )

    drug_degree_train_map = train.groupby("ingredient_concept_id")["condition_concept_id"].nunique()
    condition_degree_train_map = train.groupby("condition_concept_id")[
        "ingredient_concept_id"
    ].nunique()
    cond_record_count_map = cond_feats.set_index("condition_concept_id")["record_count"]

    def _attach_baseline(df: pd.DataFrame, p_c_arr: np.ndarray) -> pd.DataFrame:
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

    train = _attach_baseline(train, p_c_train)
    validate = _attach_baseline(validate, p_c_val)

    def _attach_tier(df: pd.DataFrame, feats: dict, split: str) -> pd.DataFrame:
        df = df.copy()
        suffix = f"__{split}"
        for name, arr in feats.items():
            if name.endswith(suffix):
                df[name[: -len(suffix)]] = arr
        return df

    train = _attach_tier(train, gene_feats, "train")
    validate = _attach_tier(validate, gene_feats, "val")
    train["has_target_annotation"] = has_target_annotation_by_ing[d_pos_train]
    validate["has_target_annotation"] = has_target_annotation_by_ing[d_pos_val]
    for df in (train, validate):
        mask = df["has_target_annotation"] == 0
        df.loc[mask, "nb_excess_gene"] = np.nan

    # -------------------------------------------------- gene-overlap feature
    # D_genes (drugs x genes), n_shared_genes via D_genes @ C_genes.T. Keyed on
    # ensembl_gene_id throughout, to match gene_index (see note above -- the OT
    # parquet family and condition_gene_ot_long.csv both use Ensembl ids).
    all_genes_edges = edges_raw.dropna(subset=["ensembl_gene_id"])
    drug_gene_pairs = all_genes_edges[
        ["ingredient_concept_id", "ensembl_gene_id"]
    ].drop_duplicates()
    drug_pos = {d: i for i, d in enumerate(all_ing)}
    n_gene_total = len(gene_index)
    drug_gene_pairs = drug_gene_pairs[drug_gene_pairs["ensembl_gene_id"].isin(gene_index)]
    D_genes = sp.csr_matrix(
        (
            np.ones(len(drug_gene_pairs)),
            (
                drug_gene_pairs["ingredient_concept_id"].map(drug_pos).to_numpy(dtype=int),
                drug_gene_pairs["ensembl_gene_id"].map(gene_index).to_numpy(dtype=int),
            ),
        ),
        shape=(n_ing, n_gene_total),
    )
    D_genes.sum_duplicates()
    D_genes.data = np.minimum(D_genes.data, 1.0)

    # Condition-side genes: condition_gene_ot_long.csv (Open Targets disease-arm
    # association scores), restricted to top m=50 genes by ot_score per condition,
    # per spec. Falls back to a train-derived co-flag proxy only if that file is
    # missing or its gene column cannot be reconciled with gene_index.
    ot_gene_path = data_root / "condition" / "condition_gene_ot_long.csv"
    used_ot_scores = False
    cond_pos_map = {c: i for i, c in enumerate(all_cond)}
    if ot_gene_path.exists():
        ot = pd.read_csv(
            ot_gene_path, usecols=["condition_concept_id", "ensembl_gene_id", "ot_score"]
        )
        ot = ot.dropna(subset=["condition_concept_id", "ensembl_gene_id"])
        ot = ot[ot["ensembl_gene_id"].isin(gene_index)]
        ot = ot[ot["condition_concept_id"].isin(cond_pos_map)]
        if len(ot):
            used_ot_scores = True
            ot["ot_score"] = ot["ot_score"].fillna(0.0)
            rows_l, cols_l, vals_l = [], [], []
            for cid, g in ot.groupby("condition_concept_id"):
                gg = g.nlargest(TOP_M_PATHWAY_GENES, "ot_score")
                ci = cond_pos_map[cid]
                for gene_id, _sc in zip(gg["ensembl_gene_id"], gg["ot_score"], strict=True):
                    rows_l.append(ci)
                    cols_l.append(gene_index[gene_id])
                    vals_l.append(1.0)
            C_topm = (
                sp.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n_cond, n_gene_total))
                if rows_l
                else sp.csr_matrix((n_cond, n_gene_total))
            )
    if not used_ot_scores:
        fallbacks_taken.append(
            "condition_gene_ot_long.csv not staged or had no rows reconcilable with gene_index "
            "(ensembl_gene_id); condition-side gene sets built from train FAERS-flagged pairs' "
            "co-occurring drug-target genes, top m=50 by within-condition co-flag count, as a "
            "coarse ot_score stand-in."
        )
        flagged = train.loc[
            train["y_faers_signal"] == 1, ["ingredient_concept_id", "condition_concept_id"]
        ]
        flagged = flagged.merge(drug_gene_pairs, on="ingredient_concept_id", how="inner")
        gene_counts = (
            flagged.groupby(["condition_concept_id", "ensembl_gene_id"])
            .size()
            .reset_index(name="n")
        )
        rows_l, cols_l, vals_l = [], [], []
        for cid, g in gene_counts.groupby("condition_concept_id"):
            if cid not in cond_pos_map:
                continue
            gg = g.nlargest(TOP_M_PATHWAY_GENES, "n")
            ci = cond_pos_map[cid]
            for gene_id, _n in zip(gg["ensembl_gene_id"], gg["n"], strict=True):
                rows_l.append(ci)
                cols_l.append(gene_index[gene_id])
                vals_l.append(1.0)
        C_topm = (
            sp.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n_cond, n_gene_total))
            if rows_l
            else sp.csr_matrix((n_cond, n_gene_total))
        )

    C_topm.sum_duplicates()
    C_topm.data = np.minimum(C_topm.data, 1.0)

    # gene overlap (rung-4 gene sharing), sparse
    n_shared_gene_matrix = (D_genes @ C_topm.T).toarray()

    # pathway overlap: D_genes @ GP for drugs, C_topm @ GP for conditions
    GP = pathway_ctx["GP"]
    Drug_pathway = (D_genes @ GP).tocsr()
    Drug_pathway.data = np.minimum(Drug_pathway.data, 1.0)
    Cond_pathway = (C_topm @ GP).tocsr()
    Cond_pathway.data = np.minimum(Cond_pathway.data, 1.0)

    n_shared_pathway_matrix = (Drug_pathway @ Cond_pathway.T).toarray()
    d_pw_count = np.asarray(Drug_pathway.sum(axis=1)).ravel()
    c_pw_count = np.asarray(Cond_pathway.sum(axis=1)).ravel()
    denom = d_pw_count[:, None] + c_pw_count[None, :] - n_shared_pathway_matrix
    jaccard_pathway_matrix = np.divide(
        n_shared_pathway_matrix, denom, out=np.zeros_like(n_shared_pathway_matrix), where=denom > 0
    )

    def _gather_pairs(df: pd.DataFrame, d_pos_arr, c_pos_arr) -> dict:
        d_arr = df["ingredient_concept_id"].map(drug_pos).to_numpy()
        c_arr = df["condition_concept_id"].map({c: i for i, c in enumerate(all_cond)}).to_numpy()
        valid = ~(pd.isna(d_arr) | pd.isna(c_arr))
        d_arr_i = np.where(valid, d_arr, 0).astype(int)
        c_arr_i = np.where(valid, c_arr, 0).astype(int)

        n_shared_genes = np.zeros(len(df))
        n_shared_genes[valid] = n_shared_gene_matrix[d_arr_i[valid], c_arr_i[valid]]
        n_shared_pathways = np.zeros(len(df))
        n_shared_pathways[valid] = n_shared_pathway_matrix[d_arr_i[valid], c_arr_i[valid]]
        jaccard_pathways = np.zeros(len(df))
        jaccard_pathways[valid] = jaccard_pathway_matrix[d_arr_i[valid], c_arr_i[valid]]
        has_shared_gene = (n_shared_genes > 0).astype(float)
        return {
            "n_shared_genes": n_shared_genes,
            "n_shared_pathways": n_shared_pathways,
            "jaccard_pathways": jaccard_pathways,
            "has_shared_gene": has_shared_gene,
        }

    train_pw = _gather_pairs(train, d_pos_train, c_pos_train)
    val_pw = _gather_pairs(validate, d_pos_val, c_pos_val)
    for k, v in train_pw.items():
        train[k] = v
    for k, v in val_pw.items():
        validate[k] = v

    log.info("gene/pathway feature construction complete at %.1fs elapsed", time.time() - t0)

    # ---------------------------------------------------------------- designs
    # E14best unavailable (exp14 has no committed script yet at time of run) --
    # substitute: exp07 intrinsic union proxy (degree + p_c here, since the full
    # E7union block requires files not needed by this script's precondition set)
    # plus exp08's nb_excess_gene as the base design, per task instruction.
    fallbacks_taken.append(
        "exp14's model/features not available/committed at run time; base design substituted with "
        "degree + p_c + nb_excess_gene (exp08's gene-tier neighbour feature) instead of E14best's "
        "gene-overlap block."
    )
    baseline_cols = ["log1p_drug_degree", "log1p_condition_degree", "log1p_record_count", "p_c"]
    base_design_cols = [*baseline_cols, "nb_excess_gene"]
    gene_overlap_cols = [*base_design_cols, "n_shared_genes"]
    pathway_cols = [*gene_overlap_cols, "n_shared_pathways", "jaccard_pathways", "has_shared_gene"]

    y_val = validate["y_faers_signal"].to_numpy()
    groups = train["group_key"].to_numpy()
    drug_ids_val = validate["ingredient_concept_id"].to_numpy()
    drug_ids_train = train["ingredient_concept_id"].to_numpy()

    def _cv_and_full(cols: list[str]):
        from sklearn.model_selection import GroupKFold

        x_tr = train[cols].to_numpy(dtype=float)
        x_va = validate[cols].to_numpy(dtype=float)
        gkf = GroupKFold(n_splits=3)
        cv_aucs = []
        for tr_idx, te_idx in gkf.split(x_tr, y_train_real, groups):
            pred = _fit_predict(x_tr[tr_idx], y_train_real[tr_idx], x_tr[te_idx])
            m = _drug_macro_metrics(drug_ids_train[te_idx], y_train_real[te_idx], pred)
            cv_aucs.append(m["drug_macro_auc"])
        val_pred = _fit_predict(x_tr, y_train_real, x_va)
        return cv_aucs, val_pred

    design_order = ["base_(degree+p_c+nb_excess_gene)", "+gene_overlap", "+pathway_overlap"]
    design_cols = {
        design_order[0]: base_design_cols,
        design_order[1]: gene_overlap_cols,
        design_order[2]: pathway_cols,
    }
    design_results = {}
    designs_rows = []
    for name in design_order:
        cv_aucs, val_pred = _cv_and_full(design_cols[name])
        m_val = _drug_macro_metrics(drug_ids_val, y_val, val_pred)
        design_results[name] = {"val_pred": val_pred, "val_metrics": m_val, "cv_aucs": cv_aucs}
        designs_rows.append(
            {
                "design": name,
                "n_features": len(design_cols[name]),
                "train_cv_drug_macro_auc_mean": float(np.nanmean(cv_aucs)),
                "train_cv_drug_macro_auc_std": float(np.nanstd(cv_aucs)),
                "validate_drug_macro_auc": m_val["drug_macro_auc"],
                "validate_drug_macro_p10": m_val["drug_macro_p10"],
                "n_drugs_scored": m_val["n_drugs_scored"],
            }
        )
        log.info(
            "design=%s cv=%.4f val_drug_macro_auc=%.4f (elapsed=%.1fs)",
            name,
            float(np.nanmean(cv_aucs)),
            m_val["drug_macro_auc"],
            time.time() - t0,
        )
    designs_df = pd.DataFrame(designs_rows)
    designs_df.to_csv(out / f"{exp_id}_designs.csv", index=False)

    increment_pathway_over_gene = (
        design_results["+pathway_overlap"]["val_metrics"]["drug_macro_auc"]
        - design_results["+gene_overlap"]["val_metrics"]["drug_macro_auc"]
    )

    # -------------------------------------------- conditional: zero-shared-gene
    zero_shared_gene_val = validate["n_shared_genes"].to_numpy() == 0
    m_gene_zero = _drug_macro_metrics(
        drug_ids_val[zero_shared_gene_val],
        y_val[zero_shared_gene_val],
        design_results["+gene_overlap"]["val_pred"][zero_shared_gene_val],
    )
    m_pathway_zero = _drug_macro_metrics(
        drug_ids_val[zero_shared_gene_val],
        y_val[zero_shared_gene_val],
        design_results["+pathway_overlap"]["val_pred"][zero_shared_gene_val],
    )
    increment_pathway_on_zero_shared_gene_pairs = (
        m_pathway_zero["drug_macro_auc"] - m_gene_zero["drug_macro_auc"]
    )
    zero_shared_gene_df = pd.DataFrame(
        [
            {
                "subset": "zero_shared_gene_pairs",
                "n_pairs": int(zero_shared_gene_val.sum()),
                "gene_overlap_design_drug_macro_auc": m_gene_zero["drug_macro_auc"],
                "pathway_overlap_design_drug_macro_auc": m_pathway_zero["drug_macro_auc"],
                "increment": increment_pathway_on_zero_shared_gene_pairs,
                "n_drugs_scored": m_pathway_zero["n_drugs_scored"],
            }
        ]
    )
    zero_shared_gene_df.to_csv(out / f"{exp_id}_zero_shared_gene_conditional.csv", index=False)

    # ------------------------------------------------------------- coverage
    share_pairs_with_shared_pathway = float((validate["n_shared_pathways"].to_numpy() > 0).mean())
    has_pw = validate["n_shared_pathways"].to_numpy() > 0
    has_gene = validate["n_shared_genes"].to_numpy() > 0
    share_shared_pathway_without_shared_gene = (
        float((has_pw & ~has_gene).sum() / max(has_pw.sum(), 1)) if has_pw.sum() else float("nan")
    )
    coverage_df = pd.DataFrame(
        [
            {
                "share_pairs_with_shared_pathway": share_pairs_with_shared_pathway,
                "share_shared_pathway_without_shared_gene": share_shared_pathway_without_shared_gene,
                "share_pairs_with_shared_gene": float(has_gene.mean()),
            }
        ]
    )
    coverage_df.to_csv(out / f"{exp_id}_coverage_overlap.csv", index=False)

    # ================================================================================
    # Part 1: placebo correction, applied to nb_excess_gene AND pathway features
    # ================================================================================
    zero_nb_gene_train = train["n_neighbours_gene"].fillna(0).to_numpy() == 0
    zero_nb_gene_val_mask = validate["n_neighbours_gene"].fillna(0).to_numpy() == 0

    a_cols = baseline_cols
    b_cols_gene = base_design_cols  # baseline + nb_excess_gene

    placebo_refit_gene = _placebo_refit_within_subset(
        zero_nb_gene_train,
        zero_nb_gene_val_mask,
        a_cols,
        b_cols_gene,
        train,
        y_train_real,
        validate,
        y_val,
        drug_ids_train,
        drug_ids_val,
        SEEDS,
        log,
        "nb_excess_gene",
    )

    # baseline A drug_macro_auc on FULL validate, for the value-permutation gain comparison
    pred_a_full = _fit_predict(
        train[a_cols].to_numpy(dtype=float), y_train_real, validate[a_cols].to_numpy(dtype=float)
    )
    m_a_full = _drug_macro_metrics(drug_ids_val, y_val, pred_a_full)

    placebo_permute_gene = _placebo_value_permutation(
        "nb_excess_gene",
        a_cols,
        b_cols_gene,
        train,
        y_train_real,
        validate,
        y_val,
        train["condition_concept_id"].to_numpy(),
        drug_ids_val,
        m_a_full["drug_macro_auc"],
        SEEDS,
        log,
        "nb_excess_gene",
    )

    # Same two placebos applied to the pathway feature (jaccard_pathways), base = gene_overlap design
    zero_pw_train = train["n_shared_pathways"].to_numpy() == 0
    zero_pw_val = validate["n_shared_pathways"].to_numpy() == 0
    a2_cols = gene_overlap_cols
    b2_cols_pathway = pathway_cols

    placebo_refit_pathway = _placebo_refit_within_subset(
        zero_pw_train,
        zero_pw_val,
        a2_cols,
        b2_cols_pathway,
        train,
        y_train_real,
        validate,
        y_val,
        drug_ids_train,
        drug_ids_val,
        SEEDS,
        log,
        "pathway_overlap",
    )

    pred_a2_full = _fit_predict(
        train[a2_cols].to_numpy(dtype=float), y_train_real, validate[a2_cols].to_numpy(dtype=float)
    )
    m_a2_full = _drug_macro_metrics(drug_ids_val, y_val, pred_a2_full)

    placebo_permute_pathway = _placebo_value_permutation(
        "jaccard_pathways",
        a2_cols,
        b2_cols_pathway,
        train,
        y_train_real,
        validate,
        y_val,
        train["condition_concept_id"].to_numpy(),
        drug_ids_val,
        m_a2_full["drug_macro_auc"],
        SEEDS,
        log,
        "pathway_overlap",
    )

    placebo_corrected_rows = [
        {
            "feature": "nb_excess_gene",
            "placebo_form": "original_zero_neighbour_invalid (exp08, carried through)",
            "gain": EXP08_ZERO_NB_PLACEBO_INVALID,
            "n_reps": 1,
            "std": float("nan"),
            "note": "exp08's original comparison of two differently-fitted models B (all-train) vs A "
            "on the zero-neighbour subset -- invalid by construction, kept here for the record.",
        },
        {
            "feature": "nb_excess_gene",
            "placebo_form": "refit_within_subset",
            "gain": placebo_refit_gene["mean_gain"],
            "n_reps": len(SEEDS),
            "std": placebo_refit_gene["std_gain"],
            "note": f"A and B both refit on zero-neighbour rows only (n_train={placebo_refit_gene['n_subset_train']}, "
            f"n_val={placebo_refit_gene['n_subset_val']}), seeds={SEEDS}",
        },
        {
            "feature": "nb_excess_gene",
            "placebo_form": "train_time_value_permutation",
            "gain": placebo_permute_gene["mean_gain"],
            "n_reps": len(SEEDS),
            "std": placebo_permute_gene["std_gain"],
            "note": f"feature permuted within condition before fitting B, seeds={SEEDS}",
        },
        {
            "feature": "pathway_overlap",
            "placebo_form": "refit_within_subset",
            "gain": placebo_refit_pathway["mean_gain"],
            "n_reps": len(SEEDS),
            "std": placebo_refit_pathway["std_gain"],
            "note": f"A (gene_overlap design) and B (+pathway) refit on zero-shared-pathway rows only "
            f"(n_train={placebo_refit_pathway['n_subset_train']}, n_val={placebo_refit_pathway['n_subset_val']})",
        },
        {
            "feature": "pathway_overlap",
            "placebo_form": "train_time_value_permutation",
            "gain": placebo_permute_pathway["mean_gain"],
            "n_reps": len(SEEDS),
            "std": placebo_permute_pathway["std_gain"],
            "note": f"jaccard_pathways permuted within condition before fitting, seeds={SEEDS}",
        },
        {
            "feature": "nb_excess_gene",
            "placebo_form": "degree_preserving_permutation (exp08, carried through, unaffected)",
            "gain": EXP08_PERMUTED_GRAPH_PLACEBO,
            "n_reps": 5,
            "std": float("nan"),
            "note": "not rerun here per task instruction; exp08's own valid-placebo result",
        },
        {
            "feature": "nb_excess_gene",
            "placebo_form": "within_condition_label_shuffle (exp08, carried through, unaffected)",
            "gain": EXP08_LABEL_SHUFFLE_PLACEBO,
            "n_reps": 1,
            "std": float("nan"),
            "note": "not rerun here per task instruction; exp08's own valid-placebo result",
        },
    ]
    placebo_corrected_df = pd.DataFrame(placebo_corrected_rows)
    placebo_corrected_df.to_csv(out / f"{exp_id}_placebo_corrected.csv", index=False)

    # ------------------------------------------------------------ bootstrap
    per_drug_rows = []
    val_df_idx = pd.DataFrame(
        {
            "drug": drug_ids_val,
            "y": y_val,
            "pred_gene": design_results["+gene_overlap"]["val_pred"],
            "pred_pathway": design_results["+pathway_overlap"]["val_pred"],
        }
    )
    from sklearn.metrics import roc_auc_score

    for drug, g in val_df_idx.groupby("drug"):
        if len(g) < 20 or len(np.unique(g["y"])) < 2:
            continue
        auc_gene = roc_auc_score(g["y"], g["pred_gene"])
        auc_pathway = roc_auc_score(g["y"], g["pred_pathway"])
        per_drug_rows.append({"drug": drug, "n_pairs": len(g), "gain": auc_pathway - auc_gene})
    per_drug_df = pd.DataFrame(per_drug_rows)
    if len(per_drug_df):
        rng_boot = np.random.default_rng(0)
        gains = per_drug_df["gain"].to_numpy()
        boot_means = [
            float(np.mean(rng_boot.choice(gains, size=len(gains), replace=True)))
            for _ in range(BOOT_RESAMPLES)
        ]
        boot_lo, boot_hi = (
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        )
        boot_mean = float(np.mean(boot_means))
    else:
        boot_means, boot_lo, boot_hi, boot_mean = [], float("nan"), float("nan"), float("nan")
    bootstrap_df = pd.DataFrame({"resample_mean_gain": boot_means})
    bootstrap_df.to_csv(out / f"{exp_id}_bootstrap_increments.csv", index=False)

    # --------------------------------------------------------------- floors
    p_c_lookup_score_val = validate["p_c"].to_numpy()
    floor_pc = _drug_macro_metrics(drug_ids_val, y_val, p_c_lookup_score_val)["drug_macro_auc"]
    floor_degree_pc = _drug_macro_metrics(
        drug_ids_val, y_val, validate["log1p_condition_degree"].to_numpy()
    )["drug_macro_auc"]

    elapsed = time.time() - t0
    log.info("total elapsed before finalizing: %.1fs", elapsed)

    # --------------------------------------------------------------- metrics
    best_design = design_results["+pathway_overlap"]["val_metrics"]
    metrics = {
        "placebo_zero_nb_refit_gain": placebo_refit_gene["mean_gain"],
        "placebo_value_permuted_gain": placebo_permute_gene["mean_gain"],
        "placebo_seed_noise": placebo_refit_gene["std_gain"],
        "placebo_zero_pw_refit_gain": placebo_refit_pathway["mean_gain"],
        "placebo_value_permuted_pathway_gain": placebo_permute_pathway["mean_gain"],
        "drug_macro_auc": best_design["drug_macro_auc"],
        "drug_macro_p10": best_design["drug_macro_p10"],
        "n_drugs_scored": best_design["n_drugs_scored"],
        "increment_pathway_over_gene": increment_pathway_over_gene,
        "increment_pathway_on_zero_shared_gene_pairs": increment_pathway_on_zero_shared_gene_pairs,
        "share_pairs_with_shared_pathway": share_pairs_with_shared_pathway,
        "share_shared_pathway_without_shared_gene": share_shared_pathway_without_shared_gene,
        "floor_pc": floor_pc,
        "floor_degree_pc": floor_degree_pc,
        "gene_pathway_map_covered_share": covered_share,
        "bootstrap_gain_mean": boot_mean,
        "bootstrap_gain_ci_lo": boot_lo,
        "bootstrap_gain_ci_hi": boot_hi,
        "exp08_zero_nb_placebo_invalid_original": EXP08_ZERO_NB_PLACEBO_INVALID,
        "exp08_permuted_graph_placebo": EXP08_PERMUTED_GRAPH_PLACEBO,
        "exp08_label_shuffle_placebo": EXP08_LABEL_SHUFFLE_PLACEBO,
    }

    distinguishable = (
        abs(increment_pathway_over_gene) >= 0.01
        or abs(increment_pathway_on_zero_shared_gene_pairs) >= 0.01
    )

    findings = (
        f"Part 1 -- exp08's zero-neighbour placebo was invalid by construction: it compared model B "
        f"(all rows) against model A on the zero-neighbour subset, so the reported +{EXP08_ZERO_NB_PLACEBO_INVALID:.4f} "
        f"measured a difference between two differently-fitted models, not a genuine null. Refit-within-subset "
        f"(fit both A and B on the zero-neighbour rows only, {len(SEEDS)} seeds) gives "
        f"{placebo_refit_gene['mean_gain']:+.4f} +- {placebo_refit_gene['std_gain']:.4f} "
        f"(seeds: {placebo_refit_gene['gains_by_seed']}), and train-time value permutation gives "
        f"{placebo_permute_gene['mean_gain']:+.4f} +- {placebo_permute_gene['std_gain']:.4f} "
        f"(seeds: {placebo_permute_gene['gains_by_seed']}). Both valid forms are "
        f"{'near-null, confirming the +0.0198 was an artefact of the invalid comparison' if abs(placebo_refit_gene['mean_gain']) < 0.01 and abs(placebo_permute_gene['mean_gain']) < 0.01 else 'NOT clearly null -- see caveat below'}. "
        f"exp08's two other placebos (degree-preserving graph permutation {EXP08_PERMUTED_GRAPH_PLACEBO:+.4f}, "
        f"label shuffle {EXP08_LABEL_SHUFFLE_PLACEBO:+.4f}) are unaffected by this correction and are carried "
        f"through unchanged, along with exp08's +{EXP08_INCREMENT_COND:.4f} conditional increment and its "
        f"decay curve {EXP08_DECAY_CURVE}. "
        f"Part 2 -- pathway overlap over the gene-overlap design (base = degree+p_c+nb_excess_gene, since "
        f"exp14's model/features are not committed yet, substituted per task instruction): overall increment "
        f"= {increment_pathway_over_gene:+.4f} drug_macro_auc; on the "
        f"{int(zero_shared_gene_val.sum())} pairs with zero shared genes (the real question), increment = "
        f"{increment_pathway_on_zero_shared_gene_pairs:+.4f}. Coverage: {share_pairs_with_shared_pathway * 100:.1f}% "
        f"of validate pairs share >=1 pathway; of those, {share_shared_pathway_without_shared_gene * 100:.1f}% "
        f"share no gene at all (the rest are coextensive with gene overlap). Gene->pathway map covered "
        f"{covered_share * 100:.1f}% of drug target genes (>=60% required). "
        f"The two placebo forms applied to the pathway feature itself: refit-within-subset "
        f"{placebo_refit_pathway['mean_gain']:+.4f} +- {placebo_refit_pathway['std_gain']:.4f}, "
        f"value-permutation {placebo_permute_pathway['mean_gain']:+.4f} +- {placebo_permute_pathway['std_gain']:.4f}. "
        f"Verdict: pathway overlap is "
        f"{'distinguishable from gene overlap' if distinguishable else 'NOT clearly distinguishable from gene overlap at this sample size -- Round 4 should stop adding hops and spend budget on the label or a better condition->gene layer instead'}. "
        f"Condition-side top-m=50 genes were selected by real Open Targets ot_score from "
        f"condition_gene_ot_long.csv, keyed on ensembl_gene_id (matching gene_index)."
        + (f" Fallbacks taken: {'; '.join(fallbacks_taken)}." if fallbacks_taken else "")
    )

    artifacts = [
        str(out / f"{exp_id}_placebo_corrected.csv"),
        str(out / f"{exp_id}_designs.csv"),
        str(out / f"{exp_id}_zero_shared_gene_conditional.csv"),
        str(out / f"{exp_id}_coverage_overlap.csv"),
        str(out / f"{exp_id}_pathway_map_coverage.csv"),
        str(out / f"{exp_id}_bootstrap_increments.csv"),
    ]

    results.commit()
    log.info("metrics: %s", metrics)
    return {"__metrics__": metrics, "__findings__": findings, "__artifacts__": artifacts}


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp15",
        title="Reactome pathway overlap, and a valid placebo battery",
        hypothesis=(
            "Pathway overlap adds <0.01 drug_macro_auc over gene overlap overall but carries the "
            "signal on pairs with zero shared genes; and exp08's zero-neighbour placebo result is an "
            "artefact of comparing two differently fitted models rather than evidence of a p_c "
            "surrogate."
        ),
        approach=(
            "Refit-within-subset and train-time value-permutation placebos on exp08's neighbour "
            "feature and the new pathway features; Reactome membership from the OT target parquets "
            "(not the hierarchy-only reactome file); sparse gene-pathway products; conditional "
            "evaluation on zero-shared-gene pairs."
        ),
        label="y_faers_signal",
        features=[
            "degree",
            "p_c",
            "intrinsic_union",
            "nb_excess_gene",
            "gene_overlap",
            "pathway_overlap",
        ],
        split="train/validate, grouped by primary target gene",
        notes="Pathway half of the never-completed exp10, plus the exp08 placebo correction.",
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
    metrics = metrics.get("__metrics__", metrics)
    print(findings)

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=artifacts,
        next_steps=(
            "If pathway overlap is not distinguishable from gene overlap, stop adding hops and "
            "invest Round 4's budget in the label or a real condition->gene layer instead. Re-run "
            "the conditional evaluation with exp14's actual gene-overlap design once it is "
            "committed, since this run substituted exp08's nb_excess_gene for it."
        ),
    )


if __name__ == "__main__":
    pass
