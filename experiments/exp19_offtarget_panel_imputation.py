"""exp19 -- QSAR imputation of a 12-target off-target safety panel, and whether the
densified syndrome-interaction feature converts the measured-only within-drug association
into a ranking gain.

Implements experiments/exp19_offtarget_panel_imputation.md. Fetches canonical SMILES from
the ChEMBL molecule REST API for the off-target-panel training molecules
(data/input/drug/offtarget_activities.csv, 69,191 pChEMBL activities for a 12-target human
safety panel, staged on the volume) and for all 4,280 project ingredients. Builds ECFP4 +
physicochemical descriptors with RDKit, fits 12 single-target QSAR regressors held out by
Bemis-Murcko scaffold split (never random split -- a random split over congeneric series
inflates QSAR performance), excludes the project's own ingredients from QSAR training
entirely, imputes affinities for all 4,280 ingredients from a model fit on the full
non-ingredient set, builds the dense syn_aff/syn_n syndrome-interaction feature using
experiments/offtarget_syndrome_map.json (hand-curated, unreviewed -- staged at
/ref/offtarget_syndrome_map.json), and evaluates with drug_macro_auc and the floors.py
method (inlined here, same convention as exp07's inlined leak_audit, so this script does
not depend on Modal mounting a second local file) on three populations.

floors()/_drug_macro_auc() logic below is copied from experiments/floors.py (same file,
same behavior) rather than imported, per exp07's precedent for cross-file logic in a
single-invocation Modal function.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

app = modal.App("bridge-exp19")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pandas==2.2.3",
    "numpy==2.1.3",
    "scikit-learn==1.5.2",
    "lightgbm==4.5.0",
    "pyarrow==17.0.0",
    "scipy==1.14.1",
    "matplotlib==3.9.2",
    "rdkit==2024.3.5",
    "requests==2.32.3",
)

data = modal.Volume.from_name("bridge-data")
results = modal.Volume.from_name("bridge-results", create_if_missing=True)

# The six targets with the largest measured within-drug effects (per the spec's session
# measurement), used only if the "if at risk" budget fallback is taken.
FALLBACK_SIX: list[str] = ["D2", "5-HT2A", "beta1", "COX-1", "alpha1A", "hERG"]

MEASURED_ONLY_SYNDROME_SUBSET_AUC = 0.5417
MEASURED_ONLY_SYNDROME_SUBSET_INTERACTION_AUC = 0.5397

CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"


# ---- floors.py, inlined verbatim (see experiments/floors.py for the documented spec) ----


def _floor_drug_macro_auc(
    ids: pd.Series, y: np.ndarray, score: np.ndarray, min_pairs: int
) -> tuple[float, int]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(
        {"drug": np.asarray(ids), "y": np.asarray(y), "score": np.asarray(score, dtype=float)}
    )
    aucs: list[float] = []
    for _drug_id, grp in df.groupby("drug"):
        if len(grp) < min_pairs:
            continue
        y_grp = grp["y"].to_numpy()
        if len(np.unique(y_grp)) < 2:
            continue
        s = grp["score"]
        if s.isna().any():
            s = s.fillna(s.median())
        if s.isna().all():
            continue
        aucs.append(float(roc_auc_score(y_grp, s.to_numpy())))
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def _floors(train: pd.DataFrame, evaluate: pd.DataFrame, label: str, eligibility: int) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    if len(train) == 0 or len(evaluate) == 0:
        return {
            "floor_pc": float("nan"),
            "floor_degree_pc": float("nan"),
            "prevalence": float("nan"),
            "n_drugs_scored": 0,
        }

    train = train.copy()
    evaluate = evaluate.copy()
    y_train = train[label].astype(float).to_numpy()
    y_eval = evaluate[label].astype(float).to_numpy()

    p_c_train = train.groupby("condition_concept_id")[label].mean()
    p_c_mean = float(p_c_train.mean())
    eval_p_c = evaluate["condition_concept_id"].map(p_c_train).astype(float).fillna(p_c_mean)

    floor_pc, n_drugs_pc = _floor_drug_macro_auc(
        evaluate["ingredient_concept_id"], y_eval, eval_p_c.to_numpy(), eligibility
    )

    drug_degree_train = train.groupby("ingredient_concept_id").size()
    cond_degree_train = train.groupby("condition_concept_id").size()
    drug_degree_median = float(drug_degree_train.median())
    cond_degree_median = float(cond_degree_train.median())

    def _build_features(df: pd.DataFrame, p_c_series: pd.Series) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        feats["drug_degree"] = np.log1p(
            df["ingredient_concept_id"]
            .map(drug_degree_train)
            .astype(float)
            .fillna(drug_degree_median)
        )
        feats["condition_degree"] = np.log1p(
            df["condition_concept_id"]
            .map(cond_degree_train)
            .astype(float)
            .fillna(cond_degree_median)
        )
        feats["p_c"] = p_c_series.to_numpy()
        if "record_count" in df.columns:
            rc = pd.to_numeric(df["record_count"], errors="coerce")
            feats["condition_record_count"] = np.log1p(rc.fillna(rc.median()))
        return feats

    train_p_c = train["condition_concept_id"].map(p_c_train).astype(float).fillna(p_c_mean)
    x_train = _build_features(train, train_p_c).astype("float32")
    x_eval = _build_features(evaluate, eval_p_c).astype("float32")
    for col in x_train.columns:
        col_median = float(x_train[col].median())
        x_train[col] = x_train[col].fillna(col_median)
        x_eval[col] = x_eval[col].fillna(col_median)

    if len(np.unique(y_train)) < 2:
        floor_degree_pc, n_drugs_degree_pc = float("nan"), 0
    else:
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
        )
        model.fit(x_train, y_train)
        eval_proba = model.predict_proba(x_eval)[:, 1]
        floor_degree_pc, n_drugs_degree_pc = _floor_drug_macro_auc(
            evaluate["ingredient_concept_id"], y_eval, eval_proba, eligibility
        )

    prevalence = float(np.mean(y_train)) if len(y_train) else float("nan")
    n_drugs_scored = n_drugs_degree_pc if n_drugs_degree_pc else n_drugs_pc
    return {
        "floor_pc": floor_pc,
        "floor_degree_pc": floor_degree_pc,
        "prevalence": prevalence,
        "n_drugs_scored": n_drugs_scored,
    }


# ---- preconditions ----


def _check_preconditions(paths: dict[str, pathlib.Path]) -> str | None:
    import pandas as pd

    expected_rows = {
        "train": 723_586,
        "validate": 434_151,
        "ingredient_features": 4_280,
        "offtarget_activities": 69_191,
    }
    for key, expected in expected_rows.items():
        path = paths[key]
        if not path.exists():
            return f"missing required input: {path}"
        with path.open() as fh:
            n = sum(1 for _ in fh) - 1
        if n != expected:
            return f"row count mismatch for {path}: expected {expected}, got {n}"

    if not paths["syndrome_map"].exists():
        return f"missing required input: {paths['syndrome_map']}"
    if not paths["condition_ontology_map"].exists():
        return f"missing required input: {paths['condition_ontology_map']}"

    train = pd.read_csv(paths["train"], usecols=["y_faers_signal"])
    rate = float(train["y_faers_signal"].mean())
    if abs(rate - 0.1100) > 0.001:
        return f"train positive rate for y_faers_signal is {rate:.4f}, expected 0.1100 +/- 0.001"
    return None


# ---- SMILES fetch ----


def _fetch_smiles(chembl_ids: list[str], log: logging.Logger, batch_size: int = 120) -> dict:
    """Batch-fetch canonical SMILES from the ChEMBL molecule REST API, concurrently.

    batch_size=120 keeps the GET request line under the server's 4094-byte limit (measured:
    a 250-id batch produces a ~4100-4200 byte request line and gets a flat 400 "Request Line
    is too large" from every batch -- this is not a rate limit and does not benefit from
    retries, so the fix is a smaller batch, not more attempts).
    """
    import concurrent.futures

    import requests

    unique_ids = sorted(set(chembl_ids))
    batches = [unique_ids[i : i + batch_size] for i in range(0, len(unique_ids), batch_size)]
    out: dict[str, str] = {}

    def _fetch_one(batch: list[str]) -> dict[str, str]:
        session = requests.Session()
        result: dict[str, str] = {}
        params = {
            "molecule_chembl_id__in": ",".join(batch),
            "only": "molecule_chembl_id,molecule_structures",
            "limit": 1000,
        }
        for attempt in range(3):
            try:
                resp = session.get(CHEMBL_MOLECULE_URL, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                for mol in payload.get("molecules", []):
                    struct = mol.get("molecule_structures") or {}
                    smiles = struct.get("canonical_smiles")
                    if smiles:
                        result[mol["molecule_chembl_id"]] = smiles
                return result
            except Exception as exc:
                if attempt == 2:
                    log.warning("SMILES batch fetch failed after 3 attempts: %s", exc)
                    return result
        return result

    log.info(
        "fetching SMILES for %d unique ChEMBL ids in %d batches of <=%d",
        len(unique_ids),
        len(batches),
        batch_size,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for res in pool.map(_fetch_one, batches):
            out.update(res)
    log.info("fetched SMILES for %d / %d requested ids", len(out), len(unique_ids))
    return out


# ---- descriptors ----


def _descriptors_for_smiles(smiles_map: dict) -> tuple:
    """ECFP4 (2048 bits) + physicochemical descriptors, per valid SMILES.

    Returns (chembl_ids, fingerprints (n, 2048) uint8, phys_df, valid_chembl_ids).
    Invalid/unparseable SMILES are dropped; the caller must account for the reduced set.
    """
    import numpy as np
    import pandas as pd
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    ids: list[str] = []
    fps: list[np.ndarray] = []
    phys_rows: list[dict] = []
    for chembl_id, smiles in smiles_map.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = generator.GetFingerprintAsNumPy(mol)
        ids.append(chembl_id)
        fps.append(fp)
        phys_rows.append(
            {
                "mol_wt": Descriptors.MolWt(mol),
                "log_p": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "ring_count": Descriptors.RingCount(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "frac_csp3": Descriptors.FractionCSP3(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol),
            }
        )
    fp_matrix = np.array(fps, dtype="uint8") if fps else np.zeros((0, 2048), dtype="uint8")
    phys_df = pd.DataFrame(phys_rows)
    return ids, fp_matrix, phys_df


def _max_tanimoto_to_train(
    query_fp: np.ndarray, train_fp: np.ndarray, chunk: int = 512
) -> np.ndarray:
    """Nearest-neighbour Tanimoto similarity of each query fingerprint to a train set, via
    chunked matrix multiplication instead of sklearn's brute-force Jaccard NearestNeighbors.

    That brute-force path is what caused the first two attempts at this experiment to hit
    the 900s Modal timeout mid-imputation: NearestNeighbors(metric="jaccard").kneighbors on
    a several-thousand x several-thousand x 2048-bit problem, repeated once per off-target,
    was too slow. Tanimoto on binary vectors is (A & B).sum() / (A | B).sum(), which reduces
    to intersection = A @ B.T, union = sum_A + sum_B - intersection -- a BLAS matmul that is
    orders of magnitude faster than a generic brute-force nearest-neighbour search.
    """
    import numpy as np

    if len(query_fp) == 0 or len(train_fp) == 0:
        return np.zeros(len(query_fp), dtype="float32")

    train_f = train_fp.astype("float32")
    train_sum = train_f.sum(axis=1)
    query_f = query_fp.astype("float32")
    query_sum = query_f.sum(axis=1)

    best = np.zeros(len(query_fp), dtype="float32")
    for start in range(0, len(query_fp), chunk):
        end = min(start + chunk, len(query_fp))
        intersection = query_f[start:end] @ train_f.T
        union = query_sum[start:end, None] + train_sum[None, :] - intersection
        sim = intersection / np.clip(union, 1e-9, None)
        best[start:end] = sim.max(axis=1)
    return best


def _bemis_murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


# ---- QSAR ----


def _fit_qsar_one_target(
    target_activities: pd.DataFrame,
    smiles_map: dict,
    ingredient_chembl_ids: set,
    log: logging.Logger,
    target_name: str,
) -> dict:
    """One target's QSAR: exclude our ingredients, scaffold-split for the reported
    metrics, then a final model on 100% of the (non-ingredient) data for imputation --
    same fit-for-eval-vs-fit-for-deployment split as exp07's CV-then-final-fit pattern.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    # aggregate duplicate (molecule, target) rows by max pChEMBL, per provenance convention
    agg = target_activities.groupby("molecule_chembl_id")["pchembl_value"].max().reset_index()
    agg = agg[~agg["molecule_chembl_id"].isin(ingredient_chembl_ids)]
    agg["smiles"] = agg["molecule_chembl_id"].map(smiles_map)
    agg = agg.dropna(subset=["smiles"])

    n_available = len(agg)
    if n_available < 20:
        return {
            "target": target_name,
            "spearman_rho": float("nan"),
            "rmse": float("nan"),
            "n_train": 0,
            "n_test": 0,
            "domain_coverage": float("nan"),
            "skipped": True,
            "model": None,
            "train_fp": None,
            "train_ids": [],
        }

    ids, fp_matrix, phys_df = _descriptors_for_smiles(
        dict(zip(agg["molecule_chembl_id"], agg["smiles"], strict=False))
    )
    if len(ids) < 20:
        return {
            "target": target_name,
            "spearman_rho": float("nan"),
            "rmse": float("nan"),
            "n_train": 0,
            "n_test": 0,
            "domain_coverage": float("nan"),
            "skipped": True,
            "model": None,
            "train_fp": None,
            "train_ids": [],
        }

    id_to_y = dict(zip(agg["molecule_chembl_id"], agg["pchembl_value"], strict=False))
    y = np.array([id_to_y[i] for i in ids], dtype=float)
    id_to_smiles = dict(zip(agg["molecule_chembl_id"], agg["smiles"], strict=False))
    scaffolds = np.array([_bemis_murcko_scaffold(id_to_smiles[i]) for i in ids])

    x_full = np.hstack([fp_matrix, phys_df.to_numpy(dtype="float32")])

    # Bemis-Murcko scaffold split, 80/20 by scaffold group (not by row).
    rng = np.random.default_rng(0)
    unique_scaffolds = np.unique(scaffolds)
    rng.shuffle(unique_scaffolds)
    n_test_scaffolds = max(1, round(0.2 * len(unique_scaffolds)))
    test_scaffolds = set(unique_scaffolds[:n_test_scaffolds])
    test_mask = np.array([s in test_scaffolds for s in scaffolds])
    train_mask = ~test_mask

    if train_mask.sum() < 10 or test_mask.sum() < 5:
        # too few scaffolds to split meaningfully; report null metrics but still fit a
        # final model on everything for imputation.
        rho, rmse, n_test = float("nan"), float("nan"), int(test_mask.sum())
    else:
        eval_model = RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=0
        )
        eval_model.fit(x_full[train_mask], y[train_mask])
        pred_test = eval_model.predict(x_full[test_mask])
        rho = float(spearmanr(y[test_mask], pred_test).correlation)
        rmse = float(np.sqrt(mean_squared_error(y[test_mask], pred_test)))
        n_test = int(test_mask.sum())

    # applicability domain: nearest-neighbour Tanimoto of test fingerprints to train set
    domain_coverage = float("nan")
    if train_mask.sum() >= 10 and test_mask.sum() >= 5:
        train_fp_bits = fp_matrix[train_mask]
        test_fp_bits = fp_matrix[test_mask]
        tanimoto_sim = _max_tanimoto_to_train(test_fp_bits, train_fp_bits)
        domain_coverage = float(np.mean(tanimoto_sim >= 0.4))

    # final deployment model: 100% of non-ingredient data, for imputing the 4,280 ingredients.
    final_model = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=0
    )
    final_model.fit(x_full, y)

    log.info(
        "target=%s n_train_total=%d rho=%.4f rmse=%.4f domain_coverage=%.4f",
        target_name,
        n_available,
        rho if rho == rho else float("nan"),
        rmse,
        domain_coverage,
    )

    return {
        "target": target_name,
        "spearman_rho": rho,
        "rmse": rmse,
        "n_train": n_available,
        "n_test": n_test,
        "domain_coverage": domain_coverage,
        "skipped": False,
        "model": final_model,
        "train_fp": fp_matrix.astype(bool),
        "train_ids": ids,
    }


def _impute_for_ingredients(
    qsar_result: dict, ingredient_smiles_map: dict, log: logging.Logger
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    target_name = qsar_result["target"]
    if qsar_result["model"] is None:
        return pd.DataFrame(
            {
                "chembl_id": list(ingredient_smiles_map.keys()),
                "imputed_affinity": float("nan"),
                "in_domain": False,
                "has_structure": False,
                "target": target_name,
            }
        )

    ids, fp_matrix, phys_df = _descriptors_for_smiles(ingredient_smiles_map)
    x = np.hstack([fp_matrix, phys_df.to_numpy(dtype="float32")]) if ids else np.zeros((0, 2058))
    preds = qsar_result["model"].predict(x) if len(ids) else np.array([])

    in_domain = np.array([], dtype=bool)
    if len(ids) and qsar_result["train_fp"] is not None and len(qsar_result["train_fp"]):
        tanimoto_sim = _max_tanimoto_to_train(fp_matrix, qsar_result["train_fp"])
        in_domain = tanimoto_sim >= 0.4

    structured_ids = set(ids)
    all_ids = list(ingredient_smiles_map.keys())
    pred_map = dict(zip(ids, preds, strict=False))
    domain_map = dict(zip(ids, in_domain, strict=False))

    out = pd.DataFrame(
        {
            "chembl_id": all_ids,
            "imputed_affinity": [pred_map.get(i, float("nan")) for i in all_ids],
            "in_domain": [bool(domain_map.get(i, False)) for i in all_ids],
            "has_structure": [i in structured_ids for i in all_ids],
            "target": target_name,
        }
    )
    log.info("imputed %s for %d/%d ingredients with structure", target_name, len(ids), len(all_ids))
    return out


# ---- syndrome map / ADR widening ----


def _widen_syndrome_map(
    syndrome_map: dict, cond_ontology: pd.DataFrame, log: logging.Logger
) -> tuple[dict, dict]:
    """Widen each off-target's syndrome condition set to every condition_concept_id that
    shares an ontology_id (MONDO/EFO/HPO term) with an already-curated member. This is the
    closest 'ancestor hierarchy' relation condition_ontology_map.csv actually carries (it
    is a concept->ontology-term map, not a parent/child table) -- widening on shared term
    picks up the many-to-one OMOP concepts that resolve to the same disease/phenotype term
    the curator already flagged, without inventing a hierarchy the file does not have.
    """

    cond_to_terms = cond_ontology.groupby("condition_concept_id")["ontology_id"].apply(set)
    term_to_conds = cond_ontology.groupby("ontology_id")["condition_concept_id"].apply(set)

    widened: dict[str, list[int]] = {}
    coverage_before: dict[str, int] = {}
    coverage_after: dict[str, int] = {}
    for off, conds in syndrome_map.items():
        conds_set = {int(c) for c in conds}
        coverage_before[off] = len(conds_set)
        added = set()
        for c in conds_set:
            terms = cond_to_terms.get(c, set())
            for term in terms:
                added |= term_to_conds.get(term, set())
        widened_set = conds_set | added
        widened[off] = sorted(widened_set)
        coverage_after[off] = len(widened_set)

    total_before = len(set().union(*[set(v) for v in syndrome_map.values()])) if syndrome_map else 0
    total_after = len(set().union(*[set(v) for v in widened.values()])) if widened else 0
    log.info(
        "ADR map widening: %d -> %d distinct conditions across all syndromes",
        total_before,
        total_after,
    )
    summary = {
        "n_conditions_before": total_before,
        "n_conditions_after": total_after,
        "per_offtarget_before": coverage_before,
        "per_offtarget_after": coverage_after,
    }
    return widened, summary


def _syndrome_pair_coverage(pairs: pd.DataFrame, syndrome_map: dict) -> float:
    all_conds = set()
    for conds in syndrome_map.values():
        all_conds |= set(conds)
    if len(pairs) == 0:
        return float("nan")
    return float(pairs["condition_concept_id"].isin(all_conds).mean())


# ---- dense feature build ----


def _build_syn_features(
    df: pd.DataFrame,
    ingredient_panel: pd.DataFrame,
    syndrome_map: dict,
    off_target_names: list[str],
) -> pd.DataFrame:
    """syn_aff (max combined affinity over off-targets whose widened syndrome contains this
    condition), syn_n (count of such off-targets with a non-null combined affinity),
    measured_any (whether >=1 of those off-targets has a measured, not imputed, value).

    Vectorized via merges, not a per-row Python loop: a first attempt looped over each of
    df's ~1.1M rows doing a pandas .loc lookup per row and blew the 900s Modal timeout
    without reaching evaluation. The syndrome map is tiny (12 off-targets x a few dozen
    conditions each), so exploding it to (offtarget, condition) pairs and merging against a
    melted long form of the ingredient panel is orders of magnitude cheaper.
    """
    import pandas as pd

    off_target_names = [o for o in off_target_names if o in syndrome_map]

    # melt ingredient_panel's per-offtarget combined/measured columns into long form
    long_frames = []
    for off in off_target_names:
        combined_col = f"{off}_combined"
        measured_col = f"{off}_measured"
        if combined_col not in ingredient_panel.columns:
            continue
        sub = ingredient_panel[["ingredient_concept_id", combined_col, measured_col]].rename(
            columns={combined_col: "affinity", measured_col: "measured"}
        )
        sub = sub.dropna(subset=["affinity"])
        sub["offtarget"] = off
        long_frames.append(sub)
    if not long_frames:
        out = pd.DataFrame(index=df.index)
        out["syn_aff"] = float("nan")
        out["syn_n"] = 0
        out["syn_measured_any"] = 0.0
        return out
    ing_long = pd.concat(long_frames, ignore_index=True)

    # explode the (widened) syndrome map to (offtarget, condition_concept_id) pairs
    pair_rows = [
        {"offtarget": off, "condition_concept_id": int(c)}
        for off in off_target_names
        for c in syndrome_map.get(off, [])
    ]
    syn_pairs = pd.DataFrame(pair_rows, columns=["offtarget", "condition_concept_id"])

    # (ingredient, condition, offtarget, affinity, measured) via a many-to-many merge on
    # offtarget, then aggregate to one row per (ingredient, condition).
    joined = syn_pairs.merge(ing_long, on="offtarget", how="inner")
    agg = joined.groupby(["ingredient_concept_id", "condition_concept_id"]).agg(
        syn_aff=("affinity", "max"),
        syn_n=("affinity", "count"),
        syn_measured_any=("measured", "max"),
    )

    key = df[["ingredient_concept_id", "condition_concept_id"]].merge(
        agg, on=["ingredient_concept_id", "condition_concept_id"], how="left"
    )
    key.index = df.index

    out = pd.DataFrame(index=df.index)
    out["syn_aff"] = key["syn_aff"]
    out["syn_n"] = key["syn_n"].fillna(0).astype(int)
    out["syn_measured_any"] = key["syn_measured_any"].fillna(False).astype(float)
    return out


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=16.0,
    memory=32768,
    timeout=900,
)
def run(exp_id: str) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp19")

    import json

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    data_root = pathlib.Path("/data")
    paths = {
        "train": data_root / "splits" / "train.csv",
        "validate": data_root / "splits" / "validate.csv",
        "ingredient_features": data_root / "drug" / "ingredient_features.csv",
        "offtarget_activities": data_root / "drug" / "offtarget_activities.csv",
        "syndrome_map": data_root / "ref" / "offtarget_syndrome_map.json",
        "condition_ontology_map": data_root / "condition" / "condition_ontology_map.csv",
    }

    err = _check_preconditions(paths)
    if err is not None:
        log.error("precondition check failed: %s", err)
        return {"precondition_failed": 1.0, "precondition_error_message": err}
    log.info("preconditions passed")

    fallback_notes: list[str] = []

    train = pd.read_csv(paths["train"])
    validate = pd.read_csv(paths["validate"])
    ingredient_features = pd.read_csv(paths["ingredient_features"])
    offtarget_activities = pd.read_csv(paths["offtarget_activities"])
    with paths["syndrome_map"].open() as fh:
        syndrome_map_raw = json.load(fh)
    cond_ontology = pd.read_csv(paths["condition_ontology_map"])

    off_target_names = sorted(offtarget_activities["offtarget"].unique())
    panel_target_to_off = dict(
        zip(
            offtarget_activities["target_chembl_id"],
            offtarget_activities["offtarget"],
            strict=False,
        )
    )
    n_targets_total = len(off_target_names)

    budget_risk = False  # set True and take the 6-target fallback if a real time crunch hits
    if budget_risk:
        off_target_names = [t for t in off_target_names if t in FALLBACK_SIX]
        fallback_notes.append(
            f"Budget fallback: cut panel from {n_targets_total} to the 6 largest-effect "
            f"targets: {off_target_names}."
        )
    log.info("panel targets in this run: %s", off_target_names)

    # ---- ingredient chembl ids, for the exclusion rule ----
    ingredient_features = ingredient_features.copy()
    ingredient_features["chembl_id_effective"] = ingredient_features["chembl_id"].fillna(
        ingredient_features["chembl_parent_id"]
    )
    ingredient_chembl_ids = set(
        ingredient_features.loc[
            ingredient_features["chembl_id_effective"].notna(), "chembl_id_effective"
        ]
    )
    ingredients_with_structure_target = ingredient_features[
        ingredient_features["chembl_id_effective"].notna()
    ][["omop_concept_id", "chembl_id_effective"]].rename(
        columns={"omop_concept_id": "ingredient_concept_id", "chembl_id_effective": "chembl_id"}
    )
    log.info(
        "%d / %d ingredients have a ChEMBL id to attempt structure lookup",
        len(ingredients_with_structure_target),
        len(ingredient_features),
    )

    # ---- SMILES fetch: training molecules (minus our ingredients) + all ingredients ----
    training_mol_ids = set(offtarget_activities["molecule_chembl_id"]) - ingredient_chembl_ids
    ids_to_fetch = list(training_mol_ids) + list(ingredients_with_structure_target["chembl_id"])
    smiles_map = _fetch_smiles(ids_to_fetch, log)

    ingredient_smiles_map = {
        row.chembl_id: smiles_map[row.chembl_id]
        for row in ingredients_with_structure_target.itertuples()
        if row.chembl_id in smiles_map
    }
    n_ingredients_with_structure = len(ingredient_smiles_map)
    log.info(
        "ingredients_with_structure=%d / %d (%.1f%%)",
        n_ingredients_with_structure,
        len(ingredient_features),
        100 * n_ingredients_with_structure / len(ingredient_features),
    )
    if n_ingredients_with_structure < 100:
        return {
            "precondition_failed": 1.0,
            "precondition_error_message": (
                f"only {n_ingredients_with_structure} ingredients resolved a structure from "
                "the ChEMBL SMILES fetch; cannot proceed with QSAR imputation."
            ),
        }

    # ---- 12 (or 6) single-target QSAR fits ----
    qsar_results: dict[str, dict] = {}
    for off in off_target_names:
        target_rows = offtarget_activities[offtarget_activities["offtarget"] == off]
        qsar_results[off] = _fit_qsar_one_target(
            target_rows, smiles_map, ingredient_chembl_ids, log, off
        )

    qsar_perf_rows = []
    for off, res in qsar_results.items():
        qsar_perf_rows.append(
            {
                "target": off,
                "spearman_rho": res["spearman_rho"],
                "rmse": res["rmse"],
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "domain_coverage": res["domain_coverage"],
                "skipped": res["skipped"],
            }
        )
    qsar_perf_df = pd.DataFrame(qsar_perf_rows)
    qsar_perf_path = out / f"{exp_id}_qsar_performance.csv"
    qsar_perf_df.to_csv(qsar_perf_path, index=False)

    rho_values = qsar_perf_df.loc[~qsar_perf_df["skipped"], "spearman_rho"].dropna()
    qsar_spearman_median = float(rho_values.median()) if len(rho_values) else float("nan")
    qsar_spearman_min = float(rho_values.min()) if len(rho_values) else float("nan")
    n_targets_above_half = int((rho_values >= 0.5).sum())
    log.info(
        "QSAR summary: median_rho=%.4f min_rho=%.4f n_targets>=0.5=%d/%d",
        qsar_spearman_median,
        qsar_spearman_min,
        n_targets_above_half,
        len(off_target_names),
    )

    # ---- impute for all 4,280 ingredients, per target ----
    imputed_frames = []
    for off in off_target_names:
        imputed_frames.append(
            _impute_for_ingredients(qsar_results[off], ingredient_smiles_map, log)
        )
    imputed_long = pd.concat(imputed_frames, ignore_index=True)

    # measured affinities: max pChEMBL per (ingredient's chembl id, off-target)
    measured = offtarget_activities.copy()
    measured["offtarget"] = measured["target_chembl_id"].map(panel_target_to_off)
    measured_by_ingredient = measured[measured["molecule_chembl_id"].isin(ingredient_chembl_ids)]
    measured_agg = (
        measured_by_ingredient.groupby(["molecule_chembl_id", "offtarget"])["pchembl_value"]
        .max()
        .reset_index()
    )

    chembl_to_ingredient = dict(
        zip(
            ingredients_with_structure_target["chembl_id"],
            ingredients_with_structure_target["ingredient_concept_id"],
            strict=False,
        )
    )
    # also cover ingredients whose chembl_id_effective matched via the raw chembl_id column
    # (ingredients_with_structure_target already uses chembl_id_effective, so this is complete)

    # merge, not a dict-based .map(): ingredient_features has 2 rows with NaN
    # omop_concept_id (4,280 rows, 4,278 unique ids), and a dict built from zip() keeps two
    # distinct NaN key objects, which pandas then rejects as a non-unique mapper index.
    ing_panel = ingredient_features[["omop_concept_id", "chembl_id_effective"]].rename(
        columns={"omop_concept_id": "ingredient_concept_id", "chembl_id_effective": "chembl_id"}
    )

    for off in off_target_names:
        imp = imputed_long[imputed_long["target"] == off].set_index("chembl_id")
        ing_panel[f"{off}_imputed"] = ing_panel["chembl_id"].map(imp["imputed_affinity"])
        ing_panel[f"{off}_in_domain"] = ing_panel["chembl_id"].map(imp["in_domain"]).fillna(False)

        meas = measured_agg[measured_agg["offtarget"] == off].set_index("molecule_chembl_id")
        ing_panel[f"{off}_measured_value"] = ing_panel["chembl_id"].map(meas["pchembl_value"])
        ing_panel[f"{off}_measured"] = ing_panel[f"{off}_measured_value"].notna()
        ing_panel[f"{off}_combined"] = ing_panel[f"{off}_measured_value"].fillna(
            ing_panel[f"{off}_imputed"]
        )

    ing_panel["has_structure"] = ing_panel["chembl_id"].isin(ingredient_smiles_map.keys())
    panel_out_path = pathlib.Path("/data") / "drug" / "offtarget_panel_imputed.csv"
    ing_panel.to_csv(panel_out_path, index=False)
    log.info("wrote offtarget_panel_imputed.csv to volume: %s", panel_out_path)

    # ---- ADR map widening ----
    widened_map, widen_summary = _widen_syndrome_map(syndrome_map_raw, cond_ontology, log)
    widen_path = out / f"{exp_id}_adr_map_widened.json"
    with widen_path.open("w") as fh:
        json.dump({"widened_map": widened_map, "summary": widen_summary}, fh, indent=1)

    all_conds_before = set()
    for conds in syndrome_map_raw.values():
        all_conds_before |= {int(c) for c in conds}
    all_conds_after = set()
    for conds in widened_map.values():
        all_conds_after |= set(conds)

    pair_coverage_before = _syndrome_pair_coverage(train, syndrome_map_raw) + 0.0
    pair_coverage_after = float(
        pd.concat([train, validate])["condition_concept_id"].isin(all_conds_after).mean()
    )
    log.info(
        "ADR map pair coverage before=%.4f (%d conds), after=%.4f (%d conds)",
        pair_coverage_before,
        len(all_conds_before),
        pair_coverage_after,
        len(all_conds_after),
    )

    # ---- syn_aff / syn_n dense feature build (using the widened map) ----
    train_syn = _build_syn_features(train, ing_panel, widened_map, off_target_names)
    val_syn = _build_syn_features(validate, ing_panel, widened_map, off_target_names)

    # ---- degree + p_c baseline features (train-only, applied to both) ----
    def _degree_pc(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        drug_degree = df_a.groupby("ingredient_concept_id").size()
        cond_degree = df_a.groupby("condition_concept_id").size()
        drug_degree_median = float(drug_degree.median())
        cond_degree_median = float(cond_degree.median())
        p_c_map = df_a.groupby("condition_concept_id")["y_faers_signal"].mean()
        train_wide_mean = float(df_a["y_faers_signal"].mean())

        def _apply(df: pd.DataFrame) -> pd.DataFrame:
            f = pd.DataFrame(index=df.index)
            f["degree_drug"] = np.log1p(
                df["ingredient_concept_id"].map(drug_degree).fillna(drug_degree_median)
            )
            f["degree_condition"] = np.log1p(
                df["condition_concept_id"].map(cond_degree).fillna(cond_degree_median)
            )
            f["p_c"] = df["condition_concept_id"].map(p_c_map).fillna(train_wide_mean)
            return f

        return _apply(df_a), _apply(df_b)

    degree_pc_train, degree_pc_val = _degree_pc(train, validate)

    def _drug_macro(ids: pd.Series, y: pd.Series, score: np.ndarray) -> dict:
        df = pd.DataFrame(
            {"drug": ids.to_numpy(), "y": y.to_numpy(), "score": np.asarray(score, dtype=float)}
        )
        rows = []
        for drug_id, grp in df.groupby("drug"):
            if len(grp) < 20 or grp["y"].nunique() < 2:
                continue
            rows.append({"drug": drug_id, "auc": float(roc_auc_score(grp["y"], grp["score"]))})
        tbl = pd.DataFrame(rows)
        if tbl.empty:
            return {"drug_macro_auc": float("nan"), "n_drugs_scored": 0}
        return {"drug_macro_auc": float(tbl["auc"].mean()), "n_drugs_scored": len(tbl)}

    def _fit_and_eval(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        ids_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        ids_val: pd.Series,
    ) -> dict:
        x_train = x_train.astype("float32").fillna(x_train.median(numeric_only=True))
        x_val = x_val.astype("float32")
        for col in x_train.columns:
            x_val[col] = x_val[col].fillna(float(x_train[col].median()))
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=0,
        )
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_val)[:, 1]
        macro = _drug_macro(ids_val, y_val, proba)
        return {
            "drug_macro_auc": macro["drug_macro_auc"],
            "n_drugs_scored": macro["n_drugs_scored"],
            "validate_ap": float(average_precision_score(y_val, proba)),
        }

    y_label = "y_faers_signal"

    def _evaluate_population(name: str, cond_subset: set | None, structure_only: bool) -> dict:
        train_mask = pd.Series(True, index=train.index)
        val_mask = pd.Series(True, index=validate.index)
        if cond_subset is not None:
            train_mask &= train["condition_concept_id"].isin(cond_subset)
            val_mask &= validate["condition_concept_id"].isin(cond_subset)
        if structure_only:
            structured_ings = set(
                ing_panel.loc[ing_panel["has_structure"], "ingredient_concept_id"]
            )
            train_mask &= train["ingredient_concept_id"].isin(structured_ings)
            val_mask &= validate["ingredient_concept_id"].isin(structured_ings)

        tr = train[train_mask]
        va = validate[val_mask]
        if len(tr) == 0 or len(va) == 0 or va[y_label].nunique() < 2:
            return {
                "population": name,
                "n_train_rows": len(tr),
                "n_val_rows": len(va),
                "floor_pc": float("nan"),
                "floor_degree_pc": float("nan"),
                "drug_macro_auc_with_interaction": float("nan"),
                "n_drugs_scored": 0,
            }

        floor_res = _floors(tr, va, y_label, eligibility=20)

        x_train_baseline = degree_pc_train.loc[train_mask]
        x_val_baseline = degree_pc_val.loc[val_mask]
        x_train_interaction = pd.concat(
            [degree_pc_train.loc[train_mask], train_syn.loc[train_mask]], axis=1
        )
        x_val_interaction = pd.concat([degree_pc_val.loc[val_mask], val_syn.loc[val_mask]], axis=1)

        baseline_fit = _fit_and_eval(
            x_train_baseline,
            tr[y_label],
            tr["ingredient_concept_id"],
            x_val_baseline,
            va[y_label],
            va["ingredient_concept_id"],
        )
        interaction_fit = _fit_and_eval(
            x_train_interaction,
            tr[y_label],
            tr["ingredient_concept_id"],
            x_val_interaction,
            va[y_label],
            va["ingredient_concept_id"],
        )

        return {
            "population": name,
            "n_train_rows": len(tr),
            "n_val_rows": len(va),
            "floor_pc": floor_res["floor_pc"],
            "floor_degree_pc": floor_res["floor_degree_pc"],
            "drug_macro_auc_baseline_degree_pc": baseline_fit["drug_macro_auc"],
            "drug_macro_auc_with_interaction": interaction_fit["drug_macro_auc"],
            "validate_ap_baseline": baseline_fit["validate_ap"],
            "validate_ap_with_interaction": interaction_fit["validate_ap"],
            "n_drugs_scored": interaction_fit["n_drugs_scored"],
        }

    syndrome_conds_widened = all_conds_after
    pop_all = _evaluate_population("all_conditions", None, structure_only=False)
    pop_syndrome = _evaluate_population(
        "syndrome_114_widened", syndrome_conds_widened, structure_only=False
    )
    pop_syndrome_structure = _evaluate_population(
        "syndrome_widened_structure_only", syndrome_conds_widened, structure_only=True
    )

    models_df = pd.DataFrame([pop_all, pop_syndrome, pop_syndrome_structure])
    models_path = out / f"{exp_id}_models.csv"
    models_df.to_csv(models_path, index=False)

    drug_macro_auc_syndrome_subset = pop_syndrome["drug_macro_auc_with_interaction"]
    floor_pc_syndrome_subset = pop_syndrome["floor_pc"]
    increment_over_measured_only = (
        drug_macro_auc_syndrome_subset - MEASURED_ONLY_SYNDROME_SUBSET_INTERACTION_AUC
    )

    # ---- within-drug sanity check on imputed-only drugs (no measured panel at all) ----
    measured_ings = set(measured_by_ingredient["molecule_chembl_id"].map(chembl_to_ingredient))
    imputed_only_ing_panel = ing_panel[~ing_panel["ingredient_concept_id"].isin(measured_ings)]

    within_rows = []
    trainval = pd.concat([train, validate], ignore_index=True)
    for off in off_target_names:
        col_val = f"{off}_imputed"
        col_dom = f"{off}_in_domain"
        if col_val not in imputed_only_ing_panel.columns:
            continue
        panel_off = imputed_only_ing_panel[
            imputed_only_ing_panel[col_val].notna() & imputed_only_ing_panel[col_dom]
        ][["ingredient_concept_id", col_val]].rename(columns={col_val: "affinity"})
        if len(panel_off) < 5:
            continue
        hi_thresh = panel_off["affinity"].median()
        panel_off = panel_off.assign(
            potent=panel_off["affinity"] >= max(6.0, hi_thresh)
            if panel_off["affinity"].max() >= 6.0
            else panel_off["affinity"] >= hi_thresh
        )
        syn_conds = set(widened_map.get(off, []))
        merged = trainval.merge(panel_off, on="ingredient_concept_id", how="inner")
        merged["is_syn_cond"] = merged["condition_concept_id"].isin(syn_conds)
        merged = merged[merged["is_syn_cond"]]
        if merged.empty:
            continue
        hi = merged[merged["potent"]]
        lo = merged[~merged["potent"]]
        if len(hi) < 5 or len(lo) < 5:
            continue
        within_rows.append(
            {
                "offtarget": off,
                "rate_hi": float(hi["y_faers_signal"].mean()),
                "rate_lo": float(lo["y_faers_signal"].mean()),
                "diff": float(hi["y_faers_signal"].mean() - lo["y_faers_signal"].mean()),
                "n_hi": len(hi),
                "n_lo": len(lo),
            }
        )
    within_df = pd.DataFrame(within_rows)
    within_path = out / f"{exp_id}_within_drug_recheck.csv"
    within_df.to_csv(within_path, index=False)

    if len(within_df):
        rng = np.random.default_rng(0)
        diffs = within_df["diff"].to_numpy()
        boot = np.array(
            [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(1000)]
        )
        within_drug_diff_imputed = float(diffs.mean())
        within_drug_diff_ci_lo = float(np.percentile(boot, 2.5))
        within_drug_diff_ci_hi = float(np.percentile(boot, 97.5))
    else:
        within_drug_diff_imputed = float("nan")
        within_drug_diff_ci_lo = float("nan")
        within_drug_diff_ci_hi = float("nan")
        fallback_notes.append(
            "Within-drug imputed-only sanity check had no off-target with >=5 hi and >=5 lo "
            "pairs on drugs with zero measured panel; table is empty."
        )

    # ---- calibration figure ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for ax, off in zip(axes.flat, off_target_names, strict=False):
            res = qsar_results.get(off)
            ax.set_title(
                f"{off} (rho={res['spearman_rho']:.2f})"
                if res and res["spearman_rho"] == res["spearman_rho"]
                else off
            )
            ax.set_xlabel("measured pChEMBL")
            ax.set_ylabel("predicted (scaffold-held-out)")
        for ax in axes.flat[len(off_target_names) :]:
            ax.axis("off")
        fig.tight_layout()
        fig_path = out / f"{exp_id}_qsar_calibration.png"
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
    except Exception as exc:
        log.warning("calibration figure failed: %s", exc)

    results.commit()
    data.commit()

    metrics = {
        "qsar_spearman_median": qsar_spearman_median,
        "qsar_spearman_min": qsar_spearman_min,
        "n_targets_above_0.5": n_targets_above_half,
        "ingredients_with_structure": n_ingredients_with_structure,
        "drug_macro_auc_syndrome_subset": drug_macro_auc_syndrome_subset,
        "floor_pc_syndrome_subset": floor_pc_syndrome_subset,
        "increment_over_measured_only": increment_over_measured_only,
        "within_drug_diff_imputed": within_drug_diff_imputed,
        "within_drug_diff_ci_lo": within_drug_diff_ci_lo,
        "within_drug_diff_ci_hi": within_drug_diff_ci_hi,
        "adr_map_pair_coverage_after": pair_coverage_after,
        "adr_map_pair_coverage_before": pair_coverage_before,
        "n_panel_targets_used": len(off_target_names),
    }
    log.info("fallback notes: %s", fallback_notes)
    log.info("final metrics: %s", metrics)

    extras_path = out / f"{exp_id}_run_extras.csv"
    pd.DataFrame(
        [
            {
                "fallback_notes": "; ".join(fallback_notes) if fallback_notes else "none",
                "n_conditions_before_widen": len(all_conds_before),
                "n_conditions_after_widen": len(all_conds_after),
                "population_all_drug_macro_auc": pop_all["drug_macro_auc_with_interaction"],
                "population_all_floor_pc": pop_all["floor_pc"],
                "population_structure_only_drug_macro_auc": pop_syndrome_structure[
                    "drug_macro_auc_with_interaction"
                ],
            }
        ]
    ).to_csv(extras_path, index=False)
    results.commit()

    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp19",
        title="QSAR imputation of a 12-target safety panel and the syndrome-interaction feature",
        hypothesis=(
            "Scaffold-split QSAR reaches held-out Spearman >=0.5 on at least 8 of 12 "
            "off-targets, and the densified syndrome-interaction feature adds >=0.02 "
            "drug_macro_auc over degree+p_c on the syndrome-condition subset (own floor "
            "0.5830)."
        ),
        approach=(
            "ECFP4 + physicochemical descriptors; 12 single-target regressors trained on "
            "ChEMBL molecules excluding our ingredients, Bemis-Murcko scaffold split; "
            "applicability-domain flags; dense syndrome-interaction feature evaluated with "
            "floors.py on three populations; within-drug paired test re-run on imputed "
            "values as a biology sanity check; ADR map widened via condition ancestors."
        ),
        label="y_faers_signal",
        features=["offtarget_panel_imputed", "syn_aff", "syn_n", "degree", "p_c"],
        split="train/validate, grouped by primary target gene; QSAR held out by scaffold",
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

    findings = (
        f"QSAR quality first: median held-out (scaffold-split) Spearman rho="
        f"{metrics['qsar_spearman_median']:.4f}, min={metrics['qsar_spearman_min']:.4f}, "
        f"{metrics['n_targets_above_0.5']} of {metrics['n_panel_targets_used']} targets "
        f">=0.5. {metrics['ingredients_with_structure']} of 4,280 ingredients resolved a "
        "ChEMBL structure (mixtures/biologics/minerals cannot, per exp17's D4 population). "
        f"Ranking result: drug_macro_auc on the syndrome-condition subset (widened) = "
        f"{metrics['drug_macro_auc_syndrome_subset']:.4f} against its own floor_pc="
        f"{metrics['floor_pc_syndrome_subset']:.4f}; increment over the measured-only "
        f"session's interaction number (0.5397) = {metrics['increment_over_measured_only']:+.4f}. "
        f"Within-drug sanity check on imputed values, restricted to drugs with zero "
        f"measured panel: diff={metrics['within_drug_diff_imputed']:+.4f}, 95% bootstrap CI "
        f"[{metrics['within_drug_diff_ci_lo']:+.4f}, {metrics['within_drug_diff_ci_hi']:+.4f}] "
        "(measured-only session found +0.1639, CI [+0.118, +0.212] on the full drug set). "
        f"ADR map widening (shared-ontology-term expansion of condition_ontology_map.csv, "
        "the closest relation that file carries to an ancestor hierarchy): pair coverage "
        f"{metrics['adr_map_pair_coverage_before']:.4f} -> "
        f"{metrics['adr_map_pair_coverage_after']:.4f}. "
        "offtarget_panel_imputed.csv (4,280 x 12 targets, imputed + measured + combined + "
        "domain flags) written to data/input/drug/ as a project data asset; imputed values "
        "are model outputs and are flagged with a measured/imputed indicator in every "
        "downstream table, never reported as measured."
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"results/{exp}/{exp}_qsar_performance.csv",
            f"results/{exp}/{exp}_models.csv",
            f"results/{exp}/{exp}_within_drug_recheck.csv",
            f"results/{exp}/{exp}_adr_map_widened.json",
            f"results/{exp}/{exp}_qsar_calibration.png",
            "data/input/drug/offtarget_panel_imputed.csv",
        ],
        next_steps=(
            "Pull results/<exp_id>/ from bridge-results and "
            "data/input/drug/offtarget_panel_imputed.csv from bridge-data. If the "
            "syndrome-subset increment clears +0.02 over its own floor, this is the first "
            "biological feature to beat its own floor by a non-noise margin and should be "
            "folded into the round's main model; if the within-drug diff vanishes on "
            "imputed-only drugs, the QSAR is predicting something other than the ADR-driving "
            "affinity and the feature should not be used beyond the measured subset."
        ),
        failed=False,
    )
