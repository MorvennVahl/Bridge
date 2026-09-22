"""floors.py -- reusable p_c-lookup and degree+p_c floors for any (label, subset,
eligibility) combination, per experiments/exp13_subset_floors_and_residual_referee.md.

Pure pandas/numpy/sklearn. Deliberately has NO import of bridge.labnotebook so it can be
imported inside a Modal container (the labnotebook writes must stay on the local machine,
never inside a remote function).

Expected inputs
----------------
`train` and `evaluate` are DataFrames, already subset by the caller to whatever population
is being scored (e.g. all rows, `faers_case_count >= 3`, a case-count stratum, a gene-
neighbour rung). Both must contain at minimum:

  - `ingredient_concept_id` -- the drug side of the pair (OMOP concept id)
  - `condition_concept_id`  -- the condition side of the pair (OMOP concept id)
  - `group_key`             -- present for API symmetry with the rest of the repo; not
                               used inside floors() itself (no fitting is grouped-CV'd here,
                               it is a single train fit scored once on `evaluate`)
  - `<label>`                -- the binary label column named by the `label` argument

`evaluate["record_count"]` / `train["record_count"]` (condition record_count) are used if
present; if absent the degree+p_c model is fit on drug degree, condition degree and p_c
only.

Design choices, stated explicitly
----------------------------------
- Degree (drug degree = pairs per ingredient, condition degree = pairs per condition) and
  p_c (a condition's TRAIN flag rate for `label`) are always computed from `train` only,
  never from `evaluate`, and mapped onto `evaluate`. A condition present in `evaluate` but
  unseen in `train` falls back to the train median (degree) or train mean (p_c) -- the same
  convention exp07/exp09 use.
- `prevalence` is the label's positive rate in **train**, not evaluate: the floor table is
  meant to describe what a population looked like going into a fit, and train is what any
  model in this population would have been fit on. Evaluate's own prevalence can differ
  under a grouped, non-uniform split and is not what floor_pc's p_c lookup was built from.
- `n_drugs_scored` comes from the drug_macro_auc computation on floor_degree_pc (the model
  that also produces floor_pc's evaluation eligibility set is the same eligibility rule
  applied to the same `evaluate` rows, so the two floors share one n_drugs_scored).
- Kept deliberately cheap (a single HistGradientBoostingClassifier fit on <=4 features, no
  internal CV) because floors() is called once per population and this module is expected
  to be invoked many times across many subsets within a single ~600s Modal function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def _drug_macro_auc(
    ids: pd.Series,
    y: np.ndarray,
    score: np.ndarray,
    min_pairs: int,
) -> tuple[float, int]:
    """Macro-average ROC-AUC of score vs binary y within each drug with
    >= min_pairs rows and both classes present (METRIC.md's eligibility rule).

    Known recurring bugs handled here: bool-dtype score cast to float before ranking,
    NaN scores filled with the per-drug median before roc_auc_score.
    """
    df = pd.DataFrame(
        {
            "drug": np.asarray(ids),
            "y": np.asarray(y),
            "score": np.asarray(score, dtype=float),
        }
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
        auc = roc_auc_score(y_grp, s.to_numpy())
        aucs.append(float(auc))
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def floors(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    label: str,
    eligibility: int,
) -> dict[str, float]:
    """p_c-lookup and degree+p_c floors, prevalence, and n_drugs_scored for one
    (label, subset, eligibility) combination.

    Parameters
    ----------
    train : DataFrame subset the caller wants degree/p_c fit on. Must contain
        `ingredient_concept_id`, `condition_concept_id`, `group_key`, `<label>`.
    evaluate : DataFrame subset the caller wants scored (may be validate, or a stratum of
        it). Same required columns as `train`.
    label : name of the binary label column shared by both frames.
    eligibility : minimum pairs per drug for drug_macro_auc (METRIC.md's eligibility rule
        is >=N pairs and both classes present; this experiment reports both N=5 and N=20
        at the call site -- floors() itself just takes whatever N the caller passes).

    Returns
    -------
    dict with keys: floor_pc, floor_degree_pc, prevalence, n_drugs_scored.
    """
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

    # ---- p_c: condition's TRAIN flag rate for `label` -----------------------------
    p_c_train = train.groupby("condition_concept_id")[label].mean()
    p_c_mean = float(p_c_train.mean())
    eval_p_c = evaluate["condition_concept_id"].map(p_c_train).astype(float)
    eval_p_c = eval_p_c.fillna(p_c_mean)

    floor_pc, n_drugs_pc = _drug_macro_auc(
        evaluate["ingredient_concept_id"], y_eval, eval_p_c.to_numpy(), eligibility
    )

    # ---- degree + p_c: light fitted model ------------------------------------------
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
    x_train = _build_features(train, train_p_c)
    x_eval = _build_features(evaluate, eval_p_c)

    # bool/NaN safety: cast to float32, fill any remaining NaN with the train column median.
    x_train = x_train.astype("float32")
    x_eval = x_eval.astype("float32")
    for col in x_train.columns:
        col_median = float(x_train[col].median())
        x_train[col] = x_train[col].fillna(col_median)
        x_eval[col] = x_eval[col].fillna(col_median)

    if len(np.unique(y_train)) < 2:
        floor_degree_pc = float("nan")
        n_drugs_degree_pc = 0
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

        floor_degree_pc, n_drugs_degree_pc = _drug_macro_auc(
            evaluate["ingredient_concept_id"], y_eval, eval_proba, eligibility
        )

    prevalence = float(np.mean(y_train)) if len(y_train) else float("nan")

    # n_drugs_scored: report the degree+p_c eligibility count (identical eligibility rule
    # applied to the same evaluate rows as floor_pc; the two only differ if one score is
    # entirely NaN for a drug, which does not happen here since both are always defined).
    n_drugs_scored = n_drugs_degree_pc if n_drugs_degree_pc else n_drugs_pc

    return {
        "floor_pc": floor_pc,
        "floor_degree_pc": floor_degree_pc,
        "prevalence": prevalence,
        "n_drugs_scored": n_drugs_scored,
    }
