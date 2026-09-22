"""Reusable leak-audit module for Bridge experiments.

Plain functions, importable both locally and from inside a Modal container. Not a Modal
app. Written for exp07 (experiments/exp07_ablation_rerun_leak_audit.md) and intended to be
imported by every later experiment (exp08, exp09, exp10) before fitting any model.

exp02's degree block included `log1p(faers_case_count)` -- a per-pair column that is one of
the three clauses of the `y_faers_signal` label (`prr>=2 & chi2>=4 & case_count>=3`). This
module exists to catch that class of mistake automatically rather than relying on a human
re-reading the feature list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

BLACKLIST_EXACT: set[str] = {
    "faers_case_count",
    "faers_prr",
    "faers_ror",
    "faers_chi_square",
    "in_faers",
    "in_semmeddb",
    "in_eu_label",
}
BLACKLIST_PREFIXES: tuple[str, ...] = ("semmeddb_", "y_")


def audit_columns(columns: list[str]) -> list[str]:
    """Return the blacklisted column names found in `columns` (empty list = clean).

    Checks BLACKLIST_EXACT (exact name match) and BLACKLIST_PREFIXES (startswith match).
    Does not check `ingredient_features_label_adjacent.csv` columns, since that file's
    columns are never loaded into a dataframe here -- the caller must separately ensure
    that file is never read at all.
    """
    hits: list[str] = []
    for col in columns:
        if col in BLACKLIST_EXACT or any(col.startswith(p) for p in BLACKLIST_PREFIXES):
            hits.append(col)
    return hits


def per_feature_auc_screen(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Single-feature pooled AUC for every numeric column in `x` against binary `y`.

    Diagnostic only, not modeling: each column is filled with its own median before
    scoring (a simple approach adequate for a screen). Non-numeric / all-NaN / constant
    columns are skipped (cannot be scored) rather than raising.

    Returns a DataFrame with columns ['feature', 'pooled_auc'], sorted descending by
    pooled_auc.

    Design choice: this function does NOT raise on a high score. It returns the full
    dataframe and leaves the "fail loudly above 0.75" decision to the caller, because the
    caller (exp07's `run()`) needs to write the screen to disk and return a structured
    `precondition_failed` dict rather than have a bare exception propagate out of a Modal
    function and be reported as an infrastructure failure. `check_train_validate_gap`
    below follows the opposite, raising, convention instead -- documented there -- because
    that check runs after all deliverables are already written, so raising is safe and
    simpler for the caller.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    y_arr = y.to_numpy()
    if len(np.unique(y_arr)) < 2:
        return pd.DataFrame(columns=["feature", "pooled_auc"])

    rows: list[dict[str, float | str]] = []
    for col in x.columns:
        series = x[col]
        if series.dtype == object or series.dtype == bool:
            try:
                series = series.astype(float)
            except (TypeError, ValueError):
                continue
        series = series.astype(float)
        if series.notna().sum() == 0:
            continue
        median = series.median()
        filled = series.fillna(median)
        if filled.nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y_arr, filled.to_numpy())
        except ValueError:
            continue
        # orient so pooled_auc reports discriminative power regardless of sign
        auc = max(auc, 1.0 - auc)
        rows.append({"feature": col, "pooled_auc": float(auc)})

    screen = pd.DataFrame(rows, columns=["feature", "pooled_auc"])
    return screen.sort_values("pooled_auc", ascending=False).reset_index(drop=True)


def check_train_validate_gap(
    train_cv_metric: float, validate_metric: float, max_gap: float = 0.15
) -> None:
    """Raise AssertionError if train_cv_metric exceeds validate_metric by more than max_gap.

    Scale is assumed higher-is-better (e.g. drug_macro_auc, AP). A large positive gap
    (train much better than validate) indicates overfitting or a leaked feature that does
    not generalize the same way to validate's disjoint groups.
    """
    gap = train_cv_metric - validate_metric
    if gap > max_gap:
        raise AssertionError(
            f"train-validate gap {gap:.4f} exceeds max_gap={max_gap:.4f} "
            f"(train_cv={train_cv_metric:.4f}, validate={validate_metric:.4f})"
        )
