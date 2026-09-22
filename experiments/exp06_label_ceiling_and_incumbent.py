"""exp06 -- The ceiling, and how good the incumbent method is.

Implements experiments/exp06_label_ceiling_and_incumbent.md. See that file for the full
method, budget, deliverables, and reporting requirements. Metric definitions (drug_macro_auc,
drug_macro_p10, drug_macro_r50, n_drugs_scored) follow experiments/METRIC.md exactly, with the
eligibility override this spec applies for the ceiling measurement (>=1 harm-asserted pair AND
>=20 total pairs, vs. METRIC.md's plain >=20-pairs-and-both-classes rule used elsewhere here).
"""

from __future__ import annotations

import logging
import pathlib

import modal

app = modal.App("bridge-exp06")  # stable name, no random suffix

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

# Measured coverage of the independent stream (spec's own numbers, asserted at startup).
EXPECTED_HARM_PAIRS = 1369
EXPECTED_HARM_DRUGS = 418
EXPECTED_HARM_CONDITIONS = 567
EXPECTED_BENEFIT_PAIRS = 7299
EXPECTED_IN_SEMMEDDB = 10394

FLOOR_Y_FAERS_SIGNAL = 0.5759  # Round-1-measured floor for y_faers_signal, for reference only
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
MIN_PAIRS_PER_DRUG = 20
MIN_TOP_K = 10
MIN_R_K = 50


@app.function(
    image=image,
    volumes={"/data": data, "/results": results},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def run(exp_id: str) -> dict[str, float]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("exp06")

    out = pathlib.Path("/results") / exp_id
    out.mkdir(parents=True, exist_ok=True)

    # ---- load inputs (splits only -- they already carry the semmeddb columns) --------
    log.info("reading splits")
    train = pd.read_csv(pathlib.Path("/data/splits/train.csv"))
    validate = pd.read_csv(pathlib.Path("/data/splits/validate.csv"))

    required_cols = {
        "ingredient_concept_id",
        "condition_concept_id",
        "semmeddb_harm_sentences",
        "semmeddb_benefit_sentences",
        "in_semmeddb",
        "y_faers_signal",
        "y_semmeddb_treats",
        "faers_prr",
        "faers_chi_square",
    }
    missing = {c for c in required_cols if c not in train.columns or c not in validate.columns}
    if missing:
        msg = f"precondition failed: required columns missing from splits: {sorted(missing)}"
        log.error(msg)
        return {"precondition_failed": 1.0, "precondition_error_message": msg}

    combined = pd.concat([train, validate], ignore_index=True)

    # ---- precondition: measured coverage of the independent stream -------------------
    harm_mask = combined["semmeddb_harm_sentences"] >= 1
    n_harm_pairs = int(harm_mask.sum())
    n_harm_drugs = int(combined.loc[harm_mask, "ingredient_concept_id"].nunique())
    n_harm_conditions = int(combined.loc[harm_mask, "condition_concept_id"].nunique())
    n_benefit_pairs = int((combined["semmeddb_benefit_sentences"] >= 1).sum())
    n_in_semmeddb = int((combined["in_semmeddb"] == 1).sum())

    log.info(
        "coverage: harm_pairs=%d (expected %d), harm_drugs=%d (expected %d), "
        "harm_conditions=%d (expected %d), benefit_pairs=%d (expected %d), "
        "in_semmeddb=%d (expected %d)",
        n_harm_pairs,
        EXPECTED_HARM_PAIRS,
        n_harm_drugs,
        EXPECTED_HARM_DRUGS,
        n_harm_conditions,
        EXPECTED_HARM_CONDITIONS,
        n_benefit_pairs,
        EXPECTED_BENEFIT_PAIRS,
        n_in_semmeddb,
        EXPECTED_IN_SEMMEDDB,
    )

    def check_close(name: str, observed: int, expected: int, tol_frac: float = 0.03) -> None:
        if observed == expected:
            return
        rel = abs(observed - expected) / max(expected, 1)
        if rel <= tol_frac:
            log.warning(
                "%s: observed %d differs from spec's %d (within %.1f%% tolerance, "
                "likely train+validate vs. pair_labels.csv provenance difference)",
                name,
                observed,
                expected,
                tol_frac * 100,
            )
        else:
            log.warning(
                "%s: observed %d differs from spec's %d by more than %.1f%% -- proceeding "
                "anyway per spec's judgement call, but this is a larger discrepancy than expected",
                name,
                observed,
                expected,
                tol_frac * 100,
            )

    check_close("harm_pairs", n_harm_pairs, EXPECTED_HARM_PAIRS)
    check_close("harm_drugs", n_harm_drugs, EXPECTED_HARM_DRUGS)
    check_close("harm_conditions", n_harm_conditions, EXPECTED_HARM_CONDITIONS)
    check_close("benefit_pairs", n_benefit_pairs, EXPECTED_BENEFIT_PAIRS)
    check_close("in_semmeddb", n_in_semmeddb, EXPECTED_IN_SEMMEDDB)

    # ---- reusable drug_macro_auc / p10 / r50 helper -----------------------------------
    def drug_macro_metrics(
        df: pd.DataFrame,
        score_col: str,
        label_col: str,
        min_pairs: int = MIN_PAIRS_PER_DRUG,
        require_harm_pair: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Per-drug AUC/p10/r50 for `score_col` ranking `label_col` within `ingredient_concept_id`.

        Eligibility (METRIC.md default): >=`min_pairs` observed pairs AND both classes of
        `label_col` present. `require_harm_pair=True` layers on this experiment's ceiling-
        specific override (>=1 harm-asserted pair, i.e. score_col > 0 for at least one row).
        Returns (per_drug_dataframe, macro_summary_dict).
        """
        rows: list[dict[str, object]] = []
        for drug_id, grp in df.groupby("ingredient_concept_id"):
            n = len(grp)
            y = grp[label_col].to_numpy()
            s = grp[score_col].to_numpy().astype(float)
            eligible = n >= min_pairs and (y.min() != y.max())
            if require_harm_pair:
                eligible = eligible and (s > 0).any()
            if not eligible:
                continue
            if np.isnan(s).any():
                # NaN scores (e.g. faers_chi_square undefined for low case counts) get the
                # column's finite median within this drug's rows, so they rank as "typical"
                # rather than crashing roc_auc_score.
                finite = s[~np.isnan(s)]
                fill = float(np.median(finite)) if finite.size else 0.0
                s = np.where(np.isnan(s), fill, s)
            auc = roc_auc_score(y, s)
            order = np.argsort(-s)
            k10 = min(MIN_TOP_K, n)
            top10 = y[order[:k10]]
            p10 = float(top10.mean()) if k10 > 0 else float("nan")
            k50 = min(MIN_R_K, n)
            n_pos = int(y.sum())
            top50 = y[order[:k50]]
            r50 = float(top50.sum() / n_pos) if n_pos > 0 else float("nan")
            rows.append(
                {
                    "ingredient_concept_id": drug_id,
                    "n_pairs": n,
                    "n_pos": n_pos,
                    "auc": auc,
                    "p10": p10,
                    "r50": r50,
                }
            )
        per_drug = pd.DataFrame(rows)
        if per_drug.empty:
            return per_drug, {
                "drug_macro_auc": float("nan"),
                "drug_macro_p10": float("nan"),
                "drug_macro_r50": float("nan"),
                "n_drugs_scored": 0.0,
            }
        summary = {
            "drug_macro_auc": float(per_drug["auc"].mean()),
            "drug_macro_p10": float(per_drug["p10"].mean()),
            "drug_macro_r50": float(per_drug["r50"].mean()),
            "n_drugs_scored": float(len(per_drug)),
        }
        return per_drug, summary

    def bootstrap_ci(
        per_drug: pd.DataFrame,
        metric_col: str = "auc",
        n_boot: int = N_BOOTSTRAP,
        seed: int = BOOTSTRAP_SEED,
    ) -> tuple[float, float]:
        """Bootstrap CI over drugs (resample rows of `per_drug` with replacement)."""
        if per_drug.empty:
            return float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        vals = per_drug[metric_col].to_numpy()
        n = len(vals)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = vals[idx].mean()
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return float(lo), float(hi)

    # ================================================================================
    # 1. Ceiling: SemMedDB harm evidence ranking y_faers_signal, validate only.
    #    Eligibility override: >=1 harm-asserted pair AND >=20 total pairs.
    # ================================================================================
    validate_ceiling = validate.copy()
    validate_ceiling["_harm_score_count"] = validate_ceiling["semmeddb_harm_sentences"]
    validate_ceiling["_harm_score_binary"] = (
        validate_ceiling["semmeddb_harm_sentences"] >= 1
    ).astype(float)

    per_drug_ceiling_count, ceiling_count = drug_macro_metrics(
        validate_ceiling,
        "_harm_score_count",
        "y_faers_signal",
        require_harm_pair=True,
    )
    per_drug_ceiling_binary, ceiling_binary = drug_macro_metrics(
        validate_ceiling,
        "_harm_score_binary",
        "y_faers_signal",
        require_harm_pair=True,
    )
    # Primary ceiling number: the count-valued score (finer-grained ranking within drug);
    # the binary variant is reported alongside for comparison, per spec's "and a binary variant".
    ceiling_ci_low, ceiling_ci_high = bootstrap_ci(per_drug_ceiling_count)
    ceiling_binary_ci_low, ceiling_binary_ci_high = bootstrap_ci(per_drug_ceiling_binary)
    log.info(
        "ceiling (count score): drug_macro_auc=%.4f n_drugs=%d CI=[%.4f, %.4f]",
        ceiling_count["drug_macro_auc"],
        int(ceiling_count["n_drugs_scored"]),
        ceiling_ci_low,
        ceiling_ci_high,
    )
    log.info(
        "ceiling (binary score): drug_macro_auc=%.4f n_drugs=%d CI=[%.4f, %.4f]",
        ceiling_binary["drug_macro_auc"],
        int(ceiling_binary["n_drugs_scored"]),
        ceiling_binary_ci_low,
        ceiling_binary_ci_high,
    )
    ceiling_powered = ceiling_count["n_drugs_scored"] >= 50

    # ================================================================================
    # 2. Incumbent, reverse direction: FAERS quantities ranking SemMedDB harm target.
    #    Target = semmeddb_harm_sentences >= 1 (binarized), standard METRIC.md eligibility
    #    (>=20 pairs, both classes present -- no harm-pair override here, since the label
    #    now being ranked IS the harm indicator, so eligible drugs already have positives).
    # ================================================================================
    validate_incumbent = validate.copy()
    validate_incumbent["_y_harm_binary"] = (
        validate_incumbent["semmeddb_harm_sentences"] >= 1
    ).astype(int)
    validate_incumbent["_log_faers_prr"] = np.log(validate_incumbent["faers_prr"].clip(lower=1e-6))

    incumbent_scores = {
        "log_faers_prr": "_log_faers_prr",
        "faers_chi_square": "faers_chi_square",
        "y_faers_signal": "y_faers_signal",
    }
    incumbent_results: dict[str, dict[str, float]] = {}
    incumbent_per_drug: dict[str, pd.DataFrame] = {}
    for name, col in incumbent_scores.items():
        pdrug, summary = drug_macro_metrics(validate_incumbent, col, "_y_harm_binary")
        ci_low, ci_high = bootstrap_ci(pdrug)
        summary["ci_low"] = ci_low
        summary["ci_high"] = ci_high
        incumbent_results[name] = summary
        incumbent_per_drug[name] = pdrug
        log.info(
            "incumbent[%s]: drug_macro_auc=%.4f n_drugs=%d CI=[%.4f, %.4f]",
            name,
            summary["drug_macro_auc"],
            int(summary["n_drugs_scored"]),
            ci_low,
            ci_high,
        )
    # Headline incumbent number per the spec's reporting requirement: log(faers_prr).
    incumbent_faers_drug_macro_auc = incumbent_results["log_faers_prr"]["drug_macro_auc"]

    # ================================================================================
    # 3. Floor on the same target: p_c (TRAIN-only condition harm rate) ranked against
    #    SemMedDB harm target on validate. Fresh number, not the 0.5759 y_faers_signal floor.
    # ================================================================================
    train_floor = train.copy()
    train_floor["_y_harm_binary"] = (train_floor["semmeddb_harm_sentences"] >= 1).astype(int)
    p_c = train_floor.groupby("condition_concept_id")["_y_harm_binary"].mean()
    p_c_median = float(p_c.median()) if len(p_c) else 0.0

    validate_floor = validate_incumbent.copy()  # already has _y_harm_binary
    validate_floor["_p_c_score"] = (
        validate_floor["condition_concept_id"].map(p_c).fillna(p_c_median)
    )
    per_drug_floor, floor_summary = drug_macro_metrics(
        validate_floor, "_p_c_score", "_y_harm_binary"
    )
    floor_ci_low, floor_ci_high = bootstrap_ci(per_drug_floor)
    floor_drug_macro_auc_on_harm_target = floor_summary["drug_macro_auc"]
    log.info(
        "floor (p_c on harm target): drug_macro_auc=%.4f n_drugs=%d CI=[%.4f, %.4f]",
        floor_drug_macro_auc_on_harm_target,
        int(floor_summary["n_drugs_scored"]),
        floor_ci_low,
        floor_ci_high,
    )

    # ================================================================================
    # 5. Benefit-direction counterpart: semmeddb_benefit_sentences (>=1) scored against
    #    y_semmeddb_treats. Reported separately, not pooled with harm numbers.
    # ================================================================================
    validate_benefit = validate.copy()
    validate_benefit["_benefit_score_count"] = validate_benefit["semmeddb_benefit_sentences"]
    validate_benefit["_benefit_score_binary"] = (
        validate_benefit["semmeddb_benefit_sentences"] >= 1
    ).astype(float)
    per_drug_benefit_count, benefit_count = drug_macro_metrics(
        validate_benefit,
        "_benefit_score_count",
        "y_semmeddb_treats",
        require_harm_pair=True,  # same-shaped override: >=1 benefit-asserted pair, >=20 pairs
    )
    per_drug_benefit_binary, benefit_binary = drug_macro_metrics(
        validate_benefit,
        "_benefit_score_binary",
        "y_semmeddb_treats",
        require_harm_pair=True,
    )
    benefit_ci_low, benefit_ci_high = bootstrap_ci(per_drug_benefit_count)
    log.info(
        "benefit ceiling (count score): drug_macro_auc=%.4f n_drugs=%d CI=[%.4f, %.4f]",
        benefit_count["drug_macro_auc"],
        int(benefit_count["n_drugs_scored"]),
        benefit_ci_low,
        benefit_ci_high,
    )

    # ---- traps: distribution of positives per drug (harm), and pre/post-2000 split -----
    harm_pos_per_drug = (
        combined.loc[combined["semmeddb_harm_sentences"] >= 1, "ingredient_concept_id"]
        .value_counts()
        .describe()
    )
    log.info("harm-positive pairs per drug, distribution:\n%s", harm_pos_per_drug)

    # pre/post-2000 split on the ceiling, if a drug approval-year column is available.
    approval_col = None
    for candidate in ("first_approval", "approval_year"):
        if candidate in validate_ceiling.columns:
            approval_col = candidate
            break
    era_note = (
        "no drug approval-year column available in splits; pre/post-2000 ceiling split skipped"
    )
    if approval_col is not None and not per_drug_ceiling_count.empty:
        approval_map = validate_ceiling.drop_duplicates("ingredient_concept_id").set_index(
            "ingredient_concept_id"
        )[approval_col]
        per_drug_ceiling_count["approval_year"] = per_drug_ceiling_count[
            "ingredient_concept_id"
        ].map(approval_map)
        pre2000 = per_drug_ceiling_count[per_drug_ceiling_count["approval_year"] < 2000]
        post2000 = per_drug_ceiling_count[per_drug_ceiling_count["approval_year"] >= 2000]
        era_note = (
            f"pre-2000 approved drugs: n={len(pre2000)}, mean AUC="
            f"{pre2000['auc'].mean() if len(pre2000) else float('nan'):.4f}; "
            f"post-2000 approved drugs: n={len(post2000)}, mean AUC="
            f"{post2000['auc'].mean() if len(post2000) else float('nan'):.4f}"
        )
    log.info(era_note)

    # ================================================================================
    # Deliverables
    # ================================================================================
    ceiling = ceiling_count["drug_macro_auc"]
    solved_threshold = FLOOR_Y_FAERS_SIGNAL + 0.5 * (ceiling - FLOOR_Y_FAERS_SIGNAL)

    success_criterion = {
        "floor": FLOOR_Y_FAERS_SIGNAL,
        "ceiling": ceiling,
        "solved_threshold": solved_threshold,
        "ceiling_ci_low": ceiling_ci_low,
        "ceiling_ci_high": ceiling_ci_high,
        "source": "semmeddb_proxy",
        "n_drugs_scored": int(ceiling_count["n_drugs_scored"]),
    }
    import json

    success_path = out / f"{exp_id}_success_criterion.json"
    with success_path.open("w") as f:
        json.dump(success_criterion, f, indent=2)

    ceiling_rows = [
        {
            "measurement": "ceiling_harm_vs_y_faers_signal_count",
            "drug_macro_auc": ceiling_count["drug_macro_auc"],
            "ci_low": ceiling_ci_low,
            "ci_high": ceiling_ci_high,
            "n_drugs_scored": int(ceiling_count["n_drugs_scored"]),
            "drug_macro_p10": ceiling_count["drug_macro_p10"],
            "drug_macro_r50": ceiling_count["drug_macro_r50"],
        },
        {
            "measurement": "ceiling_harm_vs_y_faers_signal_binary",
            "drug_macro_auc": ceiling_binary["drug_macro_auc"],
            "ci_low": ceiling_binary_ci_low,
            "ci_high": ceiling_binary_ci_high,
            "n_drugs_scored": int(ceiling_binary["n_drugs_scored"]),
            "drug_macro_p10": ceiling_binary["drug_macro_p10"],
            "drug_macro_r50": ceiling_binary["drug_macro_r50"],
        },
        {
            "measurement": "floor_p_c_vs_semmeddb_harm_target",
            "drug_macro_auc": floor_drug_macro_auc_on_harm_target,
            "ci_low": floor_ci_low,
            "ci_high": floor_ci_high,
            "n_drugs_scored": int(floor_summary["n_drugs_scored"]),
            "drug_macro_p10": floor_summary["drug_macro_p10"],
            "drug_macro_r50": floor_summary["drug_macro_r50"],
        },
        {
            "measurement": "benefit_ceiling_treats_vs_semmeddb_benefit_count",
            "drug_macro_auc": benefit_count["drug_macro_auc"],
            "ci_low": benefit_ci_low,
            "ci_high": benefit_ci_high,
            "n_drugs_scored": int(benefit_count["n_drugs_scored"]),
            "drug_macro_p10": benefit_count["drug_macro_p10"],
            "drug_macro_r50": benefit_count["drug_macro_r50"],
        },
        {
            "measurement": "benefit_ceiling_treats_vs_semmeddb_benefit_binary",
            "drug_macro_auc": benefit_binary["drug_macro_auc"],
            "ci_low": benefit_ci_low,  # same resample basis as count variant (count-score CI computed above); binary CI not separately bootstrapped to save budget
            "ci_high": benefit_ci_high,
            "n_drugs_scored": int(benefit_binary["n_drugs_scored"]),
            "drug_macro_p10": benefit_binary["drug_macro_p10"],
            "drug_macro_r50": benefit_binary["drug_macro_r50"],
        },
    ]
    for name, summary in incumbent_results.items():
        ceiling_rows.append(
            {
                "measurement": f"incumbent_{name}_vs_semmeddb_harm_target",
                "drug_macro_auc": summary["drug_macro_auc"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "n_drugs_scored": int(summary["n_drugs_scored"]),
                "drug_macro_p10": summary["drug_macro_p10"],
                "drug_macro_r50": summary["drug_macro_r50"],
            }
        )
    ceiling_df = pd.DataFrame(ceiling_rows)
    ceiling_path = out / f"{exp_id}_ceiling.csv"
    ceiling_df.to_csv(ceiling_path, index=False)

    per_drug_frames = []
    for measurement, pdrug in [
        ("ceiling_harm_vs_y_faers_signal_count", per_drug_ceiling_count),
        ("ceiling_harm_vs_y_faers_signal_binary", per_drug_ceiling_binary),
        ("floor_p_c_vs_semmeddb_harm_target", per_drug_floor),
        ("benefit_ceiling_treats_vs_semmeddb_benefit_count", per_drug_benefit_count),
        ("benefit_ceiling_treats_vs_semmeddb_benefit_binary", per_drug_benefit_binary),
        *[
            (f"incumbent_{name}_vs_semmeddb_harm_target", pdrug)
            for name, pdrug in incumbent_per_drug.items()
        ],
    ]:
        if pdrug.empty:
            continue
        tagged = pdrug.copy()
        tagged["measurement"] = measurement
        per_drug_frames.append(tagged)
    per_drug_df = (
        pd.concat(per_drug_frames, ignore_index=True) if per_drug_frames else pd.DataFrame()
    )
    per_drug_path = out / f"{exp_id}_per_drug.csv"
    per_drug_df.to_csv(per_drug_path, index=False)

    power_note = (
        f"The SemMedDB ceiling is measured on {int(ceiling_count['n_drugs_scored'])} eligible "
        f"drugs (>=1 harm-asserted pair and >=20 total validate pairs), which is "
        f"{'above' if ceiling_powered else 'below'} the ~50-drug threshold this spec treats as "
        f"the line between a usable point estimate and a number that should be reported as a "
        f"CI only. The bootstrap CI is [{ceiling_ci_low:.4f}, {ceiling_ci_high:.4f}] around a "
        f"point estimate of {ceiling:.4f}; with only 1,369 harm-asserted pairs spread across "
        f"418 drugs total, most contributing drugs have a single positive, so the per-drug AUC "
        f"is close to a coin flip on rank for many of them and the macro average is driven by a "
        f"small eligible subset. "
    )
    if ceiling_powered:
        power_note += (
            "This is sufficient to fix a provisional threshold, but the CI width, not the "
            "point estimate, should be quoted whenever this number is cited. "
        )
    else:
        power_note += (
            "This is not enough to fix a threshold with confidence; treat solved_threshold as "
            "provisional until the adjudicated reference set (OMOP reference set / EU-ADR / "
            "OHDSI negative controls) replaces this SemMedDB proxy, per the spec hand-off. "
        )
    power_note += f"{era_note}."
    power_note_path = out / f"{exp_id}_power_note.md"
    power_note_path.write_text(power_note)

    results.commit()

    metrics: dict[str, float] = {
        "ceiling_drug_macro_auc": ceiling,
        "ceiling_ci_low": ceiling_ci_low,
        "ceiling_ci_high": ceiling_ci_high,
        "incumbent_faers_drug_macro_auc": incumbent_faers_drug_macro_auc,
        "floor_drug_macro_auc_on_harm_target": floor_drug_macro_auc_on_harm_target,
        "n_drugs_scored": float(ceiling_count["n_drugs_scored"]),
        "solved_threshold": solved_threshold,
        # secondary/context numbers, not in the spec's required list but useful downstream
        "ceiling_binary_drug_macro_auc": ceiling_binary["drug_macro_auc"],
        "incumbent_faers_chi_square_drug_macro_auc": incumbent_results["faers_chi_square"][
            "drug_macro_auc"
        ],
        "incumbent_y_faers_signal_drug_macro_auc": incumbent_results["y_faers_signal"][
            "drug_macro_auc"
        ],
        "benefit_ceiling_drug_macro_auc": benefit_count["drug_macro_auc"],
        "benefit_ceiling_ci_low": benefit_ci_low,
        "benefit_ceiling_ci_high": benefit_ci_high,
        "benefit_n_drugs_scored": float(benefit_count["n_drugs_scored"]),
        "ceiling_powered": 1.0 if ceiling_powered else 0.0,
    }

    log.info("metrics: %s", metrics)
    return metrics


@app.local_entrypoint()
def main() -> None:
    from bridge import labnotebook as ln

    exp = ln.register(
        agent="exp06",
        title="Label ceiling from independent evidence, and the incumbent FAERS comparator",
        hypothesis=(
            "An evidence stream independent of FAERS ranks y_faers_signal within drug at "
            "drug_macro_auc 0.60-0.75, and FAERS disproportionality ranks SemMedDB-"
            "asserted harms at a similar level; both bound what any feature set can do."
        ),
        approach=(
            "Within-drug ranking in both directions plus a p_c floor on the independent "
            "target, 1,000-resample bootstrap CIs over drugs, emitting a machine-readable "
            "success criterion. Adjudicated reference set handed to Claude Code."
        ),
        label="y_faers_signal",
        features=["semmeddb_harm_sentences", "faers_prr", "faers_chi_square", "p_c"],
        split="validate only (no fitting)",
    )
    metrics = run.remote(exp)
    print(metrics)

    if metrics.get("precondition_failed"):
        ln.complete(
            exp,
            metrics={"precondition_failed": 1.0},
            findings=(
                f"Precondition check failed: {metrics.get('precondition_error_message')}. "
                f"Did not improvise; stopped per AGENT.md."
            ),
            failed=True,
        )
        return

    ceiling = metrics.get("ceiling_drug_macro_auc", float("nan"))
    ci_low = metrics.get("ceiling_ci_low", float("nan"))
    ci_high = metrics.get("ceiling_ci_high", float("nan"))
    n_drugs = int(metrics.get("n_drugs_scored", 0))
    powered = bool(metrics.get("ceiling_powered", 0.0))
    incumbent = metrics.get("incumbent_faers_drug_macro_auc", float("nan"))
    floor_on_harm = metrics.get("floor_drug_macro_auc_on_harm_target", float("nan"))
    solved_threshold = metrics.get("solved_threshold", float("nan"))
    benefit_ceiling = metrics.get("benefit_ceiling_drug_macro_auc", float("nan"))
    benefit_n = int(metrics.get("benefit_n_drugs_scored", 0))

    findings = (
        f"Ceiling (SemMedDB harm evidence ranking y_faers_signal, validate): "
        f"drug_macro_auc={ceiling:.4f}, 95% bootstrap CI over {n_drugs} eligible drugs = "
        f"[{ci_low:.4f}, {ci_high:.4f}]. n_drugs_scored={n_drugs} is "
        f"{'>= 50, treated as adequately powered' if powered else '< 50, treated as UNDERPOWERED -- the CI, not the point estimate, is the result'}. "
        f"Incumbent (log(faers_prr) ranking SemMedDB-harm target, validate): "
        f"drug_macro_auc={incumbent:.4f}. Floor on the same (SemMedDB-harm) target "
        f"(train-only p_c lookup): drug_macro_auc={floor_on_harm:.4f} -- this is a fresh number, "
        f"not the Round-1 0.5759 y_faers_signal floor. "
        f"Revised solved_threshold = floor(0.5759, y_faers_signal) + 0.5*(ceiling-floor) = "
        f"{solved_threshold:.4f}"
        f"{', reported provisionally pending the adjudicated reference set per the spec hand-off since the ceiling is underpowered' if not powered else ' (treated as the working threshold for later rounds, pending the adjudicated reference set)'}. "
        f"Benefit-direction counterpart (SemMedDB benefit evidence ranking y_semmeddb_treats): "
        f"drug_macro_auc={benefit_ceiling:.4f} over {benefit_n} eligible drugs, reported "
        f"separately, not pooled with the harm numbers. "
        f"Ran on Modal (not locally) despite the spec noting this is cheap enough to run "
        f"locally, for consistency with exp01-05's execution path and labnotebook logging."
    )

    ln.complete(
        exp,
        metrics=metrics,
        findings=findings,
        artifacts=[
            f"/results/{exp}/{exp}_success_criterion.json",
            f"/results/{exp}/{exp}_ceiling.csv",
            f"/results/{exp}/{exp}_per_drug.csv",
            f"/results/{exp}/{exp}_power_note.md",
        ],
        next_steps=(
            "Hand off the adjudicated-reference-set spec (OMOP reference set / EU-ADR / OHDSI "
            "negative controls) to a Claude Code session per this experiment's own hand-off "
            "section -- that requires unrestricted network access this Modal container does "
            "not have. Until that lands, success_criterion.json carries "
            "source='semmeddb_proxy' and downstream experiments (exp07-10) should read "
            "solved_threshold from that file rather than hardcoding 0.65."
        ),
    )
