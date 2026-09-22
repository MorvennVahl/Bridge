# exp07 — Honest intrinsic ablation, and a leak audit every later run inherits

**Supersedes** `exp_20260922_f6645b` (exp02). Append a `complete` naming it in `supersedes`;
do not edit its entry.

**Status.** Not started. Independent of exp06, exp08, exp09.

---

## Why exp02 has to be re-run

Its degree block was:

```python
out["degree_faers_case_count"] = np.log1p(df["faers_case_count"].fillna(train_case_count_median))
```

`faers_case_count` is a **per-pair** column and the Evans label is
`prr >= 2 & chi2 >= 4 & case_count >= 3`. The baseline therefore contained one of the three
clauses of its own target. Measured on validate:

| design | pooled AP | pooled ROC-AUC |
|---|---|---|
| drug degree + condition degree | 0.1156 | 0.5022 |
| the same + `log1p(faers_case_count)` | **0.3789** | **0.8400** |
| `log1p(faers_case_count)` alone, as a score | 0.2336 | — |

That accounts for exp02's logged `baseline_degree_only_ap = 0.3431` and its train CV AP of
0.9008 against validate 0.4182. All four designs shared the block, so the logged increments
(+0.0669 drug, +0.0468 condition) are uninterpretable — not wrong in magnitude, but measured
on top of a contaminated floor.

## Question

With the label term removed, does drug-intrinsic or condition-intrinsic biology improve a
drug's condition ranking over the `p_c` lookup?

## Hypothesis

On `drug_macro_auc`, the honest degree+`p_c` baseline sits at 0.576 ± 0.01, drug-intrinsic
features add **≥ 0.02**, condition-intrinsic features add less than that (they cannot
reorder a single drug's list except through `p_c`-like effects), and the union stays below
the exp06 ceiling.

**What would falsify it.** Condition-intrinsic features adding a large increment would be a
red flag rather than a result: within one drug's ranking, a condition feature can only help
through condition-level propensity, which `p_c` already carries. A big increment there means
a second `p_c` surrogate entered the design.

---

## Feature blocks

Exactly exp02's blocks (`experiments/exp02_intrinsic_blocks.md` §Feature blocks) with two
changes:

1. **Degree block:** drug degree, condition degree, condition `record_count`. **No
   `faers_case_count`, no `faers_prr`, no `faers_chi_square`.**
2. **Add `p_c`** — the condition's train flag rate — as an explicit baseline column present
   in *all four* designs, so every increment is measured on top of the strongest trivial
   model rather than beside it.

## Method

1. Preconditions as exp02, plus the leak audit below.
2. Four nested designs: `D+p_c`, `+drug`, `+cond`, `+drug+cond`. One fixed
   `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=63,
   early_stopping=False, random_state=0)`. No hyperparameter search.
3. Grouped 3-fold CV inside train on `group_key`, scoring **`drug_macro_auc` on held-out CV
   drugs** — not pooled AP — so selection and reporting use the same metric.
4. One validate pass per design. Report `drug_macro_auc`, `drug_macro_p10`, `drug_macro_r50`,
   `n_drugs_scored`, and pooled AP for continuity with Round 1.
5. Bootstrap CI over drugs (1,000 resamples) for each increment. With across-drug sd 0.128,
   an increment under ~0.01 is noise; say so rather than reporting it as small-but-present.
6. Keep exp02's missingness attribution: refit `+drug` restricted to the 1,716 ingredients
   with an assigned mechanism target. If the increment survives there it is pharmacology; if
   it collapses it was the ChEMBL-missingness indicator.
7. `drug_macro_auc_unseen_family` — the same headline restricted to validate drugs whose
   primary-target family never appears in train. `METRIC.md` requires it within 0.02 of the
   overall number.

## The leak audit — the reusable deliverable

Write `experiments/leak_audit.py` and call it from every later experiment before fitting:

- **Blacklist by construction.** Fail if any feature column name is in
  `{faers_case_count, faers_prr, faers_ror, faers_chi_square, semmeddb_*, y_*, in_faers,
  in_semmeddb, in_eu_label}` or comes from `ingredient_features_label_adjacent.csv`.
- **Per-feature screen.** Compute each single feature's pooled AUC against the label; write
  them all to `<exp_id>_feature_screen.csv` and **fail loudly above 0.75**. No single
  tabular covariate in this dataset should separate the label that well; exp02's leak would
  have tripped this at 0.84 before any model was fitted.
- **Gap assertion.** Fail if train-CV `drug_macro_auc` exceeds validate by more than 0.15 —
  exp02 would have tripped this too (0.90 vs 0.42 in AP terms).

The audit is the point of this experiment as much as the ablation is. Round 1 produced one
void result out of four because nothing checked.

## Budget

| step | est. |
|---|---|
| read, join, one-hot | 90 s |
| leak audit (per-feature AUC over ~170 columns, vectorised) | 30 s |
| 4 designs × (3 CV + 1 full) fits | ~4.5 min |
| per-drug metrics, bootstraps, missingness refit | ~90 s |
| **total** | **~8 min** |

**If at risk:** drop CV to 2 folds for the union design. Never drop the audit or step 7.

## Deliverables

`<exp_id>_ablation.csv`, `<exp_id>_feature_screen.csv`, `<exp_id>_bootstrap_increments.csv`,
`<exp_id>_missingness_attribution.csv`, `<exp_id>_importance_top25.csv`,
`<exp_id>_subgroups.csv` (by `is_mapped`, `arm`, `gene_arm`, `best_match_tier`,
`has_chembl_match`), and `experiments/leak_audit.py`.

## Reporting requirements

`metrics`: `drug_macro_auc`, `drug_macro_auc_unseen_family`, `drug_macro_p10`,
`drug_macro_r50`, `n_drugs_scored`, `floor_drug_macro_auc_pc_lookup`,
`ap_drug_only_increment`, `ap_condition_only_increment`, `pooled_average_precision`,
`validate_average_precision`.

`findings` must open with the corrected baseline and state explicitly that it supersedes
exp02's 0.3431, so the next agent cannot cite the old number. Then the increments with CIs,
the missingness attribution, and the unseen-family number.

## Registration

```python
exp = ln.register(
    agent="exp07",
    title="Honest nested intrinsic ablation on drug_macro_auc, with a leak audit",
    hypothesis=(
        "With the faers_case_count label term removed and p_c carried as a baseline "
        "column, drug-intrinsic features add >=0.02 drug_macro_auc over the 0.576 "
        "floor while condition-intrinsic features add less."
    ),
    approach=(
        "Re-run of exp02's four nested designs on a clean degree block; grouped 3-fold "
        "CV scored on drug_macro_auc; bootstrap CIs over drugs; mechanism-subset "
        "refit; reusable leak_audit.py with blacklist, per-feature AUC screen and a "
        "train-validate gap assertion."
    ),
    label="y_faers_signal",
    features=["degree", "p_c", "drug_intrinsic", "condition_intrinsic"],
    split="train/validate, grouped by primary target gene",
)
```
