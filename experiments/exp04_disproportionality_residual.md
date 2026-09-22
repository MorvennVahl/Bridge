# exp04 — Is the degree-free residual a better target than the raw flag?

**Lineage.** `lab/round_01.md` lane L6, unchanged in intent, with the estimator fixed
(two-way fixed effects by alternating demeaning) and the subset size measured.

**Status.** Not started. Independent of the other four.

---

## Question

The raw FAERS signal is dominated by how often a drug and a condition get reported. If we
remove drug and condition reporting propensity from the disproportionality itself and model
the **residual**, do the same features predict it better than they predict the raw binary
flag?

## Hypothesis

The same drug- and condition-intrinsic feature set predicts the two-way-demeaned
`log(faers_prr)` residual with **Spearman ρ ≥ 0.10** on validate, and that is a larger
effect — relative to what each target's own null allows — than the AP lift the same features
achieve on `y_faers_signal` in exp02.

**What would falsify it.** If the residual is essentially unpredictable (ρ near zero) while
the binary flag is predictable, then what the features were predicting in exp02 was the
reporting structure, not the pharmacology — a negative result that would reshape the whole
project, since it would mean the label as constituted cannot support the causal question and
the effort belongs on getting OHDSI calibrated effect estimates instead of better features.

**Why this is not just a metric change.** A residual target is much closer to a causal
quantity than a screening flag: it asks "is this pair disproportionate *beyond* what this
drug's and this condition's reporting volumes imply", which is the quantity a
disproportionality analysis is trying to estimate in the first place. Comparing predictions
across a continuous and a binary target requires care — see step 6.

---

## Inputs

| path on the volume | rows | use |
|---|---|---|
| `/splits/train.csv` | 723,586 | fit; `faers_prr`, `faers_case_count` |
| `/splits/validate.csv` | 434,151 | one evaluation |
| `/drug/ingredient_features.csv` | 4,280 × 159 | drug block (exp02's selection) |
| `/condition/condition_features_basic.csv` | 5,631 | condition block |
| `/condition/condition_group_long.csv` | ~10,554 | organ system / therapeutic area |

**Measured subset.** Restricting to `faers_case_count >= 3` — so the PRR is not built on
one or two reports — leaves **344,653 train rows** (47.6%) over 1,149 drugs and 3,604
conditions, and **206,586 validate rows** (47.6%). On that train subset,
`log(faers_prr)` has mean 0.283 and sd 1.052. Assert these at startup.

---

## Method

1. Precondition check, including the subset sizes and the `log(faers_prr)` moments above.
2. Restrict both folds to `faers_case_count >= 3` and `faers_prr > 0`. State the counts in
   the report; everything downstream is on this subset and the AP numbers here are therefore
   **not** comparable to exp01/exp02 headline numbers computed on all rows. Recompute the
   binary comparison (step 6) on this same subset so the comparison is internally valid.
3. **Fit the two-way structure on train only.** Target `z = log(faers_prr)`. Estimate
   additive drug and condition effects by alternating demeaning (Gauss–Seidel / iterative
   within-transformation): subtract drug means, then condition means, repeat to convergence
   (tolerance 1e-6 on the max mean shift, cap 50 iterations — it converges in well under 20
   on data this sparse). Keep the fitted drug and condition offsets.

   `residual = z - drug_effect(d) - condition_effect(c) - grand_mean`

   This is the standard within-transformation; using it rather than a mixed model is
   deliberate — `statsmodels`' random-effects fit on 345 k rows with ~4.7 k levels will not
   finish in the budget, and the point here is the residual, not inference on the effects.
4. **Apply the train offsets to validate.** Drugs and conditions unseen in train get offset
   0 (the grand mean), and are reported as a separate subgroup. Never re-estimate offsets on
   validate — that removes exactly the structure the features are supposed to predict, and
   would make the result look strong for the wrong reason.
5. Fit `HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=63,
   early_stopping=False, random_state=0, loss="squared_error")` on the residual, using
   exp02's drug + condition intrinsic blocks plus the degree terms. Grouped 3-fold CV inside
   train on `group_key`, one validate pass. Report Spearman ρ, Pearson r, and R².
6. **The comparison that the hypothesis turns on.** On the identical subset and feature set,
   also fit the binary `y_faers_signal` classifier. Two targets are not comparable by ρ vs
   AP, so report both of these:
   - AP lift over prevalence for the binary model, and ρ for the residual model, side by side
     with their nulls stated;
   - the **common footing**: Spearman ρ between each model's predictions and the *residual*,
     and AP of each model's predictions against the *binary flag*. A model trained on the
     residual that also ranks the binary flag competitively is the strong result; one that
     wins only on its own target is not.
7. Report how much of the residual's variance the drug and condition offsets removed (R² of
   the two-way structure itself). If the offsets explain very little of `log(faers_prr)`, then
   degree was not dominating this quantity and the experiment's premise is weaker than
   `AGENT.md` §5 suggests — say so.

## Budget

| step | est. |
|---|---|
| read + join + subset | 80 s |
| alternating demeaning to convergence on 345 k rows | 20 s |
| regressor: 3 CV + 1 full fit on 345 k × ~170 | ~2 min |
| classifier comparison: 3 CV + 1 full | ~2 min |
| figures + tables | 60 s |
| **total** | **~7 min** |

`timeout=600`, `cpu=8.0`, `memory=16384`.

**If the budget is at risk:** drop the classifier's CV (keep its single train fit + validate
pass), since exp02 already carries the well-estimated binary number.

## Deliverables

- `<exp_id>_residual_model.csv` — train CV ρ (mean, spread), validate ρ, Pearson r, R²,
  n rows
- `<exp_id>_target_comparison.csv` — step 6, the four-cell table (each model × each target)
- `<exp_id>_offsets.csv` — the fitted drug and condition offsets, with their n; useful to
  every later experiment, and worth committing
- `<exp_id>_twoway_fit.csv` — variance of `log(faers_prr)` explained by the two-way structure
- `<exp_id>_residual_scatter.png` — predicted vs actual residual on validate, hexbin, with
  the ρ annotated
- `<exp_id>_subgroups.csv` — validate ρ by `is_mapped`, `arm`, and for drugs/conditions
  unseen in train

## Reporting requirements

`metrics`: `validate_spearman_residual`, `train_cv_spearman_residual`,
`validate_average_precision` (the binary comparison model on this subset),
`baseline_degree_only_ap` (on this subset, so the entry is comparable),
`twoway_r2`, `n_train_rows`, `n_validate_rows`.

`findings` must state: ρ on the residual and its null; the four-cell comparison result;
how much variance the two-way structure removed; and a recommendation — should later rounds
switch target? Answer it. A hedge here costs the next agent a rerun.

## Traps specific to this experiment

- **Do not estimate the offsets on all rows.** Train only, then apply. This is the same
  leak as computing degree over the full table, wearing a different hat.
- `faers_ror` is identical to `faers_prr` here; do not treat it as a second outcome.
- The residual is heteroscedastic by case count — a PRR from 3 cases is noisier than one
  from 300. Report ρ overall and within case-count strata {3–5, 6–20, 21+}; if the signal
  lives only in the high-count stratum, the "closer to causal" claim needs that qualifier.
- Weighting by case count is *not* part of this spec. If you try it, it is a second
  experiment; register it separately rather than folding it in.
- This subset (47.6% of rows) is not a random subsample — it is biased toward
  well-reported pairs, which correlates with drug age and market size. Say so when comparing
  to exp02's numbers.

## Registration

```python
exp = ln.register(
    agent="exp04",
    title="Two-way-demeaned log PRR residual as the modelling target",
    hypothesis=("Drug and condition intrinsic features predict the degree-free "
                "log(faers_prr) residual with Spearman rho >= 0.10 on validate, a larger "
                "effect relative to its null than the same features achieve on the binary "
                "Evans flag."),
    approach=("Restrict to faers_case_count >= 3. Two-way fixed effects on log PRR by "
              "alternating demeaning, fitted on train and applied to validate. HistGBM "
              "regressor on the residual, plus the binary classifier on identical rows for "
              "a four-cell prediction-vs-target comparison."),
    label="faers_prr_residual",
    features=["degree", "drug_intrinsic", "condition_intrinsic"],
    split="train/validate, grouped by primary target gene, case_count >= 3 subset",
)
```
