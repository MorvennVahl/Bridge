# exp09 — A residual target that exists on held-out drugs

**Supersedes** `exp_20260922_3f4a30` (exp04). Its estimator was correct; its target was not
estimable under this split.

**Status.** Not started. Independent of the others.

---

## What went wrong, precisely

exp04 fitted two-way fixed effects on `log(faers_prr)` — drug effect plus condition effect —
on train, then applied them to validate:

```python
d_eff = df["ingredient_concept_id"].map(drug_effect).fillna(0.0)
```

The splits hold out **whole drugs**, so no validate drug appears in `drug_effect` and
**every validate row received drug offset 0**. On train the target had the drug level
removed; on validate it retained it in full. Those are two different quantities, and fitting
on one while scoring the other is what produced validate ρ = **−0.3064** against train CV
ρ = **+0.4889**. The two-way residual is not an estimable target under a drug-grouped split
— no amount of iteration fixes that, because the drug offset of an unseen drug is exactly
the thing we are trying to predict.

## Question

Is the *within-drug* disproportionality pattern — which conditions are disproportionate for
this drug relative to its own average — predictable from biology for a drug with no data?

## Hypothesis

Features predict the within-drug residual of `log(faers_prr)` with a **within-drug Spearman
ρ ≥ 0.10** (macro-averaged over held-out drugs), and that ranking also achieves
`drug_macro_auc` on the binary Evans flag competitively with exp07's union model. The
continuous target is better behaved than the flag because it does not throw away the
magnitude of the disproportionality.

**What would falsify it.** Macro within-drug ρ near zero while exp07's classifier clears its
floor: the continuous target adds nothing and later rounds should stay binary. A negative ρ
this time would indicate a remaining target mismatch, not a finding — check the offset
application before reporting anything.

---

## The fix

**Remove only the condition level, and evaluate within drug.**

```
target:      r = log(faers_prr) - grand_mean - condition_offset(c)      # condition offsets from TRAIN
evaluation:  macro over held-out drugs of  spearman( r[d, :], pred[d, :] )
```

The drug level never needs an offset: within-drug rank correlation is invariant to any
additive per-drug constant, so the quantity the split makes unlearnable is differenced out
of the *metric* instead of being imputed to zero in the *target*. This is the same object
`METRIC.md` measures — `drug_macro_auc` on the binary flag and macro within-drug ρ on the
continuous one are the classification and regression forms of one question.

One-way condition demeaning by simple group means on train; no alternating iteration is
needed once the drug factor is gone. Conditions unseen in train get the grand mean, flagged
in a `condition_unseen` column and reported as a subgroup.

## Method

1. Preconditions: subset `faers_case_count >= 3 & faers_prr > 0` — **344,651 train rows over
   1,149 drugs and 3,604 conditions; 206,585 validate rows** (47.6% of each fold), train
   `log(faers_prr)` mean 0.283, sd 1.052. Assert all of these; exp04 verified them.
2. Build the target above. Report the variance share removed by the condition offsets alone
   (exp04's two-way structure removed R² = 0.4700; the one-way number will be lower and that
   difference *is* the drug level, worth reporting as a quantity in its own right).
3. Features: exp07's clean blocks — degree, `p_c`, drug-intrinsic, condition-intrinsic. Run
   `leak_audit.py` from exp07 first; `faers_*` columns are the target's ingredients here, so
   the blacklist matters more than usual.
4. `HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=63,
   early_stopping=False, random_state=0)`. Grouped 3-fold CV in train scored by macro
   within-drug ρ on held-out CV drugs; one validate pass.
5. **The cross-target comparison, done coherently.** Four cells: {regressor on residual,
   classifier on Evans flag} × {evaluated by macro within-drug ρ, evaluated by
   `drug_macro_auc`}. The strong result is the regressor winning on its own target *and*
   ranking the binary flag competitively. Use the same rows and the same feature matrix for
   both so the comparison is internally valid.
6. Heteroscedasticity: report macro ρ within case-count strata {3–5, 6–20, 21+}. A PRR from
   3 cases is far noisier than one from 300, and if the signal lives only in the 21+ stratum
   the "closer to causal" claim needs that qualifier.

## Budget

| step | est. |
|---|---|
| read, subset, condition demeaning | 90 s |
| regressor 3 CV + 1 full on 345k × ~170 | 2 min |
| classifier arm on identical rows | 2 min |
| per-drug ρ, strata, figures | 60 s |
| **total** | **~6 min** |

**If at risk:** drop the classifier arm's CV and keep its single fit; exp07 carries the
well-estimated binary number.

## Deliverables

`<exp_id>_residual_model.csv`, `<exp_id>_target_comparison.csv` (the four cells),
`<exp_id>_per_drug_rho.csv`, `<exp_id>_condition_offsets.csv`,
`<exp_id>_variance_decomposition.csv` (condition level vs drug level vs residual),
`<exp_id>_strata.csv`, `<exp_id>_residual_scatter.png`.

## Reporting requirements

`metrics`: `macro_within_drug_spearman`, `train_cv_macro_within_drug_spearman`,
`drug_macro_auc_from_residual_model`, `drug_macro_auc_from_classifier`,
`condition_variance_share`, `drug_variance_share`, `n_train_rows`, `n_validate_rows`,
`n_drugs_scored`.

`findings` must open by stating that exp04's ρ = −0.3064 was a target-mismatch artefact of
assigning drug offset 0 to every held-out drug, and that this entry supersedes it. Then the
macro within-drug ρ, the four-cell comparison, the strata breakdown, and a direct
recommendation on whether later rounds model the continuous target or the flag.

## Registration

```python
exp = ln.register(
    agent="exp09",
    title="Within-drug log-PRR residual: a target that exists on held-out drugs",
    hypothesis=(
        "Removing only the condition level and evaluating within drug yields macro "
        "within-drug Spearman >= 0.10 on validate, and that ranking scores "
        "drug_macro_auc on the Evans flag competitively with the intrinsic union."
    ),
    approach=(
        "One-way condition demeaning on train applied to validate; within-drug rank "
        "correlation as the metric so the unlearnable drug level is differenced out of "
        "the metric rather than imputed to zero in the target; four-cell cross-target "
        "comparison on identical rows; case-count strata."
    ),
    label="faers_prr_within_drug_residual",
    features=["degree", "p_c", "drug_intrinsic", "condition_intrinsic"],
    split="train/validate, grouped by primary target gene, case_count >= 3 subset",
)
```
