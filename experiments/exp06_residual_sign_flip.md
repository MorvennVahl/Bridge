# exp06 — Why the residual Spearman flips sign between train CV and validate

Follow-up diagnostic on `exp_20260922_3f4a30` (exp04). Experiment id `exp_20260922_7a4500`.

## The observation

exp04 reported, for the same model on the same target:

| | |
|---|---|
| train CV Spearman | **+0.489** |
| validate Spearman | **−0.306** |

A reversal that large is usually a construction error rather than a weak signal. Weak
generalisation decays toward zero; it does not cross it and keep going.

## Hypothesis

The target is built as

```
residual = z − grand_mean − drug_offset − condition_offset
```

with `drug_offset` from a train-fitted effect and `.fillna(0.0)` for drugs absent from
train (`exp04_disproportionality_residual.py:232-239`).

The splits hold out **whole ingredients**. So no validate drug appears in train, every
validate row takes `drug_offset = 0`, and the drug effect that was removed from the
training target is still present in the validate target. Model and target would then be
measuring different quantities.

Three falsifiable predictions:

1. The share of validate rows with `drug_unseen` is ~100%, not a small tail.
2. The train-fitted drug effect accounts for a large share of validate target variance.
3. An evaluation invariant to a per-drug offset removes the sign flip.

If validate rho stayed negative under all three, the hypothesis would be wrong and the
anti-correlation real.

## Result — confirmed

| check | train | validate |
|---|---|---|
| rows with `drug_unseen` | 0.0% | **100.0%** |
| between-drug share of target variance | 0.0% | **40.3%** |

| evaluation of the same predictions | Spearman |
|---|---|
| train CV (exp04 target) | +0.475 |
| validate, exp04 as written | **−0.318** |
| validate, **within drug** (497 drugs with ≥20 rows) | **+0.400** |
| prediction vs the omitted per-drug offset | **−0.481** |

Reproducing exp04's setup gives +0.475 / −0.318 against its reported +0.489 / −0.306, so
this is a faithful reproduction rather than a different experiment.

The sign flip disappears under within-drug evaluation. The mechanism is the last row:
predictions anti-correlate with the offset that was never removed, and because that offset
carries 40% of the validate target's variance, the pooled correlation follows it instead of
the within-drug ranking.

## What this does not show

- **It does not rescue the residual as a modelling target.** +0.400 within-drug is measured
  on drugs whose per-drug level is unknowable by construction. A pooled score on unseen
  drugs has no well-defined value under this target. The finding is that −0.306 is
  uninformative, not that the residual works.
- **exp04's other metrics are unaffected** — two-way R² = 0.470 and the AP comparisons are
  not revisited here.
- **Features are the three degree terms only**, not exp04's full drug+condition block. The
  artefact is target-side and reproduces regardless, but this is not a re-run of exp04's
  model.
- One planned check was dropped as uninformative: a "condition-only demeaned target"
  variant is *identical* to the as-written target on validate, because `drug_offset` is
  zero on every validate row. It tests nothing. The within-drug evaluation is what carries
  the result.
- No test-set contact.

## Implication for anyone using this target

Under ingredient-grouped splits, a two-way-demeaned target has to say what the drug offset
means for a drug with no data — that *is* the prediction problem. Two coherent options:

1. Predict the drug offset from drug features as a separate head and add it back.
2. Demean by condition only, so the target is well defined for every drug, and state that
   the drug-level term is not modelled.

Either way, report within-drug and pooled correlation separately. A single pooled number
hides which one moved.

## Reproduce

```bash
uv run python experiments/exp06_residual_sign_flip.py
```

Runs locally against `data/splits/`; writes `results/exp_20260922_7a4500_metrics.json`.
