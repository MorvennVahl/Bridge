# exp16 — Does biology predict approved indications?

Experiment id `exp_20260922_132b7b`. Depends on the ChEMBL indication layer (PR #21).

## Why this was not possible before

`exp11` tried to evaluate against an adjudicated reference set and failed its precondition:
only **3 drugs** had ≥5 reference pairs against a required 30, because the OMOP/EU-ADR sets
are drug-by-health-outcome, not a dense drug-by-condition matrix.

The ChEMBL indication layer supplies **106 drugs** with ≥5 approved-indication pairs and ≥5
non-indication pairs (549 at any phase). 281 drugs clear METRIC.md eligibility here.

This is also the first test of biology against a *curated efficacy* label. `exp14` tested
gene overlap on the harm label; `exp05` tested mechanism features on SemMedDB assertions.

## Result

| model | drug_macro_auc |
|---|---|
| `p_c` lookup (the floor) | **0.8469** |
| fitted degree + `p_c` | 0.8257 |
| **+ gene overlap, genetic-weighted** | **0.8617** |
| gene overlap alone | 0.6015 |

**Increment over the floor: +0.0148**, bootstrap 95% CI over 281 drugs **[+0.0040, +0.0261]**.
Excludes zero, but not by much. Secondary: `drug_macro_p10` 0.0619 against 0.0452 for the
floor; pooled AP 0.0479 against 0.0341.

## The floor is very high on this label

`p_c` scores **0.8469** here, against **0.5759** on `y_faers_signal`. Approved indications
concentrate on a handful of conditions, so a condition's base rate alone ranks most of a
drug's list correctly. **Any future efficacy result must be read against 0.8469**, not
against the 0.5759 the harm-label work uses.

The fitted degree+`p_c` model scores 0.8257 — *below* the one-line lookup, repeating the
pattern Round 2 found on the harm label.

### A correction I caught before logging

I first bootstrapped the increment against the fitted floor (0.8257) and got **+0.0360**
[+0.0189, +0.0528]. That is exactly exp13's error. Against the correct floor the effect is
**less than half as large**. Both numbers are in the metrics file.

## Leak demonstration

Adding Open Targets `dt_clinical` to the overlap weight raises drug_macro_auc to **0.8875**,
an inflation of **+0.0258 — 1.7× the honest effect**.

`dt_clinical` is derived from the same ChEMBL table as this label, and
`condition_gene_ot_long.csv`'s `ot_score` blends it in. **Anyone weighting overlap by
`ot_score` against an indication label will report roughly triple the real gain.** Use the
per-datatype columns and drop `dt_clinical`.

## What this does not show

- **Coverage is thin.** Only **2.74%** of validate pairs have any gene overlap, so the
  increment comes from a small slice and says nothing about the other 97%.
- **Negatives are constructed, not adjudicated** — a CEM pair for a ChEMBL-curated drug that
  is not on its indication list. Better than absence from spontaneous reporting, but not
  verified. Drugs with no curated indication are excluded rather than labelled negative.
- **No transportability ladder.** exp12 found the harm-label advantage decays to the floor by
  D1–D2; this says nothing about whether the efficacy advantage survives target-space distance.
- The indication layer's own precision is imperfect — see the `dataset_state` note.
- No test-set contact.

## Reproduce

```bash
uv run python experiments/exp16_indication_label_biology.py
```

Writes `results/exp_20260922_132b7b_metrics.json`.
