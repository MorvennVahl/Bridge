# exp16 — The external yardstick, on a metric the reference sets can support

**Supersedes** exp11's failed precondition (`exp_20260922_250b1f`) — not by loosening it, but
by changing the metric to match the data's shape. Register fresh; name exp11 in `notes`.

**Status.** Not started. Run first.

---

## Why exp11 could not proceed, and why that was a spec error

exp11 fetched the reference sets correctly and then could not use them: only 3 of 223 drugs
had ≥5 reference pairs, against the 30 the spec demanded. The cause is structural and was
predictable from what these sets are. Adjudicated pharmacovigilance reference sets are built
**drug-by-health-outcome-of-interest**: many drugs against a handful of carefully adjudicated
outcomes. Our per-drug metric needs the transpose — many conditions per drug.

What actually landed in `data/input/reference/reference_set.csv` (472 pairs, 223 drugs,
8 conditions, 198 positives / 274 negatives; 398 from Ryan et al. 2013, 74 from Coloma et
al. 2013):

| condition_concept_id | pairs | positives |
|---|---|---|
| 4026032 | 127 | 85 |
| 4329847 | 112 | 41 |
| 192671 | 101 | 29 |
| 197320 | 97 | 28 |
| 137829 / 141651 / 440689 | 10 each | 5 each |
| 4281749 | 5 | 0 |

The four large outcomes carry **437 pairs and 183 positives across ~100 drugs each**. That is
a perfectly adequate evaluation — of a *different* ranking task.

## Question

For a given adverse outcome, how well do our scorers rank drugs — against adjudicated truth
with real negatives — and how does that compare to FAERS disproportionality, the method they
would replace?

## The metric for this experiment

```
condition_macro_auc = mean over outcomes c of  ROC-AUC( y[:, c], score[:, c] )
```

Restricted to outcomes with ≥50 pairs and both classes — the four above. This is the
transpose of `drug_macro_auc` and it is the standard framing of the pharmacovigilance
benchmark: for one outcome, rank the drugs.

**Its floor is the transpose too.** Within a single outcome, every drug's `p_c` is identical,
so `p_c` cannot contribute; the trivial baseline is the **drug's own train flag rate `p_d`**
(and drug degree). Compute both, per `floors.py`, and report them as
`floor_pd` / `floor_degree_pd`. Do not reuse any floor measured on `y_faers_signal` — this is
a different label on a different population.

**Pooled metrics are legitimate here and nowhere else in the project.** The negatives are
adjudicated rather than merely unobserved, so report pooled ROC-AUC and AP alongside, and say
so explicitly — it is the only place in this project where a pooled number means what a
reader assumes it means.

## Method

1. Precondition: file present; ≥4 outcomes with ≥50 pairs and both classes; both labels
   present overall. Assert the counts above and fail loudly on mismatch.
2. Score every drug–outcome pair with: `p_d` floor, degree + `p_d` floor, the incumbent FAERS
   quantities (`log(faers_prr)`, `faers_chi_square`, the Evans flag), exp07's intrinsic union,
   exp08's neighbour model, exp14's gene-overlap model. Refit nothing — score the fitted
   models. Any drug or pair absent from our grid is reported as coverage loss, never imputed.
3. `condition_macro_auc` + pooled AUC/AP for each, with 1,000-resample bootstrap CIs over
   **drugs** (resample drugs, recompute both metrics).
4. Per-outcome table — four outcomes is few enough to read individually, and a method that
   works on GI bleed and fails on liver injury is a finding, not an average.
5. **The ceiling, finally computable.** Take the incumbent FAERS score's
   `condition_macro_auc` as the operational ceiling: it is what a reviewer gets today from
   RWD. Emit `success_criterion.json` v2 with `source: "adjudicated_per_condition"`,
   `floor_pd`, `ceiling_incumbent`, and `solved_threshold = floor + 0.5*(ceiling - floor)`.
   If a biology model **matches the incumbent on adjudicated outcomes without using that
   drug's RWD**, the project's core promise is met — state it in exactly those terms.
6. Report the rank correlation between each model's reference-set ordering and its CEM-label
   ordering on the same drug–outcome pairs. A large gap is the number that quantifies how
   much of our CEM performance is observation process rather than pharmacology.

## Budget

No fitting. ~5 min including bootstraps. `timeout=900`.

## Deliverables

`<exp_id>_reference_eval.csv`, `<exp_id>_per_outcome.csv`, `<exp_id>_coverage.csv`,
`success_criterion.json` (v2), `<exp_id>_cem_vs_reference_rank.csv`,
`<exp_id>_bootstrap.csv`.

## Reporting requirements

`metrics`: `condition_macro_auc_union`, `condition_macro_auc_neighbour`,
`condition_macro_auc_overlap`, `incumbent_condition_macro_auc`, `floor_pd`,
`floor_degree_pd`, `pooled_auc_union`, `pooled_ap_union`, `n_outcomes`, `n_pairs`,
`ceiling_incumbent`, `solved_threshold`.

`findings` must open with the floor on this set, then each model against it and against the
incumbent, then the per-outcome spread, then the revised `solved_threshold`. State plainly
whether any model matches the incumbent, because that is the project's actual claim.

## Traps

- **Never fit on this set.** It is the only external check the project has.
- Four outcomes is a small sample of outcome space, all of them serious acute events. Do not
  generalise to the 5,631-condition grid; say which four.
- Negative controls are "believed to have no causal relation", not "verified safe". Quote the
  source's own definition in the coverage file.
- Drugs in the reference set skew old and widely used. Report their `first_approval`
  distribution against the full ingredient set so the selection is visible.

## Registration

```python
exp = ln.register(
    agent="exp16",
    title="Adjudicated reference-set evaluation with a per-condition metric",
    hypothesis=("On four adjudicated outcomes with real negatives, our scorers exceed the p_d "
                "floor computed on that set, and the incumbent FAERS score's "
                "condition_macro_auc gives the operational ceiling for the success criterion."),
    approach=("Transpose the metric to condition_macro_auc (rank drugs within an outcome) to "
              "match the drug-by-HOI shape of Ryan 2013 / Coloma 2013; score existing fitted "
              "models plus p_d and degree+p_d floors; pooled AUC/AP legitimate here because "
              "negatives are adjudicated; emit success_criterion.json v2."),
    label="reference_set_label",
    features=["p_d", "faers_prr", "intrinsic_union", "nb_excess_gene", "gene_overlap"],
    split="reference pairs, no fitting",
    notes="Supersedes exp11's failed precondition by changing the metric, not the threshold.",
)
```
