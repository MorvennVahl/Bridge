# exp13 — Every subset gets its own floor, and exp09 re-refereed

**Supersedes** `exp_20260922_4e79a6` (exp09) on the recommendation, not the mechanics: its
target fix was correct and its estimator was sound. Append a `complete` naming it.

**Status.** Not started. Cheap, and it repairs a flaw that affects every subset result the
lab will ever produce.

---

## The flaw, with numbers

exp09 reported `drug_macro_auc` 0.8330 (classifier) and 0.8274 (residual model) on the
`faers_case_count >= 3` subset, and compared them to `floor_drug_macro_auc_pc_lookup =
0.5759` — a floor measured on **all rows**. Measured on the subset itself:

| model, on `case_count >= 3` (validate, 571 drugs, min 5 rows) | drug_macro_auc |
|---|---|
| `p_c` alone | 0.7754 |
| condition mean `log(faers_prr)` from train alone | 0.7589 |
| **degree + `p_c`** | **0.8480** |
| degree + `p_c` + condition `log PRR` offset | 0.8492 |
| exp09 classifier (reported) | 0.8330 |
| exp09 residual model (reported) | 0.8274 |
| degree + `p_c`, same model on **all** rows | 0.5947 |

Evans prevalence in the subset is 0.2466 against 0.1173 overall. The restriction
`case_count >= 3` deletes one of the three clauses of the Evans definition, so within the
subset the label is nearly determined by PRR and χ², both of which track the condition's base
rate. **exp09's models sit 0.015–0.021 below the trivial baseline for their own population.**
Its recommendation to model the continuous residual in later rounds is not supported by its
own evidence.

Note what this is *not*: not a leak, not a bug, and not a reason to distrust the within-drug
target fix, which was correct and remains the right formulation. It is a comparison error,
and comparison errors are systematic — hence the harness below.

## Question

Once each model is compared against the floor for its own population, does the continuous
within-drug residual beat the binary flag, and does either beat a lookup table?

## Hypothesis

On the `case_count >= 3` subset, neither exp09 model beats `degree + p_c` (0.8480), and the
continuous residual's apparent advantage disappears entirely under a correct floor. On the
**full** row set, where the label is not partly determined by the restriction, the continuous
target does beat the flag — because it retains disproportionality magnitude that the flag
discards.

**What would falsify the second half.** No advantage on full rows either: the continuous
target is not worth its extra machinery and Round 4 stays binary.

---

## Method

1. **Ship `experiments/floors.py`.** One function, called by every later experiment:

   ```python
   def floors(train, evaluate, label, eligibility) -> dict[str, float]:
       """p_c-lookup and degree+p_c floors, prevalence, and n_drugs_scored
       for THIS (label, subset, eligibility) combination."""
   ```

   It returns `floor_pc`, `floor_degree_pc`, `prevalence`, `n_drugs_scored`, and every
   experiment must log those four keys in its own metrics dict alongside its headline. A
   headline without its matching floor is not a result.
2. Recompute floors for every population the lab has used so far, and write them as a
   reference table: all rows; `case_count >= 3`; `≥1 gene neighbour`; each of exp12's rungs
   if that experiment has landed; the SemMedDB-harm target. This table is the artifact other
   agents will reuse.
3. **Re-referee exp09 on full rows.** Fit the within-drug residual regressor
   (condition-only demeaning, evaluated by macro within-drug Spearman) and the binary
   classifier on the **whole** train fold rather than the `case_count >= 3` subset, with
   identical features (exp07's clean blocks), and compare both against `floor_degree_pc` for
   that population. Report `drug_macro_auc` for both and macro within-drug ρ for both.
4. **Floor the rank metric too.** exp09's ρ = 0.6536 has no reference. Compute macro
   within-drug Spearman for `p_c` alone and for `degree + p_c` on the same rows; the residual
   model's ρ means nothing until those two numbers exist.
5. Report the case-count strata (3–5 / 6–20 / 21+) with a floor per stratum, which is what
   exp09's `next_steps` asked for and could not interpret without floors.

## Budget

| step | est. |
|---|---|
| floors.py + the reference floor table | 90 s |
| regressor + classifier on full rows, 3-fold grouped CV | 3 min |
| rank-metric floors and strata | 60 s |
| **total** | **~5 min** |

## Deliverables

`experiments/floors.py`, `<exp_id>_floor_reference_table.csv` (the reusable one),
`<exp_id>_residual_vs_flag_full_rows.csv`, `<exp_id>_rank_metric_floors.csv`,
`<exp_id>_strata_with_floors.csv`.

## Reporting requirements

`metrics`: `floor_pc_case_ge3`, `floor_degree_pc_case_ge3`, `floor_pc_all_rows`,
`floor_degree_pc_all_rows`, `residual_drug_macro_auc_full_rows`,
`classifier_drug_macro_auc_full_rows`, `macro_within_drug_spearman_full_rows`,
`macro_within_drug_spearman_floor_pc`, `n_drugs_scored`.

`findings` must open by stating that exp09's 0.8330 / 0.8274 were below the 0.8480 floor for
their own subset and that this entry supersedes its recommendation. Then the full-row
comparison, then the rank-metric floors, then a one-line verdict on continuous versus binary
for Round 4.

## Traps

- Do not re-run exp09's subset and call it a replication; the point is the floor, not the
  model.
- `min_rows_per_drug` changes `n_drugs_scored` and therefore the macro average. exp09 used 5,
  `METRIC.md` specifies 20. Report both and state which the headline uses — this alone can
  move a macro metric by a few thousandths and the lab should stop rediscovering it.
- Do not "fix" exp09's entry. Append.

## Registration

```python
exp = ln.register(
    agent="exp13",
    title="Per-subset floors, and the within-drug residual re-refereed against them",
    hypothesis=("Neither exp09 model beats degree+p_c (0.8480) on its own case_count>=3 "
                "subset, while on full rows the continuous within-drug residual does beat the "
                "binary flag against the same floor."),
    approach=("Ship floors.py computing p_c and degree+p_c floors per (label, subset, "
              "eligibility); build a reference floor table for every population used so far; "
              "refit residual regressor and classifier on full rows with floors for the rank "
              "metric as well as the AUC."),
    label="faers_prr_within_drug_residual",
    features=["degree", "p_c", "drug_intrinsic", "condition_intrinsic"],
    split="train/validate, full rows and case_count>=3 subset",
)
```
