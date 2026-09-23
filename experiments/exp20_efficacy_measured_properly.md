# exp20 — The efficacy half, on the metric and the floors the safety half uses

**Follows** exp05 (`exp_20260922_7c2e17`), which completed and produced the most interesting
number in the project under measurement conditions we no longer accept.

**Status.** Not started.

---

## Why exp05 needs redoing rather than extending

It reported mechanism-only validate **AP 0.1275** against a **0.48%** prevalence — a 26×
lift — with the mechanism-versus-indication gap at +0.0686 above a 3-seed spread of 0.0324,
`first_approval` importance ranked 13th, and summed ATC importance of **exactly 0**. If that
holds up it is the strongest result the project has, and it says the efficacy half is not only
viable but *more* viable than the safety half.

But it was measured in pooled AP with:

- **no floor.** A 26× lift on a 0.48% label says nothing until the condition's own train
  TREATS rate is ranked against the same target. On the safety side, the `p_c` lookup beat
  every fitted model for two rounds before anyone checked.
- **the wrong metric.** `drug_macro_auc` is the project's primary and was not computed;
  pooled AP at 0.48% prevalence is dominated by high-degree drugs.
- **CV skipped** on the M+I design (314 columns, budget exhausted), 3 seeds instead of 5, the
  `y_semmeddb_causes` secondary dropped, and its step 5 never attempted.

## Question

On the primary metric and against its own floors, is the efficacy label predictable from
mechanism — and does the same-target neighbour signal that dominates the safety side dominate
here too?

## Hypothesis

`drug_macro_auc` for the mechanism-only design exceeds the TREATS `p_c` floor by **≥ 0.03**,
and the mechanism-versus-indication gap survives on the per-drug metric. The neighbour feature
adds less here than its +0.0443 on the safety side, because an indication is a property of the
molecule's intended target rather than of what other drugs on that target happened to be
reported for.

**What would falsify it.** The TREATS `p_c` lookup matching the mechanism model — the same
outcome that deflated three safety experiments. Given how strong condition-level base rates
have been throughout this project, that is the outcome to expect and to design against.

---

## Method

1. Preconditions: `y_semmeddb_treats` positives 3,926 train / 2,092 validate (0.54% / 0.48%);
   `leak_audit.py`; and a hard assertion that **no SemMedDB column appears as a feature** —
   the round-3 convention, sharpened by exp06's `benefit_ceiling = 1.0000`, which measured a
   column against itself.
2. `floors.py` for this label: `p_c` (the condition's train TREATS rate), degree + `p_c`,
   prevalence, `n_drugs_scored`. **Report these before any model number.**
3. Designs, reusing exp05's audited M/I column assignment
   (`exp_20260922_7c2e17_column_assignment.csv` — do not re-derive it):
   **M** mechanism and target biology; **I** indication-encoding (ATC, `indication_class`,
   `usan_stem_definition`, `kegg_efficacy`, `max_phase`, `first_approval`); **M+I**;
   and **M + neighbour**, where the neighbour feature is exp08's `nb_excess` recomputed
   against the TREATS label with leave-one-drug-out — exp05's unattempted step 5.
4. Metrics: `drug_macro_auc`, `drug_macro_p10`, `n_drugs_scored`, pooled AP (for continuity
   with exp05), each against the floors from step 2. Eligibility will bite hard at 0.48%
   prevalence — a drug needs ≥1 TREATS positive among its pairs to be scoreable. Report
   `n_drugs_scored` next to every number and state the eligibility rule used.
5. Grouped 3-fold CV inside train scored on `drug_macro_auc`; 5 seeds; one validate pass per
   design. If the budget bites, cut seeds before cutting CV — exp05 cut CV and left its M+I
   design with no internal estimate at all.
6. Carry `y_semmeddb_causes` (715 / 370 positives, 0.10%) as an indicative secondary with its
   own floors, clearly labelled as underpowered.
7. **Transportability**, as on the safety side: `drug_macro_auc_unseen_family` using exp12's
   rung construction, and the small-molecule restriction from exp17.

## Budget

| step | est. |
|---|---|
| read, column assignment, floors | 2 min |
| 4 designs × (3-fold CV + full) × 5 seeds, narrow designs | 6 min |
| neighbour feature vs TREATS with leave-one-drug-out | 1.5 min |
| secondary label, rungs, figure | 1.5 min |
| **total** | **~10 min** at `timeout=900` |

## Deliverables

`<exp_id>_designs.csv` (M / I / M+I / M+neighbour × metric × floor),
`<exp_id>_floor_table.csv`, `<exp_id>_seed_spread.csv`, `<exp_id>_by_rung.csv`,
`<exp_id>_causes_secondary.csv`, `<exp_id>_importance_top25.csv`,
`<exp_id>_efficacy_vs_safety.png` — mechanism-only margin over floor for TREATS beside the
same quantity for `y_faers_signal`, which is the comparison that decides where Round 5 goes.

## Reporting requirements

`metrics`: `drug_macro_auc_M`, `_I`, `_MI`, `_M_neighbour`; `floor_pc_treats`,
`floor_degree_pc_treats`; `drug_macro_p10_M`; `n_drugs_scored`;
`validate_average_precision` (M, for continuity with exp05's 0.1275);
`drug_macro_auc_unseen_family_M`; `causes_drug_macro_auc_M`; `seed_spread_drug_macro_auc`.

`findings` must open with the floor, then M's margin over it with a CI, then whether exp05's
pooled-AP claim survives translation to the per-drug metric — stated explicitly, because a
26× pooled lift that vanishes on the primary metric is exactly the pattern exp09 produced on
the safety side. Then the neighbour result, then a recommendation on whether Round 5 should
move the project's centre of gravity to efficacy.

## Traps

- SemMedDB is literature-mined: absence is weak evidence and coverage tracks publication
  volume, so never describe this label as efficacy ground truth — it is a proxy for published
  assertion.
- At 0.48% prevalence, `n_drugs_scored` may fall far below the 686 the safety side uses. A
  macro average over 80 drugs is not comparable to one over 686; say so wherever the two
  appear together.
- TREATS and PREVENTS are pooled in this label. Do not split them at this prevalence.
- ATC and `indication_class` encode the indication by construction. The M/I partition is what
  makes the claim transportable; do not let an I column drift into M.

## Registration

```python
exp = ln.register(
    agent="exp20",
    title="Efficacy label on drug_macro_auc, with floors and the neighbour feature",
    hypothesis=("Mechanism-only features exceed the TREATS p_c floor by >=0.03 drug_macro_auc "
                "and the mechanism-versus-indication gap survives on the per-drug metric, "
                "while the neighbour feature adds less here than on the safety side."),
    approach=("Reuse exp05's audited M/I column assignment; floors.py for the TREATS label "
              "reported before any model number; four designs including exp08's neighbour "
              "feature recomputed against TREATS with leave-one-drug-out; 5 seeds, grouped "
              "3-fold CV scored on drug_macro_auc; transportability by exp12's rungs."),
    label="y_semmeddb_treats",
    features=["mechanism_block", "target_biology", "indication_encoding", "nb_excess_gene",
              "p_c", "degree"],
    split="train/validate, grouped by primary target gene",
    notes="Re-measures exp05 (exp_20260922_7c2e17) on the primary metric with floors.",
)
```
