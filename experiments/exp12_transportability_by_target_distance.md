# exp12 — Transportability: how far from training can a drug be?

**Status.** Not started. **The project's central promise, and it has never been measured.**

---

## Why it has never been measured

Both rounds reported an "unseen family" number and neither was one:

- The split's grouping unit is the **primary target gene**, and folds are disjoint on it by
  construction. So "drugs whose primary-target group is absent from train" is every validate
  drug, and the restriction is the identity.
- exp07 logged `drug_macro_auc_unseen_family = 0.6147814134373707` — identical to its overall
  `drug_macro_auc` to sixteen digits — because `ingredient_target_long.csv` was not staged on
  the volume, so no coarser family unit could be computed.

Meanwhile exp08 showed that neighbour signal exists *and* that 81.9% of validate drugs have
no gene-level train neighbour at all. That combination is exactly the transportability
question: performance must be a decaying function of distance from the training set in target
space, and we have never drawn the curve.

## Question

As a held-out drug's targets get further from anything in training, how fast does performance
fall — and where does it hit the floor?

## Hypothesis

`drug_macro_auc` decays monotonically across a target-space distance ladder, from drugs
sharing a target gene with a training drug down to drugs with no target annotation at all,
and at the far end it is **indistinguishable from the `p_c` floor** of 0.5759.

**What would falsify it, in both directions.**

- **Flat curve.** Performance does not depend on target-space proximity, so what the models
  learned is not target-mediated at all — it is drug-intrinsic chemistry or a condition-side
  effect, and exp08's +0.0443 needs another explanation.
- **No collapse at the far end.** Biology transports even to drugs with no annotated target.
  That would be the project's best possible result and would need a hard look for a
  confound (drug age, market size, route) before anyone believed it.

---

## The distance ladder

Five strata, each a rung further from training, computed with `ingredient_target_long.csv`
(8,088 edges, 1,541 drugs, 790 genes) and the ChEMBL protein-class columns:

| rung | definition | expected share of validate drugs |
|---|---|---|
| D0 | shares ≥1 **target gene** with a train drug | ~18% of drugs / 33.6% of pairs |
| D1 | no shared gene, shares a **protein-class leaf** | — |
| D2 | no shared leaf, shares a **protein-class L1** | — |
| D3 | has target annotation, shares nothing above | — |
| D4 | **no ChEMBL target annotation at all** | 812 of 1,284 drugs |

Measure and report the realised shares rather than trusting the table; exp03's coverage
numbers (33.6% gene / 55.5% leaf / 75.3% L1 at pair level) are the reference.

**Also build one genuinely coarser split.** Regroup ingredients by
`chembl_protein_class_leaf` and rebuild train/validate with
`scripts/build_dataset.py`-equivalent logic, so a *fold boundary* falls at the family level
rather than the gene level. Report the fold sizes achieved — Round 1 found that full
connected components merge 638 drugs into one component and make 50/30/20 unreachable, so
state whether the leaf-level split hits usable proportions, and if not, report the closest
achievable split and its label balance.

## Method

1. Precondition: assert `ingredient_target_long.csv` is present with 8,088 rows — the exact
   check exp07 lacked. Fail loudly rather than substituting the overall number.
2. Score the two models we already trust on each rung: **exp07's intrinsic union** and
   **exp08's neighbour model** (degree + `p_c` + gene-tier `nb_excess`). Refit nothing; use
   the same fitted models so the curve is about the evaluation population, not about training
   variation.
3. Per rung: `drug_macro_auc`, `drug_macro_p10`, `n_drugs_scored`, and the **`p_c` floor
   recomputed on that rung's drugs** (per the Round 2 correction — a rung is a subset and
   gets its own floor).
4. Bootstrap CI over drugs within each rung. Rungs will have few drugs at the far end; if a
   rung has under 20 scoreable drugs, report the CI and mark it underpowered rather than
   reporting a point.
5. Fit-and-evaluate once on the **leaf-level split** for the neighbour model, so we have one
   number where the fold boundary itself is at family level. This is the honest
   transportability headline, and it is likely lower than 0.6268.
6. Continuous version: for each validate drug, compute the **minimum distance to any train
   drug in target space** (0 = shared gene, 1 = shared leaf, 2 = shared L1, 3 = none) and
   regress per-drug AUC on it. Report the slope with its CI — one number for "how much
   performance costs per rung of distance".

## Budget

| step | est. |
|---|---|
| read, build the ladder, assert coverage | 2 min |
| score two fitted models across five rungs + per-rung floors | 2 min |
| leaf-level re-split and one refit of the neighbour model | 3.5 min |
| bootstraps, slope regression, figure | 1.5 min |
| **total** | **~9 min** |

**If at risk:** drop step 5 (the re-split refit) and log that the leaf-level fold boundary
was not reached. Never drop the per-rung floors.

## Deliverables

`<exp_id>_decay_ladder.csv` (per rung: both models, floor, n_drugs, CI),
`<exp_id>_decay_curve.png` (the project's headline figure — AUC versus target-space distance,
with the floor drawn as a horizontal line and n_drugs annotated per rung),
`<exp_id>_leaf_split_summary.csv`, `<exp_id>_slope.csv`, `<exp_id>_per_drug.csv`.

## Reporting requirements

`metrics`: `drug_macro_auc_D0` … `_D4`, `floor_D0` … `_D4`, `n_drugs_D0` … `_D4`,
`auc_per_rung_slope`, `slope_ci_lo`, `slope_ci_hi`, `leaf_split_drug_macro_auc`,
`leaf_split_floor`.

`findings` must state the curve rung by rung against each rung's own floor, the slope, the
rung at which performance becomes indistinguishable from floor, and the leaf-level-split
number. Then answer the project's question in one sentence: how far from a known drug can a
new molecule be before this approach stops working.

## Traps

- **A rung is a different population, not just fewer drugs.** D4 drugs are mixtures,
  biologics, minerals and botanicals with a median CEM degree of 6 versus 157 for matched
  ingredients. Their floor and their prevalence differ; that is why step 3 recomputes both.
- Do not refit per rung (except step 5, which is a different split). Refitting mixes the
  population effect with a training-data effect and the curve stops being interpretable.
- Drug age confounds distance: older drugs are both better reported and more likely to have
  annotated targets. Report `first_approval` distribution per rung so the confound is visible.

## Registration

```python
exp = ln.register(
    agent="exp12",
    title="Transportability decay by target-space distance, with a family-level split",
    hypothesis=("drug_macro_auc decays monotonically across a five-rung target-space distance "
                "ladder and is indistinguishable from the per-rung p_c floor for drugs with no "
                "target annotation."),
    approach=("Score the fitted exp07 union and exp08 neighbour models across rungs D0-D4 "
              "with per-rung floors and bootstrap CIs; rebuild one train/validate split "
              "grouped by chembl_protein_class_leaf for a family-level fold boundary; "
              "regress per-drug AUC on minimum target-space distance."),
    label="y_faers_signal",
    features=["intrinsic_union", "nb_excess_gene", "p_c", "degree"],
    split="validate stratified by target-space distance, plus a leaf-level regrouped split",
)
```
