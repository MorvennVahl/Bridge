# exp03 — Does target similarity transport across the group boundary?

**Lineage.** `lab/round_01.md` lane L5, with one change forced by a measurement taken while
writing this round (see *Coverage, measured* below): the neighbour definition is a **ladder
of three similarity tiers**, not gene sharing alone, because gene sharing alone is defined
on only a third of validation pairs.

**Status.** Not started. Independent of the other four. **This is the most important
experiment in the round** — it is the project's central claim in its cheapest testable
form, and it needs no condition-side genes, so it can run while the Open Targets disease
arm is still missing.

---

## Question

For a drug *d* and condition *c*, does the rate at which **other** drugs similar to *d*
are flagged for *c* predict whether *d* is flagged for *c* — when *d*'s primary target
group was never in training?

## Hypothesis

A shrunk neighbour rate computed strictly within train beats the degree-only floor on
validate pairs that have at least one neighbour, and the gain **decays monotonically** as
the similarity tier loosens from shared gene → shared protein-class leaf → shared
protein-class L1.

**What would falsify it.**

- No gain even on pairs with ≥3 gene-level neighbours: target similarity does not transport
  across the group boundary, and the bridge hypothesis is in serious trouble — a GNN over
  the same graph would be unlikely to rescue it.
- Gain that is flat or *increasing* as the tier loosens: what transports is a broad class
  effect (all antihistamines look alike), not target-specific pharmacology. That is still
  useful, but it is a different and much weaker claim than the project's, and it would
  redirect Round 2 from pathway overlap toward class-level priors.

## Coverage, measured — read this before designing anything

Computed from the written split files and `ingredient_target_long.csv`. The splits group
ingredients by **primary target gene**, so a validation drug's neighbours mostly do *not*
share its primary gene. That is deliberate, and it is also what makes the naive version of
this lane nearly vacuous:

| neighbour tier | validate pairs with ≥1 train neighbour | with ≥3 | validate ingredients with **zero** |
|---|---|---|---|
| shared `gene_symbol` | 33.6% | 22.5% | 81.9% |
| shared `chembl_protein_class_leaf` | 55.5% | 52.7% | 70.7% |
| shared `chembl_protein_class_L1` | 75.3% | 74.7% | 62.6% |
| shared `target_chembl_id` | 25.8% | 13.7% | 84.9% |

Only 472 of 1,284 validate ingredients carry any ChEMBL target annotation at all, and 527
carry a `target_chembl_id`. So:

**The headline AP over all validate pairs is not the result of this experiment.** Two
thirds of pairs have no gene-level neighbour and the feature is undefined for them. The
result is the **conditional AP given neighbour availability**, and the decay curve. Report
the unconditional number too, clearly labelled as diluted by coverage, so nobody later
mistakes it for the effect size.

---

## Inputs

| path on the volume | rows | use |
|---|---|---|
| `/splits/train.csv` | 723,586 | neighbour rates and fitting |
| `/splits/validate.csv` | 434,151 | one evaluation |
| `/drug/ingredient_target_long.csv` | 8,088 | drug → target → gene → protein class edges |
| `/condition/condition_features_basic.csv` | 5,631 | subgroup reporting |

Nothing else. No condition genes, no Open Targets.

---

## Features

For each pair (*d*, *c*) and each tier *t* ∈ {gene, class_leaf, class_L1}:

- `n_neighbours_t` — count of **other train** ingredients sharing ≥1 tier-*t* annotation
  with *d*
- `n_neighbours_flagged_t` — how many of those are flagged for *c* in train
- `nb_rate_t` — the shrunk rate:

  ```
  nb_rate_t = (n_neighbours_flagged_t + k * p_c) / (n_neighbours_t + k),   k = 10
  ```

  where `p_c` is condition *c*'s overall **train** flag rate. A rate from 2 neighbours is
  then pulled to the condition's base rate while a rate from 40 is not. `k = 10` is fixed,
  not tuned; report sensitivity at `k ∈ {5, 20}` in the results table since it costs one
  extra vectorised pass, and use `k = 10` for the headline.
- `nb_rate_t_minus_p_c` — the rate's **excess over the condition base rate**. This is the
  feature that can carry biology; `nb_rate_t` on its own is largely a re-encoding of
  condition degree, and a model given only that will look better than it is.
- Two refinements at the gene tier only: restricted to shared **primary** gene, and
  restricted to neighbours whose `action_type` on the shared gene matches *d*'s.
- `has_target_annotation` — indicator; drugs without one have every feature undefined.

**Leakage rule, non-negotiable.** For a **train** row the neighbour set must exclude *d*
itself, and `n_neighbours_flagged_t` must exclude *d*'s own label for *c*. Compute
leave-one-drug-out by subtracting *d*'s own contribution from the tier aggregate rather
than recomputing per drug — that keeps it a couple of vectorised passes. Without this the
feature contains the label and the experiment is worthless in a way that looks excellent.

---

## Method

1. Precondition check: input paths and row counts; reproduce the coverage table above to
   ±0.005 on the ≥1-neighbour column. If it does not reproduce, the splits or the target
   table were rebuilt — stop and report.
2. Build the tier aggregates on train with leave-one-drug-out (above), and the same
   aggregates applied to validate drugs using train-only neighbour sets and train-only
   labels.
3. Fit three models, all `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
   max_leaf_nodes=63, early_stopping=False, random_state=0)`:
   - **A** degree only (exp01's three features) — the floor, refit here so the comparison
     is on identical rows
   - **B** degree + gene-tier neighbour features
   - **C** degree + all three tiers
4. Grouped 3-fold CV inside train on `group_key`; one validate pass per model.
5. **The decay curve.** Validate AP by `n_neighbours_gene` bucket {0, 1–2, 3–5, 6+}, for A
   and B. This is the transportability curve the project exists to produce; it is the main
   figure of the round.
6. **Tier decay.** Validate AP of model B restricted to pairs where the *only* available
   tier is class_leaf, and where it is class_L1, to separate target-specific transport from
   class-level transport.

## Budget

| step | est. |
|---|---|
| read splits + target table | 40 s |
| tier aggregates, leave-one-drug-out, 3 tiers × 3 shrinkage values | 90 s |
| 3 models × (3 CV + 1 full) fits, ≤20 cols — much faster than full-width fits | ~2 min |
| decay curves, figures | 60 s |
| **total** | **~6 min** |

`timeout=600`, `cpu=8.0`, `memory=16384`. The fits are narrow (under 20 columns), so the
cost here is the aggregation, not the model.

**If the budget is at risk:** drop the `k ∈ {5, 20}` sensitivity and the `action_type`
refinement. Never drop step 5.

## Deliverables

- `<exp_id>_models.csv` — A/B/C with train CV AP (mean, spread), validate AP, ROC-AUC, both
  unconditional and conditional on ≥1 gene neighbour
- `<exp_id>_decay_curve.csv` and `<exp_id>_decay_curve.png` — AP by neighbour-count bucket,
  A vs B, with pair counts per bucket on the figure
- `<exp_id>_tier_decay.csv` — step 6
- `<exp_id>_coverage.csv` — the reproduced coverage table, so the next agent does not have
  to re-measure it
- `<exp_id>_calibration.png` — model B on validate, restricted to ≥1 neighbour

## Reporting requirements

`metrics`: `validate_average_precision` (model B, **conditional on ≥1 gene neighbour** —
state this in `findings` so the leaderboard entry is not misread),
`validate_average_precision_unconditional`, `train_cv_average_precision`,
`baseline_degree_only_ap`, `ap_by_nb_bucket_0`, `_1_2`, `_3_5`, `_6plus`, `coverage_ge1_nb`.

`findings` must state: the effect size on pairs that have neighbours; the shape of the decay
curve; whether gain decays or grows as the tier loosens; and the fraction of validate pairs
where the feature exists at all. The honest summary sentence is of the form "+X AP on the
34% of pairs with a gene-level neighbour, nothing on the rest, and the gain halves when only
class-level similarity is available".

## Traps specific to this experiment

- **Self-leakage** is the whole risk here. See the leakage rule. Sanity check: model B's
  train CV AP should not be dramatically above its validate AP on the ≥1-neighbour subset.
  If it is, the leave-one-drug-out is wrong.
- `n_neighbours_flagged_t / n_neighbours_t` with no shrinkage puts 0 and 1 on the same
  footing as 0.02 and 0.98. Use the shrunk form.
- `ingredient_target_long.csv` expands ChEMBL family and complex targets: metformin carries
  51 respiratory complex I subunits, verapamil 4 calcium-channel subunits, and 609 of 4,280
  ingredients are affected. A gene-sharing neighbour set built without filtering these will
  treat every complex-I drug as a 51-way neighbour hub. Use `component_relationship` and
  `has_family_or_complex_target` to exclude subunit expansions from the gene tier, and say
  what you excluded.
- Absence of a neighbour is not evidence of safety. Pairs with zero neighbours are reported
  as a separate bucket, never imputed to the base rate and then pooled.

## Registration

```python
exp = ln.register(
    agent="exp03",
    title="Same-target neighbour transport across the primary-gene group boundary",
    hypothesis=(
        "A train-only shrunk neighbour flag-rate beats the degree floor on validate "
        "pairs with >=1 neighbour, and the gain decays as similarity loosens from "
        "shared gene to shared protein class."
    ),
    approach=(
        "Three similarity tiers (gene, protein-class leaf, protein-class L1), "
        "empirical-Bayes shrinkage k=10 toward the condition's train rate, "
        "leave-one-drug-out on train rows, HistGBM, AP by neighbour-count bucket."
    ),
    label="y_faers_signal",
    features=[
        "degree",
        "nb_rate_gene",
        "nb_rate_class_leaf",
        "nb_rate_class_L1",
        "nb_rate_primary_gene",
        "nb_rate_action_matched",
    ],
    split="train/validate, grouped by primary target gene",
)
```
