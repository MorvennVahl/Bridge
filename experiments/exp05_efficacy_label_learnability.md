# exp05 — Is the efficacy half of the project viable at all?

**Lineage.** `lab/round_01.md` lane L7, with one addition that changes what the experiment
means: the feature set is split into an **indication-encoding block** and a **mechanism
block**, and the headline is the mechanism-only number (see *The tautology problem*).

**Status.** Not started. Independent of the other four.

---

## Question

`y_semmeddb_treats` is the only efficacy signal in the dataset — 3,926 train positives,
0.54% prevalence. Is it predictable from biology at all, and is any of that predictability
transportable to a molecule with no literature and no ATC code?

## Hypothesis

`y_semmeddb_treats` is predictable well above prevalence from **mechanism and target
features alone** (no ATC, no indication class, no drug-age proxy): validate AP ≥ 3× the
0.48% validate prevalence. And the gap between the mechanism-only model and the
everything-included model measures how much of the apparent efficacy signal is really
"well-studied drug in a known therapeutic area".

**What would falsify it.** If mechanism-only AP sits at prevalence while the full model
looks strong, the efficacy label is learnable only through features a novel molecule does
not have, and the project's efficacy half is **not viable on this label**. That is a
decision-grade negative result: it says buy an indication layer (DailyMed, the OHDSI
indication set) before spending any more modelling effort on the efficacy side, which
`AGENT.md` §4 already flags as the single most valuable thing available.

## The tautology problem

ATC codes, `indication_class`, `kegg_efficacy` and `usan_stem_definition` encode a drug's
therapeutic indication **by construction**. Predicting TREATS from ATC L1 is close to
reading the label out of the feature. It is not cheating — for a drug with an assigned ATC
code it is a legitimate and strong predictor — but it is **not transportable** to the case
the project exists to serve: a new molecule that has a target and no ATC code, no USAN stem
and no literature.

So this experiment reports two numbers and the headline is the smaller one:

| block | contents | transportable to a novel molecule? |
|---|---|---|
| **M** mechanism | `mechanism` and `target biology` blocks, `chemistry`, `exposure`, `metabolism`, target classes, condition-intrinsic features, degree | yes |
| **I** indication-encoding | `atc_l1`, `n_atc_codes`, `indication_class`, `kegg_efficacy`, `usan_stem_definition`, `max_phase`, `first_approval`, `therapeutic_flag` | no |

`first_approval` and `max_phase` sit in **I** deliberately: literature assertions track
publication attention, so drug age is a proxy for how much anyone has written about the
drug, and a model that has learned "well-studied old drug" is not an efficacy model.

---

## Inputs

Same as exp02: `/splits/train.csv`, `/splits/validate.csv`, `/drug/ingredient_features.csv`,
`/condition/condition_features_basic.csv`, `/condition/condition_group_long.csv`,
`/data_dictionary.csv`. Plus `/drug/ingredient_target_long.csv` if you add the exp03
neighbour feature (step 5).

Positives: **3,926** in train (0.54%), **2,092** in validate (0.48%). `y_semmeddb_causes`
(715 / 370) is carried as a secondary target in the same run — it is the directional harm
label and costs one extra fit — but with that few positives, treat its numbers as
indicative, not as a result.

---

## Method

1. Precondition check: positive counts above, exactly.
2. Build three designs: **M**, **I**, and **M+I**. Column assignment is by the table above
   and the `block` column of the data dictionary; write the resolved column lists out as an
   artifact so the split is auditable rather than described.
3. `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
   min_samples_leaf=50, early_stopping=False, random_state=<seed>)`. Shallower and more
   regularised than the other experiments because there are 3,926 positives, not 79,554.
   **No class weighting and no resampling** — both distort calibration, and AP is
   rank-based so neither is needed for the metric. If you want to try one, register it
   separately.
4. **Repeat with 5 seeds.** At this prevalence a single grouped 3-fold CV estimate is
   noisy; the deliverable is mean AP with its across-seed spread, and the M-vs-M+I
   difference is only claimed if it exceeds that spread. This is the one place in the round
   where repetition matters more than extra features.
5. **Optional, only if step 4 leaves time:** add exp03's gene-tier neighbour rate computed
   against the TREATS label (do other drugs hitting my target treat this condition?).
   Mechanistically this is the most promising single feature for efficacy — shared-target
   drugs share indications far more tightly than they share adverse events. Register it as
   part of this experiment's approach if you include it; skip it silently if the budget is
   gone, and say so.
6. Report, separately, the importance of `first_approval` and the ATC features in M+I. If
   they dominate, state it in the headline of `findings`, not in a footnote.
7. Subgroup report: AP by `arm` and `is_mapped`, and AP restricted to the 1,716 ingredients
   with an assigned mechanism target — the population where an efficacy model would actually
   be deployed.

## Budget

| step | est. |
|---|---|
| read + join | 90 s |
| 3 designs × 5 seeds × (1 train fit + 1 validate pass), 300 iters at 31 leaves ≈ 12 s each | ~3.5 min |
| grouped 3-fold CV for the two headline designs (M, M+I), 1 seed | ~1.5 min |
| secondary target `y_semmeddb_causes`, M and M+I, 1 seed | 30 s |
| figures + tables | 60 s |
| **total** | **~7 min** |

`timeout=600`, `cpu=8.0`, `memory=16384`.

**If the budget is at risk:** cut to 3 seeds before cutting anything else, then drop the
`y_semmeddb_causes` secondary. Do not drop the M/M+I separation — without it the experiment
answers a question nobody asked.

## Deliverables

- `<exp_id>_designs.csv` — per design × seed: validate AP, ROC-AUC, prevalence, AP/prevalence
  lift; plus mean and spread across seeds
- `<exp_id>_column_assignment.csv` — which column went into M and which into I
- `<exp_id>_importance_top25.csv` — per design, with `first_approval` and ATC rows flagged
- `<exp_id>_pr_curves.png` — precision–recall curves on validate for M, I, M+I, with the
  prevalence line drawn
- `<exp_id>_calibration.png` — reliability curve for M and M+I; at 0.5% prevalence,
  calibration is where these models usually fail
- `<exp_id>_subgroups.csv` — step 7

## Reporting requirements

`metrics`: `validate_average_precision` (design **M** — the transportable number, and the
one the leaderboard should carry), `validate_average_precision_full` (M+I),
`validate_average_precision_indication_only` (I), `train_cv_average_precision`,
`ap_seed_spread`, `prevalence_validate`, and the same for `y_semmeddb_causes` under
`*_causes` keys.

`findings` must answer the viability question in its first sentence — is the efficacy half
of this project workable on this label, yes or no — then give the M vs M+I gap against the
seed spread, then the importance of drug age. Then state the recommendation on whether to
invest in a real indication layer. This experiment exists to inform that decision; leaving
it unstated wastes the run.

## Traps specific to this experiment

- **0.54% prevalence.** ROC-AUC will look excellent and mean nothing. AP first, always with
  the prevalence next to it.
- SemMedDB is literature-mined, so absence is even weaker evidence than in FAERS: a true
  indication nobody wrote a mineable sentence about is a negative in this label. Do not
  describe this as an efficacy ground truth anywhere in the report — it is a proxy for
  published assertion.
- 6,061 TREATS + 1,842 PREVENTS assertions exist across the whole 1.45M-row file;
  `y_semmeddb_treats` pools them. Do not split them at this prevalence.
- The NEG_* predicates are counted separately upstream and are never positives; do not try
  to recover them as negatives.
- Condition degree predicts this label too, for the same reporting reasons — keep the degree
  terms in **M** so the biological claim is made on top of them, not instead of them.

## Registration

```python
exp = ln.register(
    agent="exp05",
    title="Efficacy label learnability: mechanism features vs indication-encoding features",
    hypothesis=(
        "y_semmeddb_treats is predictable above 3x prevalence from mechanism and "
        "target features alone, and the gap to a model including ATC, indication "
        "class and drug age measures how much of the signal is 'well-studied drug' "
        "rather than biology."
    ),
    approach=(
        "Three designs M / I / M+I with an audited column assignment, HistGBM at 31 "
        "leaves, 5 seeds for the AP spread, grouped 3-fold CV in train, one validate "
        "pass per design-seed. y_semmeddb_causes carried as an indicative secondary."
    ),
    label="y_semmeddb_treats",
    features=[
        "mechanism_block",
        "target_biology_block",
        "condition_intrinsic",
        "degree",
        "indication_encoding_block",
    ],
    split="train/validate, grouped by primary target gene",
)
```
