# exp19 — Make the strongest measured effect usable

**Status.** Not started. Grounded in a measurement taken outside the harness (this session,
2026-09-22), reproduced below so it can be checked.

---

## The effect, and the blocker

Pulled 69,191 pChEMBL-valued activities for a 12-target human safety panel from the ChEMBL
REST API into `data/input/drug/offtarget_activities.csv` (hERG CHEMBL240, M1 CHEMBL216,
M3 CHEMBL245, α1A CHEMBL229, H1 CHEMBL231, D2 CHEMBL217, 5-HT2B CHEMBL1833, 5-HT2A CHEMBL224,
β1 CHEMBL213, SERT CHEMBL228, Nav1.5 CHEMBL1980, COX-1 CHEMBL221 — all verified human single
proteins), mapped each off-target to its ADR syndrome conditions
(`experiments/offtarget_syndrome_map.json`, hand-curated, 114 conditions, 4.13% of pairs).

**On train + validate only**, potent binders (pChEMBL ≥ 6) are flagged more for that
off-target's own syndrome on **10 of 10 targets**, ratios 1.39 (SERT) to 3.35 (D2), 8 of 10
with Fisher p < 0.05. Paired **within drug**, which removes drug-level reporting propensity
by construction: flag rate **0.3173** on the drug's own potent syndromes versus **0.1534** on
syndromes it binds weakly — difference **+0.1639**, bootstrap CI [+0.118, +0.212], Wilcoxon
p = 3.6e-09, 89 of 132 drugs positive.

For scale, the entire curated tabular drug side is worth at most +0.0095 `drug_macro_auc`.

**And yet every model design was null**, because the panel is nearly empty:

| coverage | value |
|---|---|
| ingredients with ≥1 panel affinity | 888 of 4,280 |
| with ≥3 / ≥6 / all 12 | 400 / 179 / **0** |
| validate ingredients with any | 235 of 1,284 |
| syndrome-subset pairs with the matching affinity measured | **23.2%** |

Model results: baseline 0.5990 → 0.5983 with 12 drug-level affinity columns → 0.5994 with the
syndrome-interaction feature. On the 114-condition syndrome subset the `p_c` floor is 0.5830
and beat every design (baseline 0.5349, +affinity 0.5417, +interaction 0.5397); affinity alone
ranked at 0.5060.

The signal is real and the matrix is too sparse to carry it. This experiment fills the matrix.

## Question

If the panel is imputed from structure for every ingredient, does the syndrome-interaction
feature convert a +0.164 within-drug association into a ranking gain?

## Hypothesis

QSAR-imputed panel affinities reach held-out Spearman ρ ≥ 0.5 on at least 8 of 12 targets, and
the densified syndrome-interaction feature adds **≥ 0.02 `drug_macro_auc`** over
`degree + p_c` **on the syndrome-condition subset**, measured against that subset's own floor
of 0.5830.

**What would falsify it.** Dense feature, still no ranking gain. Then the within-drug
association is real but not *ordinal* within a drug — knowing a drug binds D2 tells you its
extrapyramidal conditions are elevated, but not enough to order them against its other
conditions. That is a meaningful limit and would close this line.

---

## Method

1. **Structures.** Fetch canonical SMILES for the 39,539 training molecules and our 4,280
   ingredients from the ChEMBL molecule endpoint (allowlisted; batch by `molecule_chembl_id`).
   Report how many ingredients have a usable structure — mixtures, biologics and minerals will
   not, and they are exactly exp17's D4 population.
2. **Descriptors.** ECFP4 (Morgan radius 2, 2048 bits) plus a handful of physicochemical
   descriptors, via RDKit (`manage_packages` install; it is not in the base env).
3. **Twelve single-target QSAR regressors** on pChEMBL, one per off-target. Random forest or
   gradient boosting — no tuning, fixed hyperparameters.
   **Held out by scaffold, not at random** (Bemis–Murcko scaffold split, 80/20): a random
   split over congeneric series inflates QSAR performance badly and would make the imputation
   look better than it is.
   **Exclude our own ingredients from QSAR training entirely.** If an ingredient's own
   activity trains the model that imputes its affinity, the imputed column is a lookup of a
   measured value for some drugs and a prediction for others — two different quantities in one
   feature. Train on the ChEMBL molecules that are not our ingredients; impute for all 4,280.
4. Report per-target held-out Spearman ρ, RMSE, n_train, and an **applicability-domain** flag
   (nearest-neighbour Tanimoto to the training set). Predictions outside the domain are
   emitted with the flag, never silently.
5. **Rebuild the features densely**: 12 imputed affinities, `syn_aff` (max imputed affinity
   over the off-targets whose syndrome contains this condition), `syn_n`, plus a
   measured-versus-imputed indicator per drug.
6. Evaluate with `drug_macro_auc` and `floors.py` on three populations: all conditions, the
   114 syndrome conditions, and syndrome conditions restricted to drugs with a usable
   structure. Compare against the measured-only versions from this session (0.5417 / 0.5397)
   so the value of imputation is isolated.
7. **Sanity check the imputation against the biology**: re-run the within-drug paired test
   using imputed values on the drugs that had no measured panel. If the +0.164 pattern
   reproduces on imputed values, the imputation carries the signal; if it vanishes, the QSAR
   is predicting something else and the feature should not be used.
8. **Widen the ADR map.** H1 matched **zero** conditions because the OMOP/SNOMED condition
   vocabulary lacks symptom-level terms (somnolence, drowsiness, sedation). Expand via the
   condition ancestor hierarchy in `condition_ontology_map.csv` and report the new coverage
   against the current 114 conditions / 4.13% of pairs.

## Budget

| step | est. |
|---|---|
| SMILES fetch + descriptor generation | 4 min |
| 12 QSAR fits with scaffold splits | 5 min |
| imputation + applicability domain | 1.5 min |
| feature build, 3 populations × models + floors | 3 min |
| within-drug sanity check, ADR map widening | 1.5 min |
| **total** | **~14 min** at `timeout=900` |

**If at risk:** cut the panel to the six targets with the largest measured effects (D2, 5-HT2A,
β1, COX-1, α1A, hERG) and log the cut. Never skip step 3's scaffold split or the
ingredient-exclusion rule — both are what make the imputed column honest.

## Deliverables

`<exp_id>_qsar_performance.csv` (per target: ρ, RMSE, n_train, domain coverage),
`data/input/drug/offtarget_panel_imputed.csv` (4,280 × 12 plus flags — a project data asset,
not just an experiment output), `<exp_id>_models.csv`, `<exp_id>_within_drug_recheck.csv`,
`<exp_id>_adr_map_widened.json`, `<exp_id>_qsar_calibration.png`.

## Reporting requirements

`metrics`: `qsar_spearman_median`, `qsar_spearman_min`, `n_targets_above_0.5`,
`ingredients_with_structure`, `drug_macro_auc_syndrome_subset`, `floor_pc_syndrome_subset`,
`increment_over_measured_only`, `within_drug_diff_imputed`, `within_drug_diff_ci_lo`,
`within_drug_diff_ci_hi`, `adr_map_pair_coverage_after`.

`findings` must state QSAR quality first (an imputation that does not predict is not worth
discussing), then the ranking result against the syndrome-subset floor, then whether the
within-drug effect reproduces on imputed values.

## Traps

- **Imputed affinity is a model output and must never be reported as measured.** Carry the
  indicator into every downstream table.
- Scaffold-split ρ will be markedly lower than random-split ρ. The lower number is the honest
  one; do not report both as if comparable.
- The syndrome map is hand-curated by an agent and has not been reviewed by the PI. Treat it
  as provisional and keep it in one file so a review changes one artifact.
- `pChEMBL ≥ 6` is a 1 µM convention, not a fitted threshold. If you vary it, that is a
  separate experiment.
- Do not train QSAR on activities for our own ingredients (step 3), and do not evaluate any
  downstream model on test-fold pairs.

## Registration

```python
exp = ln.register(
    agent="exp19",
    title="QSAR imputation of a 12-target safety panel and the syndrome-interaction feature",
    hypothesis=("Scaffold-split QSAR reaches held-out Spearman >=0.5 on at least 8 of 12 "
                "off-targets, and the densified syndrome-interaction feature adds >=0.02 "
                "drug_macro_auc over degree+p_c on the syndrome-condition subset (own floor "
                "0.5830)."),
    approach=("ECFP4 + physicochemical descriptors; 12 single-target regressors trained on "
              "ChEMBL molecules excluding our ingredients, Bemis-Murcko scaffold split; "
              "applicability-domain flags; dense syndrome-interaction feature evaluated with "
              "floors.py on three populations; within-drug paired test re-run on imputed "
              "values as a biology sanity check; ADR map widened via condition ancestors."),
    label="y_faers_signal",
    features=["offtarget_panel_imputed", "syn_aff", "syn_n", "degree", "p_c"],
    split="train/validate, grouped by primary target gene; QSAR held out by scaffold",
)
```
