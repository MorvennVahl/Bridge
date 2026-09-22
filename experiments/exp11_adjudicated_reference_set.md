# exp11 — A yardstick that works

**Status.** Not started. Blocking: the success criterion in `METRIC.md` has no usable
ceiling after exp06, and every "is this good?" question in the project routes through it.

---

## Why exp06's yardstick failed

| quantity | value | n drugs |
|---|---|---|
| ceiling: SemMedDB harm ranking `y_faers_signal` | 0.4990 (CI 0.4858–0.5155) | **13** |
| incumbent: `log(faers_prr)` ranking SemMedDB harm | 0.5042 | — |
| floor on the SemMedDB-harm target (`p_c` lookup) | **0.6149** | — |

The floor exceeds the ceiling, both candidate instruments sit at chance, and the estimate
rests on 13 drugs. Two evidence streams that should partially agree do not agree at all at
this coverage — 1,369 harm pairs over 418 drugs, most contributing a single positive.

And the benefit-direction number (1.0000) was definitional: `y_semmeddb_treats` **is**
TREATS/PREVENTS, scored against the sentence counts it was derived from. It must not be
cited, and no future experiment may use a SemMedDB column to evaluate a SemMedDB-derived
label.

## Question

Against an externally adjudicated reference set — real positives and real negatives, curated
by people rather than inferred from absence — how good are the scorers we already have, and
what is the ceiling that defines success?

## Hypothesis

On an adjudicated set, exp07's intrinsic union and exp08's neighbour model both exceed the
`p_c` floor computed **on that set**, and the ceiling (agreement between two independent
adjudicated sources, or the incumbent disproportionality method's own performance) lands in
0.65–0.85 — a range that makes the provisional 0.65 target meaningful rather than arbitrary.

**What would falsify it.** Our scorers at or below the reference-set floor. That would mean
the gains measured on CEM labels are gains at predicting *CEM's observation process*, not
drug safety — and the project would need to be re-grounded on calibrated effect estimates
(the OHDSI route in `DESIGN.md`) before any more feature work.

---

## Part A — the fetch (Claude Code, not Modal)

This container has no unrestricted network. Hand off:

**Deliverable:** `data/input/reference/reference_set.csv` with columns
`ingredient_concept_id, condition_concept_id, label, source, adjudication, notes`, plus
`reference_set_provenance.json` recording source URLs, versions and the mapping method.

**Candidate sources, in priority order.**

1. **OMOP reference set** (Ryan et al. 2013, *Drug Saf*) — the standard pharmacovigilance
   benchmark: adjudicated positive and negative drug–outcome pairs across four outcomes
   (acute liver injury, acute MI, acute renal failure, upper GI bleed). Negative controls are
   the valuable half; this project has never had real negatives.
2. **EU-ADR reference set** (Coloma et al. 2013) — 10 adverse events with adjudicated pairs.
3. **OHDSI LEGEND / empirical-calibration negative-control sets** — large negative-control
   lists, usable for calibration even where positives are thin.
4. **SIDER 4.1** — label-derived indications and side effects. Not adjudicated, but it is a
   genuine *label* source and would simultaneously repair the missing indication layer
   (`in_eu_label` is false on all 1.45M rows).

**Mapping requirement.** Sources use RxNorm/ATC and MedDRA/SNOMED; both sides must land on
`ingredient_concept_id` and `condition_concept_id`. Record the mapping tier per row as
`condition_ontology_map.csv` does, and report unmapped counts — do not drop them silently.

**Report coverage** against our 4,276 × 5,631 grid: pairs retained, drugs retained,
conditions retained, and how many retained drugs fall in validate versus train.

## Part B — the evaluation (Modal, once the file exists)

1. Precondition: reference file present, non-empty, both label classes present, and at least
   30 drugs with ≥5 reference pairs — otherwise complete with `failed=True` naming the
   shortfall. Do not produce a ceiling from 13 drugs again.
2. Compute, on reference pairs only: the `p_c` floor **on this set**, the incumbent FAERS
   score, exp07's union model, exp08's neighbour model, and exp09's residual model. All five
   scored with `drug_macro_auc` plus `drug_macro_p10`, with 1,000-resample bootstrap CIs over
   drugs.
3. Because the reference set has real negatives, also report **pooled** ROC-AUC and AP — the
   one place in this project where pooled metrics are meaningful, since the negative class is
   verified rather than merely unobserved.
4. Emit `success_criterion.json` v2 with `source: "adjudicated"`, `floor`, `ceiling`,
   `solved_threshold`, `n_drugs`, and the CI. Later experiments read this file.
5. Report the correlation between each model's reference-set rank and its CEM-label rank. A
   model that ranks CEM well and the reference set poorly is learning the observation
   process, and that gap is the number to quote.

## Budget

Part B only: no fitting beyond re-scoring saved models, so ~4 min. Part A is not on the
clock — it is a separate session's work.

## Deliverables

`<exp_id>_reference_eval.csv`, `<exp_id>_per_drug.csv`, `<exp_id>_coverage.csv`,
`success_criterion.json` (v2), `<exp_id>_cem_vs_reference_rank.csv`.

## Reporting requirements

`metrics`: `reference_drug_macro_auc_union`, `reference_drug_macro_auc_neighbour`,
`reference_floor_drug_macro_auc`, `reference_pooled_auc`, `reference_pooled_ap`,
`incumbent_reference_drug_macro_auc`, `ceiling`, `solved_threshold`, `n_drugs`,
`n_reference_pairs`.

`findings` must state the floor on the reference set first (so nobody repeats exp09's
mistake of borrowing a floor from another population), then each model against it, then the
revised `solved_threshold`, then the CEM-versus-reference rank gap.

## Traps

- **Reference sets are small and outcome-restricted.** Four outcomes in the OMOP set means
  conditions are not a random sample; report which conditions and do not generalise past
  them.
- Negative controls are "believed to have no causal relation", not "verified safe". They are
  far better than absence, but they carry their own construction assumptions — record the
  source's definition verbatim in `provenance.json`.
- Do not fit anything on the reference set. It is the yardstick; fitting on it destroys the
  only external check the project has.

## Registration

```python
exp = ln.register(
    agent="exp11",
    title="Adjudicated reference-set evaluation and a usable success criterion",
    hypothesis=("On an externally adjudicated set with real negatives, the intrinsic union "
                "and neighbour models exceed the p_c floor computed on that set, and the "
                "ceiling lands in 0.65-0.85."),
    approach=("Claude Code fetches and maps the OMOP/EU-ADR/OHDSI reference sets to OMOP "
              "concept ids; Modal re-scores the existing models on reference pairs with "
              "per-drug and pooled metrics, bootstrap CIs, and emits success_criterion.json "
              "v2."),
    label="reference_set_label",
    features=["p_c", "faers_prr", "intrinsic_union", "nb_excess_gene", "residual_model"],
    split="reference pairs, no fitting",
)
```
