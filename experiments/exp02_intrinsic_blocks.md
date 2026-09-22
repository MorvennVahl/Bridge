# exp02 — Does either side's intrinsic biology add anything on top of degree?

**Lineage.** `lab/round_01.md` lanes L2 (condition-intrinsic), L3 (drug-intrinsic) and L4
(both sides, no interaction), run as one nested ablation. Separate lanes would refit the
same rows four times over in four containers; nested here, the increments are directly
comparable because they share one CV partition and one validate pass.

**Status.** Not started. Independent of the other four.

---

## Question

Once observation frequency is accounted for, does anything we know about the drug, or
anything we know about the condition, improve prediction of the FAERS harm signal — without
any pair-level interaction term?

## Hypothesis

Adding drug-intrinsic features to the degree baseline improves validate AP by **at least
0.02 absolute**, adding condition-intrinsic features adds a further increment, and the
union of both without any interaction term **does not reach** what pair-level biology will
achieve in Round 2.

**What would falsify it, and why that matters more than usual.** This experiment is the
project's null hypothesis in its concrete form. Three distinguishable outcomes:

- **Both increments ≈ 0.** The label is observation frequency and little else. The project
  would then have to attack the label (exp04's direction) rather than the features, and
  Round 2's pathway work would be premature.
- **Increments are real and large.** Good, but it raises the bar: the union model here is
  what pathway overlap and the graph model have to beat in Round 2, not the degree floor.
- **Most of the drug-side increment is the ChEMBL-missingness indicator.** Then the model
  has learned "this is an old non-drug substance", not pharmacology, and the increment is
  not transportable to a new molecule — which is the only thing the project cares about.
  This is why step 6 is not optional.

---

## Inputs

| path on the volume | rows × cols | use |
|---|---|---|
| `/splits/train.csv` | 723,586 | fit |
| `/splits/validate.csv` | 434,151 | one evaluation |
| `/drug/ingredient_features.csv` | 4,280 × 159 | drug block |
| `/condition/condition_features_basic.csv` | 5,631 × 12 | condition block |
| `/condition/condition_group_long.csv` | ~10,554 | one-hot organ system / therapeutic area |
| `/data_dictionary.csv` | 245 | select columns by `block`, do not hand-pick |

Join keys: `ingredient_concept_id` = `omop_concept_id`; `condition_concept_id`.

---

## Feature blocks

**Degree (3).** As exp01, train-derived, `log1p`.

**Drug-intrinsic (~130).** From `ingredient_features.csv`, take blocks `development`,
`safety`, `pharmacology`, `chemistry`, `exposure`, `metabolism`, `mechanism`,
`target biology`, `interactions` — selected by the `block` column of the data dictionary.
Then remove, explicitly:

- the entire `identity` block (18 cols), which includes `in_cem_list`,
  `in_indication_roster`, `roster_name`, `indication` and every `chembl_fuzzy_*` column
- free-text and high-cardinality string columns that are lists, not categories:
  `target_chembl_ids`, `target_genes`, `target_gene_symbols`, `mechanisms_of_action`,
  `action_types`, `warning_types`, `warning_classes`, `atc_codes`, `atc_l3`,
  `chembl_metab_enzymes`, `chembl_metabolite_names`, `kegg_metab_enzymes`,
  `kegg_transporters`, `kegg_interaction_genes`, `reactome_top_level_terms`,
  `target_safety_events`, `usan_stem_definition`, `indication_class`

  Keep the `n_*` counts derived from them — that is what those counts are for. If you want
  a categorical from a list column, derive `atc_l1` (already a column) rather than
  exploding the list.
- `ingredient_features_label_adjacent.csv` is not read at all, not even for the nuisance
  check; that belongs to a bias analysis, not to a predictive ablation.

Add one explicit indicator: `has_chembl_match` (`chembl_id` non-null). Missingness here is
large — 2,932 of 4,280 ingredients match a ChEMBL molecule and only 1,716 have a mechanism
with an assigned target — and it is informative, so model it rather than imputing it away.
Let the boosting model handle NaN natively; do not mean-impute.

**Condition-intrinsic (~40).** `record_count` (`log1p`), `concept_class_id` (Disorder vs
Clinical Finding), `n_omop_ancestors`, `arm`, `is_mapped`, `best_match_tier`,
`n_ontology_terms`, `n_hpo_genes`, `n_groups`, plus one-hot over the 23 HPO organ systems
and the Open Targets therapeutic areas from `condition_group_long.csv`.

`n_hpo_genes` needs care: the median is 133 among conditions that have any, because HPO
annotations propagate up the hierarchy, so a general term inherits every gene beneath it. A
high gene count therefore means **low specificity** — roughly the opposite of informative.
Include both the raw count and an IDF-style transform (`log(n_conditions / n_conditions
carrying the gene)` summed over the condition's genes, precomputable from the counts
already in the basic table), and report which one the model used.

---

## Method

1. Precondition check: every input path exists, row counts match the table above, and
   train positive rate for `y_faers_signal` is 0.1100 ± 0.001.
2. Build four nested designs: `D` (degree), `D+drug`, `D+cond`, `D+drug+cond`.
3. One fixed model for all four: `HistGradientBoostingClassifier(max_iter=200,
   learning_rate=0.06, max_leaf_nodes=63, early_stopping=False, categorical_features=...)`.
   **No hyperparameter search** — the comparison is between feature sets, and a search
   inside each would spend the budget re-discovering that boosting is robust to these
   settings. Fix `random_state=0`.
4. Grouped 3-fold CV inside train on `group_key` → mean and spread per design. Then one
   validate pass per design (four scorings, the only validate contact).
5. Report the increments as differences with the CV fold spread attached, so a +0.004
   increment is not read as an effect when the fold-to-fold spread is 0.01.
6. **Missingness attribution for the drug block.** Refit `D+drug` on the 1,716 ingredients
   that have an assigned mechanism target only (restricting rows, not columns) and report AP
   on the corresponding validate subset. If the block's increment survives there, it is
   pharmacology; if it collapses, the increment was the missingness indicator. State which.
7. Gain importances, top 25 per design, written out. Not permutation importance — it costs
   a full rescoring per feature and will not fit the budget.

## Budget

| step | est. |
|---|---|
| read + join + one-hot | 90 s |
| 4 designs × (3 CV + 1 full) fits, ≤160 cols, 17 s each | ~4.5 min |
| step 6 restricted refit | 40 s |
| scoring, importances, figures | 60 s |
| **total** | **~8 min** |

`timeout=600`, `cpu=8.0`, `memory=16384`. This is the tightest of the five.

**If the budget is at risk:** drop CV to 2 folds for `D+drug+cond` and keep 3 for the rest,
or drop step 7. Do **not** drop step 6 — the ablation's headline is not interpretable
without it. Record whichever fallback you took.

## Deliverables

- `<exp_id>_ablation.csv` — one row per design: n features, train CV AP (mean, min, max),
  validate AP, validate ROC-AUC, increment over the previous rung
- `<exp_id>_importance_top25.csv` — gain importances per design
- `<exp_id>_subgroups.csv` — validate AP per design by `is_mapped`, `arm`,
  `best_match_tier`, and by `has_chembl_match`
- `<exp_id>_calibration.png` — reliability curves, all four designs on one axis
- `<exp_id>_missingness_attribution.csv` — step 6

## Reporting requirements

`metrics`: `validate_average_precision` (union design), `train_cv_average_precision`,
`baseline_degree_only_ap`, `ap_drug_only_increment`, `ap_condition_only_increment`,
`ap_union`, `ap_union_mechanism_subset`.

`findings` must answer, in order: how big is each increment against the fold spread; where
does it concentrate (which subgroup); how much of the drug-side increment is missingness
rather than biology; and what this implies for whether Round 2's pair-level features are
worth building. Say the last part plainly — if the answer is "degree plus intrinsics is
essentially all of it", that is the most valuable sentence this round can produce.

## Traps specific to this experiment

- One-hot over `best_match_tier` and `atc_l1` is fine; one-hot over anything with hundreds
  of levels will blow both memory and the budget. Cap categorical cardinality at 30 levels
  and bucket the tail as `other`.
- Do not let the condition block silently include a CEM-derived count. Everything in
  `condition_features_basic.csv` is external except the columns derived from our own
  mapping, which are about coverage, not labels.
- The union design is *not* the pair-level model. It has no term that depends on the drug
  and the condition jointly. Keep it that way — its whole purpose is to be the honest null
  for Round 2.

## Registration

```python
exp = ln.register(
    agent="exp02",
    title="Nested ablation: degree, drug-intrinsic, condition-intrinsic, union",
    hypothesis=(
        "Drug-intrinsic features add >=0.02 AP over the degree baseline, condition-"
        "intrinsic features add a further increment, and the no-interaction union "
        "falls short of pair-level biology."
    ),
    approach=(
        "Four nested feature designs, one fixed HistGBM, grouped 3-fold CV in train, "
        "one validate pass per design, plus a mechanism-subset refit to separate "
        "ChEMBL missingness from pharmacology."
    ),
    label="y_faers_signal",
    features=["degree", "drug_intrinsic", "condition_intrinsic", "union"],
    split="train/validate, grouped by primary target gene",
)
```
