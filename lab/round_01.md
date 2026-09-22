# Round 1 — establish the floor and measure the bias structure

Eight independent lanes. None depends on another, so all eight can run in parallel.
Read `AGENT.md` first; it holds the rules and the traps. Every lane registers in the
lab notebook before running and completes afterwards, including nulls.

**Why this round contains no pathway-overlap experiment.** The condition→gene layer
is only half built: the HPO arm is ready (2.24M gene links over 3,322 conditions) but
the Open Targets association pull is still streaming, so disease-arm genes and their
evidence scores are not available yet. That is fine, because the scientifically
correct order is to establish what observation bias alone explains *before* adding
biology. Round 1 sets the number that Round 2's biology has to beat. Lane 5 is the
exception — it carries real biological signal and needs no condition genes at all.

---

## Rules for every lane in this round

**Compute degree features from `train` only.** Drug degree and condition degree are
counts over observed pairs. Computing them over the full table leaks validation and
test information into training features. Fit the counts on train, then apply the same
mapping to validate; unseen conditions get the train median, not zero.

**One validation evaluation per lane, at the end.** Do all selection with grouped
5-fold CV inside train, grouping on `group_key` from `split_assignment.csv`.

**Primary metric: average precision.** Prevalence is 11% for `y_faers_signal`, 0.5%
for `y_semmeddb_treats`, 0.1% for `y_semmeddb_causes` — ROC-AUC will flatter every
model. Report AP first, ROC-AUC second, and a calibration curve.

**Report the subgroup breakdown**, not just the headline: mapped vs unmapped
conditions, disease arm vs phenotype arm, and by `best_match_tier`. A feature that
works only on well-mapped conditions is still useful, but only if you know that.

**Do not touch `test.csv`.**

---

## Lane 1 — Degree-only baseline

**Hypothesis.** Observation frequency alone predicts the FAERS harm label well, and
sets a floor that any biological feature must clear.

**Label.** `y_faers_signal`.

**Features.** Exactly three, all train-derived: drug degree (conditions per
ingredient), condition degree (ingredients per condition), and `record_count` from
`condition_features_basic.csv` (OHDSI network occurrence frequency, which is an
external frequency measure rather than a function of the labels).

**Method.** Logistic regression on log-transformed counts, and LightGBM. Report both;
the gap tells you how much non-linearity the degree terms alone support.

**Deliverable.** The reference AP for the whole project. Every later lane reports its
own AP alongside this number. Also report AP for each feature alone, so we know
whether condition degree or drug degree carries it.

---

## Lane 2 — Condition-intrinsic only

**Hypothesis.** What we know about a condition independent of any drug predicts how
often drugs get flagged for it.

**Label.** `y_faers_signal`.

**Features.** `condition_features_basic.csv`: `record_count`, `concept_class_id`
(Disorder vs Clinical Finding), `n_omop_ancestors`, `arm`, `is_mapped`,
`best_match_tier`, `n_ontology_terms`, `n_hpo_genes`, `n_groups`; plus one-hot organ
system and therapeutic area from `condition_group_long.csv`. No drug features, no
drug degree.

**Watch for.** `n_hpo_genes` has a median of 133 for conditions that have any, because
HPO annotations are propagated up the hierarchy — a general term inherits every gene
beneath it. High gene count therefore means *low specificity*, roughly the opposite of
informative. Consider an IDF-style weighting and report whether it helps.

---

## Lane 3 — Drug-intrinsic only

**Hypothesis.** Some drugs are intrinsically more likely to be flagged, independent of
the condition, and drug chemistry and mechanism class capture it.

**Label.** `y_faers_signal`.

**Features.** `ingredient_features.csv` — mechanism class, target class, `max_phase`,
`first_approval`, physicochemical properties, ChEMBL warning flags. **Exclude
`ingredient_features_label_adjacent.csv` entirely.** Also exclude `in_cem_list` and
any column the data dictionary marks as label-derived; check `data_dictionary.csv`
before selecting.

**Watch for.** 2,932 of 4,280 ingredients match a ChEMBL molecule and only 1,716 have
a mechanism with an assigned target, so missingness is large and is itself
informative — an unmatched ingredient is usually an older or non-drug substance.
Model missingness explicitly rather than imputing it away, and report how much of the
lane's performance is the missingness indicator.

---

## Lane 4 — Drug + condition, no interaction

**Hypothesis.** Concatenating the two sides without any pair-level feature is enough;
the interaction terms Round 2 will build add nothing.

**Label.** `y_faers_signal`.

**Features.** Union of lanes 2 and 3 plus the degree terms from lane 1.

**Why it matters.** This is the null hypothesis for the entire project. If lane 4
matches what Round 2's pair-level biology achieves, the bridge hypothesis is not
supported and we would need to say so. State the result plainly either way.

---

## Lane 5 — Same-target neighbour signal

**Hypothesis.** Drugs that share a target behave alike. For a pair (drug *d*,
condition *c*), the rate at which *other* drugs hitting *d*'s targets are flagged for
*c* predicts whether *d* is flagged for *c*.

This is the project's core claim in its simplest testable form, and it needs no
condition-side genes — only `ingredient_target_long.csv` and the training labels.

**Label.** `y_faers_signal`.

**Features.** For each pair, computed **strictly within train**:
- number of other train drugs sharing ≥1 target gene with *d*
- of those, the fraction flagged for *c* (the neighbour rate)
- the same restricted to shared *primary* target, and to matching `action_type`
- shrunk toward the condition's overall train rate, so a neighbour rate from 2 drugs
  is not treated like one from 40 (empirical-Bayes or a simple additive prior;
  state which)

**Critical.** The split groups ingredients by primary target gene, so within-train
neighbours of a validation drug will usually *not* share its primary gene. That is
the point: this lane measures whether target similarity transports across the group
boundary. Report the neighbour-count distribution for validation pairs — if most have
zero neighbours, the headline AP is uninformative and the conditional AP given ≥1
neighbour is the real result.

**Also deliver.** AP as a function of neighbour count (0, 1–2, 3–5, 6+). This is the
transportability decay curve the project ultimately cares about.

---

## Lane 6 — Disproportionality residual as the target

**Hypothesis.** The raw FAERS signal is dominated by reporting frequency. Modelling
the *residual* after removing drug and condition reporting propensity yields a target
closer to a causal quantity, and biology predicts the residual better than it predicts
the raw flag.

**Label.** Continuous. Fit `log(faers_prr)` on train against drug and condition random
effects (or fixed effects on log degree), then take the residual as the target.

**Features.** Lane 3's drug features plus lane 2's condition features.

**Deliverable.** Spearman correlation on validate, plus a direct comparison: does the
same feature set predict the residual better than the raw binary flag? If yes, later
rounds should switch target. Restrict to pairs with `faers_case_count >= 3` so the PRR
is not built on one or two reports, and say how many pairs that leaves.

---

## Lane 7 — The efficacy label

**Hypothesis.** `y_semmeddb_treats` is predictable from drug mechanism and condition
biology, despite being 20× rarer than the harm label.

**Label.** `y_semmeddb_treats` (0.5% prevalence, ~7,900 positives total).

**Features.** Lanes 2 + 3 + 5.

**Watch for.** At this prevalence, a single grouped 5-fold CV estimate is noisy;
repeat with several seeds and report the spread rather than one number. Literature
assertions also track publication attention, so drug age and `first_approval` may act
as proxies for how much anyone has written about the drug — include them and report
their importance separately, since a model that has merely learned "well-studied drug"
is not an efficacy model.

**Why this lane matters disproportionately.** With the EU label layer empty, this is
the only efficacy signal in the dataset. Whether it is learnable at all determines
whether the project's efficacy half is viable before we invest in a new indication
source.

---

## Lane 8 — Label-definition sensitivity

**Hypothesis.** The headline results are an artefact of the Evans thresholds.

**Method.** No new features. Re-run lane 1 and lane 3 across label variants: Evans as
specified (PRR ≥ 2, χ² ≥ 4, cases ≥ 3); PRR ≥ 2 alone; PRR ≥ 4 with cases ≥ 5;
case-count ≥ 10 regardless of PRR; and the SemMedDB harm label. Also fit with
`y_any_harm`.

**Deliverable.** A table of AP by label definition, and a statement of how much the
project's conclusions depend on that choice. If rankings between feature sets are
stable across definitions, later rounds can stop worrying about it; if they flip, every
subsequent result needs the sensitivity reported alongside it.

---

## What Round 2 depends on

The Open Targets association pull, which gives disease-arm genes with per-datatype
evidence scores. Once that lands I will add: pathway overlap between drug target genes
and condition genes, shortest-path features over the drug→target→gene→pathway→condition
graph, genetic-evidence-weighted overlap, and the relation-aware GNN — each of which
has to beat whichever of lanes 4 and 5 wins here.

Round 2's design will also depend on what Round 1 finds. If lane 1 is already close to
lane 4, the interesting question shifts from "which features help" to "why is so much
of this label explained by observation frequency", and the next round should attack the
label rather than the features.
