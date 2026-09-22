# Bridge — design note

Status: reference data landed on `main`; the condition feature layer is built for both
halves of the condition set; model layer not started. The condition-to-OMOP join is blocked
on a vocabulary export (see [docs/vocab-export-spec.md](docs/vocab-export-spec.md)).

## The goal, as I understand it

You want a function that takes **what we know about a drug from biology** — its
targets, the pathways those targets sit in, mechanism class, chemistry — and
**what we know about a condition** — the genes and pathways implicated in it, where
it sits in the disease hierarchy, its phenotype profile — and returns, for a
drug–condition pair that **has no real-world data of its own**, an estimate of

1. whether the drug will *help* that condition (efficacy / "will it be successful"), and
2. whether it will *cause* it (safety signal).

The training signal comes from pairs where RWD *does* exist. The biology is what
makes the model transportable to a drug that has never been prescribed: a new
molecule has no FAERS reports, but it has a target, and that target has neighbours
whose RWD behaviour we have already observed.

That framing is a **link prediction / edge-labelling problem on a heterogeneous
graph**, which is why a GNN is a reasonable instinct. See "On the GNN question" below.

## What is actually in the CEM file

`cem_ingredient_condition_associations.csv` — 1,447,172 rows,
4,276 ingredients × 5,631 conditions (~6% of the 24.1M possible pairs).

| evidence source | pairs | note |
|---|---|---|
| FAERS only | 1,436,778 | 99.3% of the file |
| SemMedDB only | 9,908 | literature assertions |
| FAERS + SemMedDB | 486 | |
| EU label | **0** | `in_eu_label` is false on every row; `eu_label_count` is null on all 1.45M rows |

FAERS disproportionality is mostly noise at the row level: median case count 2,
median PRR 1.36; the 90th percentile PRR is 10.3. `faers_ror` is identical to
`faers_prr` in this extract.

SemMedDB carries the only **directional** signal, and it is thin:

- efficacy-direction: TREATS 6,061 · PREVENTS 1,842
- harm-direction: CAUSES 1,123 · PREDISPOSES 260 · COMPLICATES 23
- non-directional: AFFECTS 1,865 · ASSOCIATED_WITH 985 · DISRUPTS 686
- explicit negations: NEG_* 371 total

**Consequence for the plan.** As it stands this file is a safety-signal table with a
literature veneer. The indication side — the thing you need to learn "will this drug
work" — is represented by ~7,900 SemMedDB assertions and nothing else. Two fixes,
both worth doing:

- pull the label-indication layer back in (the EU label columns are present but
  unpopulated in this extract; RxNorm/DailyMed or the OHDSI indication set are
  alternatives), and
- bring in the effect-estimate layer you mentioned (OHDSI-style calibrated estimates
  with confidence intervals) as the high-quality label subset.

The top conditions by degree are exactly what a reporting-bias-driven table
predicts — Respiratory finding, Nausea, Pain, Fever, Vomiting, Dizziness. Any model
trained on raw FAERS flags will spend most of its capacity learning "this event gets
reported a lot", not "this drug causes this event". Condition-side and drug-side
degree have to be in the model as nuisance terms, or the labels have to be
disproportionality *residuals* rather than raw flags.

## Proposed graph

Nodes, with the source for each:

- **Ingredient** (4,276, OMOP RxNorm) — have it
- **Target / protein** — ChEMBL mechanisms, drug→target with action type
- **Gene** — target→gene identity
- **Pathway** — Reactome, gene→pathway
- **Condition** (5,631, OMOP) — have the ids; need features
- **Higher-level disease category** — SNOMED/OMOP `concept_ancestor`, or MONDO/EFO after mapping

Edges:

- ingredient →`has_mechanism`→ target (ChEMBL, with agonist/antagonist/inhibitor direction)
- target →`encoded_by`→ gene →`member_of`→ pathway
- gene →`associated_with`→ condition, **weighted by genetic evidence** (Open Targets
  association scores; this is the edge that carries most of the causal information).
  Now present: `data/ref/ot/association_by_datatype_indirect__filtered.parquet`, 4.5M
  scored edges over 2,160 diseases, broken down by evidence type. See below.
- condition →`is_a`→ condition (hierarchy; gives the model a prior that sibling
  conditions behave alike)
- ingredient —`RWD`— condition: **the label edge**, not an input feature

The condition-side features are now built — genes, hierarchy, therapeutic area and a
phenotype profile — but keyed on MONDO and HPO ids rather than `condition_concept_id`.
The section below covers what that mapping costs.

## On the GNN question

A GNN is the right *eventual* shape but the wrong *first* model, for a specific
reason: the value you want from the graph is the path

> new drug → its target → gene → pathway → condition

and that path is only two or three hops. A GNN with 2–3 message-passing layers over
this graph is, in effect, learning a weighted version of features you can compute
explicitly and cheaply — "does my target's pathway overlap this condition's
implicated pathways", "what happened in RWD to other drugs hitting this same
target". Compute those as **explicit path-count and pathway-overlap features**, fit
gradient boosting on them, and you get a baseline that is interpretable, trains in
minutes, and tells you whether the graph carries signal at all.

Then a GNN earns its place if it beats that baseline — and the specific reason it
might is that it can learn *which* relation types matter (R-GCN / HGT style
relation-specific weights), rather than you deciding that pathway overlap is the
feature that counts.

Three requirements on any version of this, GNN or not:

1. **Split by drug, not by pair.** The question is "a drug not in RWD", so
   evaluation must hold out entire ingredients — and ideally entire target families.
   A random split over 1.45M pairs will look excellent and mean nothing, because the
   same drug appears in train and test with different conditions.
2. **Inductive, not transductive.** No learned embedding per drug node; a new drug
   must be representable from its features alone. This rules out node2vec-style
   embeddings and argues for feature-initialised message passing.
3. **Absence is not evidence of safety.** A drug–condition pair missing from the file
   can mean "no effect" or "nobody ever prescribed it" or "nobody reported it".
   Negative sampling has to respect that, and negative *controls* (pairs believed to
   have no causal relation) are the standard OHDSI device for calibrating it.

## The condition set is bimodal, and the mapping is the hard part

The plan above assumes every condition can be attached to Open Targets genes and pathways.
Measured against the reference data, it cannot. Exact lowercase name matching of the 5,631
condition names reaches 21.2% of Open Targets disease labels and 19.6% of MONDO labels,
24.7% for the union.

The reason matters more than the number. The unmatched set is dominated by clinical
findings and symptoms — `abdominal bloating`, `abdominal tenderness`, `abnormal breath
sounds` — which are not diseases and will never appear in MONDO. They are HPO phenotypes.
Cases like `aarskog syndrome` are the opposite failure: real MONDO diseases that miss on
name but would hit on code.

So the condition side splits in two, and is built that way in `src/bridge/`:

| half | source | module | terms | carry an OMOP-reachable code |
|---|---|---|---|---|
| disease | MONDO SSSOM + Open Targets | `bridge.disease` | 36,498 | 26,754 (73.3%) |
| symptom | HPO + Open Targets HPO table | `bridge.hpo` | 20,482 | 11,455 (55.9%), but only 3,443 (16.8%) via SNOMED |

`disease_phenotypes` links the two over 7,585 diseases, so a disease inherits its
phenotype profile and a phenotype reaches disease-level genes.

### What the data does not give us

- **UMLS is the richest key and we cannot use it.** 22,375 disease terms (61.3%) and 12,839
  HPO terms carry a UMLS cross-reference — far more than any other vocabulary. OMOP does
  not distribute UMLS CUIs, so none of it is reachable from a `condition_concept_id`.
- **`hp.obo` has no usable cross-references** — 92 MedDRA and 38 ICD-10 across 20,482
  terms, no UMLS or SNOMED. The Open Targets HPO table is what makes the symptom half
  joinable at all.
- **The curated disease-to-gene layer skews Mendelian**, and Open Targets is what fixes it.
  HPO's `genes_to_disease` runs 8,499 MENDELIAN against 646 POLYGENIC, while FAERS reports
  overwhelmingly on common polygenic disease. Adding the Open Targets associations took
  conditions carrying at least one gene from 28.3% to 47.9%, and conditions sharing a gene
  with a drug target from 20.6% to 43.2%.
- **One Open Targets evidence type must never be a feature.** `known_drug` is derived from
  ChEMBL indications and clinical trials: it says a drug hitting this target is developed
  for this disease. Our label is whether a drug treats or causes a condition, so using it
  would let the model read the answer off its own input. It stays in the reference file for
  fidelity and is excluded at build time — `bridge.conditions.LEAKY_DATATYPES`. The other
  evidence types describe gene-disease biology, not drug-disease outcomes, and are safe.
- **HPO gene annotations arrive pre-propagated.** `phenotype_to_genes.txt` already applies
  the true path rule: across 114,080 child-ancestor pairs no ancestor was missing a
  descendant's gene, and `HP:0000118` alone carries 5,268 of the 5,276 distinct genes. A
  raw gene count therefore measures tree position, not biology, so terms carry Resnik
  information content and a per-gene specificity flag instead.

## Labels

`bridge.labels` turns the CEM table into two labels rather than one, because treating and
causing are not opposites — 325 pairs are asserted to do both.

| label | positive | explicit negative | contradicted (null) |
|---|---|---|---|
| efficacy (TREATS, PREVENTS) | 7,223 | 84 | 76 |
| harm (CAUSES, PREDISPOSES, COMPLICATES) | 1,343 | 37 | 26 |

Only **8,352 of 1,447,172 pairs (0.58%)** carry any directional label. Everything else has
FAERS counts only, and FAERS is a safety signal, not an efficacy one. The indication side
of this problem is built on eight thousand literature assertions, and no amount of feature
engineering changes that — it is the binding constraint on the whole plan.

FAERS pairs are reduced to a boolean using the standard disproportionality thresholds
(≥3 cases, PRR ≥2, chi-square ≥4): **162,881 of 1,437,260 reported pairs (11.3%)** clear
all three. The flag is null, not false, for pairs with no FAERS evidence — "not reported"
and "reported without a signal" are different claims.

Contradicted pairs (`TREATS` and `NEG_TREATS` together) get a null label and a flag rather
than a vote on sentence counts, which measure how often something was written rather than
whether it is true.

Degree lives in `pair_nuisance_label_adjacent.parquet`, named for the repo convention that
anything derived from the label table is not a predictor.

## Next step

Every input the baseline needs now exists: drug-side features, condition-side features and
genes, and labels. What remains is the model itself —

1. Assemble the pair matrix by joining ingredient and condition features on the gene pivot,
   plus explicit path features (does this drug's target sit on a gene implicated in this
   condition; what happened in RWD to other drugs hitting the same target).
2. Fit gradient boosting with **entire ingredients held out**, never a random pair split.
3. Decide negative sampling deliberately. Absence is not evidence of safety, so the 22.7M
   pairs absent from CEM are not negatives; OHDSI-style negative controls are the standard
   device.

The honest ceiling to keep in view: 43.2% of conditions share a gene with any drug target,
and 0.58% of pairs carry a directional label.
