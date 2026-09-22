# Bridge — design note

Working folder: `~/Desktop/Bridge`. Status: data layer inspected, model layer not built.

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
  association scores; this is the edge that carries most of the causal information)
- condition →`is_a`→ condition (hierarchy; gives the model a prior that sibling
  conditions behave alike)
- ingredient —`RWD`— condition: **the label edge**, not an input feature

The condition-side features you asked about are the weakest part of the current
setup and the right thing to build next: for each of the 5,631 conditions, a
feature vector of implicated genes and pathways (Open Targets), position in the
OMOP/SNOMED hierarchy, organ system, and a phenotype profile (HPO where mappable).

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

## Next step

Build the condition feature layer: 5,631 OMOP condition concepts →
gene / pathway / hierarchy / phenotype features, saved as a table keyed by
`condition_concept_id`. Nothing else in the plan can be evaluated until both sides
of the pair have features.
