# Bridge — agent scientist brief

You are one of several independent agents working the same problem in parallel. You
propose one experiment, run it, and write down what happened. The next agent reads
what you wrote. That is the entire method: the lab's progress is the notebook, not
any single model.

Read `lab/notebook.jsonl` before you do anything else. If your idea has already
been tried, pick a different one or explain in your registration why it deserves a
second look with a specific change.

---

## 1. The question

Given a drug and a condition, predict whether the drug **causes** that condition
(safety) or **treats** it (efficacy) — for drug–condition pairs where no real-world
evidence exists.

The point is transportability to a drug with no data of its own. A new molecule has
no adverse-event reports, but it has a target, that target has a gene, that gene sits
in pathways, and those pathways are implicated in diseases. The biology is the bridge
between drugs we have observed and drugs we have not.

A model that scores well by recognising the drug has answered nothing. This is why
the splits are grouped the way they are (§4), and why the most important number in
your report is performance on drugs whose **target family** never appeared in
training.

---

## 2. Data layout

```
data/
  input/
    drug/        ingredient_features.csv        4,280 ingredients x 159 cols
                 target_features.csv            1,536 targets x 61 cols
                 ingredient_target_long.csv     8,088 ingredient->target->gene edges
                 ingredient_features_label_adjacent.csv   QUARANTINED - see §3
    condition/   condition_ontology_map.csv     5,631 conditions -> MONDO/EFO/HPO
                 (condition feature tables land here as they are built)
  output/
    pair_labels.csv    1,447,172 rows, one per ingredient-condition pair
  splits/
    split_assignment.csv   ingredient -> fold and group
    train.csv / validate.csv / test.csv
    split_summary.csv, split_provenance.json
```

Join keys: `ingredient_concept_id` on the drug side (equals `omop_concept_id` in the
drug feature files), `condition_concept_id` on the condition side. Both are OMOP
concept ids.

### The splits

| fold | pairs | share | ingredients | target groups | conditions |
|---|---|---|---|---|---|
| train | 723,586 | 50% | 2,126 | 1,553 | 5,246 |
| validate | 434,151 | 30% | 1,284 | 928 | 4,955 |
| test | 289,435 | 20% | 866 | 619 | 4,692 |

Label rates are balanced across folds: FAERS signal 11.0 / 11.7 / 11.2%.

Folds are assigned to whole groups of ingredients sharing a **primary target gene**,
so no ingredient and no primary-target group spans two folds. Conditions deliberately
DO appear in multiple folds — the task is generalising to new drugs, not new diseases.
`scripts/build_dataset.py --scheme target_component` builds a stricter variant where
whole connected components of the drug–target graph are held out; it cannot hit
50/30/20 because promiscuous targets merge 638 ingredients into one component, but it
is the right robustness check for a model you are about to promote.

---

## 3. Rules that are not negotiable

**The test set is used exactly once, ever, for the whole project.** Not once per
agent. `bridge.labnotebook.check_test_seal()` raises unless a human has unsealed it.
Do not create the seal file. Do not read `test.csv` to "just check the distribution".
If you think the project is ready for its final evaluation, say so in your report and
let a human decide.

**Validation is for model selection, and it degrades with use.** Fit on train.
Select hyperparameters and features with grouped cross-validation *inside* train —
group by `group_key`, never by row. Touch `validate` once per experiment, at the end,
after your choices are fixed. `labnotebook.validate_evaluations()` shows how many
times the lab has queried it; when that number is large, stop trusting differences of
a few thousandths and prefer the simpler model.

**Never train on label-adjacent features.** `ingredient_features_label_adjacent.csv`
holds CEM degree terms — how many conditions each drug already has evidence for.
These are functions of the labels. They exist so you can use them as *nuisance
controls* when analysing bias, and to check that your model is not merely recovering
them. They are not predictors.

**Absence is not a negative.** `pair_labels.csv` contains only observed pairs. A
missing pair may mean no effect, or that the drug was never prescribed, or that
nobody reported it. If your method needs negatives, say explicitly how you
constructed them and what that assumes. Random unobserved pairs are not negatives,
and a model trained as if they were will look excellent and mean nothing.

**Log the experiment before you run it** (§6). A hypothesis registered after seeing
the result is not a hypothesis.

**The lab notebook is append-only. Write new entries; never edit or delete existing
text.** `lab/notebook.jsonl` is the lab's shared memory and several agents write to it
concurrently. Do not rewrite a line, do not delete a line, do not reformat the file,
do not "clean up" entries you think are wrong or superseded — including your own, and
including entries that look like mistakes. Use the API (`register`, `complete`,
`note`), which only ever appends.

If you believe an earlier result is wrong, append a new `complete` entry with
`supersedes="<old experiment id>"` and explain in `findings` what was wrong with it.
The original stays. A wrong result that someone later corrected is valuable evidence
about the problem; a wrong result that was quietly deleted leaves the next agent to
make the same mistake with no trace that anyone had been there.

This applies to every file under `lab/`. Round plans and notes are part of the record
too — if a plan needs revising, add the revision, don't overwrite the original.

---

## 4. What the labels actually are

These are CEM evidence flags, not causal effect estimates. Read `DESIGN.md` for the
full data profile. The parts that will mislead you:

- **`y_faers_signal`** (11% of pairs) — FAERS disproportionality by the Evans
  criteria (PRR ≥ 2, χ² ≥ 4, ≥ 3 cases). This is a *screening* convention. FAERS is
  a spontaneous-reporting system: the highest-degree conditions in the whole dataset
  are Respiratory finding, Nausea, Pain, Fever, Vomiting, Dizziness — which is what
  reporting bias looks like, not pharmacology. Median case count across all pairs is
  2 and median PRR 1.36, so most rows are noise.
- **`y_semmeddb_treats`** (0.5%) and **`y_semmeddb_causes`** (0.1%) — assertions
  mined from literature. Very sparse, and subject to publication attention.
- **`in_eu_label` is false on every one of the 1.45M rows.** The EU product-label
  evidence layer did not survive the CEM build's join to standard condition concepts.
  This is documented upstream, not a bug in the export. It means the **indication
  side of the label is almost absent** — efficacy rests on ~7,900 SemMedDB
  assertions and nothing else. An agent who can bring a real indication layer in
  (DailyMed, the OHDSI indication set, drug labels) would be doing the single most
  valuable thing available.

Because of the class imbalance — 0.1% to 11% depending on label — **report average
precision (area under the precision–recall curve), not ROC-AUC alone.** ROC-AUC looks
flattering at these prevalences. Report calibration too; a well-ranked but badly
calibrated score is not usable for the downstream decision.

---

## 5. Traps already discovered

Do not rediscover these; they cost real time.

1. **Degree explains a lot.** Both drug degree (how many conditions a drug has
   evidence for) and condition degree (how many drugs) are strongly predictive of the
   labels for reasons that are purely about observation, not biology. Any claim that
   a biological feature helps must be made *on top of* a degree-only baseline. Build
   that baseline first; the notebook has it.
2. **Condition coverage ceiling.** Only 4,444 of 5,631 conditions (78.9%) map to any
   disease/phenotype ontology term. The 1,187 unmapped ones are concepts like
   `Injection site pain`, `Late effect of contusion`, `Burning sensation` — clinical
   encounter concepts with no disease-ontology counterpart anywhere. They are
   disproportionately high-degree in CEM, so they carry a lot of pairs. Decide
   explicitly whether to model them (with hierarchy/frequency features only) or
   exclude them, and report which you did.
3. **Two gene arms with different semantics.** Disease-arm conditions reach genes
   through Open Targets (continuous 0–1 association score per evidence datatype).
   Phenotype-arm conditions reach genes through HPO annotations (curated, unweighted,
   no strength). Do not average these into one column. Keep them as separate blocks
   with a source flag, or binarise both and say so.
4. **Mapping tier quality varies and is recorded.** `condition_ontology_map.csv` has
   a `match_tier` column. Tiers `A1`/`A2` are OMOP2OBO manual curation; `A3`–`A5` are
   OMOP2OBO automatic; `B1` is an authoritative SNOMED cross-reference; `B2`/`B3` are
   lexical; `B4` is LLM-adjudicated and roughly 85% precise on cases where retrieval
   worked. A sensitivity analysis restricted to `A1`/`A2`/`B1` is cheap and tells you
   whether a result survives mapping noise.
5. **Ontology ancestor mappings are not equivalences.** OMOP2OBO's ancestor-category
   rows map a concept to a broad parent (`Neoplasm of uncertain behavior of larynx` →
   "disorder by anatomical region"). They are excluded from the gene join for this
   reason. Use them for grouping, never as the condition's identity.
6. **Only 424 of 1,406 HPO terms we map to exist in Open Targets' disease index.**
   If you plan to source everything from Open Targets you will silently lose most
   symptom-type conditions.

---

## 6. The experiment cycle

**Read.** Start with `labnotebook.notes("dataset_state")` — durable entries describing
what the data actually contains and what is known to be wrong with it, including every
caveat in §4 and §5 with the numbers attached. Then `labnotebook.read()` and
`leaderboard(label=...)` for the current best and the known dead ends. Reading these
costs a minute and routinely saves a whole run.

**Propose.** One experiment, one hypothesis, stated so it can fail. "Pathway overlap
between drug targets and condition genes predicts harm beyond degree" is a
hypothesis. "Try a GNN" is not — name what the graph buys you that explicit features
do not, and what result would convince you it does not.

**Register.** Before running:

```python
from bridge import labnotebook as ln
exp = ln.register(agent="<your id>", title=..., hypothesis=..., approach=...,
                  label="y_faers_signal", features=[...], split="...")
```

**Run.** Fit on train, select inside train with grouped CV, evaluate once on
validate. Save figures and tables under `results/` with the experiment id in the
filename.

**Log.** Always, including failures and nulls:

```python
ln.complete(exp, metrics={"validate_average_precision": ...,
                          "train_cv_average_precision": ...,
                          "baseline_degree_only_ap": ...},
            findings="what you now believe and why, in specifics",
            artifacts=["results/exp_..._calibration.png"],
            next_steps="the one thing you would do next")
```

A good `findings` states the effect size, the subgroup where it concentrates, and
what it does *not* show. "No gain overall, but +0.06 AP on conditions with ≥5
curated genes, and nothing on the phenotype arm" is worth more to the next agent
than "AP 0.31".

**A null result is a result.** Log it with the same care. The lab's main risk is
twenty agents cheerfully rediscovering the same non-effect.

---

## 7. Where to start

The baseline ladder, in order. Do not skip ahead; each rung tells you what the next
one has to beat.

1. **Degree only** — drug degree, condition degree, condition `record_count`. This
   is the number to beat and it will be higher than you expect.
2. **Drug features only** — mechanism class, target class, ATC, physicochemical.
   Tells you how much is drug-intrinsic.
3. **Condition features only** — gene count, organ system, therapeutic area,
   hierarchy position.
4. **Explicit pair features** — shared genes between drug targets and condition
   genes; pathway overlap (Jaccard, weighted by Open Targets score); shortest path
   length in the drug→target→gene→pathway→condition graph; count of *other* drugs
   hitting the same target that have evidence for this condition (a same-target
   nearest-neighbour signal, and probably the strongest single biological feature
   available).
5. **Then, and only then, a learned graph model** — R-GCN or HGT over the
   heterogeneous graph. It has to beat rung 4 to justify itself, and the argument for
   it is that it learns which relation types matter rather than you asserting that
   pathway overlap is the one that counts. Keep it inductive: no learned per-drug
   embedding, or the model cannot score a drug it has never seen.

Other avenues worth a run, none of them explored yet: matrix factorisation or
collaborative filtering with cold-start drug features; using `faers_prr` as a
continuous regression target instead of the binary flag; modelling the
disproportionality *residual* after removing drug and condition degree, which is
much closer to a causal quantity than the raw signal; multi-task learning across the
harm and treat labels; predicting SemMedDB `TREATS` as a proxy for indication while
the real indication layer is missing; transfer from ChEMBL activity data;
target-class-specific models; and quantifying how far performance decays as a
function of a drug's distance from the nearest training drug in target space, which
directly measures the transportability the project exists to achieve.

---

## 8. What a finished piece of work looks like

- An entry in `lab/notebook.jsonl` with a registered hypothesis and a completed
  outcome
- Metrics that include a comparison against the degree-only baseline on the same
  split
- Figures and tables under `results/`, named with the experiment id
- A statement of what you did *not* test, and any place you used less data than the
  task implies
- No test-set contact

Write for the agent who reads your entry three days from now and knows none of your
context.
