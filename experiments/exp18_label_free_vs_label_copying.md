# exp18 — Is this a biological method or a nearest-neighbour method?

**Status.** Not started. **This is the round's central experiment**, and the question it asks
is the one the project's premise rests on.

---

## Why it has to be asked now

Every feature that has worked reads other drugs' labels:

| feature | reads labels? | increment | verdict |
|---|---|---|---|
| `p_c`, the condition's train flag rate | yes | floor 0.5759 (beats the fitted degree model) | the strongest single predictor |
| exp08 neighbour excess | yes — other drugs' flags | **+0.0443** (placebo-adjusted ~+0.035) | the project's best result |
| exp07 intrinsic union | no | +0.039 over floor | modest, and exp17 asks what it is made of |
| exp14 gene overlap | **no** | **+0.0090** over the neighbour base | at the noise threshold |
| exp14 genetic-evidence weighting | no | +0.0025 | nothing |
| exp14 `dt_literature` weighting | no | +0.0061 | beat the genetic weighting |

A method built on `p_c` plus same-target neighbours is a nearest-neighbour model over the
drug–target graph. That is a legitimate and useful thing, but it has two properties the
project cannot live with silently: it cannot score a condition that has never been flagged for
anything, and its transportability claim rests on other drugs having been observed, not on
biology.

## Question

Stripped of every label-derived feature, does biology alone rank a held-out drug's conditions
above chance — and does it rank conditions that are **cold**, meaning absent from training?

## Hypothesis

A strictly label-free model beats chance (0.5) on all conditions and, on **cold conditions**,
beats the label-derived model — which has no `p_c` to fall back on there. The absolute number
will be low; the comparison is the result.

**What would falsify it.** Label-free biology at 0.5 on cold conditions. Then the project has
a nearest-neighbour method that cannot generalise to new outcomes, and Round 5's honest
options are to say so and build the best neighbour method we can, or to change the label to
something biology can predict (exp16's adjudicated outcomes, or the efficacy label in exp20).

---

## Three models, strictly partitioned

**L — label-derived only.** `p_c`, drug degree, condition degree, `record_count`, exp08's
`nb_excess` at all three tiers, `n_neighbours`. Nothing else.

**B — label-free biology only.** No `p_c`, no degree, no neighbour rate, nothing computed from
any `y_*` column:
- gene overlap (raw, `ot_score`-weighted, `dt_genetic_association`-weighted, IDF-weighted,
  Jaccard), disease arm and phenotype arm separately
- **pathway overlap** — now unblocked. exp15 failed with "0.000 of 977 genes"; the data is in
  `data/ref/ot/target__part-*.parquet`, column `pathways`, a list of
  `{pathwayId, pathway, topLevelTerm}` per target, keyed by `approvedSymbol` (gene symbol) and
  `id` (ENSG). Explode it and join on `approvedSymbol`. **Assert ≥60% of drug target genes map
  before building anything** — that assertion is what caught exp15, and it should catch a
  wrong key again.
- drug-intrinsic mechanism and target-biology blocks (exp07's blocks minus anything
  substance-type, per exp17)
- condition-intrinsic biology: `n_ot_genes`, `max_ot_score`, organ system, therapeutic area
- off-target panel affinities where measured (`data/input/drug/offtarget_activities.csv`,
  12 targets, 888 ingredients) and the syndrome-interaction feature from
  `offtarget_syndrome_map.json` — sparse for now; exp19 densifies it

**L+B** — both.

`leak_audit.py` must be extended with an assertion that **B contains no column derived from
any label column**, checked by construction against a whitelist, not by name matching alone.

## The cold-condition test

The real question is generalisation to outcomes with no history, so build it explicitly.

1. **Natural cold conditions.** Validate conditions absent from train — exp14 reported 349
   such rows where `p_c` fell back to the global rate. Report `n_conditions`, `n_pairs`,
   `n_drugs`; if it is a few hundred rows, it is a direction, not an estimate, and must be
   labelled as such.
2. **Constructed cold conditions — the powered version.** Rebuild the split with conditions
   grouped out as well as drugs: hold out 20% of conditions entirely, so a validate fold
   exists in which **no** validate condition appears in train. Report the fold sizes and label
   balance achieved. On that split, `p_c` is undefined by construction for every row, and L
   collapses to degree plus neighbour rates. This is the cleanest test the data can support of
   whether biology carries anything at all.
3. Floors for both populations via `floors.py`. On the doubly-held-out split the floor is
   drug degree plus condition degree — there is no `p_c` — and it will be near 0.5. Report it;
   a model that beats *that* floor has earned the claim.

## Method

1. Preconditions; `leak_audit.py` extended as above; `floors.py` on all four populations
   (all conditions, natural cold, constructed cold, warm).
2. Fit L, B, L+B on the standard split; evaluate `drug_macro_auc`, `drug_macro_p10`,
   `n_drugs_scored`, all with per-population floors.
3. Refit all three on the doubly-held-out split; evaluate the same way.
4. Bootstrap CIs over drugs for the B-versus-floor margin and the B-versus-L difference on
   cold conditions.
5. Feature importance for B on cold conditions only — if biology works anywhere, this names
   what carried it.

## Budget

| step | est. |
|---|---|
| feature construction, pathway join, audits | 3 min |
| 3 models × (3-fold CV + full fit), standard split | 4 min |
| doubly-held-out split rebuild + 3 refits | 4 min |
| floors, bootstraps, importance, figure | 1.5 min |
| **total** | **~12 min** at `timeout=900` |

**If at risk:** drop the CV folds on L+B; keep every cold-condition number. The cold-condition
comparison **is** the experiment.

## Deliverables

`<exp_id>_models.csv` (L / B / L+B × warm / natural-cold / constructed-cold, each with its
floor), `<exp_id>_cold_split_summary.csv`, `<exp_id>_B_importance_cold.csv`,
`<exp_id>_bootstrap.csv`, `<exp_id>_pathway_join_coverage.csv`,
`<exp_id>_cold_vs_warm.png`.

## Reporting requirements

`metrics`: `drug_macro_auc_L`, `_B`, `_LB` (warm); `drug_macro_auc_B_cold_natural`,
`_B_cold_constructed`, `_L_cold_constructed`, `_LB_cold_constructed`; `floor_warm`,
`floor_cold_constructed`; `n_drugs_cold`, `n_conditions_cold`; `pathway_gene_coverage`.

`findings` must answer the title question in its first sentence — biological method or
nearest-neighbour method — and then give the cold-condition numbers against their floor with
CIs. If B is at floor on cold conditions, say that the project as currently framed does not
have a biological method, and name which of Round 5's two options the evidence favours.

## Traps

- **`p_c` leaks in through the back door.** Any target encoding, any shrinkage prior toward a
  condition mean, any feature aggregated over training labels is L, not B. exp03 lost a whole
  experiment to this.
- The constructed cold split changes the label balance and the population; its floor must be
  recomputed and its `n_drugs_scored` reported. Do not compare its numbers to 0.5759.
- B will look weak in absolute terms. That is expected and is not the result — the result is B
  versus its own floor, and B versus L where L has nothing to stand on.
- Do not let B include `has_chembl_match` or other missingness indicators; exp17 shows those
  carry substance-type separation, which is not biology.

## Registration

```python
exp = ln.register(
    agent="exp18",
    title="Label-free biology versus label-copying, with a constructed cold-condition split",
    hypothesis=("A strictly label-free biology model beats chance on all conditions and beats "
                "the label-derived model on cold conditions, where p_c is undefined."),
    approach=("Three strictly partitioned designs L / B / L+B; pathway overlap unblocked via "
              "the OT target parquet 'pathways' column joined on approvedSymbol; a rebuilt "
              "split holding out 20% of conditions entirely so p_c is undefined by "
              "construction; floors for every population; importance for B on cold conditions."),
    label="y_faers_signal",
    features=["label_derived_block", "gene_overlap", "pathway_overlap", "mechanism",
              "offtarget_panel", "condition_biology"],
    split="train/validate grouped by primary target gene, plus a drug-and-condition held-out split",
)
```
