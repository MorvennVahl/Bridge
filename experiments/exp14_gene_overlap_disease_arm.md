# exp14 — Gene overlap between drug targets and condition genes

**Re-cut of `exp10_gene_pathway_overlap.md` at half scope.** exp10 was registered
(`exp_20260922_...`, agent exp10) and never completed — it was the tightest budget in Round 2
at ~9 min with both gene and pathway features plus a literature diagnostic. This experiment
takes the gene half only; `exp15` takes pathways. Do not re-register exp10's id; register
fresh and reference it in `notes`.

**Status.** Not started.

---

## Question

Do shared genes between a drug's targets and a condition's implicated genes improve a
held-out drug's condition ranking beyond degree, `p_c`, intrinsic features, and the
same-target neighbour signal?

## Hypothesis

Genetic-association-weighted gene overlap adds **≥ 0.02 `drug_macro_auc`** over exp08's
neighbour model (0.6268 on ≥1-neighbour pairs; 0.6148 for exp07's union on all pairs), and
the gain concentrates on disease-arm conditions.

**What would falsify it.** No gain on the disease arm, where Open Targets scores are richest.
That is a direct failure of the path drug → target → gene → condition, and it would mean
exp08's +0.0443 comes from "other drugs on this target got flagged for this condition" — a
label-copying signal — rather than from mechanism. Report it as the headline, not as a null.

**Why this is the sharper test than exp08.** exp08's neighbour feature reads other drugs'
*labels*. This one reads no labels at all: it is biology on both sides of the pair. If it
works, the bridge is real; if only exp08 works, the project has a nearest-neighbour method
rather than a biological one.

---

## Inputs

| path | rows | note |
|---|---|---|
| `/condition/condition_gene_ot_long.csv` | 897,142 | 3,337 conditions, 17,273 genes, `ot_score` + 8 `dt_*` columns |
| `/condition/condition_gene_hpo_long.csv` | 2,244,725 | phenotype arm, 3,322 conditions, unweighted |
| `/drug/ingredient_target_long.csv` | 8,088 | 1,541 drugs, 790 genes — **assert it is staged** |
| `/condition/condition_features_basic.csv` | 5,631 | `gene_arm`, `n_ot_genes`, `max_ot_score`, `ot_truncated` |

Specificity is the central difficulty: median **250 genes per condition** on the disease arm
and 133 on the phenotype arm, because both propagate up their hierarchies. Raw intersection
counts rank general conditions above specific ones for every drug.

## Features

- `n_shared_genes` — raw intersection
- `shared_gene_score_max`, `shared_gene_score_sum` — weighted by `ot_score`
- `shared_gene_genetic_max`, `shared_gene_genetic_sum` — weighted by **`dt_genetic_association`
  only**, the causal-flavoured datatype and the feature this project has been pointing at
  since `DESIGN.md`
- `shared_gene_idf_sum` — each shared gene weighted by `log(n_conditions / n_conditions
  carrying it)`
- `jaccard_genes` — normalised for both sides' breadth
- phenotype-arm counterparts, unweighted, with an `arm` flag. **Never averaged with the
  disease arm** — the score semantics differ (`AGENT.md` §5.3).

**Kept out of the headline model:** anything weighted by `dt_literature`. Fit it as a
separate diagnostic design and report it apart, so a publication-attention gain cannot be
mistaken for biology.

**Baselines carried in every design:** degree, `p_c`, exp07's intrinsic union, and exp08's
`nb_excess_gene`. The question is what overlap adds *on top of* the neighbour signal.

## Method

1. Preconditions: row counts above; `leak_audit.py`; `floors.py` for this population.
   Confirm whether `ot_truncated` is set for conditions inherited from the truncated
   `ot_associations.jsonl` pull (250 of up to 5,731 targets per disease) and stratify every
   result by it — a truncated gene list is a censored feature, not a small one.
2. Sparse implementation, which is what keeps this inside the budget: build `C`
   (3,337 conditions × 17,273 genes, values `ot_score` / `dt_genetic_association` / IDF) and
   `D` (1,541 drugs × 790 genes, binary), then every overlap feature for all 1.16M pairs is
   one sparse product read at the pair coordinates. **Do not loop over pairs.**
3. Nested designs: `E8base` → `+gene_overlap_raw` → `+gene_overlap_weighted`, plus the
   `dt_literature` diagnostic run separately.
4. Grouped 3-fold CV inside train scored on `drug_macro_auc`; one validate pass per design.
5. Stratify by `gene_arm` (2,570 both / 767 disease-only / 752 phenotype-only / 1,542 none)
   and by `best_match_tier`, with at least one run restricted to A1/A2/B1 so we know the
   result survives mapping noise.
6. Bootstrap CI over drugs for each increment; per-rung floors from `floors.py`.

## Budget

| step | est. |
|---|---|
| read gene tables, build sparse matrices | 2.5 min |
| overlap features for 1.16M pairs (sparse products) | 40 s |
| 3 designs + literature diagnostic × (3 CV + 1 full) | 4 min |
| strata, bootstraps, figure | 1 min |
| **total** | **~8 min** |

**If at risk:** drop the `dt_literature` diagnostic (log that it was not reached) and the
A1/A2/B1 restriction. Never drop the `gene_arm` stratification — a disease-arm-only effect is
the expected shape of a real result.

## Deliverables

`<exp_id>_designs.csv`, `<exp_id>_by_gene_arm.csv`, `<exp_id>_by_truncation.csv`,
`<exp_id>_by_match_tier.csv`, `<exp_id>_literature_diagnostic.csv`,
`<exp_id>_bootstrap_increments.csv`, `<exp_id>_importance_top25.csv`,
`<exp_id>_overlap_distributions.png` — shared-gene counts by arm, so the next agent sees how
sparse real overlap is before designing anything on top of it.

## Reporting requirements

`metrics`: `drug_macro_auc`, `drug_macro_p10`, `increment_over_exp08`,
`increment_genetic_weighted`, `increment_literature_weighted`,
`drug_macro_auc_disease_arm`, `drug_macro_auc_phenotype_arm`, `floor_pc`, `floor_degree_pc`,
`n_drugs_scored`, `share_pairs_with_any_shared_gene`.

`findings` must state the share of pairs that have any shared gene at all (if that is a few
percent, the increment is bounded by coverage and the headline must say so), the increment
with its CI, which arm carries it, and whether `dt_genetic_association` weighting beats
`ot_score` weighting — that comparison is the closest thing available to "is it causal
evidence or association".

## Registration

```python
exp = ln.register(
    agent="exp14",
    title="Gene overlap between drug target genes and condition-implicated genes",
    hypothesis=("Genetic-association-weighted gene overlap adds >=0.02 drug_macro_auc over "
                "exp08's neighbour model, concentrated on disease-arm conditions."),
    approach=("Sparse condition-gene and drug-gene matrices; raw, ot_score-weighted, "
              "genetic-association-only, IDF-weighted and Jaccard overlaps; dt_literature "
              "kept as a separate diagnostic; nested on top of degree, p_c, intrinsic union "
              "and nb_excess_gene; grouped 3-fold CV on drug_macro_auc; stratified by "
              "gene_arm, ot_truncated and match tier."),
    label="y_faers_signal",
    features=["degree", "p_c", "intrinsic_union", "nb_excess_gene", "gene_overlap"],
    split="train/validate, grouped by primary target gene",
    notes="Gene half of the never-completed exp10; pathway half is exp15.",
)
```
