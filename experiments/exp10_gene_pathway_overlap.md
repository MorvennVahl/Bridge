# exp10 — The bridge hypothesis, first direct test

**Status.** Not started. Unblocked by the arrival of
`data/input/condition/condition_gene_ot_long.csv`. This is the experiment the project was
designed around; read it after exp07, whose honest ablation sets the number it must beat.

---

## Question

Does the biology connecting a drug to a condition — shared genes between the drug's targets
and the condition's implicated genes, and shared pathways one hop out — improve a held-out
drug's condition ranking beyond `p_c`, degree, and both sides' intrinsic features?

## Hypothesis

Genetic-evidence-weighted gene overlap adds **≥ 0.02 `drug_macro_auc`** over exp07's union
model, and the gain concentrates on **disease-arm** conditions where Open Targets scores
exist rather than on phenotype-arm conditions reached only through HPO.

**What would falsify it, and what that would mean.** No gain on the disease arm, where the
gene layer is richest and the scores are strongest, is a direct failure of the project's
central claim — the path drug → target → gene → condition would carry no information about
the RWD edge beyond what drug and condition identity already carry. That outcome sends
Round 3 at the label (an indication layer, adjudicated reference sets, calibrated OHDSI
estimates) rather than at more features, and it should be reported as the headline finding,
not buried as a null.

---

## Inputs

| path | rows | note |
|---|---|---|
| `/condition/condition_gene_ot_long.csv` | **897,142** | 3,337 conditions, 17,273 genes, `ot_score` + 8 `dt_*` per-datatype scores |
| `/condition/condition_gene_hpo_long.csv` | 2,244,725 | phenotype arm, 3,322 conditions, unweighted |
| `/drug/ingredient_target_long.csv` | 8,088 | drug → target → gene, 1,541 drugs, 790 genes |
| `data/ref/ot/target__part-*.parquet` | 3 parts | **gene → Reactome pathway membership** |
| `/condition/condition_features_basic.csv` | 5,631 | `gene_arm`, `n_ot_genes`, `max_ot_score`, `ot_truncated` |

**The Reactome file is not what its name suggests.** `data/ref/ot/reactome__*.parquet` is a
pathway *hierarchy* — 2,870 rows of `id, label, ancestors, descendants, children, parents,
path` — with no gene column. Gene→pathway membership comes from the OT **target** parquets.
Verify that before building pathway features; staging the wrong file is a silent null.

**Specificity is the central difficulty.** Median **250 genes per condition** on the disease
arm and 133 on the phenotype arm, because both propagate up their hierarchies. A raw
intersection count will rank general conditions above specific ones for every drug, which is
the opposite of informative. Every overlap feature therefore has a weighted counterpart.

---

## Features

All computed per pair, with the sparse-matrix route in *Implementation* below.

**Gene overlap, disease arm:**
- `n_shared_genes` — raw intersection of the drug's target genes with the condition's OT genes
- `shared_gene_score_max`, `shared_gene_score_sum` — weighted by `ot_score`
- `shared_gene_genetic_max`, `shared_gene_genetic_sum` — weighted by **`dt_genetic_association`
  only**. This is the causal-flavoured datatype and it is the feature the design note has
  been pointing at since the start.
- `shared_gene_idf_sum` — each shared gene weighted by `log(n_conditions / n_conditions
  carrying that gene)`, so a gene implicated in 2,000 conditions counts for little
- `jaccard_genes` — normalised for both sides' breadth

**Explicitly excluded from the headline model:** any feature weighted by `dt_literature`.
Publication attention confounds both the SemMedDB labels and, more weakly, FAERS reporting.
Fit it as a separate diagnostic design and report it apart, so a literature-driven gain
cannot be mistaken for biology.

**Gene overlap, phenotype arm:** the same but unweighted (HPO carries no strength), with an
`arm` flag. Never averaged with the disease arm into one column — the score semantics
differ, as `AGENT.md` §5.3 requires.

**Pathway overlap (one hop out):** Reactome sets for the drug's target genes and for the
condition's genes (top-*m* genes by `ot_score`, `m = 50`, stated and fixed) →
`n_shared_pathways`, `jaccard_pathways`, `shared_pathway_min_level` (a shared leaf pathway
means more than a shared root).

**Baselines carried in every design:** degree, `p_c`, and exp07's winning intrinsic block.
If exp08's neighbour excess survived its placebos, carry that too — then this experiment
answers the only question that matters afterwards: does pathway biology add anything *beyond*
same-target neighbours?

## Method

1. Preconditions: row counts above; `leak_audit.py` from exp07; assert the OT gene table
   covers 3,337 conditions and that `ot_truncated` is carried so truncated pulls are
   reportable (the earlier `ot_associations.jsonl` was truncated at 250 targets per disease —
   confirm whether this table inherited that cap, and if so report every result stratified by
   `ot_truncated`).
2. Nested designs: `E7union` → `+gene_overlap` → `+pathway_overlap`; plus the
   `dt_literature` diagnostic design run separately.
3. Grouped 3-fold CV in train scored on `drug_macro_auc`; one validate pass per design.
4. Stratify every result by `gene_arm` (both / disease_only / phenotype_only / none — 2,570 /
   767 / 752 / 1,542 conditions) and by `best_match_tier`, restricted at least once to
   A1/A2/B1 so we know the result survives mapping noise.
5. Bootstrap CI over drugs for each increment.
6. `drug_macro_auc_unseen_family`, as `METRIC.md` requires.

## Implementation note — this is what keeps it inside ten minutes

Build two sparse indicator/weight matrices once:

- `C` (3,337 conditions × 17,273 genes), values `ot_score` / `dt_genetic_association` / IDF
- `D` (1,541 drugs × 790 genes), binary over target genes

Then every overlap feature for all 1.16M pairs is one sparse product `D @ C.T` read at the
pair's (drug, condition) coordinates — seconds, not a Python loop over pairs. Same trick for
pathways with a gene × pathway matrix. Do not iterate over pairs; that is the one way to
blow the budget here.

## Budget

| step | est. |
|---|---|
| read gene tables, build sparse matrices | 2.5 min |
| pathway sets from OT target parquets | 1.5 min |
| 3 designs + literature diagnostic × (3 CV + 1 full) | 4 min |
| per-drug metrics, strata, bootstraps, figures | 1 min |
| **total** | **~9 min** |

Tightest in the round. **If at risk:** drop the pathway design (keep gene overlap) and log
that pathway features were not reached — a partial answer on gene overlap is worth more than
a timed-out container. Never drop the `gene_arm` stratification or the literature-diagnostic
separation.

## Deliverables

`<exp_id>_designs.csv`, `<exp_id>_by_gene_arm.csv`, `<exp_id>_by_match_tier.csv`,
`<exp_id>_literature_diagnostic.csv`, `<exp_id>_importance_top25.csv`,
`<exp_id>_bootstrap_increments.csv`, `<exp_id>_per_drug.csv`,
`<exp_id>_overlap_distributions.png` (shared-gene counts by arm, so the next agent sees how
sparse the real overlap is).

## Reporting requirements

`metrics`: `drug_macro_auc` (best design), `drug_macro_auc_unseen_family`,
`drug_macro_p10`, `increment_gene_overlap`, `increment_pathway_overlap`,
`increment_literature_weighted`, `drug_macro_auc_disease_arm`,
`drug_macro_auc_phenotype_arm`, `n_drugs_scored`, `baseline_e7_union_drug_macro_auc`,
`solved_threshold` (read from exp06's `success_criterion.json`).

`findings` must state the increment with its CI, which arm carries it, how it compares to the
`solved_threshold`, and whether pathway features add anything over gene overlap. If the
result is null, say the bridge hypothesis failed its first direct test at this feature
resolution and name what would have to change — better condition→gene evidence, a different
label, or a different similarity notion.

## Registration

```python
exp = ln.register(
    agent="exp10",
    title="Gene and pathway overlap between drug targets and condition genes",
    hypothesis=(
        "Genetic-evidence-weighted overlap between a drug's target genes and a "
        "condition's implicated genes adds >=0.02 drug_macro_auc over the intrinsic "
        "union, concentrated on disease-arm conditions."
    ),
    approach=(
        "Sparse condition-gene and drug-gene matrices; raw, ot_score-weighted, "
        "genetic-association-only, IDF-weighted and Jaccard overlaps; Reactome pathway "
        "overlap from the OT target parquets; dt_literature weighting kept as a "
        "separate diagnostic; grouped 3-fold CV scored on drug_macro_auc; stratified "
        "by gene_arm and match tier."
    ),
    label="y_faers_signal",
    features=["degree", "p_c", "intrinsic_union", "gene_overlap", "pathway_overlap"],
    split="train/validate, grouped by primary target gene",
)
```
