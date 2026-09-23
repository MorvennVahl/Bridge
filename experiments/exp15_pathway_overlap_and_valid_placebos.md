# exp15 — Pathway overlap, and a placebo design that actually works

**Second half of the never-completed exp10**, plus the correction to exp08's placebo battery.
Both belong together: the placebo machinery is what decides whether a pathway increment is
believable, and exp08 showed we do not yet have it right.

**Status.** Not started.

---

## Part 1 — the placebo correction

exp08 reported `placebo_zero_nb_gain = +0.0198` where the spec predicted 0.000, and concluded
"a second `p_c` surrogate may be leaking in". **The placebo was invalid, not the feature.**

The design compared model B (fitted on all train rows) against model A on the zero-neighbour
subset, where B's extra features are constant. A constant feature does not make two models
agree: B's trees were grown on a different design matrix, so its fitted function differs
everywhere, including where the extra feature carries no information. The +0.0198 measures
that difference in fitted functions.

Two valid forms, both cheap:

1. **Refit-within-subset.** Fit A and B on the zero-neighbour rows only and compare there.
   With `nb_excess ≡ 0` on those rows, the designs are genuinely equivalent and the gain must
   be 0.000 up to seed noise. Report the seed noise by fitting with 3 seeds.
2. **Train-time value permutation.** Keep the design matrix identical and permute the
   feature's values across rows within each condition before fitting. The column keeps its
   marginal distribution and loses its pairing; any gain that survives is construction
   artefact.

Run both on exp08's gene-tier feature and report them beside the original +0.0198, so the
record shows the placebo was the problem. exp08's +0.0443 increment and its rising decay
curve (0.591 / 0.612 / 0.609 / 0.656) are unaffected — the two valid placebos it ran came
back near-null (+0.0096 permuted graph, +0.0047 label shuffle).

## Part 2 — pathway overlap

### Question

One hop out from genes: does shared **pathway** membership between a drug's target genes and a
condition's implicated genes add anything over gene overlap itself (exp14) and the neighbour
signal (exp08)?

### Hypothesis

Pathway overlap adds **< 0.01 `drug_macro_auc`** over exp14's gene overlap — i.e. the extra
hop is mostly redundant with direct gene sharing — but it extends **coverage**, scoring pairs
that have no shared gene at all, and the gain concentrates there.

**Why the hypothesis is deliberately modest.** `DESIGN.md` argued that a 2–3 hop path is
computable directly and that a GNN would mostly re-learn it; the same logic says pathway
membership is a smoothed version of gene sharing. The interesting quantity is not the average
increment but the increment **on pairs with zero shared genes**, where gene overlap is
undefined and pathway overlap is the only biological signal available. Report that
conditional number as the headline.

### Inputs

Everything exp14 uses, plus:

| path | note |
|---|---|
| `data/ref/ot/target__part-*.parquet` (3 parts, ~85 MB) | **gene → Reactome pathway membership** |
| `data/ref/ot/reactome__*.parquet` | pathway **hierarchy** only — 2,870 rows of `id, label, ancestors, descendants, children, parents, path`, **no gene column** |

The naming is a trap: the file called `reactome__*` cannot give you gene→pathway edges.
Staging it instead of the target parquets yields a silent null. Assert that the pathway map
has a gene identifier column before proceeding.

### Features

- `n_shared_pathways`, `jaccard_pathways` — over the drug's target-gene pathway set and the
  condition's gene pathway set (condition side restricted to its top **m = 50** genes by
  `ot_score`, fixed, stated)
- `shared_pathway_min_level` — depth of the shallowest shared pathway using the hierarchy's
  `path` field. A shared leaf pathway means more than a shared root like "Signal
  Transduction"; without this the feature is dominated by pathways that contain half the
  genome.
- `n_shared_pathways_idf` — pathway weighted by inverse condition frequency
- `has_shared_gene` — the interaction control, so the model can distinguish "pathway overlap
  where genes already overlap" from "pathway overlap alone"

### Method

1. Preconditions; `leak_audit.py`; `floors.py`. Assert the gene→pathway map covers ≥60% of the
   790 drug target genes and report the covered share.
2. Sparse route again: gene × pathway matrix, then `D_genes @ GP` for drugs and
   `C_topm @ GP` for conditions, then overlap by product. No loops over pairs.
3. Designs: `E14best` → `+pathway_overlap`. Plus the conditional evaluation on the
   **zero-shared-gene** subset, which is the real question.
4. Grouped 3-fold CV inside train on `drug_macro_auc`; one validate pass.
5. Both placebo forms from Part 1, applied to the pathway feature as well as to exp08's
   neighbour feature.
6. Coverage accounting: share of pairs with ≥1 shared pathway, and of those, how many also
   have a shared gene. If pathway overlap is nearly coextensive with gene overlap, say so —
   that is a real finding about the graph, and it retires the "add more hops" idea cheaply.

### Budget

| step | est. |
|---|---|
| read OT target parquets, build gene × pathway map | 2.5 min |
| pathway sets, sparse overlaps | 1.5 min |
| 2 designs × (3 CV + 1 full) | 2.5 min |
| placebos (2 forms × 2 features × 3 seeds, narrow fits) | 1.5 min |
| conditional evaluation, coverage, figure | 1 min |
| **total** | **~9 min** |

**If at risk:** drop the pathway designs and complete with the placebo results only — Part 1
is the part that fixes the record, and a clean placebo correction is worth more than a rushed
pathway number.

### Deliverables

`<exp_id>_placebo_corrected.csv` (invalid form beside both valid forms, for exp08's feature
and the pathway feature), `<exp_id>_designs.csv`,
`<exp_id>_zero_shared_gene_conditional.csv`, `<exp_id>_coverage_overlap.csv`,
`<exp_id>_pathway_map_coverage.csv`, `<exp_id>_bootstrap_increments.csv`.

### Reporting requirements

`metrics`: `placebo_zero_nb_refit_gain`, `placebo_value_permuted_gain`, `placebo_seed_noise`,
`drug_macro_auc`, `increment_pathway_over_gene`,
`increment_pathway_on_zero_shared_gene_pairs`, `share_pairs_with_shared_pathway`,
`share_shared_pathway_without_shared_gene`, `floor_pc`, `floor_degree_pc`, `n_drugs_scored`.

`findings` must state, in order: that exp08's zero-neighbour placebo was invalid by
construction and what the valid forms give; the pathway increment overall and on
zero-shared-gene pairs; and whether pathway overlap is distinguishable from gene overlap at
all. If it is not, recommend that Round 4 stop adding hops and spend its budget on the label
or on a better condition→gene layer instead.

## Registration

```python
exp = ln.register(
    agent="exp15",
    title="Reactome pathway overlap, and a valid placebo battery",
    hypothesis=(
        "Pathway overlap adds <0.01 drug_macro_auc over gene overlap overall but "
        "carries the signal on pairs with zero shared genes; and exp08's "
        "zero-neighbour placebo result is an artefact of comparing two differently "
        "fitted models rather than evidence of a p_c surrogate."
    ),
    approach=(
        "Refit-within-subset and train-time value-permutation placebos on exp08's "
        "neighbour feature and the new pathway features; Reactome membership from the "
        "OT target parquets (not the hierarchy-only reactome file); sparse gene-pathway "
        "products; conditional evaluation on zero-shared-gene pairs."
    ),
    label="y_faers_signal",
    features=[
        "degree",
        "p_c",
        "intrinsic_union",
        "nb_excess_gene",
        "gene_overlap",
        "pathway_overlap",
    ],
    split="train/validate, grouped by primary target gene",
    notes="Pathway half of the never-completed exp10, plus the exp08 placebo correction.",
)
```
