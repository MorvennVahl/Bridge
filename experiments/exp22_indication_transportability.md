# exp22 — Does the indication-label biology advantage survive target-space distance?

Experiment id `exp_20260923_8f4c2f`. Follows exp21 (`exp_20260922_132b7b`).

## Question

exp21 found gene overlap beating the `p_c` floor by +0.0148 `drug_macro_auc` on approved
ChEMBL indications. exp12 ran the same distance ladder on the FAERS harm label and the
advantage collapsed to the floor by D1–D2. Nobody had run it on an efficacy label, because
until the ChEMBL indication layer there was no curated one with enough drugs.

The project's claim is transportability to a molecule with **no** real-world data, so an
advantage that exists only at D0 — where the drug already shares a target gene with a
training drug — is not the claim.

Rungs follow exp12 exactly, measured against train drugs:

| rung | meaning |
|---|---|
| D0 | shares a target gene with a train drug |
| D1 | shares a ChEMBL protein-class leaf |
| D2 | shares a ChEMBL protein-class L1 |
| D3 | has target annotation, no shared class |
| D4 | no ChEMBL target annotation |

## Result — confirmed, and it is a null

Biology-specific margin: the union model against **the same fitted model with
`gene_overlap` zeroed at prediction time**. Bootstrap 95% CI over drugs.

| rung | drugs | pairs w/ overlap | margin | CI | |
|---|---|---|---|---|---|
| **D0** | 84 | 5.6% | **+0.0091** | [+0.0052, +0.0130] | **positive** |
| D1 | 78 | 1.5% | +0.0027 | [−0.0079, +0.0129] | null |
| D2 | 59 | 1.7% | +0.0007 | [−0.0021, +0.0035] | null |
| D3 | 4 | 5.9% | +0.0059 | [−0.0105, +0.0312] | underpowered |
| D4 | 56 | 0.0% | +0.0000 | [0.0000, 0.0000] | exact placebo |

**The advantage exists only at D0 and is indistinguishable from zero at every greater
distance.** This matches exp12 on the harm label. The bridge hypothesis now fails the
distance test on **both** label directions.

## This refines exp21

exp21 reported +0.0148 over the `p_c` floor. Decomposed here, only **+0.0037**
[+0.0003, +0.0069] is attributable to gene overlap; the remaining +0.0111 is the degree
features the `p_c` lookup does not carry. The +0.0148 was correct as a model-versus-floor
number, but should not be read as the size of the biological effect.

## A trap worth recording — it cost this experiment two runs

The obvious ablation, *refit the model without `gene_overlap` and compare*, **is invalid.**

Adding a feature changes the fitted trees globally, so a refit model differs from the union
model even on rows where the feature is constant. Scored that way, **D4 showed +0.0675
[+0.0160, +0.1262]** — despite no D4 drug having any target at all and `gene_overlap` being
identically zero there. Biology cannot act at D4; the margin was two differently-fitted
models being compared.

This is the same failure exp15 diagnosed in exp08's zero-neighbour placebo. Zeroing the
feature at prediction time inside a *single* fitted model fixes it and makes D4 an exact
placebo, which is how the table above was produced.

**Any future ablation on this project should use feature zeroing, not refitting — or carry
a placebo capable of detecting the difference.**

## What this does not show

- **Coverage is the binding constraint.** Only 1.5–5.9% of pairs in any rung have non-zero
  overlap, so the null at D1–D3 is partly a statement about how rarely the feature fires,
  not only about biology being uninformative where it does.
- D3 has 4 scored drugs and is uninformative either way.
- Negatives are constructed from a curated drug's non-indications, not adjudicated.
- Only gene overlap was tested; pathway-level and neighbour features are not covered.
- No test-set contact.

## Reproduce

```bash
uv run python experiments/exp22_indication_transportability.py
```

Writes `results/exp_20260923_8f4c2f_ladder.csv` and `_metrics.json`.
