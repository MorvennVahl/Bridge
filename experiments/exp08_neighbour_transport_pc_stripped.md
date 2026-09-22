# exp08 — Is exp03's gain target transport, or the condition base rate?

**Supersedes** `exp_20260922_341a64` (exp03) on the attribution question; its coverage
measurements stand and should be reused rather than re-derived.

**Status.** Not started. This is the project's core claim, now with controls.

---

## Why exp03 cannot be read as support for the bridge hypothesis

Three numbers from its own log:

1. **The decay curve is flat.** AP by `n_neighbours_gene` bucket: 1–2 → 0.189, 3–5 → 0.187,
   6+ → 0.190. A neighbour-rate feature whose quality does not improve with more neighbours
   is not behaving like an estimate of a neighbour rate.
2. **It gains where the feature is undefined.** Zero-neighbour bucket: degree-only 0.120 →
   model B 0.158, on 288,357 pairs. With no neighbours the shrunk rate
   `(0 + k·p_c)/(0 + k)` collapses **exactly** to `p_c`, the condition's train flag rate. The
   feature is a condition-level target encoding in that bucket, and that bucket is 66% of
   validate pairs.
3. **Tier order is non-monotonic.** class_leaf 0.2016 > gene 0.1889 > class_L1 0.1222, where
   the hypothesis predicted monotone decay as similarity loosens.

Together these are consistent with "most of +0.062 is `p_c`", which we now know is a strong
predictor on its own (`drug_macro_auc` 0.5759 versus 0.5691 for exp01's fitted model).

## Question

After `p_c` is held fixed, does the behaviour of *other drugs sharing a target* still
improve a held-out drug's condition ranking?

## Hypothesis

The neighbour **excess** — the shrunk neighbour rate minus `p_c` — adds ≥ 0.02
`drug_macro_auc` over a `degree + p_c` baseline on drugs with ≥1 gene-level neighbour, and
all three placebos below come back null.

**What would falsify it.** Any placebo showing the same gain as the real feature. If the
degree-preserving permutation reproduces the effect, what transports is not target identity
but the shape of the drug–target graph, and the bridge hypothesis has failed its cleanest
test. Say that plainly; it redirects Round 3 to the label rather than the features.

---

## Design

**Baseline (every model carries it):** drug degree, condition degree, `record_count`, and
`p_c`.

**The feature under test:** `nb_excess_t = shrunk_rate_t − p_c` for tiers
`t ∈ {gene, class_leaf, class_L1}`, with `k = 10`, computed strictly within train and with
leave-one-drug-out on train rows exactly as exp03 did (its self-leakage check was clean:
train CV 0.2796 versus validate conditional 0.1889). Keep exp03's exclusion of
`component_relationship ∈ {COMPLEX, FAMILY}` subunit expansions from the gene tier, and keep
its definition of "primary target" via ChEMBL's `disease_efficacy` flag — the split's
`group_key` is 0 on validate by construction and would be a degenerate feature.

Also carry `n_neighbours_t` itself, so the model can learn that a rate from 40 neighbours is
worth more than one from 2 — that, not the rate alone, is what should produce a rising decay
curve.

## The three placebos

Each is a full refit of the tested model with one thing scrambled. All three run in the same
container; they are cheap because the fits are narrow (<25 columns).

1. **Zero-neighbour placebo.** Evaluate on the zero-neighbour subset only. With `p_c` in the
   baseline, `nb_excess` is identically 0 there, so the gain over baseline **must be 0.000**.
   Any gain means a second `p_c` surrogate leaked in.
2. **Degree-preserving permutation.** Permute the drug→target-gene mapping across drugs
   while preserving each drug's target count and each gene's drug count (a configuration-model
   shuffle; use 5 permutations and report mean ± sd). Recompute neighbour features from the
   shuffled graph. Real target identity should die here while graph shape survives.
3. **Within-condition label shuffle.** Shuffle `y` within each condition in train, so
   condition base rates are preserved exactly and all drug–condition structure is destroyed.
   Refit and recompute. Any residual gain is an artefact of the feature's construction.

## Method

1. Preconditions, plus reuse of exp03's coverage table (33.6% of validate pairs have ≥1
   gene neighbour; 55.5% class_leaf; 75.3% class_L1; 81.9% of validate ingredients have zero
   gene neighbours).
2. Models: **A** baseline; **B** baseline + gene-tier excess; **C** baseline + all tiers.
3. Grouped 3-fold CV in train scored on `drug_macro_auc`; one validate pass per model.
4. Primary result: `drug_macro_auc` on drugs with ≥1 gene neighbour, with coverage stated,
   plus `drug_macro_auc_unseen_family`.
5. The decay curve, recomputed on the new metric with `p_c` controlled — this is the figure
   the project has been trying to produce. Buckets {0, 1–2, 3–5, 6+}, with pair and drug
   counts on the axis.
6. Tier decay with `p_c` controlled, to separate target-specific from class-level transport.

## Budget

| step | est. |
|---|---|
| read, tier aggregates with leave-one-drug-out | 2 min |
| 3 models × (3 CV + 1 full), narrow designs | 2 min |
| 3 placebos (5 permutations for #2) | 2 min |
| per-drug metrics, decay curve, figures | 1 min |
| **total** | **~7 min** |

**If at risk:** reduce permutation count to 3. Never drop placebo 1 — it is one line and it
is the one that would have caught exp03.

## Deliverables

`<exp_id>_models.csv`, `<exp_id>_placebos.csv` (all three, side by side with the real
feature), `<exp_id>_decay_curve.csv` + `.png`, `<exp_id>_tier_decay.csv`,
`<exp_id>_per_drug.csv`, `<exp_id>_bootstrap_increment.csv`.

## Reporting requirements

`metrics`: `drug_macro_auc` (model B, drugs with ≥1 gene neighbour),
`drug_macro_auc_unseen_family`, `drug_macro_p10`, `coverage_ge1_nb`, `n_drugs_scored`,
`baseline_pc_drug_macro_auc`, `placebo_zero_nb_gain`, `placebo_permuted_graph_gain`,
`placebo_shuffled_label_gain`, and the four `decay_bucket_*` values.

`findings` must state the increment over `degree + p_c` with its CI, all three placebo
results, and an explicit verdict on exp03's +0.062: how much of it was `p_c`. That sentence
is what the next agent needs.

## Registration

```python
exp = ln.register(
    agent="exp08",
    title="Same-target neighbour excess over p_c, with three placebo controls",
    hypothesis=(
        "With p_c held fixed, the shrunk neighbour rate's excess over p_c adds >=0.02 "
        "drug_macro_auc on drugs with >=1 gene-level neighbour, and zero-neighbour, "
        "degree-preserving-permutation and within-condition-shuffle placebos are all "
        "null."
    ),
    approach=(
        "Baseline degree+p_c; nb_excess = shrunk_rate - p_c at three similarity tiers "
        "with leave-one-drug-out on train; grouped 3-fold CV scored on drug_macro_auc; "
        "three placebo refits; decay curve recomputed with p_c controlled."
    ),
    label="y_faers_signal",
    features=[
        "degree",
        "p_c",
        "nb_excess_gene",
        "nb_excess_class_leaf",
        "nb_excess_class_L1",
        "n_neighbours_gene",
    ],
    split="train/validate, grouped by primary target gene",
)
```
