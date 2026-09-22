# Round 2 — repair the record, then test the biology

Five experiments. Same runtime contract as Round 1 (`experiments/README.md`): one Modal
invocation each, `timeout=600`, `cpu=8.0`, `memory=16384`, `bridge-data` volume staged with
train/validate only. **Primary metric is now `drug_macro_auc`** as defined in
`experiments/METRIC.md`; every experiment reports it against the measured floor of 0.5759.

| file | what it settles | supersedes | est. |
|---|---|---|---|
| `exp06_label_ceiling_and_incumbent.md` | the ceiling that completes the success criterion, and how good FAERS itself is at the target | — | ~3 min |
| `exp07_ablation_rerun_leak_audit.md` | honest intrinsic ablation + a reusable leak audit | `exp_20260922_f6645b` | ~8 min |
| `exp08_neighbour_transport_pc_stripped.md` | whether exp03's gain is target transport or condition base rate | `exp_20260922_341a64` | ~7 min |
| `exp09_within_drug_residual.md` | the residual target, on a quantity that exists on held-out drugs | `exp_20260922_3f4a30` | ~6 min |
| `exp10_gene_pathway_overlap.md` | the project's central hypothesis, first direct test | — | ~9 min |

Three of the five are repairs. That is the correct shape for this round: two of Round 1's
four completed results are not usable as written, and one baseline in the notebook is
wrong in a way that would make every later comparison look like progress.

---

## What Round 1 actually established

**The floor is prevalence, and `AGENT.md` trap #1 is wrong.** Degree-only on
`y_faers_signal`: pooled AP 0.1213 against prevalence 0.1173, pooled ROC-AUC **0.5126**,
precision@100 = 0.080 which is *below* the base rate. Independently reproduced against
exp01's logged 0.1208. "Degree explains a lot and will be higher than you expect" is not
true on this label; the bar for biology is prevalence, not a high floor.

**The strongest trivial model is a lookup table.** Ranking a drug's conditions by the
condition's train flag rate `p_c` gives `drug_macro_auc` 0.5759 / `drug_macro_p10` 0.1843,
which beats exp01's fitted three-feature model (0.5691 / 0.1504). Every model in this round
carries `p_c` as an explicit baseline column, because a feature that silently encodes it
will look like biology. Exp03's shrinkage prior is exactly such a feature.

**exp02 is void — its degree block contained a label term.**
`_build_degree_features` includes `degree_faers_case_count = log1p(faers_case_count)`, a
**per-pair** column, while the Evans label is `prr >= 2 & chi2 >= 4 & case_count >= 3`.
Measured: degree-only pooled AP 0.1156 / AUC 0.5022 → **0.3789 / 0.8400** when that column
is added; `log1p(faers_case_count)` alone scores AP 0.2336. This explains exp02's logged
`baseline_degree_only_ap = 0.3431` and its train CV AP of 0.9008. All four of its designs
share the contaminated block, so its increments (+0.0669 drug, +0.0468 condition) say
nothing. exp07 re-runs it and appends a `complete` with
`supersedes="exp_20260922_f6645b"`.

**exp04's sign flip has a structural cause, not a bug in the estimator.** The splits hold
out whole drugs, so when `_apply_offsets` does
`df["ingredient_concept_id"].map(drug_effect).fillna(0.0)`, **every validate row gets drug
offset 0** — no validate drug exists in train by construction. The validate target
therefore retains the full drug level while the train target had it removed, which is what
produced validate ρ = −0.3064 against train CV ρ = +0.4889. A two-way residual is not an
estimable target under a drug-grouped split. exp09 replaces it with a within-drug
formulation, which is the same object the primary metric measures.

**exp03's gain is unattributed.** +0.062 AP conditional on ≥1 gene neighbour, but the decay
curve is flat (0.189 / 0.187 / 0.190 for 1–2, 3–5, 6+ neighbours) and model B gains
+0.038 in the **zero-neighbour** bucket where the feature is undefined and collapses to
`p_c`. Tier order is non-monotonic (class_leaf 0.2016 > gene 0.1889 > class_L1 0.1222).
exp08 settles it with three placebo controls.

**exp05 never completed.** Six `register` entries, no `complete`. The efficacy question is
still open and is not re-cut here; run the Round 1 spec as written once the harness churn
that produced those duplicate registrations is understood.

## What became possible since Round 1

`data/input/condition/condition_gene_ot_long.csv` landed: **897,142 rows, 3,337 conditions,
17,273 genes**, with per-datatype scores (`dt_genetic_association`, `dt_literature`,
`dt_animal_model`, `dt_clinical`, …). Median **250 genes per condition**, so specificity
weighting is mandatory, and `dt_literature` must be separable from the rest because these
labels have a literature veneer. That unblocks exp10 — the first direct test of the bridge
hypothesis.

Note on the Reactome file: `data/ref/ot/reactome__*.parquet` is a pathway **hierarchy**
(2,870 pathways with ancestors/parents), not gene→pathway membership. Gene→pathway comes
from `data/ref/ot/target__part-*.parquet`.

## Order

exp06 first — the success criterion in `METRIC.md` is incomplete until the ceiling is
measured, and everything else is easier to interpret once we know what "as good as the data
allows" is. exp07 through exp09 are independent repairs. exp10 depends on nothing but is
most informative read after exp07, whose honest ablation is the number it has to beat.
