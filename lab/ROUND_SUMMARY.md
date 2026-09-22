# Findings summary — Round 1 (exp01–05) and Round 2 (exp06, exp08, exp09; exp07/exp10 still running)

Scope: drug–condition pair prediction from FAERS/SemMedDB evidence, testing whether
biology (drug targets → genes → pathways → conditions) transports to a molecule with
no real-world data. Full detail is in `lab/notebook.jsonl`; this is a narrative digest.

## The headline shift: the metric itself was wrong in Round 1

Round 1 scored everything by pooled average precision (AP) over all 434k validate pairs.
Round 2 introduced `drug_macro_auc` (mean per-drug ROC-AUC, averaged over drugs with
≥20 pairs and both classes present — see `experiments/METRIC.md`) because pooled AP is
dominated by high-volume drugs and doesn't match the deployment question ("for this one
new molecule, which conditions should a reviewer look at"). Under the new metric, the
**floor is a one-line lookup table** — ranking a drug's conditions by each condition's
training flag rate `p_c` scores `drug_macro_auc` = 0.5759, beating exp01's fitted model
(0.5691). Every Round 2 experiment carries `p_c` as an explicit baseline column for
this reason.

## Round 1 results (original AP-based scoring)

| exp | question | result |
|---|---|---|
| **exp01** | Does observation frequency alone predict the FAERS harm label? | Validate AP 0.1208 vs 11.7% prevalence — barely above chance. Single-feature ranking was **not** stable across the 5 label definitions tested, so label-choice sensitivity is now mandatory in later reporting. |
| **exp02** | Do drug/condition-intrinsic features add over degree? | **Void** — its "degree" block accidentally included `faers_case_count`, one of the three literal clauses of the label. Its logged AP=0.3431 is a label leak, not a result. Corrected by exp07 (below). |
| **exp03** | Does same-target-neighbour behavior transport across the drug split? | +0.062 AP conditional on having ≥1 gene-level neighbour, but the gain didn't decay with neighbour count and was largest in the *zero-neighbour* bucket — a red flag that the feature was mostly re-encoding `p_c`. Re-examined with controls in exp08. |
| **exp04** | Is a degree-free disproportionality residual a better target than the raw flag? | **Structurally invalid** — its two-way (drug+condition) fixed-effects target assigned every held-out drug a 0 drug-offset by construction (no validate drug appears in train), so validate scored a different quantity than train (validate ρ=-0.31 vs train CV ρ=+0.49). Fixed in exp09 (below). |
| **exp05** | Is the efficacy label (`y_semmeddb_treats`) learnable from transportable features? | **Yes** — mechanism-only features clear 3× prevalence (AP 0.1275 vs 0.0048 prevalence), and the gap to a model with ATC/indication codes (+0.069 AP) exceeds the seed-to-seed noise, so it's a real but modest indication-encoding effect on top of real mechanism signal. |

## Round 2 results so far

**exp06 — what's the ceiling?** Ranked validate pairs by an evidence stream independent
of FAERS (SemMedDB harm mentions) and vice versa. Result: **ceiling ≈ 0.50** (drug_macro_auc,
95% CI [0.486, 0.515] over only 13 eligible drugs — badly underpowered), incumbent FAERS-vs-SemMedDB
number also ≈ 0.50, and the `p_c` floor *on the SemMedDB target* is 0.615. This is close to
the "ceiling near 0.5" failure mode the spec called out as the most consequential possible
outcome: on this thin proxy, FAERS and SemMedDB barely agree with each other at all, which
would mean no feature set can score much above 0.50–0.55 on this label using only this
evidence. The result is provisional and underpowered (n=13 drugs) — the spec's own
recommendation is to fetch an adjudicated reference set (OMOP/EU-ADR/OHDSI negative controls)
before trusting a hard threshold. The benefit-direction ceiling was 1.0 but on an extremely
sparse subset — not informative.

**exp08 — is exp03's neighbour-transport gain real, once `p_c` is controlled?** Partially.
With `p_c` explicit in the baseline, the neighbour-excess feature still adds **+0.044
drug_macro_auc** (bootstrap 95% CI [+0.018, +0.035]) on drugs with ≥1 gene neighbour, and
two of three placebo controls came back null (a degree-preserving graph permutation and a
within-condition label shuffle), meaning *some* of the effect is genuinely about drug-target
identity rather than just graph shape. But the **zero-neighbour placebo did not collapse to
0** as the theory requires (+0.020 instead of ~0.000), indicating a second `p_c`-like leak
is still present somewhere in the construction. Verdict: exp03's original +0.062 AP was
overstated by leakage, but not entirely fake — there's a real, smaller same-target transport
effect underneath it that needs one more round of cleanup to fully isolate.

**exp09 — fixing the residual target (supersedes exp04).** Replacing the two-way fixed
effects with one-way (condition-only) demeaning, and scoring with within-drug Spearman
correlation (which is invariant to the unlearnable per-drug constant), produces a target
that's actually estimable on held-out drugs: **macro within-drug ρ = 0.65** on validate
(train CV 0.69) — a strong, clean result, a large improvement over exp04's structurally
broken −0.31. The regressor's continuous output also ranks the binary Evans flag well
(drug_macro_auc 0.827), competitive with a classifier trained directly on it (0.833).
Recommendation from this entry: later rounds should prefer the continuous within-drug
residual target over the raw binary flag — it's better-behaved and the split-induced
target-mismatch trap is now understood and avoidable.

## Still running (will update)

- **exp07** — honest re-run of exp02's degree+intrinsic ablation with the label leak
  removed and `p_c` added; also ships a reusable `leak_audit.py` module now used by
  exp08–exp10. Currently mid-run (on its final, largest feature design).
- **exp10** — the project's first direct test of the central hypothesis: does gene/pathway
  overlap between a drug's targets and a condition's implicated genes add signal beyond
  intrinsics? Currently mid-run (fitting the `+gene_overlap` design; baseline E7union design
  already scored drug_macro_auc 0.615 on validate).

## What this means so far

1. The project's headline claim (biology transports across drugs) has **one supporting
   result** (exp08's controlled +0.044, not yet fully clean) and **one open, potentially
   serious problem** (exp06's near-0.5 ceiling, though underpowered) — the central test
   (exp10) is what will actually settle whether pathway-level biology adds anything, and
   it's still in flight.
2. Two of Round 1's four completed results were invalidated by methodology bugs (exp02's
   label leak, exp04's split-mismatch) — both are now fixed and superseded. This is a
   reminder that the leak-audit machinery introduced in exp07 is now load-bearing for
   every later experiment.
3. The efficacy side (exp05) and the within-drug residual reformulation (exp09) are the
   two cleanest positive results of the whole project so far.
