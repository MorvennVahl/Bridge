# Round 4 — settle whether this is a biological method or a nearest-neighbour method

| file | what it settles | est. |
|---|---|---|
| `exp16_reference_set_per_condition.md` | the external yardstick, via a metric the reference sets can actually support | ~5 min |
| `exp17_what_actually_transports.md` | why the decay curve inverted, and whether the claim holds for small molecules | ~10 min |
| `exp18_label_free_vs_label_copying.md` | **the round's central question** | ~12 min |
| `exp19_offtarget_panel_imputation.md` | make the strongest measured effect usable | ~14 min |
| `exp20_efficacy_measured_properly.md` | the efficacy half, on the metric and floors the safety half uses | ~10 min |

**Runtime contract change.** Round 3 was budget-starved: exp14 spent 416 s and 553 s on
single designs and **dropped cross-validation entirely** rather than exceed ten minutes, then
logged three fallbacks. Round 4 runs at `timeout=900`, `cpu=16.0`, `memory=32768`. A fallback
that removes CV or a stratification is still logged, but it should no longer be routine.
Everything else in `experiments/README.md` and `METRIC.md` stands; `floors.py` and
`leak_audit.py` are now mandatory in every run.

---

## What Round 3 established

**The decay curve did not decay — it inverted.** exp12, margin over each rung's **own** floor:

| rung | definition | drug_macro_auc | own floor | margin | n drugs |
|---|---|---|---|---|---|
| D0 | shares a target gene with a train drug | 0.6270 | 0.6058 | +0.021 | 192 |
| D1 | shares a protein-class leaf | 0.5931 | 0.5923 | +0.001 | 106 |
| D2 | shares a protein-class L1 | 0.5618 | 0.5757 | **−0.014** | 89 |
| D3 | annotated, shares nothing | 0.5859 | 0.5614 | +0.025 | 91 |
| D4 | **no target annotation at all** | 0.6314 | 0.5463 | **+0.085** | 208 |

Slope −0.0023 per rung, CI [−0.0105, +0.0058] — flat. The largest margin is at D4, where
there is no target information whatsoever, and the model is *below* its floor at D2. Whatever
is working is not a decaying function of target-space distance. Target-mediated transport, if
it exists at all, is confined to D0 — an exactly shared gene — and does not extend to protein
class. exp17 decomposes this.

**Label-free biology came in at noise.** exp14: gene overlap adds **+0.0090** over exp08's
neighbour base (0.6225 → 0.6314) against a noise threshold of ~0.01. Genetic-association
weighting added +0.0025; `dt_literature` weighting added +0.0061 — **publication attention
beat causal genetic evidence**, which is the opposite of the project's premise. Only 10.4% of
disease-arm pairs share any gene at all.

Set that beside exp08's neighbour signal (+0.0443, placebo-adjusted ~+0.035) and the position
is uncomfortable but clear: **the only feature that works reads other drugs' labels.** A
nearest-neighbour method over the drug–target graph is a legitimate product, but it is not
the biological bridge this project set out to build, and it cannot score a condition nobody
has been flagged for. exp18 is designed to settle which one we have.

**exp13 confirmed the subset-floor finding and closed the continuous-target question.** The
`case_count >= 3` floor is 0.8449 (degree + `p_c`, min 20 pairs/drug); exp09's 0.8330 and
0.8274 sit below it. On full rows the residual model scores **0.4483** against a 0.5967 floor,
and the rank metric decomposes cleanly: `p_c` alone ρ = 0.195, degree + `p_c` ρ = 0.7448,
residual regressor ρ = 0.7455. exp09's ρ = 0.6536 was degree and `p_c` the whole time.
**Round 4 stays binary.**

**Two preconditions failed honestly, and both were the specs' fault, not the data's.**

- exp11: the reference sets are **drug-by-HOI**, not drug-by-condition — 472 pairs, 223
  drugs, **8 conditions**, so only 3 drugs have ≥5 reference pairs against the 30 the spec
  required. The per-drug metric cannot be computed on them. But the transpose is dense: four
  outcomes carry 97–127 pairs each, with **real adjudicated negatives** (274 of 472). The fix
  is a metric, not a bigger fetch. exp16.
- exp15: `0.000 of 977` drug target genes mapped to pathways. The data is present — the OT
  target parquet has a `pathways` column (`pathwayId`, `pathway`, `topLevelTerm`) keyed by
  `approvedSymbol` / `id` (ENSG). The join key was wrong. Given exp14, a large pathway effect
  is unlikely, so pathway overlap does not get its own slot this round; it is folded into
  exp18's label-free block with the join key named.

**The efficacy half looks better than the safety half.** exp05 completed: mechanism-only
validate AP **0.1275** against a 0.48% prevalence — 26× — with an M-versus-M+I gap of +0.0686
above a 3-seed spread of 0.0324, and summed ATC importance of exactly 0. But it was measured
in pooled AP with no floors, no per-drug metric, CV skipped on the M+I design, and its step 5
never attempted. That number is too interesting to leave in that state. exp20.

## Order

exp16 first — it is cheap and it is the only external check the project has. exp18 is the one
that matters; read it before exp17 and exp19, both of which are, in different ways, attempts
to give the label-free side a fair chance before we conclude anything about it.
