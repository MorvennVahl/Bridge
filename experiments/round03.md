# Round 3 — measure the promise, and fix how we compare

Runtime contract unchanged (`experiments/README.md`), primary metric unchanged
(`experiments/METRIC.md`). Five experiments, all independent.

| file | what it settles | est. |
|---|---|---|
| `exp11_adjudicated_reference_set.md` | a yardstick that works — the SemMedDB ceiling is dead | ~4 min + hand-off |
| `exp12_transportability_by_target_distance.md` | the project's central promise, never yet measured | ~9 min |
| `exp13_subset_floors_and_residual_referee.md` | every subset gets its own floor; exp09's claim re-refereed | ~5 min |
| `exp14_gene_overlap_disease_arm.md` | gene overlap (exp10 re-cut to half scope) | ~8 min |
| `exp15_pathway_overlap_and_valid_placebos.md` | pathway overlap + a placebo design that works | ~9 min |

---

## What Round 2 established

**The bridge hypothesis has its first real support.** exp08's controlled, `p_c`-explicit
increment is **+0.0443 `drug_macro_auc`** (0.6268 vs 0.5825) on the 33.6% of validate pairs
with a gene-level neighbour, and the decay curve now behaves as the hypothesis predicted:
0.591 / 0.612 / 0.609 / **0.656** across neighbour buckets 0 / 1–2 / 3–5 / 6+. The
degree-preserving graph permutation returned +0.0096 ± 0.0021 and the label shuffle +0.0047,
so roughly four fifths of the gain survives the placebos that matter. **Placebo-adjusted
transport ≈ +0.035.** This is the strongest positive result the project has produced.

**All intrinsic features together buy +0.039.** exp07's honest ablation: `drug_macro_auc`
0.6148 against the 0.5759 floor, `drug_macro_p10` 0.2252 against 0.1843, pooled AP 0.1717.
Modest, real, and well below the provisional 0.65 target.

**The in-house yardstick is dead.** exp06 measured the ceiling at `drug_macro_auc` **0.4990**
(CI 0.4858–0.5155) over **13 eligible drugs** — chance, and underpowered. The incumbent FAERS
score ranked the SemMedDB harm target at 0.5042, also chance. And the floor on that same
target was **0.6149**, i.e. *above* both. When the floor beats the ceiling, the instrument is
broken, not the models: `solved_threshold` came out at 0.5374, below the floor it was built
from. The criterion in `METRIC.md` cannot be completed from in-house data.

**One exp06 number must never be cited: `benefit_ceiling_drug_macro_auc = 1.0000`.** It is
definitional. `y_semmeddb_treats` is derived from the same TREATS/PREVENTS sentences used to
score it, so the experiment measured a column against itself. My spec asked for that
comparison; it was a design error, not an execution error.

## Three method corrections that bind every experiment in this round

**1. Every subset gets its own floor, in its own metrics dict.** exp09 reported
`drug_macro_auc` 0.8330 (classifier) and 0.8274 (residual model) on the
`faers_case_count >= 3` subset and compared them to the all-rows floor of 0.5759. Measured
on that subset: Evans prevalence is 0.2466 (not 0.117), `p_c` **alone** scores 0.7754, and
**degree + `p_c` scores 0.8480** — above both of exp09's models. The restriction deletes one
of the three Evans clauses, so the label becomes far more separable. exp09's headline is a
deficit against the correct reference, and its recommendation to "model the continuous
residual in later rounds" is **not supported**. exp13 re-referees it and ships `floors.py`.

**2. exp08's placebo 1 was invalid by construction.** Comparing model B against model A on a
subset where B's extra features are constant does not isolate the feature: B was *fitted* on
all rows, so its whole tree structure differs and its predictions differ off-support. That,
not a leak, is the +0.0198 it reported. The valid form is to refit both models on the subset
alone, or to permute the feature's values at training time. Placebos 2 and 3 were correctly
designed and both came back near-null. **exp08's verdict of "at least one placebo came back
non-null" should therefore be read as an artefact of my placebo specification**, and its
+0.0443 stands as the better estimate. exp15 re-runs the battery in the valid form.

**3. Re-verify the volume before dispatch.** exp07 could not compute
`drug_macro_auc_unseen_family` because `ingredient_target_long.csv` was not on the volume,
so it logged the overall number (0.6148) in that slot instead — two identical values in one
metrics dict that mean different things. Assert every staged path exists and matches its
expected row count, and fail rather than substituting.

## What this round is for

Two of Round 2's four results were about our instruments rather than the biology. Round 3
splits accordingly: exp11 and exp13 fix the instruments; exp12, exp14 and exp15 use them.

exp12 is the one to read first if you only read one. The project's claim is transportability
to a drug with no data of its own, and that has still never been measured — `group_key` is
the primary target gene and is disjoint across folds by construction, so "unseen family" as
Round 1 and 2 defined it was never a restriction at all. Until the decay curve in target
space exists, the headline 0.6148 and 0.6268 describe interpolation, not transport.
