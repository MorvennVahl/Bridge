# exp06 — The ceiling, and how good the incumbent method is

**Status.** Not started. **Run this first.** The success criterion in `experiments/METRIC.md`
is a formula with one unmeasured term; this experiment measures it.

---

## Question

How well can the FAERS harm label be ranked *at all* by an evidence stream that is
independent of FAERS — and how well does FAERS disproportionality itself, the method our
model is meant to replace for drugs with no data, rank an independent harm target?

## Hypothesis

`drug_macro_auc` for an independent-evidence ranking of `y_faers_signal` lands in
**0.60–0.75**, materially below 1.0, and the incumbent FAERS score ranks SemMedDB-asserted
harms at a similar level. Both numbers being modest is the expected outcome and is what
makes a biology model at 0.65 a success rather than a disappointment.

**What would falsify it, in both directions.**

- **Ceiling near 0.5.** The two evidence streams are unrelated, so `y_faers_signal` carries
  almost no transferable harm signal. Then no feature set can succeed on it, the success
  criterion cannot be met on this label, and the project's priority becomes acquiring an
  adjudicated reference set — not better features. This is the single most consequential
  outcome available in this round.
- **Ceiling near 0.9.** The label is far more reproducible than assumed, the provisional
  0.65 target is much too lax, and it should be reset to `floor + 0.5 * (ceiling - floor)`
  immediately.

---

## Inputs

| path | rows | note |
|---|---|---|
| `/splits/train.csv` | 723,586 | `p_c` floor, train-only rates |
| `/splits/validate.csv` | 434,151 | evaluation fold |
| `data/output/pair_labels.csv` | 1,447,172 | `semmeddb_harm_sentences`, `semmeddb_benefit_sentences` |

**Measured coverage of the independent stream** — this is thin and the spec is built around
that: **1,369 pairs** carry ≥1 SemMedDB harm sentence, over **418 drugs** and **567
conditions** (7,299 pairs carry benefit sentences; 10,394 are in SemMedDB at all). Assert
these counts at startup.

---

## Method

1. **Ceiling, as `METRIC.md` defines it.** Score each validate pair by its SemMedDB harm
   evidence (`semmeddb_harm_sentences`, and a binary variant), rank within drug against
   `y_faers_signal`, and compute `drug_macro_auc` over eligible drugs. Because most drugs
   have no harm assertion, restrict to drugs with ≥1 harm-asserted pair **and** ≥20 pairs,
   and report `n_drugs_scored` prominently — if that count is under ~50, the ceiling is
   underpowered and must be reported as a CI, not a point.
2. **Incumbent, reverse direction.** Score pairs by the FAERS quantities (`log(faers_prr)`,
   `faers_chi_square`, and the binary Evans flag) and rank within drug against the
   **SemMedDB harm** target. This is the operationally meaningful number: how well does
   today's disproportionality screening identify literature-asserted harms? It is the
   benchmark a biology model must match to claim it substitutes for RWD.
3. **Floor on the same target.** `p_c` lookup (train harm rate per condition) ranked against
   the SemMedDB harm target, so the independent-target evaluation has its own floor. Do not
   reuse 0.5759 — that floor belongs to `y_faers_signal`.
4. **Bootstrap CI** over drugs, 1,000 resamples, for every number above. With n_drugs likely
   in the tens, the CI is the result and the point estimate is decoration.
5. **Benefit-direction counterpart.** Same as (1) with `semmeddb_benefit_sentences` against
   `y_semmeddb_treats`, to establish whether the efficacy half has any measurable ceiling
   at all. Report separately; do not pool with harm.
6. **Write the criterion.** Emit `<exp_id>_success_criterion.json` with `floor`, `ceiling`,
   the resulting `solved_threshold = floor + 0.5 * (ceiling - floor)`, and the CI. Every
   later experiment reads that file rather than hardcoding a target.

## Hand-off: the adjudicated version (Claude Code, not Modal)

The SemMedDB ceiling is a proxy and it is thin. The defensible version uses a curated
reference set with adjudicated positives *and* negatives — the OMOP reference set (Ryan et
al. 2013), EU-ADR, or the OHDSI negative-control sets used for empirical calibration. Those
require unrestricted network access, so they are a spec for the Claude Code session rather
than work for this container:

- fetch the reference set, map its drug and outcome concepts to `ingredient_concept_id` /
  `condition_concept_id`, and write `data/input/reference/reference_set.csv` with columns
  `ingredient_concept_id, condition_concept_id, label, source, adjudication`
- report coverage against our 4,276 × 5,631 grid — how many reference pairs survive the join

Once that file exists, rerun steps 1–4 against it and treat *that* as the ceiling of record.
Until then the `success_criterion.json` carries `source: "semmeddb_proxy"` and everything
downstream is provisional.

## Budget

Small — no model fitting, only ranking and bootstrapping. ~3 min including IO.
`timeout=600`. This one is also cheap enough to run locally if Modal is busy; say which you
used.

## Deliverables

- `<exp_id>_success_criterion.json` — the machine-readable criterion for all later work
- `<exp_id>_ceiling.csv` — ceiling, incumbent and floor numbers with bootstrap CIs and
  `n_drugs_scored` for each
- `<exp_id>_per_drug.csv` — per-drug AUCs behind each macro number
- `<exp_id>_power_note.md` — one paragraph on whether the SemMedDB ceiling is powered enough
  to fix a threshold, stated plainly

## Reporting requirements

`metrics`: `ceiling_drug_macro_auc`, `ceiling_ci_low`, `ceiling_ci_high`,
`incumbent_faers_drug_macro_auc`, `floor_drug_macro_auc_on_harm_target`, `n_drugs_scored`,
`solved_threshold`.

`findings` must state the ceiling with its CI, whether it is powered, the incumbent number,
and — explicitly — the revised `solved_threshold` or a statement that it remains provisional
pending the adjudicated set.

## Traps

- **Do not use the SemMedDB columns as predictors anywhere else after this.** They are the
  independent yardstick; a model that trains on them forfeits the only external check
  available.
- 1,369 harm pairs across 567 conditions means many drugs contribute a single positive. A
  per-drug AUC from one positive is a coin flip on rank; weight by nothing, but report the
  distribution of positives per drug.
- SemMedDB tracks publication attention, so its coverage correlates with drug age. Report
  the ceiling separately for drugs approved before and after 2000 if n permits.

## Registration

```python
exp = ln.register(
    agent="exp06",
    title="Label ceiling from independent evidence, and the incumbent FAERS comparator",
    hypothesis=(
        "An evidence stream independent of FAERS ranks y_faers_signal within drug at "
        "drug_macro_auc 0.60-0.75, and FAERS disproportionality ranks SemMedDB-"
        "asserted harms at a similar level; both bound what any feature set can do."
    ),
    approach=(
        "Within-drug ranking in both directions plus a p_c floor on the independent "
        "target, 1,000-resample bootstrap CIs over drugs, emitting a machine-readable "
        "success criterion. Adjudicated reference set handed to Claude Code."
    ),
    label="y_faers_signal",
    features=["semmeddb_harm_sentences", "faers_prr", "faers_chi_square", "p_c"],
    split="validate only (no fitting)",
)
```
