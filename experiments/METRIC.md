# The metric we optimize, and what "solved" means

Status: **proposed**, floor measured. The success threshold is provisional until the
ceiling is measured (see §4) — it cannot be set honestly before then.

---

## 1. The metric

```
drug_macro_auc = mean over held-out drugs d of  ROC-AUC( y[d, :], score[d, :] )
```

For each held-out drug, rank **its** conditions and take the AUC over that ranking.
Macro-average across drugs. Eligibility: a drug enters the average if it has ≥ 20 observed
pairs and both classes present. On validate that is **686 of 1,284 drugs**; report the
count every time, and report pooled AP over all pairs as a coverage-complete secondary so
excluded drugs are not silently dropped from the story.

**Why per-drug, and why AUC inside the drug.**

- **It is the deployment unit.** The project's question is "here is a molecule with no RWD
  — which conditions should a reviewer look at". That is one ranking per drug. Pooled
  metrics over 434,151 pairs answer a question nobody asks, and are dominated by volume:
  the top decile of drugs by pair count holds 35.3% of validate pairs.
- **The drug-level reporting confound cancels exactly.** Within a drug, every condition
  shares the same drug reporting propensity, so drug degree — the thing `AGENT.md` §5 warns
  carries the label — cannot contribute. It is differenced out by construction rather than
  controlled for by hoping the model treats it as a nuisance term.
- **The null is fixed at 0.5 for every drug**, whatever its base rate. Per-drug positive
  rates vary (median 0.109, IQR 0.068–0.153), so a macro-averaged AP or AP-lift mixes
  discrimination with base rate; macro AUC does not. Measured across-drug spread: AUC
  sd 0.128 versus AP-lift sd 3.95. A metric you optimize must not be hostage to a handful
  of drugs with two positives — **AP-lift is rejected as the primary for that reason.**

## 2. Secondary metrics, always reported beside it

- `drug_macro_p10` — mean precision@10 per drug. AUC is insensitive to the top of the
  ranking and the decision lives there; this is the reviewer-facing number.
- `drug_macro_r50` — mean recall@50 per drug.
- `pooled_average_precision` — all validate pairs, for continuity with exp01–exp05 and the
  notebook leaderboard.
- `n_drugs_scored` — coverage.

## 3. The floor, measured

`y_faers_signal`, validate fold, 686 eligible drugs. Two trivial models and exp01's
learned baseline:

| model | drug_macro_auc | median | drug_macro_p10 |
|---|---|---|---|
| rank by condition's train flag rate `p_c` | **0.5759** | 0.5969 | **0.1843** |
| rank by condition train degree | 0.5583 | 0.5405 | 0.0981 |
| exp01 learned degree model (3 features, HistGBM) | 0.5691 | 0.5368 | 0.1504 |
| per-drug base rate (chance) | 0.5000 | — | 0.1142 |

**The floor is the `p_c` lookup table, not exp01's model.** A one-line ranking by each
condition's training flag rate beats the fitted three-feature model on both the primary and
the secondary metric. Any claim of progress is measured against 0.5759 / 0.1843.

This also disposes of a related question: `p_c` must be an explicit baseline column in every
later model, because a feature that silently encodes it (exp03's shrinkage prior does) will
look like biology.

Reference points for context: pooled ROC-AUC for the same exp01 model is 0.5126 and pooled
AP 0.1213 against prevalence 0.1173 — the pooled view reads as pure noise while the per-drug
view shows there is a little within-drug structure. That difference is the argument for the
metric, not an inconsistency.

## 4. What "solved" means — and why the threshold is not ours to invent

A biology model cannot predict the part of the label that is not reproducible. So the
success criterion is stated as a fraction of the distance between two measured numbers:

```
floor    = 0.5759   (condition-prior lookup, measured)
ceiling  =    ?     (label reproducibility, to be measured)
solved   when drug_macro_auc >= floor + 0.5 * (ceiling - floor)
```

The ceiling is how well an evidence stream that is **independent of FAERS** ranks the same
label within drug — SemMedDB harm assertions against `y_faers_signal` is the cheap version
available today, and a split-half or temporal reproducibility of the disproportionality flag
is the better version if report-level FAERS can be obtained. Whatever that number is, it
bounds what any feature set can achieve, and reaching half the floor-to-ceiling gap with
biology alone — no RWD for the drug in question — is a defensible claim of success.

**Measuring the ceiling is therefore the first experiment of the next round**, not an
afterthought.

**Provisional target, to be replaced by the formula above:** `drug_macro_auc >= 0.65` with
`drug_macro_p10 >= 0.25` — a reviewer who opens ten conditions finds 2.5 true signals
instead of 1.8. Stated so it can fail.

**Two requirements that come with it, not after it:**

1. **Transportability.** The headline must hold on drugs whose primary-target family never
   appears in train, within 0.02 of the overall number. A model that works only on drugs
   resembling training drugs does not address the project's question, whatever its average.
2. **Margin over the floor must exceed noise.** Report a bootstrap CI over drugs (1,000
   resamples of the 686) for the difference against the `p_c` lookup. With sd 0.128 across
   drugs, a +0.01 macro AUC difference is not a result.

## 5. How to optimize it

Optimize `drug_macro_auc` **inside train** under grouped 3-fold CV on `group_key`, using the
same per-drug macro computation on held-out CV drugs. Validate keeps its role: one
confirmation per experiment, after the choices are fixed. Do not tune against validate —
`labnotebook.validate_evaluations()` already counts the queries.

Log these keys so the notebook stays comparable across rounds (append, never replace —
`leaderboard()` ranks on `validate_average_precision`):

```
drug_macro_auc, drug_macro_auc_unseen_family, drug_macro_p10, drug_macro_r50,
n_drugs_scored, pooled_average_precision, validate_average_precision,
floor_drug_macro_auc_pc_lookup
```
