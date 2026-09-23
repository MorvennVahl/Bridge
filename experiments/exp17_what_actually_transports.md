# exp17 — Why the decay curve inverted, and whether the claim holds where it matters

**Follows** exp12 (`exp_20260922_c9b979`), which measured the curve and could not explain it.

**Status.** Not started.

---

## The observation to explain

Margin over each rung's own floor: D0 **+0.021**, D1 **+0.001**, D2 **−0.014**, D3 **+0.025**,
D4 **+0.085**. Slope −0.0023 per rung with CI [−0.0105, +0.0058]. The model does best,
relative to its floor, on the 208 drugs with **no target annotation at all**, and worst on
drugs that share only a protein class.

If performance were target-mediated, this is close to the reverse of what we would see.

## Question

Is the D4 margin biology, or is it the model separating a different *kind of substance* —
mixtures, minerals, botanicals, biologics — from small-molecule drugs? And does the D0 margin
survive in the population the project actually targets: a novel small molecule with a known
target?

## Hypothesis

The D4 margin is substance-type separation, not transport: it collapses when drug-type
indicators are removed and when the evaluation is restricted to small molecules. The D0
margin (+0.021) survives that restriction. Under a small-molecules-only restriction the curve
becomes **monotone decreasing** as originally hypothesised in exp12.

**What would falsify it.**

- **D0's margin does not survive.** Then no rung shows target-mediated transport in the
  target population, and exp08's +0.0443 is carried by drugs the project does not care about
  — a decision-grade negative that redirects Round 5 entirely.
- **D4's margin survives the restriction and the indicator removal.** Then something
  genuinely transportable is being learned about substances with no target annotation, and we
  have mischaracterised our own model. Find out what it is before building on it.

---

## Method

1. Characterise D4. Cross-tabulate its 208 drugs by `molecule_type`, `availability_label`,
   `exposure_type`, route flags, `first_approval` decade and CEM degree, against D0–D3. The
   expectation from Round 1 is a median CEM degree near 6 versus 157 for matched ingredients;
   confirm or correct it.
2. **Indicator-removal refit.** Refit exp07's union with every substance-type column removed
   (`molecule_type`, `availability_label`, `has_chembl_match`, route and exposure flags,
   anything whose non-null pattern encodes "not a conventional small molecule"). Recompute all
   five rungs with per-rung floors. The D4 margin should collapse; report by how much.
3. **Small-molecules-only curve.** Restrict every rung to ingredients with
   `molecule_type == 'Small molecule'` (or a ChEMBL structure present), recompute per-rung
   floors on that restricted population, and redraw the curve. Report the slope and its CI.
   This is the headline: the project's claim, evaluated on the population the project serves.
4. **Missingness as a feature, made explicit.** Fit a model whose *only* features are
   missingness indicators over the drug block plus degree and `p_c`. If it reaches the union
   model's D4 number, the D4 margin is entirely "we can tell this is not a regular drug".
5. Repeat rungs D0–D3 with exp08's neighbour model as well as the union, so the two are
   comparable at every rung under the same restrictions.
6. Report per-rung `first_approval` and CEM-degree distributions alongside every AUC, so drug
   age is visible as the standing confounder it is.

## Budget

| step | est. |
|---|---|
| rung construction, characterisation tables | 2 min |
| indicator-removal refit + 5 rungs with floors | 3 min |
| small-molecule restriction: refit + 5 rungs + floors | 3 min |
| missingness-only model, bootstraps, figure | 2 min |
| **total** | **~10 min** at `timeout=900` |

**If at risk:** drop step 4. Steps 2 and 3 are the experiment.

## Deliverables

`<exp_id>_rung_characterisation.csv`, `<exp_id>_curve_indicator_removed.csv`,
`<exp_id>_curve_small_molecules.csv`, `<exp_id>_missingness_only.csv`,
`<exp_id>_decay_curves.png` (three curves on one axis — original, indicator-removed,
small-molecules-only — each with its own floor line and n_drugs annotated),
`<exp_id>_per_drug.csv`.

## Reporting requirements

`metrics`: `margin_D0` … `margin_D4` for each of the three curves (`_orig`, `_noind`,
`_smallmol`), `slope_smallmol`, `slope_ci_lo_smallmol`, `slope_ci_hi_smallmol`,
`missingness_only_drug_macro_auc_D4`, `n_drugs_smallmol_D0` … `_D4`.

`findings` must answer two questions in its first three sentences: what the D4 margin is made
of, and whether D0's margin survives restriction to small molecules. Then the slope of the
restricted curve with its CI, and a one-sentence verdict on whether target-space distance
predicts performance in the population the project targets.

## Traps

- Restricting to small molecules shrinks every rung; report `n_drugs` per rung and mark any
  rung under 20 scoreable drugs as underpowered rather than reporting a point estimate.
- Substance type and drug age are entangled (minerals and botanicals are old and sparsely
  reported). Removing type indicators does not remove age; report both.
- Do not refit per rung (step 3 refits once on the restricted *training* population, then
  scores all rungs). Refitting per rung mixes population and training effects.

## Registration

```python
exp = ln.register(
    agent="exp17",
    title="Decomposing the inverted transport curve: substance type versus target distance",
    hypothesis=("D4's +0.085 margin is substance-type separation that collapses when type "
                "indicators are removed and under a small-molecule restriction, while D0's "
                "+0.021 margin survives and the restricted curve becomes monotone decreasing."),
    approach=("Characterise rungs by molecule type, availability, route, approval decade and "
              "degree; refit with all substance-type columns removed; rebuild the curve "
              "restricted to small molecules with per-rung floors; fit a missingness-only "
              "model as the explicit alternative hypothesis."),
    label="y_faers_signal",
    features=["intrinsic_union", "nb_excess_gene", "p_c", "degree", "missingness_indicators"],
    split="validate stratified by target-space distance, small-molecule restriction",
)
```
