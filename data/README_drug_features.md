# Bridge — ingredient-side feature dataset

Drug-side features for the Bridge drug–condition problem, at **OMOP RxNorm Ingredient
grain**: one row per ingredient concept, selected for whether the feature plausibly
carries information about how the drug behaves when given to real patients.

This is the input side of the `ingredient —RWD— condition` edge. It deliberately
contains **no** CEM-derived predictors; the CEM-derived columns live in a separate
file so they cannot be used as features by accident.

## Files

| file | rows × cols | grain |
|---|---|---|
| `ingredient_features.csv` | 4,280 × 159 | one row per OMOP ingredient concept |
| `target_features.csv` | 1,536 × 61 | one row per ChEMBL target × protein component |
| `ingredient_target_long.csv` | 8,088 × 18 | one row per ingredient × target × gene × action type |
| `ingredient_features_label_adjacent.csv` | 4,280 × 7 | **not predictors** — CEM degree terms |
| `data_dictionary.csv` | 245 | every column in every table, with source, release and endpoint |
| `provenance.json` | — | per-source endpoint, query parameters, row counts, retrieval timestamps |

4,280 rows = the 4,276 CEM ingredients, plus semaglutide and ertugliflozin (in the
indication roster but absent from `cem_ingredients.csv`), plus the two non-drug
exposures from the depression block (electroconvulsive therapy, psychotherapy), which
carry identity columns only.

## Feature blocks

| block | cols | what it is | why it may matter in patients |
|---|---|---|---|
| identity | 18 | OMOP / ChEMBL / KEGG ids and how each was resolved | joins and audit |
| development | 6 | max phase, first approval year, therapeutic flag, availability | approval era and Rx/OTC status shape who receives the drug |
| exposure | 4 | oral / parenteral / topical route flags, route count | route governs systemic exposure |
| pharmacology | 9 | ATC codes and level-1/level-4 groupings, USAN stem, indication class | breadth of clinical use across organ systems |
| chemistry | 24 | ChEMBL physicochemical properties, rule-of-five | absorption, distribution, CNS penetration |
| mechanism | 30 | targets, action types, mechanism strings, target-class composition | the transportable biology |
| target biology | 38 | Open Targets tractability, gnomAD constraint, curated target safety liabilities, DepMap essentiality, Reactome membership | on-target consequences of perturbing the target |
| metabolism | 14 | CYP/UGT metabolising enzymes, transporters, named metabolites | clearance, variability, accumulation |
| interactions | 5 | KEGG DDI partner genes, CYP inhibition/induction flags | drug–drug interaction potential |
| safety | 11 | ChEMBL boxed-warning and withdrawal records with toxicity classes, openFDA boxed-warning presence | prior regulatory signal |

## Coverage (of 4,280 rows)

| | n | % |
|---|---|---|
| matched to a ChEMBL molecule | 2,932 | 68.5% |
| physicochemical properties | 2,471 | 57.7% |
| ≥1 ATC code | 2,239 | 52.3% |
| max_phase = 4 (approved) | 2,013 | 47.0% |
| ChEMBL mechanism with an assigned target | 1,716 | 40.1% |
| mechanism recorded but target unassigned | 247 | 5.8% |
| matched to a KEGG DRUG entry | 2,557 | 59.7% |
| KEGG metabolising enzymes | 425 | 9.9% |
| KEGG DDI partner genes | 150 | 3.5% |
| ≥1 ChEMBL drug_warning record | 682 | 15.9% |
| ChEMBL black-box warning flag | 523 | 12.2% |
| withdrawn in ≥1 jurisdiction | 179 | 4.2% |
| ≥1 Open Targets target safety liability | 1,015 | 23.7% |
| openFDA label aggregation hit | 372 | 8.7% |

Target side: 619 ChEMBL targets from the mechanism join, of which 558 have protein
components, resolving to 1,317 UniProt accessions and **790 human genes** (all 790
returned by Open Targets).

The 31.5% of ingredients with no ChEMBL match are mixtures ("Multivitamin
preparation"), biologics and blood products (insulins, immunoglobulins, glatiramer),
minerals and ions (iron, zinc, lactate), botanicals, excipient-like substances
(polyethylene glycols), and `placebo`. They hold **5.8% of CEM ingredient–condition
pairs**; median CEM degree is 6 for unmatched vs 157 for matched ingredients, so the
match covers the model-relevant mass of the label file.

## Provenance

Every column in `data_dictionary.csv` carries `source`, `source_release`, `endpoint`,
`dtype` and `non_null_fraction`. Sources used:

- **ChEMBL_37** (released 2026-05-01), EBI REST API — molecule records and
  physicochemical properties, `drug_mechanism`, `drug_warning`, `metabolism`,
  `target`, `target_component`, `protein_classification`.
- **UniProtKB** REST — protein names, gene symbols, length, keywords, subcellular
  location, function text.
- **Ensembl** REST — HGNC symbol → Ensembl gene id.
- **Open Targets Platform** GraphQL — tractability buckets, gnomAD genetic constraint,
  curated safety liabilities, DepMap essentiality, Reactome pathway membership.
- **KEGG DRUG** REST — metabolising enzymes, transporters, interaction partners, efficacy.
- **openFDA** SPL label count aggregations — presence of boxed-warning, pregnancy and
  drug-interaction label sections.
- **OMOP/RxNorm via CEM** — ingredient concept ids and names.
- **User-supplied indication roster** — the 97 depression / T2D / hypertension exposures
  with their pharmacologic classes.

Identifier columns (`chembl_id`, `chembl_parent_id`, `accession`, `ensembl_gene_id`,
`kegg_drug_id`) are retained so any value can be traced to its source record. Raw API
responses are kept under `handoff/`.

## Deliberate exclusions and caveats

1. **No fuzzy name matching.** A conservative fuzzy pass (difflib ≥ 0.93) produced 68
   candidates, of which many were wrong in ways that inject false biology —
   *4-hydroxybenzoic acid* → salicylic acid, *bromine* → betaine, *factor IX* → factor X,
   *methylene chloride* → vinyl chloride. All 68 are preserved in
   `chembl_fuzzy_candidate_*` for manual review and **no feature reads from them**. A
   handful are genuine misspellings worth promoting by hand (`Chlopheniramine`,
   `Ibadronate`, `providone-Iodine`, `solfenacin`, `Levabuterol`).
2. **Salts resolved to parents.** Name matches are mapped through ChEMBL
   `molecule_hierarchy` to the parent molecule before joining mechanisms and warnings;
   without this, e.g. saxagliptin's DPP-4 mechanism is missed because it sits on the
   hydrochloride record.
3. **Family and complex targets inflate gene counts.** ChEMBL assigns some mechanisms
   to a protein family or complex rather than a single protein — metformin's target
   expands to 51 respiratory-complex-I subunits, verapamil's to 4 calcium-channel
   α-subunits. `has_family_or_complex_target` (609 ingredients) flags these;
   `component_relationship` in `target_features.csv` distinguishes subunits from group
   members. Do not read `n_target_genes` as polypharmacology breadth without it.
4. **`mechanism_recorded_but_target_unassigned`** separates "ChEMBL has no mechanism"
   from "ChEMBL documents a mechanism but names no target" (247 ingredients, e.g.
   hydralazine). These are different kinds of missingness for imputation.
5. **Label-adjacent columns are quarantined.** `ingredient_features_label_adjacent.csv`
   holds CEM drug-side degree (`cem_n_conditions`, FAERS pair counts, median PRR).
   These share the FAERS lineage of the labels; per DESIGN.md they belong in a model as
   nuisance/degree terms, never as biology predictors.
6. **openFDA count aggregations return the top 1,000 terms only.** Absence from an
   aggregation is not zero. FAERS report-volume aggregations were **not** retrieved —
   openFDA now requires an API key for large count queries on `/drug/event.json`
   (`API_KEY_MISSING`); the label endpoints do not.
7. **No off-target secondary pharmacology yet.** hERG, muscarinic, α1, H1, D2 and
   5-HT2B affinities are the features most directly tied to real-world adverse events
   (QT prolongation, falls, delirium, sedation), and they require the ChEMBL
   `activities` table (24.5M rows) — not feasible over the REST API, but a single join
   against a local ChEMBL copy.
8. Nothing is filled in from model recall. Where a source has no value the cell is null
   and the dictionary records which source was queried.

## Reproducing

```
python fetch_chembl_bulk.py        # ChEMBL molecule / mechanism / warning / metabolism tables
python fetch_target_annotations.py # ChEMBL targets -> UniProt -> Ensembl -> Open Targets
python fetch_protein_classes.py    # ChEMBL target components + protein classification
python fetch_kegg.py               # KEGG DRUG name index + entry records
python fetch_openfda.py            # openFDA SPL label count aggregations
python build_features.py           # assembles all tables + data_dictionary.csv
```

`build_features.py` fails loudly rather than silently if a new column has no registered
source, and asserts that the final row count equals the ingredient roster length.
