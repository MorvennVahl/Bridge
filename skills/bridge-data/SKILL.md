---
name: bridge-data
description: How to load and join the Bridge repo's data files (CEM ingredient–condition associations, drug-side feature tables, ontology reference data). Use when a task reads anything under data/ in this repo, builds features or labels, or asks which file or join key to use.
---

# Bridge data

All of `data/` is Git LFS. Run `git lfs pull` first; an unpulled file is a ~130-byte pointer.

## Files and keys

| file | grain | key |
|---|---|---|
| `cem_ingredient_condition_associations.csv` | ingredient × condition (1.45M) | `ingredient_concept_id`, `condition_concept_id` |
| `cem_ingredients.csv` | ingredient (4,276) | `ingredient_concept_id` |
| `ingredient_features.csv` | ingredient (4,280) | `omop_concept_id` |
| `ingredient_target_long.csv` | ingredient × target × gene × action | `omop_concept_id`, `target_chembl_id` |
| `target_features.csv` | target × protein component | `target_chembl_id` |
| `ingredient_features_label_adjacent.csv` | ingredient | `omop_concept_id` |
| `ref/` | MONDO SSSOM, HPO, Open Targets parquet | ontology ids |
| `ref/omop/concept.csv` | condition (5,631) | `condition_concept_id`, `concept_code` (SNOMED) |
| `ref/omop/concept_source_codes.csv` | condition × source vocabulary | `condition_concept_id` |
| `ref/omop/concept_ancestor.csv` | condition × ancestor | `condition_concept_id` |
| `ref/omop/concept_synonym.csv` | condition × synonym | `condition_concept_id` |

`omop_concept_id` and `ingredient_concept_id` are the same OMOP RxNorm Ingredient id. Column
definitions are in `data_dictionary.csv`.

## Rules

- **Labels vs features.** The CEM association file is the label source. Never join it, or
  `ingredient_features_label_adjacent.csv`, into model inputs.
- **Split by ingredient**, never by pair.
- **Associations are not harms.** `semmeddb_relationships` includes TREATS, PREVENTS, NEG_*.
  Parse it before labelling direction.
- **FAERS is noisy.** Median case count is 2. Threshold `faers_case_count` and `faers_prr`.
- **`in_eu_label` is false on every row.** Ignore that column.
- **Conditions are SNOMED only.** Only ~35% map to MONDO by exact SNOMED code. Mapping goes
  through `bridge/condition_map.py`; pass it `condition_concept_codes()`, which reads the
  OMOP vocabulary export at `data/ref/omop/concept.csv`. Omit it and tier 1 is skipped
  silently.

## Load

```python
import pandas as pd

assoc = pd.read_csv("data/cem_ingredient_condition_associations.csv")
feats = pd.read_csv("data/ingredient_features.csv")
X = assoc.merge(feats, left_on="ingredient_concept_id", right_on="omop_concept_id")
```

Regeneration and provenance: `data/README.md`, `data/provenance.json`.
