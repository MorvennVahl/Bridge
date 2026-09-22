# data

Large files are stored with Git LFS. Run `git lfs pull` after cloning.

## Files

| file | rows | content |
|---|---|---|
| `cem_ingredient_condition_associations.csv` | 1,447,172 | One row per ingredient–condition pair from the OHDSI Common Evidence Model: FAERS disproportionality stats, EU label counts, SemMedDB relationship types. LFS, 115 MB. |
| `cem_ingredients.csv` | 4,276 | Distinct ingredients in the association file (concept id, name). |
| `ingredient_features.csv` | 4,280 | Drug-side features, one row per OMOP ingredient. LFS. |
| `target_features.csv` | 1,536 | One row per ChEMBL target × protein component. LFS. |
| `ingredient_target_long.csv` | 8,088 | One row per ingredient × target × gene × action type. LFS. |
| `ingredient_features_label_adjacent.csv` | 4,280 | CEM degree terms. Not predictors. |
| `data_dictionary.csv` | 245 | Every column in every feature table, with source and release. |
| `provenance.json` | | Per-source endpoints, parameters, row counts, retrieval timestamps. |
| `ref/` | | Downloaded reference data: MONDO SSSOM mappings, HPO ontology and gene/disease tables, Open Targets disease, phenotype, target, and Reactome parquet files. LFS. |

The feature tables are documented in [README_drug_features.md](README_drug_features.md).

## CEM files

Both CEM files are restricted to RxNorm `Ingredient` concepts on the drug side and standard
`Condition`-domain concepts on the outcome side. Source table: `cem_output.cem_unified` in
the local Postgres `cem` database.

Caveats:

- SemMedDB includes TREATS, PREVENTS, and NEG_* relationships, so a row is any documented
  association, not necessarily a harm.
- FAERS rows with a single case are included.
- Evidence on ancestor or descendant condition concepts is not rolled up.
- `in_eu_label` is false on every row in the current extract.

### Regenerating the association file

Requires the local Postgres `cem` database. Run from the repo root.

```bash
psql -d cem -c "\copy (select u.concept_id_1 as ingredient_concept_id, c1.concept_name as ingredient_name, u.concept_id_2 as condition_concept_id, c2.concept_name as condition_name, bool_or(u.source_id='aeolus') as in_faers, bool_or(u.source_id='eu_pl_adr') as in_eu_label, bool_or(u.source_id='semmeddb') as in_semmeddb, max(u.statistic_value) filter (where u.source_id='aeolus' and u.evidence_type='COUNT') as faers_case_count, max(u.statistic_value) filter (where u.source_id='aeolus' and u.evidence_type='PRR') as faers_prr, max(u.statistic_value) filter (where u.source_id='aeolus' and u.evidence_type='ROR') as faers_ror, max(u.statistic_value) filter (where u.source_id='aeolus' and u.evidence_type='CHI_SQUARE') as faers_chi_square, max(u.statistic_value) filter (where u.source_id='eu_pl_adr') as eu_label_count, string_agg(distinct u.relationship_id||'='||u.statistic_value::int, ';') filter (where u.source_id='semmeddb') as semmeddb_relationships from cem_output.cem_unified u join staging_vocabulary.concept c1 on c1.concept_id=u.concept_id_1 join staging_vocabulary.concept c2 on c2.concept_id=u.concept_id_2 where c1.concept_class_id='Ingredient' and c2.domain_id='Condition' and c2.standard_concept='S' group by 1,2,3,4 order by 2,4) to 'data/cem_ingredient_condition_associations.csv' csv header"
```

`cem_ingredients.csv` is the distinct `ingredient_concept_id, ingredient_name` pairs from
that file.
