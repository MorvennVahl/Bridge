# OMOP vocabulary export — request

Four `\copy` exports from the local `cem` Postgres, needed to attach biology features to the
condition side of the CEM table. Each is a single self-contained command; they can be run in
any order. Total output is roughly 30 MB.

**Why this is needed.** `cem_ingredient_condition_associations.csv` carries only
`condition_concept_id` and `condition_name`. Every external reference source we want to join
(MONDO, Open Targets, HPO) keys on ontology *codes*, not OMOP concept ids. Without these
exports there is no join key, and name matching alone reaches only 24.7% of the 5,631
conditions.

All four reuse the same `conditions` CTE, which re-derives the exact 5,631-concept set using
the same filter as the association-file query in [README.md](../README.md) — so the sets are
guaranteed to line up.

Write outputs to `data/ref/omop/`. Anything under `data/` is Git LFS-tracked, so commit them
normally; no special handling.

```bash
mkdir -p data/ref/omop
```

---

## 1. Concept attributes — `concept.csv`

The core ask: `concept_code` and `vocabulary_id` for each condition.

Expected: **5,631 rows**. If it differs, stop and tell us — it means the concept set has
drifted from the association file and everything downstream is misaligned.

```bash
psql -d cem -c "\copy (with conditions as (select distinct u.concept_id_2 as condition_concept_id from cem_output.cem_unified u join staging_vocabulary.concept c1 on c1.concept_id=u.concept_id_1 join staging_vocabulary.concept c2 on c2.concept_id=u.concept_id_2 where c1.concept_class_id='Ingredient' and c2.domain_id='Condition' and c2.standard_concept='S') select c.concept_id as condition_concept_id, c.concept_name, c.domain_id, c.vocabulary_id, c.concept_class_id, c.standard_concept, c.concept_code, c.valid_start_date, c.valid_end_date, c.invalid_reason from conditions k join staging_vocabulary.concept c on c.concept_id=k.condition_concept_id order by 1) to 'data/ref/omop/concept.csv' csv header"
```

## 2. Source-vocabulary codes — `concept_source_codes.csv`

**This is the one that decides how much coverage we get**, so please don't add a vocabulary
filter — export all of them.

The reasoning: MONDO's mapping file carries 9,124 SNOMED xrefs but also 8,183 MeSH, 2,117
ICD10CM, 10,045 OMIM and 9,785 Orphanet. OMOP standard Condition concepts are SNOMED, so
SNOMED alone caps our join. Walking `concept_relationship` backwards to every non-standard
source concept that maps *to* each condition gives us four or five join keys per concept
instead of one.

Expected: roughly 50k–150k rows, wide variance depending on which vocabularies are loaded.

```bash
psql -d cem -c "\copy (with conditions as (select distinct u.concept_id_2 as condition_concept_id from cem_output.cem_unified u join staging_vocabulary.concept c1 on c1.concept_id=u.concept_id_1 join staging_vocabulary.concept c2 on c2.concept_id=u.concept_id_2 where c1.concept_class_id='Ingredient' and c2.domain_id='Condition' and c2.standard_concept='S') select cr.concept_id_2 as condition_concept_id, c.vocabulary_id as source_vocabulary_id, c.concept_code as source_concept_code, c.concept_name as source_concept_name, c.concept_class_id as source_concept_class_id from conditions k join staging_vocabulary.concept_relationship cr on cr.concept_id_2=k.condition_concept_id join staging_vocabulary.concept c on c.concept_id=cr.concept_id_1 where cr.relationship_id='Maps to' and cr.invalid_reason is null order by 1,2,3) to 'data/ref/omop/concept_source_codes.csv' csv header"
```

**One thing worth checking while you're in there:** does this vocabulary build include an
`HPO` vocabulary_id? Run
`select vocabulary_id, count(*) from staging_vocabulary.concept group by 1 order by 2 desc;`
and send us the output. Roughly three quarters of the unmapped conditions are symptoms and
findings ("abdominal bloating", "abnormal breath sounds") rather than diseases — those will
never map to MONDO, but they map cleanly to HPO, and knowing whether OMOP can hand us HPO
codes directly changes how we build that half of the pipeline.

## 3. Hierarchy — `concept_ancestor.csv`

Gives each condition its position in the SNOMED hierarchy, so sibling conditions can share
statistical strength. Restricted to standard Condition ancestors to keep it manageable.

Expected: roughly 150k–400k rows.

```bash
psql -d cem -c "\copy (with conditions as (select distinct u.concept_id_2 as condition_concept_id from cem_output.cem_unified u join staging_vocabulary.concept c1 on c1.concept_id=u.concept_id_1 join staging_vocabulary.concept c2 on c2.concept_id=u.concept_id_2 where c1.concept_class_id='Ingredient' and c2.domain_id='Condition' and c2.standard_concept='S') select ca.descendant_concept_id as condition_concept_id, ca.ancestor_concept_id, a.concept_name as ancestor_name, a.concept_code as ancestor_code, a.vocabulary_id as ancestor_vocabulary_id, ca.min_levels_of_separation, ca.max_levels_of_separation from conditions k join staging_vocabulary.concept_ancestor ca on ca.descendant_concept_id=k.condition_concept_id join staging_vocabulary.concept a on a.concept_id=ca.ancestor_concept_id where a.standard_concept='S' and a.domain_id='Condition' order by 1, ca.min_levels_of_separation) to 'data/ref/omop/concept_ancestor.csv' csv header"
```

## 4. Synonyms — `concept_synonym.csv`

Fallback for concepts that fail code-based mapping: synonyms roughly double the surface for
fuzzy label matching against MONDO and Open Targets.

Expected: roughly 10k–25k rows. The `language_concept_id = 4180186` filter is English — if
that id isn't present in this build, drop the filter and send everything.

```bash
psql -d cem -c "\copy (with conditions as (select distinct u.concept_id_2 as condition_concept_id from cem_output.cem_unified u join staging_vocabulary.concept c1 on c1.concept_id=u.concept_id_1 join staging_vocabulary.concept c2 on c2.concept_id=u.concept_id_2 where c1.concept_class_id='Ingredient' and c2.domain_id='Condition' and c2.standard_concept='S') select cs.concept_id as condition_concept_id, cs.concept_synonym_name from conditions k join staging_vocabulary.concept_synonym cs on cs.concept_id=k.condition_concept_id where cs.language_concept_id=4180186 order by 1) to 'data/ref/omop/concept_synonym.csv' csv header"
```

---

## Sanity check before sending

```bash
wc -l data/ref/omop/*.csv
```

`concept.csv` must be 5,632 lines (5,631 + header). The other three are informational — if
any comes back empty, that table probably isn't loaded in this build, which is useful to know
and not a blocker for the others.

## Not in scope here

The drug side (ChEMBL mechanisms, target and pathway data) needs no database access — we pull
it from public sources directly. This request is condition-side only.
