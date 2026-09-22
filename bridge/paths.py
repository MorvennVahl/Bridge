"""Canonical locations for the repo's data files.

Reference inputs live under `data/ref/` and are Git LFS-tracked. Derived tables go to
`data/derived/`, which is gitignored — everything there is regenerable from the inputs.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA = REPO_ROOT / "data"
REF = DATA / "ref"
DERIVED = DATA / "derived"

# Inputs
HPO_OBO = REF / "hpo__hp.obo"
HPO_PHENOTYPE_TO_GENES = REF / "hpo__phenotype_to_genes.txt"
HPO_GENES_TO_DISEASE = REF / "hpo__genes_to_disease.txt"
OT_DISEASE = REF / "ot" / "disease__disease.parquet"
OT_DISEASE_HPO = REF / "ot" / "disease_hpo__disease_hpo.parquet"
OT_DISEASE_PHENOTYPE = REF / "ot" / "disease_phenotype__disease_phenotype.parquet"
MONDO_SSSOM = REF / "mondo.sssom.tsv"
CEM_ASSOCIATIONS = DATA / "cem_ingredient_condition_associations.csv"

# Outputs
HPO_TERMS_OUT = DERIVED / "hpo_terms.parquet"
HPO_XREFS_OUT = DERIVED / "hpo_xrefs.parquet"
HPO_GENES_OUT = DERIVED / "hpo_genes.parquet"
DISEASE_TERMS_OUT = DERIVED / "disease_terms.parquet"
DISEASE_XREFS_OUT = DERIVED / "disease_xrefs.parquet"
DISEASE_GENES_OUT = DERIVED / "disease_genes.parquet"
DISEASE_PHENOTYPES_OUT = DERIVED / "disease_phenotypes.parquet"
