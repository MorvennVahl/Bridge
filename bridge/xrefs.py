"""Cross-reference normalisation shared by the phenotype and disease halves.

Every reference source spells vocabularies differently — SNOMED arrives as `SCTID`,
`SNOMEDCT_US` and `SNOMEDCT`; Orphanet as both `Orphanet` and `ORPHA`; MeSH in either case.
Left alone, the same code fails to join itself, so everything funnels through here.
"""

import re

# Open Targets writes obo ids with an underscore (MONDO_0005148); the ontology files use a
# colon (MONDO:0005148).
_OBO_ID = re.compile(r"^([A-Za-z][A-Za-z0-9.]*)_(\d+)$")

#: Raw source label (upper-cased) to the canonical name used across the derived tables.
CANONICAL_SOURCES: dict[str, str] = {
    "SCTID": "SNOMED",
    "SNOMEDCT": "SNOMED",
    "SNOMEDCT_US": "SNOMED",
    "UMLS": "UMLS",
    "MEDGEN": "MEDGEN",
    "MESH": "MESH",
    "MEDDRA": "MEDDRA",
    "OMIM": "OMIM",
    "OMIMPS": "OMIMPS",
    "ORPHA": "Orphanet",
    "ORPHANET": "Orphanet",
    "DOID": "DOID",
    "NCIT": "NCIT",
    "ICD10": "ICD10",
    "ICD-10": "ICD10",
    "ICD10CM": "ICD10CM",
    "ICD9": "ICD9",
    "ICD-9": "ICD9",
    "ICD9CM": "ICD9CM",
    "ICD11.FOUNDATION": "ICD11",
    "EFO": "EFO",
    "GARD": "GARD",
}

#: Vocabularies OMOP ships, and so the ones that can join through to a condition concept.
#: UMLS and MedGen are deliberately absent — they are the richest keys in both MONDO and
#: HPO, but OMOP does not distribute UMLS CUIs, so they are unreachable from our side.
#: `ICD9` and `ICD10` here are the bare-code spellings used by MONDO and Open Targets;
#: OMOP calls the corresponding vocabularies `ICD9CM` and `ICD10`/`ICD10CM`. The vocabulary
#: census requested in docs/vocab-export-spec.md will confirm which are actually loaded.
OMOP_REACHABLE = frozenset(
    {
        "SNOMED",
        "MESH",
        "MEDDRA",
        "OMIM",
        "Orphanet",
        "ICD10",
        "ICD10CM",
        "ICD9",
        "ICD9CM",
        "NCIT",
    }
)


def normalise_id(raw: str) -> str:
    """Turn an Open Targets-style `MONDO_0005148` into the canonical `MONDO:0005148`."""
    match = _OBO_ID.match(raw)
    return f"{match.group(1)}:{match.group(2)}" if match else raw


def canonical_source(raw: str) -> str | None:
    """Canonical vocabulary name for an xref prefix, or None if it is not one we keep."""
    return CANONICAL_SOURCES.get(raw.strip().upper())


def parse_xref(xref: str) -> tuple[str, str] | None:
    """Split `SCTID:82525005` into `("SNOMED", "82525005")`, or None if unrecognised."""
    source, _, code = xref.partition(":")
    canonical = canonical_source(source)
    code = code.strip()
    return (canonical, code) if canonical and code else None
