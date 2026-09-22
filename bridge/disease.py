"""Assemble the disease side of the condition feature layer.

Counterpart to `bridge.hpo`. Roughly a quarter of the 5,631 CEM conditions are diseases
rather than symptoms, and those route through MONDO and Open Targets instead of HPO.

Two sources are combined because neither is sufficient alone: Open Targets carries the
hierarchy and therapeutic areas but only some of the cross-references, while the MONDO SSSOM
file carries far more cross-references but no hierarchy. Terms present only in SSSOM are
kept with a null hierarchy rather than dropped — a term we can join to OMOP but cannot place
in the tree is still worth having.
"""

import dataclasses
import logging
from collections.abc import Iterable, Mapping, Sequence

from bridge.xrefs import normalise_id, parse_xref

logger = logging.getLogger(__name__)

#: Ontology prefixes that denote a disease. Open Targets' disease file also contains
#: biological attributes (OBA), GO processes and HPO phenotypes; the first two are not
#: diseases and the third is handled by `bridge.hpo`.
DISEASE_PREFIXES = ("MONDO:", "EFO:", "Orphanet:", "DOID:")


@dataclasses.dataclass(slots=True)
class DiseaseTerm:
    """A disease concept, from Open Targets where available and SSSOM otherwise."""

    disease_id: str
    name: str
    parents: list[str]
    ancestors: list[str]
    descendants: list[str]
    therapeutic_areas: list[str]
    in_open_targets: bool


def is_disease(term_id: str) -> bool:
    return term_id.startswith(DISEASE_PREFIXES)


def build_terms(
    ids: Iterable[str],
    names: Iterable[str],
    parents: Iterable[Sequence[str] | None],
    ancestors: Iterable[Sequence[str] | None],
    therapeutic_areas: Iterable[Sequence[str] | None],
) -> dict[str, DiseaseTerm]:
    """Disease terms from the Open Targets disease table, keyed by normalised id."""
    terms: dict[str, DiseaseTerm] = {}
    for raw_id, name, par, anc, areas in zip(
        ids, names, parents, ancestors, therapeutic_areas, strict=True
    ):
        disease_id = normalise_id(raw_id)
        if not is_disease(disease_id):
            continue
        terms[disease_id] = DiseaseTerm(
            disease_id=disease_id,
            name=name or "",
            parents=[normalise_id(p) for p in par or ()],
            ancestors=[normalise_id(a) for a in anc or ()],
            descendants=[],
            therapeutic_areas=[normalise_id(a) for a in areas or ()],
            in_open_targets=True,
        )
    fill_descendants(terms)
    return terms


def fill_descendants(terms: dict[str, DiseaseTerm]) -> None:
    """Populate `descendants` by inverting the ancestor lists, in place."""
    collected: dict[str, list[str]] = {}
    for term in terms.values():
        for ancestor in term.ancestors:
            collected.setdefault(ancestor, []).append(term.disease_id)
    for term_id, kids in collected.items():
        if term_id in terms:
            terms[term_id].descendants = kids


def add_sssom_terms(
    terms: dict[str, DiseaseTerm], subject_ids: Iterable[str], subject_labels: Iterable[str]
) -> int:
    """Add MONDO terms that appear in SSSOM but not in Open Targets. Returns how many."""
    added = 0
    for subject_id, label in zip(subject_ids, subject_labels, strict=True):
        disease_id = normalise_id(subject_id)
        if disease_id in terms or not is_disease(disease_id):
            continue
        terms[disease_id] = DiseaseTerm(
            disease_id=disease_id,
            name=label or "",
            parents=[],
            ancestors=[],
            descendants=[],
            therapeutic_areas=[],
            in_open_targets=False,
        )
        added += 1
    return added


def xrefs_from_open_targets(
    ids: Iterable[str], xref_lists: Iterable[Sequence[str] | None]
) -> list[tuple[str, str, str]]:
    """`(disease_id, source, code)` triples from the Open Targets `dbXRefs` column."""
    out: list[tuple[str, str, str]] = []
    for raw_id, xrefs in zip(ids, xref_lists, strict=True):
        disease_id = normalise_id(raw_id)
        if not is_disease(disease_id):
            continue
        for xref in xrefs or ():
            if parsed := parse_xref(xref):
                out.append((disease_id, parsed[0], parsed[1]))
    return out


def xrefs_from_sssom(
    subject_ids: Iterable[str], object_ids: Iterable[str], predicates: Iterable[str]
) -> list[tuple[str, str, str]]:
    """`(disease_id, source, code)` triples from the MONDO SSSOM file.

    Only `skos:exactMatch` is kept. The file also holds 89 `skos:broadMatch` rows, where the
    target is a *more general* concept — treating those as equivalent would silently merge a
    specific disease into its parent.
    """
    out: list[tuple[str, str, str]] = []
    for subject_id, object_id, predicate in zip(subject_ids, object_ids, predicates, strict=True):
        if predicate != "skos:exactMatch":
            continue
        disease_id = normalise_id(subject_id)
        if not is_disease(disease_id):
            continue
        if parsed := parse_xref(object_id):
            out.append((disease_id, parsed[0], parsed[1]))
    return out


def genes_by_disease(
    xrefs: Iterable[tuple[str, str, str]],
    gene_disease_ids: Iterable[str],
    gene_ids: Iterable[int],
    association_types: Iterable[str],
) -> list[tuple[str, int, str]]:
    """Attach HPO's gene-to-disease annotations to disease ids, via OMIM and Orphanet.

    The Open Targets target-disease association file is not in the repo, so this is the only
    disease-to-gene evidence we hold. It reaches diseases that carry an OMIM or Orphanet
    cross-reference, which skews Mendelian — worth remembering when the features are used,
    because common polygenic disease is thinly covered.
    """
    by_code: dict[tuple[str, str], set[str]] = {}
    for disease_id, source, code in xrefs:
        if source in ("OMIM", "Orphanet"):
            by_code.setdefault((source, code), set()).add(disease_id)

    out: list[tuple[str, int, str]] = []
    for raw_disease, gene_id, assoc in zip(
        gene_disease_ids, gene_ids, association_types, strict=True
    ):
        parsed = parse_xref(raw_disease)
        if not parsed:
            continue
        for disease_id in by_code.get(parsed, ()):
            out.append((disease_id, gene_id, assoc))
    return out


def disease_phenotype_pairs(
    diseases: Iterable[str], phenotypes: Iterable[str]
) -> list[tuple[str, str]]:
    """Normalised `(disease_id, hpo_id)` pairs linking the disease half to the HPO half."""
    out: list[tuple[str, str]] = []
    for raw_disease, raw_phenotype in zip(diseases, phenotypes, strict=True):
        disease_id = normalise_id(raw_disease)
        hpo_id = normalise_id(raw_phenotype)
        if is_disease(disease_id) and hpo_id.startswith("HP:"):
            out.append((disease_id, hpo_id))
    return out


def coverage(terms: Mapping[str, DiseaseTerm], xrefs: Iterable[tuple[str, str, str]]) -> dict:
    """Summary counts for logging: how many terms are joinable and how."""
    by_source: dict[str, set[str]] = {}
    for disease_id, source, _ in xrefs:
        by_source.setdefault(source, set()).add(disease_id)
    return {
        "n_terms": len(terms),
        "n_open_targets": sum(1 for t in terms.values() if t.in_open_targets),
        "by_source": {source: len(ids) for source, ids in sorted(by_source.items())},
    }
