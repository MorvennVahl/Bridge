"""Parse the HPO ontology and its gene annotations into join-ready structures.

The condition side of the CEM table is bimodal: roughly a quarter of the 5,631 conditions
are diseases that map to MONDO, and most of the rest are clinical findings and symptoms
("abdominal bloating", "abnormal breath sounds") that only exist in HPO. This module covers
the HPO half.

`hp.obo` carries the ontology structure but almost no code cross-references (92 MedDRA and
38 ICD-10 across 20,482 terms), so it cannot be joined to OMOP on its own. The Open Targets
HPO table supplies the missing keys — see `extract_xrefs`.
"""

import dataclasses
import logging
import math
import re
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = "HP:0000001"

_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')

# Open Targets writes obo ids with an underscore (HP_0000118); HPO files use a colon.
_OT_ID = re.compile(r"^([A-Za-z]+)_(\d+)$")

# Xref sources worth keeping, normalised to a single label per vocabulary. SNOMEDCT_US and
# SCTID are the same vocabulary under two spellings and must collapse, or a SNOMED code
# joins on only one of them.
_XREF_SOURCES = {
    "SNOMEDCT_US": "SNOMED",
    "SNOMEDCT": "SNOMED",
    "SCTID": "SNOMED",
    "UMLS": "UMLS",
    "MESH": "MESH",
    "MEDDRA": "MEDDRA",
    "ICD10": "ICD10",
    "ICD-10": "ICD10",
    "NCIT": "NCIT",
    "OMIM": "OMIM",
}


@dataclasses.dataclass(slots=True)
class HpoTerm:
    """One `[Term]` stanza from hp.obo."""

    hpo_id: str
    name: str
    parents: list[str]
    synonyms: list[str]
    alt_ids: list[str]
    is_obsolete: bool
    replaced_by: str | None


def normalise_id(raw: str) -> str:
    """Turn an Open Targets-style `HP_0000118` into the canonical `HP:0000118`."""
    match = _OT_ID.match(raw)
    return f"{match.group(1)}:{match.group(2)}" if match else raw


def parse_obo(path: Path) -> dict[str, HpoTerm]:
    """Read hp.obo into a mapping of HPO id to term.

    Obsolete terms are kept: they still appear in older annotation files, and `replaced_by`
    is what lets a caller forward them to a live term.
    """
    terms: dict[str, HpoTerm] = {}
    stanza: str | None = None
    fields: dict[str, list[str]] = {}

    def flush() -> None:
        if stanza != "Term" or "id" not in fields:
            return
        hpo_id = fields["id"][0]
        terms[hpo_id] = HpoTerm(
            hpo_id=hpo_id,
            name=fields.get("name", [""])[0],
            parents=[v.split("!")[0].strip() for v in fields.get("is_a", [])],
            synonyms=[m.group(1) for v in fields.get("synonym", []) if (m := _QUOTED.search(v))],
            alt_ids=list(fields.get("alt_id", [])),
            is_obsolete=fields.get("is_obsolete", ["false"])[0] == "true",
            replaced_by=next(iter(fields.get("replaced_by", [])), None),
        )

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                flush()
                stanza = line[1:-1]
                fields = {}
                continue
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields.setdefault(key.strip(), []).append(value.strip())
    flush()

    logger.info("parsed %d HPO terms from %s", len(terms), path.name)
    return terms


def compute_ancestors(terms: Mapping[str, HpoTerm]) -> dict[str, set[str]]:
    """Transitive `is_a` closure for every term, excluding the term itself.

    Iterative rather than recursive: HPO is ~20k terms deep enough to matter, and a
    recursive walk would need the interpreter's limit raised.
    """
    ancestors: dict[str, set[str]] = {}
    for start in terms:
        if start in ancestors:
            continue
        stack = [start]
        while stack:
            node = stack[-1]
            if node in ancestors:
                stack.pop()
                continue
            pending = [p for p in terms[node].parents if p in terms and p not in ancestors]
            if pending:
                stack.extend(pending)
                continue
            acc: set[str] = set()
            for parent in terms[node].parents:
                if parent in terms:
                    acc.add(parent)
                    acc |= ancestors[parent]
            ancestors[node] = acc
            stack.pop()
    return ancestors


def invert(ancestors: Mapping[str, set[str]]) -> dict[str, set[str]]:
    """Descendants, derived from the ancestor closure."""
    descendants: dict[str, set[str]] = {}
    for term, ancs in ancestors.items():
        for anc in ancs:
            descendants.setdefault(anc, set()).add(term)
    return descendants


def compute_children(terms: Mapping[str, HpoTerm]) -> dict[str, list[str]]:
    """Direct children of each term, inverted from the `is_a` edges."""
    children: dict[str, list[str]] = {}
    for term in terms.values():
        for parent in term.parents:
            children.setdefault(parent, []).append(term.hpo_id)
    return children


def compute_depths(terms: Mapping[str, HpoTerm], root: str = ROOT) -> dict[str, int]:
    """Shortest `is_a` distance from the root, by breadth-first search over children."""
    children = compute_children(terms)
    depths: dict[str, int] = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in children.get(node, ()):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def specific_genes(
    annotated: Mapping[str, set[int]], terms: Mapping[str, HpoTerm]
) -> dict[str, set[int]]:
    """Genes annotated to a term that none of its children also carry.

    `phenotype_to_genes.txt` arrives already rolled up the ontology (the HPO "true path
    rule" is applied upstream), verified here: across 114,080 child-ancestor pairs, no
    ancestor was missing a descendant's gene, and `HP:0000118` alone carries 5,268 of the
    5,276 distinct genes. So propagating again is a no-op, and a raw gene count says more
    about a term's position in the tree than about the phenotype.

    Subtracting the children's genes recovers the signal the roll-up buried: the genes
    attached at this level of specificity rather than inherited from below.
    """
    children = compute_children(terms)
    specific: dict[str, set[int]] = {}
    for term, genes in annotated.items():
        inherited: set[int] = set()
        for child in children.get(term, ()):
            inherited |= annotated.get(child, set())
        specific[term] = genes - inherited
    return specific


def information_content(
    annotated: Mapping[str, set[int]], root: str = "HP:0000118"
) -> dict[str, float]:
    """Resnik information content, `-ln(genes(term) / genes(root))`.

    Near-root terms are annotated with almost every gene and should count for little; a
    term annotated with a handful of genes is highly specific. Terms with no annotation are
    omitted rather than assigned an arbitrary value.

    Clamped at zero: `HP:0000001` ("All") sits above the root used here and carries a
    handful more genes, which would otherwise give it a slightly negative score.
    """
    total = len(annotated.get(root, ()))
    if not total:
        logger.warning("root %s carries no gene annotations; skipping IC", root)
        return {}
    return {
        term: max(0.0, -math.log(len(genes) / total)) for term, genes in annotated.items() if genes
    }


def extract_xrefs(
    ids: Iterable[str], xref_lists: Iterable[list[str] | None]
) -> list[tuple[str, str, str]]:
    """Pull `(hpo_id, source, code)` triples out of Open Targets `dbXRefs` lists.

    This is the only route from HPO to OMOP in the data we hold: hp.obo has no usable code
    cross-references, while the Open Targets HPO table carries SNOMED and UMLS for most
    terms. Non-HP ids are dropped — that table also contains the anatomy ontologies HPO
    imports (FMA, UBERON, MA), which are not phenotypes.
    """
    out: list[tuple[str, str, str]] = []
    for raw_id, xrefs in zip(ids, xref_lists, strict=True):
        hpo_id = normalise_id(raw_id)
        if not hpo_id.startswith("HP:") or not xrefs:
            continue
        for xref in xrefs:
            source, _, code = xref.partition(":")
            canonical = _XREF_SOURCES.get(source.upper())
            if canonical and code:
                out.append((hpo_id, canonical, code.strip()))
    return out
