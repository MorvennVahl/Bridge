"""Map OMOP condition concepts onto disease/phenotype ontology terms.

Two ontology arms, because the CEM outcome set is a mix of diseases and clinical
findings and no single resource covers both:

  disease arm    OMOP condition -> MONDO / EFO / Orphanet -> Open Targets associations
  phenotype arm  OMOP condition -> HPO                    -> HPO gene annotations

Match tiers, highest precision first. Every row in the output records which tier
produced it, so downstream features can be restricted to whatever precision an
analysis needs.

  1  sctid_xref     SNOMED concept_code == an SCTID cross-reference in Open
                    Targets' disease index or in the MONDO SSSOM mapping set.
                    Requires data/condition_concept.csv (OMOP vocabulary export).
  2  label_exact    normalised condition name == normalised ontology label or
                    exact/narrow synonym.
  3  token_exact    same after dropping stopwords and sorting tokens, which
                    absorbs SNOMED's "Pain of joint" vs HPO's "Joint pain".
  4  llm_adjudicated  a lexical shortlist adjudicated by an LLM (adjudicate.py).

Reference data is fetched by fetch_reference.py into data/ref/.
"""

from __future__ import annotations

import collections
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
REF = DATA / "ref"

STOPWORDS = {"of", "the", "a", "an", "in", "on", "with", "and", "to", "by"}

# Ontology prefixes kept in the disease arm. OBA (Ontology of Biological
# Attributes) is measurement-like, and GO/UBERON/PATO are not diseases, so they
# are excluded to avoid mapping a condition onto an assay or an anatomical part.
DISEASE_PREFIXES = ("MONDO_", "EFO_", "Orphanet_")


def normalise(s: str) -> str:
    """Casefold, strip accents and punctuation, drop parentheticals.

    SNOMED fully-specified names carry a semantic tag in parentheses
    ("Nausea (finding)") which must not participate in matching.
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_key(normalised: str) -> str:
    return " ".join(sorted(w for w in normalised.split() if w not in STOPWORDS))


# --------------------------------------------------------------------------- #
# reference indexes
# --------------------------------------------------------------------------- #


def load_open_targets_diseases() -> pd.DataFrame:
    return pd.read_parquet(REF / "ot" / "disease__disease.parquet")


def load_hpo_terms() -> dict[str, dict]:
    """Parse hp.obo into {HP:id: {name, syn, xref, is_a}}, obsolete terms dropped."""
    terms: dict[str, dict] = {}
    cur: dict | None = None

    def flush(c):
        if c and c.get("id") and not c["obsolete"]:
            terms[c["id"]] = c

    with open(REF / "hpo__hp.obo") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush(cur)
                cur = {
                    "id": None,
                    "name": None,
                    "syn": [],
                    "xref": [],
                    "is_a": [],
                    "obsolete": False,
                }
            elif line.startswith("[") and line.endswith("]"):
                flush(cur)
                cur = None
            elif cur is not None and ": " in line:
                key, _, val = line.partition(": ")
                if key == "id" and val.startswith("HP:"):
                    cur["id"] = val
                elif key == "name":
                    cur["name"] = val
                elif key == "synonym":
                    m = re.match(r'"(.*?)"\s+(\w+)', val)
                    if m and m.group(2) in ("EXACT", "NARROW"):
                        cur["syn"].append(m.group(1))
                elif key == "xref":
                    cur["xref"].append(val.split(" ")[0])
                elif key == "is_a":
                    cur["is_a"].append(val.split(" ")[0])
                elif key == "is_obsolete" and val == "true":
                    cur["obsolete"] = True
    flush(cur)
    return terms


def load_mondo_sssom() -> pd.DataFrame:
    path = REF / "mondo.sssom.tsv"
    skip = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, low_memory=False)


def build_indexes(ot: pd.DataFrame, hpo: dict[str, dict], sssom: pd.DataFrame):
    """Return (label_index, token_index, sctid_index, term_meta).

    Label and token indexes map a normalised string to a set of ontology ids and
    cover both arms. sctid_index covers the disease arm only: HPO carries no
    SNOMED cross-references, its UMLS/SNOMED xrefs having been removed for
    licensing reasons.
    """
    label: dict[str, set[str]] = collections.defaultdict(set)
    token: dict[str, set[str]] = collections.defaultdict(set)
    sctid: dict[str, set[str]] = collections.defaultdict(set)
    meta: dict[str, dict] = {}

    disease = ot[ot.id.str.startswith(DISEASE_PREFIXES)]
    for row in disease.itertuples(index=False):
        meta[row.id] = {"label": row.name, "arm": "disease", "ontology": row.id.split("_")[0]}
        syns = [] if row.exactSynonyms is None else list(row.exactSynonyms)
        for k in [row.name, *syns]:
            nk = normalise(k)
            if nk:
                label[nk].add(row.id)
                token[token_key(nk)].add(row.id)
        for xref in [] if row.dbXRefs is None else list(row.dbXRefs):
            if str(xref).startswith("SCTID:"):
                sctid[str(xref).split(":", 1)[1]].add(row.id)

    for hid, term in hpo.items():
        meta[hid] = {"label": term["name"], "arm": "phenotype", "ontology": "HP"}
        for k in [term["name"]] + term["syn"]:
            nk = normalise(k)
            if nk:
                label[nk].add(hid)
                token[token_key(nk)].add(hid)

    # MONDO's SSSOM set widens the SNOMED bridge beyond Open Targets' own xrefs.
    snomed_rows = sssom[sssom.object_id.str.startswith("SCTID", na=False)]
    for row in snomed_rows.itertuples(index=False):
        mondo_id = row.subject_id.replace(":", "_")
        sctid[row.object_id.split(":", 1)[1]].add(mondo_id)
        meta.setdefault(
            mondo_id, {"label": row.subject_label, "arm": "disease", "ontology": "MONDO"}
        )

    return label, token, sctid, meta


# --------------------------------------------------------------------------- #
# tiered matching
# --------------------------------------------------------------------------- #


def map_conditions(
    conditions: pd.DataFrame, label, token, sctid, meta, concept_codes: pd.Series | None = None
) -> pd.DataFrame:
    """One row per (condition, matched ontology term), tagged with its tier.

    `conditions` needs condition_concept_id and condition_name.
    `concept_codes` is an optional condition_concept_id -> SNOMED concept_code
    mapping; without it tier 1 is skipped and coverage is lexical only.
    """
    rows = []
    for row in conditions.itertuples(index=False):
        cid = row.condition_concept_id
        nname = normalise(row.condition_name)

        matches: dict[str, str] = {}
        if concept_codes is not None:
            code = concept_codes.get(cid)
            if isinstance(code, str) and code:
                for oid in sctid.get(code, ()):
                    matches.setdefault(oid, "1_sctid_xref")
        for oid in label.get(nname, ()):
            matches.setdefault(oid, "2_label_exact")
        for oid in token.get(token_key(nname), ()):
            matches.setdefault(oid, "3_token_exact")

        if not matches:
            rows.append(
                {
                    "condition_concept_id": cid,
                    "condition_name": row.condition_name,
                    "ontology_id": None,
                    "ontology": None,
                    "arm": None,
                    "ontology_label": None,
                    "match_tier": "0_unmatched",
                }
            )
            continue
        for oid, tier in matches.items():
            m = meta.get(oid, {})
            rows.append(
                {
                    "condition_concept_id": cid,
                    "condition_name": row.condition_name,
                    "ontology_id": oid,
                    "ontology": m.get("ontology"),
                    "arm": m.get("arm"),
                    "ontology_label": m.get("label"),
                    "match_tier": tier,
                }
            )
    return pd.DataFrame(rows)


def coverage_report(mapped: pd.DataFrame):
    """Return (best-tier-per-condition, tier counts)."""
    best = (
        mapped.sort_values("match_tier")
        .groupby("condition_concept_id", as_index=False)
        .first()[
            [
                "condition_concept_id",
                "condition_name",
                "match_tier",
                "arm",
                "ontology",
                "ontology_id",
                "ontology_label",
            ]
        ]
    )
    counts = (
        best.match_tier.value_counts().rename("conditions").rename_axis("best_tier").reset_index()
    )
    counts["share"] = (counts.conditions / len(best)).round(4)
    return best, counts
