"""Fetch OMOP vocabulary metadata for the CEM condition concepts from Hecate.

Hecate (https://hecate.pantheon-hds.com) is a public, no-auth semantic search API
over the OHDSI vocabularies. It replaces a direct `staging_vocabulary` query for
the two things the condition feature layer needs:

  concept detail    concept_id -> concept_code (SNOMED), concept_class_id,
                    standard_concept, record_count (occurrences across the OHDSI
                    network, a usage proxy that doubles as a condition feature)
  hierarchy         concept_id -> ancestors, via /expand with parentlevels

There is no batch concept endpoint, so this is one request per concept with a
small thread pool. Results are written as JSONL so a partial run can be resumed.

Outputs (data/ref/):
  hecate_concepts.jsonl   one Concept record per condition
  hecate_ancestors.jsonl  {concept_id, ancestors: [{concept_id, name, code, level}]}
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://hecate.pantheon-hds.com"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REF = DATA / "ref"
PARENT_LEVELS = 3
WORKERS = 8


def get_json(url: str, tries: int = 4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return None


def condition_ids() -> list[int]:
    ids = pd.read_csv(DATA / "cem_ingredient_condition_associations.csv",
                      usecols=["condition_concept_id"])
    return sorted(ids.condition_concept_id.unique().tolist())


def flatten_ancestors(node: dict, acc: list, depth: int = 1) -> None:
    """Walk the nested /expand response, recording each ancestor once."""
    acc.append({"concept_id": node.get("concept_id"),
                "concept_name": node.get("concept_name"),
                "concept_code": node.get("concept_code"),
                "concept_class_id": node.get("concept_class_id"),
                "level": node.get("level") if node.get("level") is not None else depth})
    for child in (node.get("children") or []):
        # children of an ancestor are siblings/self, not ancestors; only recurse
        # through nodes that are themselves ancestors of the seed concept.
        if child.get("children"):
            flatten_ancestors(child, acc, depth + 1)


def done_ids(path: Path, key: str = "concept_id") -> set[int]:
    if not path.exists():
        return set()
    seen = set()
    with open(path) as fh:
        for line in fh:
            try:
                seen.add(json.loads(line)[key])
            except Exception:
                pass
    return seen


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    ids = condition_ids()
    print(f"conditions: {len(ids)}", flush=True)

    cpath, apath = REF / "hecate_concepts.jsonl", REF / "hecate_ancestors.jsonl"

    todo = [i for i in ids if i not in done_ids(cpath)]
    print(f"concept detail to fetch: {len(todo)}", flush=True)

    def fetch_concept(cid: int):
        rows = get_json(f"{BASE}/api/concepts/{cid}")
        if not rows:
            return {"concept_id": cid, "missing": True}
        rec = rows[0] if isinstance(rows, list) else rows
        rec["concept_id"] = cid
        return rec

    if todo:
        with open(cpath, "a") as out, ThreadPoolExecutor(WORKERS) as pool:
            for n, rec in enumerate(pool.map(fetch_concept, todo), 1):
                out.write(json.dumps(rec) + "\n")
                if n % 500 == 0:
                    out.flush()
                    print(f"  concepts {n}/{len(todo)}", flush=True)

    todo_a = [i for i in ids if i not in done_ids(apath)]
    print(f"hierarchy to fetch: {len(todo_a)}", flush=True)

    def fetch_ancestors(cid: int):
        resp = get_json(f"{BASE}/api/concepts/{cid}/expand"
                        f"?childlevels=0&parentlevels={PARENT_LEVELS}")
        acc: list = []
        if resp:
            for root in (resp.get("concepts") or []):
                flatten_ancestors(root, acc)
        acc = [a for a in acc if a.get("concept_id") != cid]
        return {"concept_id": cid, "n_ancestors": len(acc), "ancestors": acc}

    if todo_a:
        with open(apath, "a") as out, ThreadPoolExecutor(WORKERS) as pool:
            for n, rec in enumerate(pool.map(fetch_ancestors, todo_a), 1):
                out.write(json.dumps(rec) + "\n")
                if n % 500 == 0:
                    out.flush()
                    print(f"  hierarchy {n}/{len(todo_a)}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
