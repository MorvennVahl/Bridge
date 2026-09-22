"""Fetch Open Targets disease-target associations via GraphQL, per disease.

Replaces scripts/fetch_ot_associations.py, which streamed the 2.9 GB
association_by_datatype_indirect parquet dataset. EBI's FTP-over-HTTPS delivers
roughly 100 KB/s from this sandbox, which put that route at about eight hours.
Querying the API for only the diseases we actually map onto is a small fraction
of the traffic, and returns per-datatype scores nested per target rather than one
row per datatype.

Scores are the "indirect" (hierarchy-rolled-up) associations that the GraphQL
`associatedTargets` field returns by default, so a condition mapped to a parent
term still sees evidence recorded on its children.

Each disease is capped at TOP_N targets ordered by overall association score.
The full count is recorded per disease in the `n_total` field so truncation is
never silent: a disease with n_total > TOP_N has a tail that was not fetched.
The tail is mostly low-score literature co-mention and is not useful for
overlap features, but the flag lets a downstream analysis check that.

Output is JSONL, one object per disease, appended as results arrive, so an
interrupted run resumes without refetching.

Usage:
    python scripts/fetch_ot_associations_api.py <disease_ids.txt> <out.jsonl>
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
TOP_N = 250
WORKERS = 8
RETRIES = 3

QUERY = """query($efoId:String!,$index:Int!,$size:Int!){
  disease(efoId:$efoId){ id name
    associatedTargets(page:{index:$index,size:$size}){
      count
      rows{ target{ id approvedSymbol } score datatypeScores{ id score } } } } }"""


def gql(variables: dict, timeout: int = 90) -> dict:
    payload = json.dumps({"query": QUERY, "variables": variables}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_one(disease_id: str) -> dict:
    """One disease -> its top-N target associations, or an error record."""
    for attempt in range(RETRIES):
        try:
            data = gql({"efoId": disease_id, "index": 0, "size": TOP_N})
            dz = (data.get("data") or {}).get("disease")
            if dz is None:
                # Not an error: many of our mapped terms are not in the Open
                # Targets disease index at all (notably most HPO terms).
                return {"disease_id": disease_id, "absent": True,
                        "n_total": 0, "targets": []}
            assoc = dz.get("associatedTargets") or {}
            targets = [{
                "ensembl_gene_id": row["target"]["id"],
                "gene_symbol": row["target"].get("approvedSymbol"),
                "score": row["score"],
                "datatype_scores": {s["id"]: s["score"]
                                    for s in (row.get("datatypeScores") or [])},
            } for row in (assoc.get("rows") or [])]
            return {"disease_id": disease_id, "disease_name": dz.get("name"),
                    "absent": False, "n_total": assoc.get("count", 0),
                    "n_fetched": len(targets),
                    "truncated": assoc.get("count", 0) > len(targets),
                    "targets": targets}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == RETRIES - 1:
                return {"disease_id": disease_id, "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(2 ** attempt)
    return {"disease_id": disease_id, "error": "exhausted retries"}


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("error"):
                out.add(rec["disease_id"])
    return out


def main() -> int:
    ids_file, out_file = Path(sys.argv[1]), Path(sys.argv[2])
    wanted = [ln.strip() for ln in open(ids_file) if ln.strip()]
    already = done_ids(out_file)
    todo = [i for i in wanted if i not in already]
    print(f"wanted {len(wanted)} | already done {len(already)} | to fetch {len(todo)}",
          flush=True)

    n_ok = n_absent = n_err = 0
    with open(out_file, "a") as out, ThreadPoolExecutor(WORKERS) as pool:
        for n, rec in enumerate(pool.map(fetch_one, todo), 1):
            out.write(json.dumps(rec) + "\n")
            if rec.get("error"):
                n_err += 1
            elif rec.get("absent"):
                n_absent += 1
            else:
                n_ok += 1
            if n % 100 == 0:
                out.flush()
                print(f"  {n}/{len(todo)} | with targets {n_ok} | "
                      f"absent {n_absent} | errors {n_err}", flush=True)

    print(f"done: with targets {n_ok} | absent {n_absent} | errors {n_err}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
