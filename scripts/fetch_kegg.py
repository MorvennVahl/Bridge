"""KEGG DRUG annotation for CEM ingredients: metabolising enzymes, transporters,
drug-drug interaction partners, ATC codes and efficacy strings.

Name resolution uses the bulk `list/drug` index (one request) rather than per-name
`find` queries. Entry records are then fetched 10 at a time.

Outputs: handoff/kegg_list.txt, handoff/kegg_entries.json,
         handoff/kegg_name_match.csv, handoff/provenance_kegg.json
"""

import datetime
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


K = "https://rest.kegg.jp"


def norm(s):
    if not isinstance(s, str):
        return None
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\.(alpha|beta|gamma|delta|omega|l|d)\.", r"\1", s)
    s = re.sub(r"\(\+/-\)|\(\+-\)|\(rs\)|\(\+\)|\(-\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or None


def kget(path, tries=4):
    for a in range(tries):
        r = requests.get(f"{K}/{path}", timeout=180)
        if r.status_code == 200:
            return r.text
        if r.status_code == 404:
            return ""
        time.sleep(3 * (a + 1))
    return ""


# ---- 1. bulk name index ------------------------------------------------------
listing = kget("list/drug")
Path("handoff/kegg_list.txt").write_text(listing)
idx = {}
for line in listing.rstrip("\n").split("\n"):
    if not line.strip():
        continue
    did, names = line.split("\t", 1)
    did = did.replace("dr:", "")
    for nm in names.split(";"):
        nm = re.sub(r"\((USP|JAN|INN|USAN|BAN|TN|DCF|NF|JP\d*)[^)]*\)", " ", nm)
        k = norm(nm)
        if k:
            idx.setdefault(k, []).append(did)
print("kegg entries", len(listing.strip().split("\n")), "| name keys", len(idx), flush=True)

# ---- 2. match ingredients ----------------------------------------------------
ing = pd.read_csv("handoff/ingredient_chembl_map.csv")
matches = []
for nm in ing.ingredient_name:
    k = norm(nm)
    cands = idx.get(k, [])
    matches.append(
        {
            "ingredient_name": nm,
            "kegg_drug_id": sorted(cands)[0] if cands else None,
            "kegg_n_candidates": len(cands),
            "kegg_match_method": "exact_normalized_name" if cands else None,
        }
    )
mdf = pd.DataFrame(matches)
mdf.to_csv("handoff/kegg_name_match.csv", index=False)
dids = sorted(set(mdf.kegg_drug_id.dropna().unique()))
print(
    "matched ingredients",
    mdf.kegg_drug_id.notna().sum(),
    "| unique kegg ids",
    len(dids),
    flush=True,
)

# ---- 3. entry records --------------------------------------------------------
FIELDS = (
    "ENTRY",
    "NAME",
    "FORMULA",
    "EFFICACY",
    "TARGET",
    "METABOLISM",
    "INTERACTION",
    "REMARK",
    "COMMENT",
    "BRITE",
    "DBLINKS",
    "STR_MAP",
)
entries = {}
for i in range(0, len(dids), 10):
    batch = dids[i : i + 10]
    txt = kget("get/" + "+".join("dr:" + d for d in batch))
    for rec in txt.split("\n///\n"):
        if not rec.strip():
            continue
        cur, data = None, {}
        for line in rec.split("\n"):
            if line[:12].strip():
                cur = line[:12].strip()
                data.setdefault(cur, []).append(line[12:].strip())
            elif cur:
                data[cur].append(line[12:].strip())
        eid = (data.get("ENTRY") or [""])[0].split()[0] if data.get("ENTRY") else None
        if eid:
            entries[eid] = {k: v for k, v in data.items() if k in FIELDS}
    if (i // 10) % 20 == 0:
        print("  entries", len(entries), flush=True)
    time.sleep(0.25)
Path("handoff/kegg_entries.json").write_text(json.dumps(entries))
Path("handoff/provenance_kegg.json").write_text(
    json.dumps(
        {
            "source": "KEGG DRUG",
            "base_url": K,
            "endpoints": {
                "name_index": f"{K}/list/drug",
                "records": f"{K}/get/dr:<id> (10 per request)",
            },
            "n_entries_in_release": len(listing.strip().split("\n")),
            "n_ingredients_matched": int(mdf.kegg_drug_id.notna().sum()),
            "n_records_fetched": len(entries),
            "license_note": "KEGG is free for academic use; see https://www.kegg.jp/kegg/legal.html",
            "retrieved_utc": utc_now(),
        },
        indent=2,
    )
)
print("done. records", len(entries), flush=True)
