"""Annotate the ChEMBL targets hit by CEM ingredients.

Inputs : handoff/target_ids.json  (list of target_chembl_id)
Outputs: handoff/targets_chembl.json, handoff/targets_uniprot.tsv,
         handoff/targets_sym2ensg.json, handoff/targets_opentargets.json,
         handoff/provenance_targets.json
"""

import datetime
import json
import os
import re
import time
from pathlib import Path

import requests


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


S = requests.Session()
S.headers.update({"Accept": "application/json"})
CH = "https://www.ebi.ac.uk/chembl/api/data"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
prov = {"retrieved_utc_start": utc_now(), "sources": {}}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def cget(path, **params):
    for a in range(5):
        r = S.get(f"{CH}/{path}", params=params, timeout=180)
        if r.status_code == 200:
            return r.json()
        time.sleep(3 * (a + 1))
    r.raise_for_status()


def gql(q, v):
    for a in range(5):
        r = requests.post(OT, json={"query": q, "variables": v}, timeout=180)
        if r.status_code == 200:
            j = r.json()
            if "errors" not in j:
                return j["data"]
            return {"__errors__": j["errors"]}
        time.sleep(3 * (a + 1))
    r.raise_for_status()


tids = json.loads(Path("handoff/target_ids.json").read_text())

# ---- 1. ChEMBL target records -------------------------------------------------
if os.path.exists("handoff/targets_chembl.json"):
    trecs = json.loads(Path("handoff/targets_chembl.json").read_text())
else:
    trecs = []
    for c in chunks(tids, 20):
        trecs += cget("target.json", target_chembl_id__in=",".join(c), limit=1000)["targets"]
    Path("handoff/targets_chembl.json").write_text(json.dumps(trecs))
prov["sources"]["chembl_target"] = {
    "endpoint": f"{CH}/target.json",
    "n_requested": len(tids),
    "n_returned": len(trecs),
    "retrieved_utc": utc_now(),
}
print("chembl targets", len(trecs), flush=True)

UP_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")
raw_accs = sorted(
    {
        c["accession"]
        for t in trecs
        for c in (t.get("target_components") or [])
        if c.get("accession")
    }
)
accs = [a for a in raw_accs if UP_RE.match(a)]
dropped = [a for a in raw_accs if a not in set(accs)]
prov["non_uniprot_component_accessions_dropped"] = dropped
print("accessions", len(accs), "| dropped non-UniProt", dropped, flush=True)

# ---- 2. UniProt --------------------------------------------------------------
FIELDS = "accession,id,protein_name,gene_primary,length,keyword,cc_subcellular_location,cc_function"
rows, header = [], None
if os.path.exists("handoff/targets_uniprot.tsv") and os.path.exists(
    "handoff/targets_sym2ensg.json"
):
    lines = Path("handoff/targets_uniprot.tsv").read_text().rstrip("\n").split("\n")
    header, rows = lines[0], lines[1:]
    sym2ensg = json.loads(Path("handoff/targets_sym2ensg.json").read_text())
    accs = []
for c in chunks(accs, 90):
    r = requests.get(
        "https://rest.uniprot.org/uniprotkb/stream",
        params={
            "query": " OR ".join(f"accession:{a}" for a in c),
            "fields": FIELDS,
            "format": "tsv",
        },
        timeout=300,
    )
    r.raise_for_status()
    lines = r.text.rstrip("\n").split("\n")
    header = lines[0]
    rows += lines[1:]
    time.sleep(0.5)
with open("handoff/targets_uniprot.tsv", "w") as fh:
    fh.write("\n".join([header, *rows]))
prov["sources"]["uniprot"] = {
    "endpoint": "https://rest.uniprot.org/uniprotkb/stream",
    "fields": FIELDS,
    "n_accessions": len(accs),
    "n_rows": len(rows),
    "retrieved_utc": utc_now(),
}
print("uniprot rows", len(rows), flush=True)

syms = sorted({r.split("\t")[3] for r in rows if len(r.split("\t")) > 3 and r.split("\t")[3]})

# ---- 3. Ensembl symbol -> gene id -------------------------------------------
if "sym2ensg" not in dir():
    sym2ensg = {}
for c in chunks(syms if not sym2ensg else [], 800):
    r = requests.post(
        "https://rest.ensembl.org/lookup/symbol/homo_sapiens",
        json={"symbols": c},
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    r.raise_for_status()
    sym2ensg.update({k: v.get("id") for k, v in r.json().items()})
    time.sleep(0.5)
Path("handoff/targets_sym2ensg.json").write_text(json.dumps(sym2ensg))
prov["sources"]["ensembl"] = {
    "endpoint": "https://rest.ensembl.org/lookup/symbol/homo_sapiens",
    "n_symbols": len(syms),
    "n_mapped": sum(v is not None for v in sym2ensg.values()),
    "retrieved_utc": utc_now(),
}
print("ensembl mapped", len(sym2ensg), flush=True)

# ---- 4. Open Targets ---------------------------------------------------------
Q = """query T($ids:[String!]!){ targets(ensemblIds:$ids){
  id approvedSymbol biotype isEssential
  tractability{modality label value}
  geneticConstraint{constraintType score upperBin}
  safetyLiabilities{event datasource}
  pathways{pathwayId pathway topLevelTerm} } }"""
genes = sorted({v for v in sym2ensg.values() if v})
ot = {}
for c in chunks(genes, 50):
    d = gql(Q, {"ids": c})
    if "__errors__" in d:
        raise SystemExit(f"OT batch query failed: {d['__errors__']}")
    for t in d["targets"]:
        ot[t["id"]] = t
    time.sleep(0.3)
Path("handoff/targets_opentargets.json").write_text(json.dumps(ot))
prov["sources"]["opentargets"] = {
    "endpoint": OT,
    "query": "targets(ensemblIds)",
    "n_genes": len(genes),
    "n_returned": len(ot),
    "retrieved_utc": utc_now(),
}
Path("handoff/provenance_targets.json").write_text(json.dumps(prov, indent=2))
print("opentargets", len(ot), flush=True)
