"""ChEMBL protein-classification hierarchy for the target components.

Outputs: handoff/target_components.json, handoff/protein_classification.json,
         handoff/provenance_protein_class.json
"""

import datetime
import json
import time
from pathlib import Path

import requests


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


B = "https://www.ebi.ac.uk/chembl/api/data"
S = requests.Session()
S.headers.update({"Accept": "application/json"})


def get(path, **params):
    for a in range(5):
        r = S.get(f"{B}/{path}", params=params, timeout=180)
        if r.status_code == 200:
            return r.json()
        time.sleep(3 * (a + 1))
    r.raise_for_status()


trecs = json.loads(Path("handoff/targets_chembl.json").read_text())
cids = sorted({c["component_id"] for t in trecs for c in (t.get("target_components") or [])})

comps = []
for i in range(0, len(cids), 20):
    j = get(
        "target_component.json",
        component_id__in=",".join(map(str, cids[i : i + 20])),
        limit=100,
        only="component_id,accession,description,component_type,organism,"
        "protein_classifications,go_slims",
    )
    comps += j["target_components"]
Path("handoff/target_components.json").write_text(json.dumps(comps))
print("components", len(comps), flush=True)

pcs, offset = [], 0
while True:
    j = get("protein_classification.json", limit=1000, offset=offset)
    pcs += j["protein_classifications"]
    offset += 1000
    if offset >= j["page_meta"]["total_count"]:
        break
Path("handoff/protein_classification.json").write_text(json.dumps(pcs))
Path("handoff/provenance_protein_class.json").write_text(
    json.dumps(
        {
            "source": "ChEMBL",
            "release": "ChEMBL_37",
            "endpoints": {
                "target_component": f"{B}/target_component.json",
                "protein_classification": f"{B}/protein_classification.json",
            },
            "n_components": len(comps),
            "n_classes": len(pcs),
            "retrieved_utc": utc_now(),
        },
        indent=2,
    )
)
print("protein classes", len(pcs), flush=True)
