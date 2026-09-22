"""Bulk-download ChEMBL reference tables used for ingredient-level feature annotation.

Writes newline-delimited JSON to handoff/ plus a provenance sidecar recording the
ChEMBL release, endpoint, query parameters and retrieval timestamp for each table.
"""

import datetime
import json
import os
import time
from pathlib import Path

import requests

B = "https://www.ebi.ac.uk/chembl/api/data"
S = requests.Session()
S.headers.update({"Accept": "application/json"})
os.makedirs("handoff", exist_ok=True)

MOL_FIELDS = (
    "molecule_chembl_id,pref_name,molecule_synonyms,max_phase,first_approval,"
    "molecule_type,oral,parenteral,topical,black_box_warning,prodrug,withdrawn_flag,"
    "natural_product,therapeutic_flag,availability_type,dosed_ingredient,"
    "atc_classifications,molecule_properties,molecule_hierarchy,indication_class,"
    "usan_stem_definition,chirality,inorganic_flag,polymer_flag"
)

TABLES = {
    "molecule_phase": ("molecule", {"max_phase__gte": 1, "only": MOL_FIELDS}, "molecules"),
    "molecule_therap": ("molecule", {"therapeutic_flag": "true", "only": MOL_FIELDS}, "molecules"),
    "mechanism": ("mechanism", {}, "mechanisms"),
    "drug_warning": ("drug_warning", {}, "drug_warnings"),
    "metabolism": ("metabolism", {}, "metabolisms"),
}


def fetch_page(ep, params, tries=5):
    for a in range(tries):
        r = S.get(f"{B}/{ep}.json", params=params, timeout=300)
        if r.status_code == 200:
            return r.json()
        time.sleep(3 * (a + 1))
    r.raise_for_status()


def bulk(name, ep, params, key, limit=1000):
    out, offset, total = [], 0, None
    while True:
        j = fetch_page(ep, {**params, "limit": limit, "offset": offset})
        total = j["page_meta"]["total_count"]
        rows = j[key]
        out += rows
        offset += limit
        if offset >= total or not rows:
            break
    with open(f"handoff/chembl_{name}.json", "w") as fh:
        json.dump(out, fh)
    return {
        "table": name,
        "endpoint": f"{B}/{ep}.json",
        "query_params": params,
        "n_rows": len(out),
        "total_count_reported": total,
        "retrieved_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    status = fetch_page("status", {})
    prov = {
        "source": "ChEMBL",
        "release": status.get("chembl_db_version"),
        "release_date": status.get("chembl_release_date"),
        "base_url": B,
        "tables": [],
    }
    for name, (ep, params, key) in TABLES.items():
        prov["tables"].append(bulk(name, ep, params, key))
        print(name, prov["tables"][-1]["n_rows"], flush=True)
    Path("handoff/provenance_chembl.json").write_text(json.dumps(prov, indent=2))
    print("release", prov["release"], prov["release_date"])
