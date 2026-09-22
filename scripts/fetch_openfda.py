"""openFDA aggregate pulls keyed on generic (ingredient) name.

Only count-aggregations are used: openFDA allows 240 req/min and 1,000 req/day
without a key, so per-ingredient queries over 4,276 names are not feasible.
Each aggregation returns the top 1,000 generic names for that query.

Outputs: handoff/openfda_<name>.json, handoff/provenance_openfda.json
"""
import json, time, datetime
import requests

NOW = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
BASE = "https://api.fda.gov"

QUERIES = {
    # FAERS spontaneous-report volume per ingredient (label-adjacent: same lineage as CEM/AEOLUS)
    "faers_reports_total":  ("/drug/event.json",
                             {"count": "patient.drug.openfda.generic_name.exact"}),
    "faers_reports_serious": ("/drug/event.json",
                             {"search": "serious:1", "count": "patient.drug.openfda.generic_name.exact"}),
    "faers_reports_death":  ("/drug/event.json",
                             {"search": "seriousnessdeath:1", "count": "patient.drug.openfda.generic_name.exact"}),
    # SPL label structure per ingredient
    "label_any":            ("/drug/label.json",
                             {"count": "openfda.generic_name.exact"}),
    "label_boxed_warning":  ("/drug/label.json",
                             {"search": "_exists_:boxed_warning", "count": "openfda.generic_name.exact"}),
    "label_pregnancy":      ("/drug/label.json",
                             {"search": "_exists_:pregnancy", "count": "openfda.generic_name.exact"}),
    "label_drug_interactions": ("/drug/label.json",
                             {"search": "_exists_:drug_interactions", "count": "openfda.generic_name.exact"}),
}

prov = {"source": "openFDA", "base_url": BASE, "queries": {},
        "note": ("openFDA count aggregations return the top 1,000 terms only; ingredients outside "
                 "the top 1,000 for a given query are absent, which is not the same as zero. "
                 "FAERS counts reflect reporting volume, not incidence.")}

for name, (path, params) in QUERIES.items():
    p = {**params, "limit": 1000}
    for a in range(4):
        r = requests.get(BASE + path, params=p, timeout=180)
        if r.status_code == 200:
            break
        time.sleep(4 * (a + 1))
    if r.status_code != 200:
        prov["queries"][name] = {"endpoint": BASE + path, "params": p,
                                 "error": f"HTTP {r.status_code}", "retrieved_utc": NOW()}
        print(name, "FAILED", r.status_code, flush=True)
        continue
    j = r.json()
    rows = j.get("results", [])
    json.dump(rows, open(f"handoff/openfda_{name}.json", "w"))
    prov["queries"][name] = {"endpoint": BASE + path, "params": p, "n_terms": len(rows),
                             "disclaimer": j.get("meta", {}).get("disclaimer"),
                             "last_updated": j.get("meta", {}).get("last_updated"),
                             "retrieved_utc": NOW()}
    print(name, len(rows), flush=True)
    time.sleep(1)

json.dump(prov, open("handoff/provenance_openfda.json", "w"), indent=2)
