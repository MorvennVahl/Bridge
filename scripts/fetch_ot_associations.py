"""Pull Open Targets disease-target associations for the mapped CEM conditions.

The association_by_datatype_indirect dataset is ~2.9 GB across 41 parquet parts,
but only the diseases our conditions map onto are needed. Each part is downloaded
to a scratch directory, filtered to the wanted disease ids, and deleted before the
next part, so peak disk is one part (~80 MB) rather than the whole dataset.

"indirect" is the right table: its scores are rolled up the MONDO/EFO hierarchy, so
a condition mapped to a parent term still sees evidence recorded on its children.

Usage:
    python scripts/fetch_ot_associations.py <disease_ids.txt> <out.parquet> [scratch_dir]
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BASE = (
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output/"
    "association_by_datatype_indirect/"
)


def part_names() -> list[str]:
    with urllib.request.urlopen(BASE, timeout=90) as r:
        html = r.read().decode("utf8", "replace")
    return sorted(set(re.findall(r'href="([^"]+\.parquet)"', html)))


def main() -> int:
    ids_file, out_file = sys.argv[1], sys.argv[2]
    scratch = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(tempfile.mkdtemp())
    scratch.mkdir(parents=True, exist_ok=True)

    with open(ids_file) as fh:
        wanted = {ln.strip() for ln in fh if ln.strip()}
    print(f"wanted disease ids: {len(wanted)}", flush=True)

    parts = part_names()
    print(f"parts: {len(parts)}", flush=True)

    kept = []
    for i, name in enumerate(parts, 1):
        tmp = scratch / name
        urllib.request.urlretrieve(BASE + name, tmp)
        try:
            if i == 1:
                print("schema:", pq.ParquetFile(tmp).schema_arrow.names, flush=True)
            df = pd.read_parquet(tmp)
            sub = df[df.diseaseId.isin(wanted)]
            if len(sub):
                kept.append(sub)
            print(f"  part {i}/{len(parts)}: {len(df)} rows -> kept {len(sub)}", flush=True)
        finally:
            # scratch lives in the session workspace, never in a granted host folder
            os.unlink(tmp)

    res = (
        pd.concat(kept, ignore_index=True)
        if kept
        else pd.DataFrame(columns=["diseaseId", "targetId", "datatypeId", "score"])
    )
    res.to_parquet(out_file, index=False)
    print(
        f"total rows: {len(res)} | diseases: {res.diseaseId.nunique()} "
        f"| targets: {res.targetId.nunique()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
