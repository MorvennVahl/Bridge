"""Bridge MCP server — exposes drug x condition tools for Claude Science.

Runs over stdio. Register in Claude Science's UI under Customize → MCP Connectors,
or add to ``~/.claude-science/mcp/servers.json`` (see README).

Tools exposed:
  - search_ingredients(query, limit)  — RxNorm drug lookup
  - search_conditions(query, limit)   — OMOP condition lookup
  - get_pair_evidence(ing, cond)      — FAERS / SemMedDB rows for a specific pair
  - get_condition_features(cond)      — Frank's derived condition_features row
  - score_pair_with_swarm(ing, cond)  — run the 5-agent Anthropic swarm now, return consensus + rationales
  - list_recent_swarm_runs(limit)     — read data/experiments/swarm.jsonl
  - workflow_status()                 — KPIs + latest workflow runs

All I/O is real — reads Frank's parquets and Adam's CSVs directly. No mocks.
Requires ANTHROPIC_API_KEY in environment for score_pair_with_swarm.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure our package tree is importable when Claude Science spawns us with its own cwd.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("bridge")


def _lazy_backend():
    """Import backend modules lazily so a missing dep on tool-not-used stays quiet."""
    from backend.app import data as bdata
    from backend.app.routers import swarm as bswarm
    from backend.app.routers import workflow as bworkflow

    return bdata, bswarm, bworkflow


@mcp.tool()
def search_ingredients(query: str = "", limit: int = 20) -> str:
    """Search RxNorm drug ingredients by case-insensitive name substring.

    Returns up to ``limit`` rows as JSON. Use ``ingredient_concept_id`` for
    joins and follow-up tool calls.
    """
    bdata, _, _ = _lazy_backend()
    df = bdata.load_ingredients()
    if df.empty:
        return json.dumps({"error": "cem_ingredients.csv not readable"})
    if query:
        df = df[df["ingredient_name"].str.contains(query, case=False, na=False)]
    df = df.head(max(1, min(limit, 200)))
    return json.dumps(df.to_dict(orient="records"), default=str)


@mcp.tool()
def search_conditions(query: str = "", limit: int = 20) -> str:
    """Search OMOP conditions by case-insensitive name substring.

    Returns up to ``limit`` rows as JSON keyed on ``condition_concept_id``.
    """
    bdata, _, _ = _lazy_backend()
    df = bdata.load_conditions()
    if df.empty:
        return json.dumps({"error": "associations file not readable"})
    if query:
        df = df[df["condition_name"].str.contains(query, case=False, na=False)]
    df = df.head(max(1, min(limit, 200)))
    return json.dumps(df.to_dict(orient="records"), default=str)


@mcp.tool()
def get_pair_evidence(ingredient_concept_id: int, condition_concept_id: int) -> str:
    """Return raw evidence for a specific drug-condition pair.

    Reads from data/cem_ingredient_condition_associations.csv. Includes
    FAERS disproportionality (PRR, case count) and SemMedDB relationship
    types (TREATS, PREVENTS, CAUSES, PREDISPOSES, ...). Empty if the pair
    is not in the CEM extract.
    """
    bdata, _, _ = _lazy_backend()
    a = bdata.load_associations()
    if a.empty:
        return json.dumps({"error": "associations empty or missing"})
    try:
        m = a[
            (a["ingredient_concept_id"] == ingredient_concept_id)
            & (a["condition_concept_id"] == condition_concept_id)
        ]
    except KeyError as e:
        return json.dumps({"error": f"missing column: {e}"})
    if m.empty:
        return json.dumps({"found": False})
    row = {k: (None if str(v) == "nan" else v) for k, v in m.iloc[0].to_dict().items()}
    return json.dumps({"found": True, "row": row}, default=str)


@mcp.tool()
def get_condition_features(condition_concept_id: int) -> str:
    """Return Frank's derived features for one condition.

    Reads data/derived/condition_features.parquet. Fields include match_tier,
    arm (disease|phenotype), ontology (MONDO|HPO), n_genes,
    n_genes_with_ensembl, ontology_depth, information_content, etc.
    """
    features = REPO_ROOT / "data" / "derived" / "condition_features.parquet"
    if not features.is_file():
        return json.dumps(
            {"error": "condition_features.parquet missing — run build_condition_features"}
        )
    import pyarrow.parquet as pq

    df = pq.read_table(features).to_pandas()
    row = df[df["condition_concept_id"] == condition_concept_id]
    if row.empty:
        return json.dumps({"found": False})
    r = {k: (None if str(v) == "nan" else v) for k, v in row.iloc[0].to_dict().items()}
    return json.dumps({"found": True, "row": r}, default=str)


@mcp.tool()
async def score_pair_with_swarm(ingredient_concept_id: int, condition_concept_id: int) -> str:
    """Run the 5-agent Anthropic swarm on this pair right now.

    Requires ANTHROPIC_API_KEY. Five Claude Sonnet agents (mechanism,
    similar_drugs, ontology_depth, evidence, contrarian) score treats/causes
    in parallel. Returns consensus + each agent's rationale. Also appends to
    data/experiments/swarm.jsonl.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return json.dumps({"error": "ANTHROPIC_API_KEY not set in the environment MCP inherits"})
    _, bswarm, _ = _lazy_backend()
    req = bswarm.SwarmRequest(
        ingredient_concept_id=ingredient_concept_id,
        condition_concept_id=condition_concept_id,
    )
    run = await bswarm.run_swarm(req)
    return json.dumps(run.model_dump(), default=str)


@mcp.tool()
def list_recent_swarm_runs(limit: int = 10) -> str:
    """List recent swarm scoring runs from data/experiments/swarm.jsonl."""
    _, bswarm, _ = _lazy_backend()
    runs = bswarm.list_runs(limit=max(1, min(limit, 100)))
    return json.dumps([r.model_dump() for r in runs], default=str)


@mcp.tool()
def workflow_status() -> str:
    """Return current pipeline KPIs and the last 10 workflow runs.

    KPIs include conditions_mapped, conditions_with_genes,
    conditions_with_drug_target_overlap (the honest 20.6% modelable
    ceiling), and swarm_runs. See data/experiments/runs.jsonl for detail.
    """
    _, _, bworkflow = _lazy_backend()
    kpis = bworkflow.get_kpis()
    recent = bworkflow.list_runs(limit=10)
    return json.dumps(
        {
            "kpis": kpis.model_dump(),
            "recent_runs": [r.model_dump() for r in recent],
        },
        default=str,
    )


@mcp.tool()
def repo_overview() -> str:
    """One-shot orientation for a Claude Science agent hitting this server the first time."""
    return json.dumps(
        {
            "project": "Bridge — drug/condition prediction",
            "premise": "Predict whether a drug treats or causes a condition using biology (targets → genes → pathways → disease) as the bridge.",
            "data_universe": {
                "ingredients": 4276,
                "conditions": 5631,
                "possible_pairs": 24_078_456,
                "labeled_pairs_semmeddb": 7900,
                "conditions_biologically_reasonable": 1158,
                "ceiling_pct": 20.6,
            },
            "key_files": {
                "labels": "data/cem_ingredient_condition_associations.csv",
                "drug_features": "data/ingredient_features.csv",
                "condition_features": "data/derived/condition_features.parquet",
                "condition_genes": "data/derived/condition_genes.parquet",
                "drug_targets": "data/ingredient_target_long.csv",
            },
            "swarm_agents": [
                "mechanism",
                "similar_drugs",
                "ontology_depth",
                "evidence",
                "contrarian",
            ],
            "how_to_use_me": (
                "1) search_ingredients / search_conditions to find IDs. "
                "2) get_pair_evidence + get_condition_features to inspect. "
                "3) score_pair_with_swarm to invoke our 5-agent scorer. "
                "4) workflow_status for pipeline KPIs."
            ),
        },
        indent=2,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio (for CLI testing and stdio-capable clients)
