"""Claude Science — a parallel swarm of specialized agents.

Given a drug-condition pair, fan out 5 Claude Sonnet calls, each with a
distinct analytical role. All results are appended to
``data/experiments/swarm.jsonl`` and returned to the caller as a bundle so
the UI can render them side-by-side.

Nothing is mocked. If ``ANTHROPIC_API_KEY`` is unset or the SDK is missing,
the endpoint fails loudly with 503 — no synthesized fake responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.data import DATA_DIR, load_associations, load_conditions, load_ingredients

router = APIRouter(prefix="/api/swarm", tags=["swarm"])

EXPERIMENTS_DIR: Path = DATA_DIR / "experiments"
SWARM_LOG: Path = EXPERIMENTS_DIR / "swarm.jsonl"
_append_lock = Lock()

MODEL = "claude-sonnet-4-6"


class Agent(BaseModel):
    role: str
    system: str


AGENTS: list[Agent] = [
    Agent(
        role="mechanism",
        system=(
            "You are the MECHANISM analyst on a drug-repurposing team. Given a drug "
            "and a condition, reason from targets → pathways → disease biology. "
            "You value gene-set overlap and pathway congruence over reporting bias."
        ),
    ),
    Agent(
        role="similar_drugs",
        system=(
            "You are the SIMILAR-DRUG scout. Given a drug, think of other drugs that "
            "share its mechanism class (same target family, same effect direction) and "
            "reason whether their known relationships to this condition transfer."
        ),
    ),
    Agent(
        role="ontology_depth",
        system=(
            "You are the ONTOLOGY analyst. Reason about where the condition sits in "
            "the HPO / MONDO hierarchy. A very specific condition (deep in the tree, "
            "high information content) with a clean gene link is a strong signal; a "
            "broad umbrella term (Phenotypic abnormality) is not."
        ),
    ),
    Agent(
        role="evidence",
        system=(
            "You are the EVIDENCE reviewer. You only reason from real-world evidence: "
            "FAERS disproportionality, EU label mentions, published SemMedDB assertions. "
            "You explicitly discount FAERS PRR < 2 and single-case reports as noise."
        ),
    ),
    Agent(
        role="contrarian",
        system=(
            "You are the CONTRARIAN / safety officer. Your job is to find the case "
            "AGAINST treatment and FOR causation. Consider adverse pharmacology, "
            "off-target effects, and reasons the pair looks safe that are really "
            "reporting artifacts. Push back on the mechanism agent's assumptions."
        ),
    ),
]

_RESPONSE_INSTRUCTIONS = """
Reply with STRICT JSON only, no prose before or after, with this shape:

{
  "score_treats": 0.0-1.0,
  "score_causes": 0.0-1.0,
  "confidence": 0.0-1.0,
  "one_sentence_verdict": "...",
  "rationale": "...",
  "key_evidence": ["...", "..."]
}
"""


class SwarmRequest(BaseModel):
    ingredient_concept_id: int
    condition_concept_id: int


class AgentResult(BaseModel):
    role: str
    status: str  # ok | failed
    score_treats: float | None = None
    score_causes: float | None = None
    confidence: float | None = None
    verdict: str | None = None
    rationale: str | None = None
    key_evidence: list[str] = Field(default_factory=list)
    elapsed_ms: int
    error: str | None = None


class Consensus(BaseModel):
    mean_treats: float
    mean_causes: float
    stdev_treats: float
    stdev_causes: float
    n_agents: int
    direction: str  # "treats" | "causes" | "mixed" | "no_signal"


class SwarmRun(BaseModel):
    id: str
    started_at: str
    finished_at: str
    ingredient_concept_id: int
    ingredient_name: str
    condition_concept_id: int
    condition_name: str
    context: dict[str, Any]
    agents: list[AgentResult]
    consensus: Consensus


def _append_run(run: SwarmRun) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with _append_lock, SWARM_LOG.open("a", encoding="utf-8") as fh:
        fh.write(run.model_dump_json() + "\n")


def _build_context(ingredient_concept_id: int, condition_concept_id: int) -> dict[str, Any]:
    """Real signals from the data files that all agents get as user-message context."""
    ctx: dict[str, Any] = {}

    ings = load_ingredients()
    conds = load_conditions()
    ing = ings[ings["ingredient_concept_id"] == ingredient_concept_id] if not ings.empty else ings
    cond = (
        conds[conds["condition_concept_id"] == condition_concept_id] if not conds.empty else conds
    )
    if ing.empty:
        raise HTTPException(status_code=404, detail=f"unknown ingredient {ingredient_concept_id}")
    if cond.empty:
        raise HTTPException(status_code=404, detail=f"unknown condition {condition_concept_id}")

    ctx["ingredient_name"] = str(ing.iloc[0]["ingredient_name"])
    ctx["condition_name"] = str(cond.iloc[0]["condition_name"])

    associations = load_associations()
    if not associations.empty:
        try:
            match = associations[
                (associations["ingredient_concept_id"] == ingredient_concept_id)
                & (associations["condition_concept_id"] == condition_concept_id)
            ]
            if not match.empty:
                row = match.iloc[0]
                ctx["in_faers"] = bool(row.get("in_faers", False))
                ctx["in_semmeddb"] = bool(row.get("in_semmeddb", False))
                for col in ("faers_prr", "faers_case_count", "semmeddb_relationships"):
                    v = row.get(col)
                    if v is not None and str(v) != "nan":
                        ctx[col] = str(v)
        except KeyError:
            pass

    # Best-effort: condition ontology position from Frank's condition_features
    features_path = DATA_DIR / "derived" / "condition_features.parquet"
    if features_path.is_file():
        try:
            import pyarrow.parquet as pq

            tbl = pq.read_table(features_path)
            df = tbl.to_pandas()
            row = df[df["condition_concept_id"] == condition_concept_id]
            if not row.empty:
                for col in (
                    "ontology_id",
                    "ontology",
                    "match_tier",
                    "n_genes",
                    "n_genes_shared_with_drug_target",
                    "n_drug_target_genes",
                ):
                    if col in row.columns:
                        v = row.iloc[0][col]
                        if v is not None and str(v) != "nan":
                            ctx[col] = str(v)
        except Exception:
            pass

    return ctx


def _parse_agent_json(text: str) -> dict[str, Any] | None:
    """Extract JSON from the model reply; be lenient about ```json``` fences."""
    text = text.strip()
    # strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find a JSON object inside
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def _call_agent(client: Any, agent: Agent, user_content: str) -> AgentResult:
    t0 = time.monotonic()
    try:
        resp = await asyncio.to_thread(
            client.messages.create,
            model=MODEL,
            max_tokens=600,
            system=agent.system + "\n\n" + _RESPONSE_INSTRUCTIONS,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "\n".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        parsed = _parse_agent_json(text)
        if parsed is None:
            return AgentResult(
                role=agent.role,
                status="failed",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                error="response was not valid JSON",
                rationale=text[:500],
            )
        return AgentResult(
            role=agent.role,
            status="ok",
            score_treats=_clamp(parsed.get("score_treats")),
            score_causes=_clamp(parsed.get("score_causes")),
            confidence=_clamp(parsed.get("confidence")),
            verdict=str(parsed.get("one_sentence_verdict") or "")[:400] or None,
            rationale=str(parsed.get("rationale") or "")[:2000] or None,
            key_evidence=[str(x)[:200] for x in (parsed.get("key_evidence") or [])][:8],
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )
    except Exception as e:
        return AgentResult(
            role=agent.role,
            status="failed",
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=f"{type(e).__name__}: {e}",
        )


def _clamp(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


def _consensus(agents: list[AgentResult]) -> Consensus:
    treats = [a.score_treats for a in agents if a.score_treats is not None]
    causes = [a.score_causes for a in agents if a.score_causes is not None]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def stdev(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    mt, mc = mean(treats), mean(causes)
    direction = "no_signal"
    if mt >= 0.6 and mc < 0.4:
        direction = "treats"
    elif mc >= 0.6 and mt < 0.4:
        direction = "causes"
    elif mt >= 0.4 and mc >= 0.4:
        direction = "mixed"
    return Consensus(
        mean_treats=round(mt, 3),
        mean_causes=round(mc, 3),
        stdev_treats=round(stdev(treats), 3),
        stdev_causes=round(stdev(causes), 3),
        n_agents=len([a for a in agents if a.status == "ok"]),
        direction=direction,
    )


@router.get("/agents", response_model=list[Agent])
def list_agents() -> list[Agent]:
    return AGENTS


# ---------- continuous-loop background task ----------


class LoopStatus(BaseModel):
    running: bool
    iterations: int
    started_at: str | None
    last_pair: dict[str, Any] | None
    last_finished_at: str | None
    last_error: str | None
    interval_seconds: float


_loop_state: dict[str, Any] = {
    "running": False,
    "iterations": 0,
    "started_at": None,
    "last_pair": None,
    "last_finished_at": None,
    "last_error": None,
    "interval_seconds": 3.0,
    "stop_requested": False,
    "task": None,
}


def _already_scored_pairs() -> set[tuple[int, int]]:
    """Set of (ingredient, condition) pairs already in swarm.jsonl."""
    if not SWARM_LOG.is_file():
        return set()
    scored: set[tuple[int, int]] = set()
    with SWARM_LOG.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                scored.add((int(d["ingredient_concept_id"]), int(d["condition_concept_id"])))
            except Exception:
                continue
    return scored


def _pick_candidates(limit: int = 200) -> list[tuple[int, int]]:
    """Prefer pairs with SemMedDB directional evidence — the ~7,900 labeled rows."""
    associations = load_associations()
    if associations.empty:
        return []
    if "semmeddb_relationships" not in associations.columns:
        return []
    labeled = associations[associations["semmeddb_relationships"].notna()]
    pairs = list(
        zip(
            labeled["ingredient_concept_id"].astype(int).tolist(),
            labeled["condition_concept_id"].astype(int).tolist(),
            strict=False,
        )
    )
    return pairs[:limit]


async def _loop_body(client: Any) -> None:
    _loop_state["stop_requested"] = False
    _loop_state["iterations"] = 0
    _loop_state["started_at"] = datetime.now(UTC).isoformat()
    _loop_state["running"] = True
    _loop_state["last_error"] = None

    try:
        while not _loop_state["stop_requested"]:
            scored = _already_scored_pairs()
            candidates = [p for p in _pick_candidates(500) if p not in scored]
            if not candidates:
                _loop_state["last_error"] = "no unscored candidate pairs left"
                break

            ing_id, cond_id = candidates[0]
            try:
                context = _build_context(ing_id, cond_id)
                user_content = (
                    f"Drug: {context['ingredient_name']} (ingredient_concept_id={ing_id})\n"
                    f"Condition: {context['condition_name']} (condition_concept_id={cond_id})\n\n"
                    f"Signals from the data:\n{json.dumps({k: v for k, v in context.items() if k not in ('ingredient_name', 'condition_name')}, indent=2)}\n\n"
                    "Score treats and causes each on [0,1]. Return the JSON schema."
                )
                started = datetime.now(UTC).isoformat()
                results = await asyncio.gather(
                    *[_call_agent(client, a, user_content) for a in AGENTS]
                )
                finished = datetime.now(UTC).isoformat()

                run = SwarmRun(
                    id=uuid.uuid4().hex[:12],
                    started_at=started,
                    finished_at=finished,
                    ingredient_concept_id=ing_id,
                    ingredient_name=context["ingredient_name"],
                    condition_concept_id=cond_id,
                    condition_name=context["condition_name"],
                    context={
                        k: v
                        for k, v in context.items()
                        if k not in ("ingredient_name", "condition_name")
                    },
                    agents=list(results),
                    consensus=_consensus(list(results)),
                )
                _append_run(run)

                _loop_state["iterations"] = int(_loop_state["iterations"]) + 1
                _loop_state["last_pair"] = {
                    "ingredient_concept_id": ing_id,
                    "condition_concept_id": cond_id,
                    "ingredient_name": context["ingredient_name"],
                    "condition_name": context["condition_name"],
                    "direction": run.consensus.direction,
                }
                _loop_state["last_finished_at"] = finished
                _loop_state["last_error"] = None
            except Exception as e:
                _loop_state["last_error"] = f"{type(e).__name__}: {e}"

            await asyncio.sleep(float(_loop_state["interval_seconds"]))
    finally:
        _loop_state["running"] = False
        _loop_state["task"] = None


@router.post("/loop/start", response_model=LoopStatus)
async def start_loop(interval_seconds: float = 3.0) -> LoopStatus:
    if _loop_state["running"]:
        return LoopStatus(**{k: _loop_state[k] for k in LoopStatus.model_fields})
    try:
        import anthropic
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"anthropic SDK missing: {e}") from e
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set — export it and restart uvicorn",
        )
    _loop_state["interval_seconds"] = max(0.5, float(interval_seconds))
    client = anthropic.Anthropic(api_key=key)
    _loop_state["task"] = asyncio.create_task(_loop_body(client))
    await asyncio.sleep(0)  # yield so the task registers
    return LoopStatus(**{k: _loop_state[k] for k in LoopStatus.model_fields})


@router.post("/loop/stop", response_model=LoopStatus)
async def stop_loop() -> LoopStatus:
    _loop_state["stop_requested"] = True
    return LoopStatus(**{k: _loop_state[k] for k in LoopStatus.model_fields})


@router.get("/loop/status", response_model=LoopStatus)
def loop_status() -> LoopStatus:
    return LoopStatus(**{k: _loop_state[k] for k in LoopStatus.model_fields})


@router.get("/runs", response_model=list[SwarmRun])
def list_runs(limit: int = 20) -> list[SwarmRun]:
    if not SWARM_LOG.is_file():
        return []
    out: list[SwarmRun] = []
    with SWARM_LOG.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(SwarmRun.model_validate_json(line))
            except Exception:
                continue
    return list(reversed(out))[:limit]


@router.post("/run", response_model=SwarmRun)
async def run_swarm(req: SwarmRequest) -> SwarmRun:
    try:
        import anthropic
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"anthropic SDK missing: {e}") from e

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set — export it in the shell running uvicorn",
        )

    context = _build_context(req.ingredient_concept_id, req.condition_concept_id)
    user_content = (
        f"Drug: {context['ingredient_name']} (ingredient_concept_id={req.ingredient_concept_id})\n"
        f"Condition: {context['condition_name']} (condition_concept_id={req.condition_concept_id})\n\n"
        f"Signals from the data:\n{json.dumps({k: v for k, v in context.items() if k not in ('ingredient_name', 'condition_name')}, indent=2)}\n\n"
        "Score treats (drug helps this condition) and causes (drug worsens this condition) "
        "each on [0,1]. Return the JSON schema in your system prompt."
    )

    client = anthropic.Anthropic(api_key=key)
    started = datetime.now(UTC).isoformat()
    results = await asyncio.gather(*[_call_agent(client, a, user_content) for a in AGENTS])
    finished = datetime.now(UTC).isoformat()

    run = SwarmRun(
        id=uuid.uuid4().hex[:12],
        started_at=started,
        finished_at=finished,
        ingredient_concept_id=req.ingredient_concept_id,
        ingredient_name=context["ingredient_name"],
        condition_concept_id=req.condition_concept_id,
        condition_name=context["condition_name"],
        context={
            k: v for k, v in context.items() if k not in ("ingredient_name", "condition_name")
        },
        agents=list(results),
        consensus=_consensus(list(results)),
    )
    _append_run(run)
    return run
