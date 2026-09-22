import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type SwarmLoopStatus,
  type SwarmRun,
  type WorkflowKPIs,
  type WorkflowRun,
} from "../api";

const STAGES = [
  { id: "data_check", label: "Data", hint: "verify required files" },
  { id: "build_features", label: "Features", hint: "run bridge.build_*_features" },
  {
    id: "iterate",
    label: "Iterate",
    hint: "Claude reviews the latest swarm results and picks the next batch",
  },
] as const;

type StageId = (typeof STAGES)[number]["id"];

const RADIUS = 190;
const CENTER = 240;
const NODE_R = 46;

function statusColor(status?: string): { fill: string; stroke: string; text: string } {
  switch (status) {
    case "ok":
      return { fill: "#dcfce7", stroke: "#059669", text: "#065f46" };
    case "failed":
      return { fill: "#fee2e2", stroke: "#dc2626", text: "#7f1d1d" };
    case "running":
      return { fill: "#fef3c7", stroke: "#d97706", text: "#78350f" };
    case "not_implemented":
      return { fill: "#f1f5f9", stroke: "#94a3b8", text: "#475569" };
    default:
      return { fill: "#ffffff", stroke: "#cbd5e1", text: "#475569" };
  }
}

export default function WorkflowPage() {
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [kpis, setKpis] = useState<WorkflowKPIs | null>(null);
  const [busy, setBusy] = useState<StageId | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<WorkflowRun | null>(null);
  const [loopStatus, setLoopStatus] = useState<SwarmLoopStatus | null>(null);
  const [swarmRuns, setSwarmRuns] = useState<SwarmRun[]>([]);
  const [loopError, setLoopError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [r, k, ls, sr] = await Promise.all([
        api.workflowRuns(50),
        api.workflowKPIs(),
        api.swarmLoopStatus().catch(() => null),
        api.swarmRuns(10).catch(() => []),
      ]);
      setRuns(r);
      setKpis(k);
      setLoopStatus(ls);
      setSwarmRuns(sr);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  async function toggleLoop() {
    setLoopError(null);
    try {
      if (loopStatus?.running) {
        await api.swarmLoopStop();
      } else {
        await api.swarmLoopStart(3);
      }
      await refresh();
    } catch (e) {
      setLoopError(e instanceof Error ? e.message : String(e));
    }
  }

  const latestByStage = useMemo(() => {
    const map: Record<string, WorkflowRun | undefined> = {};
    for (const r of runs) {
      if (!map[r.stage]) map[r.stage] = r;
    }
    return map;
  }, [runs]);

  async function trigger(stage: StageId) {
    setBusy(stage);
    setError(null);
    try {
      const run = await api.triggerStage(stage);
      setSelectedRun(run);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  const nodes = STAGES.map((s, i) => {
    const angle = (i / STAGES.length) * 2 * Math.PI - Math.PI / 2;
    return {
      ...s,
      x: CENTER + RADIUS * Math.cos(angle),
      y: CENTER + RADIUS * Math.sin(angle),
    };
  });

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <header>
        <h2 className="text-lg font-semibold text-slate-900">Continuous improvement loop</h2>
        <p className="mt-1 text-sm text-slate-500">
          Data → features → model → evaluate → Claude Science reviews → iterate. Each node
          runs a real handler and appends to{" "}
          <span className="font-mono text-xs">data/experiments/runs.jsonl</span>.
        </p>
      </header>

      {/* KPI strip — real science + loop metrics, not workflow plumbing */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi
          label="Conditions mapped"
          value={
            kpis && kpis.conditions_total
              ? `${kpis.conditions_mapped.toLocaleString()} / ${kpis.conditions_total.toLocaleString()}`
              : "—"
          }
          hint={
            kpis && kpis.conditions_total
              ? `${((kpis.conditions_mapped / kpis.conditions_total) * 100).toFixed(1)}%`
              : ""
          }
        />
        <Kpi
          label="Share drug-target gene"
          value={kpis ? kpis.conditions_with_drug_target_overlap.toLocaleString() : "—"}
          hint={
            kpis && kpis.conditions_total
              ? `${((kpis.conditions_with_drug_target_overlap / kpis.conditions_total) * 100).toFixed(1)}% ceiling`
              : ""
          }
        />
        <Kpi label="Swarm runs" value={kpis?.swarm_runs ?? 0} />
        <Kpi
          label="Last stage"
          value={kpis?.last_stage ?? "—"}
          hint={kpis?.last_stage_at ? relative(kpis.last_stage_at) : ""}
          mono
        />
      </div>

      {error && (
        <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
          {error}
        </div>
      )}

      {/* The circle */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <svg viewBox="0 0 480 480" className="mx-auto h-[520px] w-full max-w-[520px]">
            {/* connecting arcs */}
            {nodes.map((n, i) => {
              const next = nodes[(i + 1) % nodes.length];
              const path = arcPath(n.x, n.y, next.x, next.y, NODE_R, CENTER, CENTER);
              return (
                <path
                  key={`arc-${i}`}
                  d={path}
                  fill="none"
                  stroke="#cbd5e1"
                  strokeWidth={1.5}
                  markerEnd="url(#arrow)"
                />
              );
            })}

            <defs>
              <marker
                id="arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="8"
                markerHeight="8"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
              </marker>
            </defs>

            {/* center label */}
            <text
              x={CENTER}
              y={CENTER - 8}
              textAnchor="middle"
              className="fill-slate-400 text-[11px] uppercase tracking-widest"
            >
              Bridge
            </text>
            <text
              x={CENTER}
              y={CENTER + 12}
              textAnchor="middle"
              className="fill-slate-500 text-[10px]"
            >
              Modal · Claude Science
            </text>

            {/* nodes */}
            {nodes.map((n) => {
              const latest = latestByStage[n.id];
              const isBusy = busy === n.id;
              const status = isBusy ? "running" : latest?.status;
              const c = statusColor(status);
              return (
                <g
                  key={n.id}
                  onClick={() => !busy && trigger(n.id)}
                  className={busy ? "cursor-wait" : "cursor-pointer"}
                >
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={NODE_R}
                    fill={c.fill}
                    stroke={c.stroke}
                    strokeWidth={2}
                    strokeDasharray={status === "not_implemented" ? "4 4" : "0"}
                    className="transition"
                  />
                  {isBusy && (
                    <circle
                      cx={n.x}
                      cy={n.y}
                      r={NODE_R + 4}
                      fill="none"
                      stroke={c.stroke}
                      strokeWidth={2}
                      opacity={0.35}
                      className="animate-ping"
                    />
                  )}
                  <text
                    x={n.x}
                    y={n.y - 4}
                    textAnchor="middle"
                    className="pointer-events-none text-[13px] font-semibold"
                    fill={c.text}
                  >
                    {n.label}
                  </text>
                  <text
                    x={n.x}
                    y={n.y + 12}
                    textAnchor="middle"
                    className="pointer-events-none text-[9px]"
                    fill={c.text}
                  >
                    {isBusy
                      ? "running…"
                      : latest
                        ? `${latest.status} · ${latest.elapsed_ms}ms`
                        : "click to run"}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* right panel: selected run detail + iterate review */}
        <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="text-sm font-semibold text-slate-700">Latest / selected run</h3>
          {(selectedRun ?? runs[0]) ? (
            <RunDetail run={selectedRun ?? runs[0]} />
          ) : (
            <div className="text-sm text-slate-400">
              no runs yet — click a node to trigger one
            </div>
          )}
        </div>
      </div>

      {/* Latest Iterate review — surfaces Claude's synthesis of the swarm results */}
      <LatestReview runs={runs} />

      {/* Continuous swarm loop */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-700">
              Claude Science — 5-agent swarm loop
            </h3>
            <p className="mt-1 text-xs text-slate-500">
              While running, five Sonnet agents (mechanism · similar drugs · ontology ·
              evidence · contrarian) score the next unlabeled drug–condition pair in
              parallel, then the loop advances. All results append to{" "}
              <span className="font-mono">data/experiments/swarm.jsonl</span>.
            </p>
          </div>
          <div className="flex flex-col items-end gap-1">
            <button
              onClick={toggleLoop}
              className={`rounded-md px-4 py-2 text-xs font-medium text-white transition ${
                loopStatus?.running
                  ? "bg-rose-600 hover:bg-rose-500"
                  : "bg-emerald-600 hover:bg-emerald-500"
              }`}
            >
              {loopStatus?.running ? "Stop loop" : "Start loop"}
            </button>
            {loopStatus?.running && (
              <span className="text-[10px] text-slate-500">
                iterating every {loopStatus.interval_seconds}s
              </span>
            )}
          </div>
        </div>

        {loopError && (
          <div className="mb-3 rounded-md border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
            {loopError}
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MiniStat
            label="Status"
            value={loopStatus?.running ? "RUNNING" : "idle"}
            good={loopStatus?.running}
          />
          <MiniStat label="Iterations" value={loopStatus?.iterations ?? 0} />
          <MiniStat
            label="Last finished"
            value={loopStatus?.last_finished_at ? relative(loopStatus.last_finished_at) : "—"}
          />
          <MiniStat
            label="Last direction"
            value={loopStatus?.last_pair?.direction ?? "—"}
            mono
          />
        </div>

        {loopStatus?.last_pair && (
          <div className="mt-3 rounded border border-slate-200 bg-slate-50 p-3 text-xs">
            <span className="text-slate-500">latest:</span>{" "}
            <span className="font-medium">{loopStatus.last_pair.ingredient_name}</span>
            <span className="mx-2 text-slate-400">→</span>
            <span className="font-medium">{loopStatus.last_pair.condition_name}</span>
          </div>
        )}

        {loopStatus?.last_error && !loopStatus.running && (
          <div className="mt-3 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
            {loopStatus.last_error}
          </div>
        )}

        {swarmRuns.length > 0 && (
          <div className="mt-4 overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] uppercase text-slate-500">
                  <th className="py-1.5">When</th>
                  <th className="py-1.5">Drug</th>
                  <th className="py-1.5">Condition</th>
                  <th className="py-1.5">Direction</th>
                  <th className="py-1.5 text-right">Treats</th>
                  <th className="py-1.5 text-right">Causes</th>
                  <th className="py-1.5 text-right">σ</th>
                </tr>
              </thead>
              <tbody>
                {swarmRuns.slice(0, 8).map((r) => (
                  <tr key={r.id} className="border-t border-slate-100">
                    <td className="py-1.5 font-mono text-xs text-slate-500">
                      {r.started_at.slice(11, 19)}
                    </td>
                    <td className="py-1.5 truncate">{r.ingredient_name}</td>
                    <td className="py-1.5 truncate">{r.condition_name}</td>
                    <td className="py-1.5">
                      <DirectionPill direction={r.consensus.direction} />
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs">
                      {r.consensus.mean_treats.toFixed(2)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs">
                      {r.consensus.mean_causes.toFixed(2)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs text-slate-500">
                      {Math.max(r.consensus.stdev_treats, r.consensus.stdev_causes).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Runs history */}
      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="mb-3 text-sm font-semibold text-slate-700">Experiment log</h3>
        {runs.length === 0 ? (
          <div className="text-sm text-slate-400">nothing yet</div>
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase text-slate-500">
                  <th className="py-1.5">When</th>
                  <th className="py-1.5">Stage</th>
                  <th className="py-1.5">Status</th>
                  <th className="py-1.5 text-right">Elapsed</th>
                  <th className="py-1.5">Notes</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr
                    key={r.id}
                    onClick={() => setSelectedRun(r)}
                    className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                      selectedRun?.id === r.id ? "bg-slate-50" : ""
                    }`}
                  >
                    <td className="py-1.5 font-mono text-xs text-slate-500">
                      {r.started_at.slice(11, 19)}
                    </td>
                    <td className="py-1.5">{r.stage}</td>
                    <td className="py-1.5">
                      <StatusPill status={r.status} />
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs">
                      {r.elapsed_ms.toLocaleString()} ms
                    </td>
                    <td className="py-1.5 text-xs text-slate-500">
                      {r.error ?? r.notes ?? ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function LatestReview({ runs }: { runs: WorkflowRun[] }) {
  const latest = runs.find(
    (r) => r.stage === "iterate" && r.status === "ok" && typeof r.metrics?.review === "string",
  );
  if (!latest) return null;
  const review = String(latest.metrics.review);
  const dc = (latest.metrics.direction_counts ?? {}) as Record<string, number>;
  const n = Number(latest.metrics.n_swarm_runs_reviewed ?? 0);
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">
          Claude review — from the last Iterate
        </h3>
        <span className="text-[10px] text-slate-500">
          synthesised over {n} recent swarm run{n === 1 ? "" : "s"}
        </span>
      </div>
      {Object.keys(dc).length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2 text-[11px]">
          {Object.entries(dc).map(([k, v]) => (
            <span
              key={k}
              className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-slate-600"
            >
              {k.replace("_", " ")}: <span className="font-mono">{v}</span>
            </span>
          ))}
        </div>
      )}
      <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-slate-700">
        {review}
      </pre>
    </section>
  );
}

function Kpi({
  label,
  value,
  hint,
  mono = false,
}: {
  label: string;
  value: string | number;
  hint?: string;
  mono?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div
        className={`mt-1 text-xl font-semibold text-slate-900 ${mono ? "font-mono text-base" : ""}`}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

function MiniStat({
  label,
  value,
  mono = false,
  good,
}: {
  label: string;
  value: string | number;
  mono?: boolean;
  good?: boolean;
}) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div
        className={`mt-0.5 font-semibold ${mono ? "font-mono text-sm" : "text-base"} ${
          good === true
            ? "text-emerald-700"
            : good === false
              ? "text-slate-600"
              : "text-slate-900"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function DirectionPill({ direction }: { direction: string }) {
  const map: Record<string, string> = {
    treats: "bg-emerald-50 text-emerald-700",
    causes: "bg-rose-50 text-rose-700",
    mixed: "bg-amber-50 text-amber-800",
    no_signal: "bg-slate-100 text-slate-600",
  };
  const cls = map[direction] ?? map.no_signal;
  return (
    <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${cls}`}>
      {direction.replace("_", " ")}
    </span>
  );
}

function StatusPill({ status }: { status: string }) {
  const cls =
    status === "ok"
      ? "bg-emerald-50 text-emerald-700"
      : status === "failed"
        ? "bg-rose-50 text-rose-700"
        : status === "not_implemented"
          ? "bg-slate-100 text-slate-600"
          : "bg-amber-50 text-amber-800";
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>{status}</span>
  );
}

function RunDetail({ run }: { run: WorkflowRun }) {
  return (
    <div className="space-y-2 text-sm">
      <div className="flex justify-between">
        <span className="font-medium text-slate-700">{run.stage}</span>
        <StatusPill status={run.status} />
      </div>
      <div className="font-mono text-[10px] text-slate-500">
        {run.started_at} · {run.elapsed_ms}ms
      </div>
      {run.error && (
        <div className="rounded border border-rose-200 bg-rose-50 p-2 text-xs text-rose-700">
          {run.error}
        </div>
      )}
      {Object.keys(run.metrics).length > 0 && (
        <pre className="max-h-72 overflow-auto rounded bg-slate-900 p-3 text-[10px] leading-relaxed text-slate-100">
          {JSON.stringify(run.metrics, null, 2)}
        </pre>
      )}
    </div>
  );
}

function relative(iso: string): string {
  const then = new Date(iso).getTime();
  const now = Date.now();
  const s = Math.round((now - then) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function arcPath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  nodeR: number,
  cx: number,
  cy: number,
): string {
  // shrink endpoints so arrows meet node borders, not centers
  const a1 = Math.atan2(y1 - cy, x1 - cx);
  const a2 = Math.atan2(y2 - cy, x2 - cx);
  const off = (nodeR / Math.hypot(x1 - cx, y1 - cy)) * 0.9;
  const sx = x1 + (cx - x1) * off * 0.15 + Math.cos(a1 + Math.PI / 2) * nodeR * 0.9;
  const sy = y1 + (cy - y1) * off * 0.15 + Math.sin(a1 + Math.PI / 2) * nodeR * 0.9;
  const ex = x2 + (cx - x2) * off * 0.15 + Math.cos(a2 - Math.PI / 2) * nodeR * 0.9;
  const ey = y2 + (cy - y2) * off * 0.15 + Math.sin(a2 - Math.PI / 2) * nodeR * 0.9;
  const mid = (Math.hypot(x1 - cx, y1 - cy) + Math.hypot(x2 - cx, y2 - cy)) / 2;
  return `M ${sx} ${sy} A ${mid} ${mid} 0 0 1 ${ex} ${ey}`;
}
