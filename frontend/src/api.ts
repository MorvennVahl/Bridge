export interface Ingredient {
  ingredient_concept_id: number;
  ingredient_name: string;
}

export interface Condition {
  condition_concept_id: number;
  condition_name: string;
}

export interface EvidenceRow {
  source: string;
  relationship: string | null;
  value: number | null;
}

export interface PredictResponse {
  ingredient_concept_id: number;
  ingredient_name: string;
  condition_concept_id: number;
  condition_name: string;
  treats_score: number | null;
  causes_score: number | null;
  confidence: number | null;
  evidence: EvidenceRow[];
  status: string;
  model: string | null;
}

export interface DrugSummaryTopRow {
  condition_concept_id: number;
  condition_name: string;
  faers_prr: number | null;
  semmeddb_relationships: string | null;
}

export interface DrugSummaryResponse {
  ingredient_concept_id: number;
  rows_scanned: number;
  conditions: number;
  faers_pairs: number;
  semmeddb_pairs: number;
  top: DrugSummaryTopRow[];
  elapsed_ms: number;
  executor: string;
}

export interface TableMeta {
  name: string;
  path: string;
  fmt: string;
  columns: string[];
  row_count: number;
  editable: boolean;
  warn: string | null;
}

export type TableRow = Record<string, unknown> & { __index__: number };

export interface TablePage {
  name: string;
  columns: string[];
  rows: TableRow[];
  offset: number;
  limit: number;
  total: number;
}

export interface CommitBody {
  edits: { index: number; values: Record<string, string> }[];
  adds: { values: Record<string, string> }[];
}

export interface CommitResult {
  edited: number;
  added: number;
  total: number;
}

const BASE = "";

async function jsonReq<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

export interface WorkflowRun {
  id: string;
  stage: string;
  status: string;
  started_at: string;
  finished_at: string;
  elapsed_ms: number;
  metrics: Record<string, unknown>;
  notes: string | null;
  error: string | null;
}

export interface WorkflowKPIs {
  total_runs: number;
  last_stage: string | null;
  last_stage_at: string | null;
  stages_run: Record<string, number>;
  conditions_total: number;
  conditions_mapped: number;
  conditions_with_genes: number;
  conditions_with_drug_target_overlap: number;
  swarm_runs: number;
  latest_metrics: Record<string, unknown>;
}

export interface SwarmAgentResult {
  role: string;
  status: string;
  score_treats: number | null;
  score_causes: number | null;
  confidence: number | null;
  verdict: string | null;
  rationale: string | null;
  key_evidence: string[];
  elapsed_ms: number;
  error: string | null;
}

export interface SwarmConsensus {
  mean_treats: number;
  mean_causes: number;
  stdev_treats: number;
  stdev_causes: number;
  n_agents: number;
  direction: string;
}

export interface SwarmRun {
  id: string;
  started_at: string;
  finished_at: string;
  ingredient_concept_id: number;
  ingredient_name: string;
  condition_concept_id: number;
  condition_name: string;
  context: Record<string, unknown>;
  agents: SwarmAgentResult[];
  consensus: SwarmConsensus;
}

export interface SwarmLoopStatus {
  running: boolean;
  iterations: number;
  started_at: string | null;
  last_pair: {
    ingredient_concept_id: number;
    condition_concept_id: number;
    ingredient_name: string;
    condition_name: string;
    direction: string;
  } | null;
  last_finished_at: string | null;
  last_error: string | null;
  interval_seconds: number;
}

export const api = {
  ingredients: (q?: string, limit = 50) =>
    jsonReq<Ingredient[]>(
      `/api/ingredients?limit=${limit}${q ? `&q=${encodeURIComponent(q)}` : ""}`,
    ),
  conditions: (q?: string, limit = 50) =>
    jsonReq<Condition[]>(
      `/api/conditions?limit=${limit}${q ? `&q=${encodeURIComponent(q)}` : ""}`,
    ),
  predict: (ingredient_concept_id: number, condition_concept_id: number) =>
    jsonReq<PredictResponse>(
      `/api/predict?ingredient_concept_id=${ingredient_concept_id}&condition_concept_id=${condition_concept_id}`,
    ),
  drugSummary: (ingredient_concept_id: number, top = 20) =>
    jsonReq<DrugSummaryResponse>(
      `/api/compute/drug-summary?ingredient_concept_id=${ingredient_concept_id}&top=${top}`,
      { method: "POST" },
    ),
  listTables: () => jsonReq<TableMeta[]>(`/api/tables`),
  readTable: (name: string, offset: number, limit: number) =>
    jsonReq<TablePage>(`/api/tables/${name}?offset=${offset}&limit=${limit}`),
  commitTable: (name: string, body: CommitBody) =>
    jsonReq<CommitResult>(`/api/tables/${name}/commit`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  workflowRuns: (limit = 50) => jsonReq<WorkflowRun[]>(`/api/workflow/runs?limit=${limit}`),
  workflowKPIs: () => jsonReq<WorkflowKPIs>(`/api/workflow/kpis`),
  triggerStage: (stage: string, notes?: string) =>
    jsonReq<WorkflowRun>(`/api/workflow/trigger`, {
      method: "POST",
      body: JSON.stringify({ stage, notes: notes ?? null }),
    }),
  runSwarm: (ingredient_concept_id: number, condition_concept_id: number) =>
    jsonReq<SwarmRun>(`/api/swarm/run`, {
      method: "POST",
      body: JSON.stringify({ ingredient_concept_id, condition_concept_id }),
    }),
  swarmRuns: (limit = 20) => jsonReq<SwarmRun[]>(`/api/swarm/runs?limit=${limit}`),
  swarmLoopStatus: () => jsonReq<SwarmLoopStatus>(`/api/swarm/loop/status`),
  swarmLoopStart: (interval_seconds = 3) =>
    jsonReq<SwarmLoopStatus>(
      `/api/swarm/loop/start?interval_seconds=${interval_seconds}`,
      { method: "POST" },
    ),
  swarmLoopStop: () =>
    jsonReq<SwarmLoopStatus>(`/api/swarm/loop/stop`, { method: "POST" }),
};
