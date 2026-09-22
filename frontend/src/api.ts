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
};
