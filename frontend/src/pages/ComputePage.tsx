import { useEffect, useState } from "react";
import { api, type Ingredient, type DrugSummaryResponse } from "../api";

export default function ComputePage() {
  const [query, setQuery] = useState("");
  const [drugs, setDrugs] = useState<Ingredient[]>([]);
  const [selected, setSelected] = useState<Ingredient | null>(null);
  const [result, setResult] = useState<DrugSummaryResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      api.ingredients(query || undefined, 50).then(setDrugs).catch(() => setDrugs([]));
    }, 200);
    return () => clearTimeout(t);
  }, [query]);

  async function run() {
    if (!selected) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.drugSummary(selected.ingredient_concept_id);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <header>
        <h2 className="text-lg font-semibold text-slate-900">Compute — drug evidence summary</h2>
        <p className="mt-1 text-sm text-slate-500">
          Fan out over the 1.45M CEM associations on Modal and return per-condition evidence
          counts for the selected drug. Real Modal function, real data — if Modal is not
          reachable, this endpoint fails loudly.
        </p>
      </header>

      <section className="grid grid-cols-1 gap-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm sm:grid-cols-3">
        <div className="sm:col-span-2">
          <label className="mb-1 block text-xs font-medium text-slate-600">Drug (ingredient)</label>
          <input
            className="mb-2 w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500"
            placeholder="search ingredients…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="max-h-56 overflow-auto rounded-md border border-slate-200">
            {drugs.length === 0 ? (
              <div className="p-3 text-xs text-slate-400">no results</div>
            ) : (
              drugs.map((d) => (
                <button
                  key={d.ingredient_concept_id}
                  onClick={() => setSelected(d)}
                  className={`block w-full truncate px-3 py-1.5 text-left text-sm hover:bg-slate-50 ${
                    selected?.ingredient_concept_id === d.ingredient_concept_id
                      ? "bg-slate-100 font-medium"
                      : ""
                  }`}
                >
                  {d.ingredient_name}
                </button>
              ))
            )}
          </div>
        </div>
        <div className="flex flex-col justify-end">
          <button
            disabled={!selected || running}
            onClick={run}
            className="rounded-md bg-slate-900 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {running ? "Running on Modal…" : "Compute summary"}
          </button>
          {selected && (
            <div className="mt-2 truncate text-xs text-slate-500">
              selected: {selected.ingredient_name}
            </div>
          )}
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        {error && (
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
            {error}
          </div>
        )}
        {!result && !error && (
          <div className="flex h-40 items-center justify-center text-sm text-slate-400">
            pick a drug and click Compute
          </div>
        )}
        {result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat label="Rows scanned" value={result.rows_scanned.toLocaleString()} />
              <Stat label="Conditions" value={result.conditions.toString()} />
              <Stat label="FAERS pairs" value={result.faers_pairs.toLocaleString()} />
              <Stat label="SemMedDB pairs" value={result.semmeddb_pairs.toLocaleString()} />
            </div>
            <div className="text-xs text-slate-500">
              executed via <span className="font-mono">{result.executor}</span> in {result.elapsed_ms} ms
            </div>
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-700">
                Top {result.top.length} conditions by combined evidence
              </h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase text-slate-500">
                    <th className="py-1">Condition</th>
                    <th className="py-1 text-right">FAERS PRR</th>
                    <th className="py-1 text-right">SemMedDB</th>
                  </tr>
                </thead>
                <tbody>
                  {result.top.map((r) => (
                    <tr key={r.condition_concept_id} className="border-t border-slate-100">
                      <td className="py-1.5">{r.condition_name}</td>
                      <td className="py-1.5 text-right font-mono text-xs">
                        {r.faers_prr !== null ? r.faers_prr.toFixed(2) : "—"}
                      </td>
                      <td className="py-1.5 text-right font-mono text-xs">
                        {r.semmeddb_relationships ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-lg font-semibold text-slate-900">{value}</div>
    </div>
  );
}
