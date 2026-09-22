import { useEffect, useMemo, useState } from "react";
import { api, type Condition, type Ingredient, type PredictResponse } from "../api";
import { useDebounced } from "../hooks/useDebounced";

function ScoreBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | null;
  tone: "good" | "bad";
}) {
  const pct = value !== null ? Math.round(value * 100) : null;
  const bg = tone === "good" ? "bg-emerald-500" : "bg-rose-500";
  return (
    <div>
      <div className="flex justify-between text-xs font-medium text-slate-600">
        <span>{label}</span>
        <span>{pct !== null ? `${pct}%` : "—"}</span>
      </div>
      <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200">
        {pct !== null && <div className={`h-full ${bg}`} style={{ width: `${pct}%` }} />}
      </div>
    </div>
  );
}

export default function PredictPage() {
  const [drugQuery, setDrugQuery] = useState("");
  const [conditionQuery, setConditionQuery] = useState("");
  const debouncedDrug = useDebounced(drugQuery);
  const debouncedCondition = useDebounced(conditionQuery);

  const [drugs, setDrugs] = useState<Ingredient[]>([]);
  const [conditions, setConditions] = useState<Condition[]>([]);
  const [selectedDrug, setSelectedDrug] = useState<Ingredient | null>(null);
  const [selectedCondition, setSelectedCondition] = useState<Condition | null>(null);

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.ingredients(debouncedDrug || undefined, 50)
      .then(setDrugs)
      .catch(() => setDrugs([]));
  }, [debouncedDrug]);

  useEffect(() => {
    api.conditions(debouncedCondition || undefined, 50)
      .then(setConditions)
      .catch(() => setConditions([]));
  }, [debouncedCondition]);

  const canPredict = useMemo(
    () => selectedDrug !== null && selectedCondition !== null,
    [selectedDrug, selectedCondition],
  );

  async function runPredict() {
    if (!selectedDrug || !selectedCondition) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const p = await api.predict(
        selectedDrug.ingredient_concept_id,
        selectedCondition.condition_concept_id,
      );
      setPrediction(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 p-6 lg:grid-cols-3">
      <section className="space-y-5 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Drug (ingredient)</label>
          <input
            className="mb-2 w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500"
            placeholder="search ingredients…"
            value={drugQuery}
            onChange={(e) => setDrugQuery(e.target.value)}
          />
          <div className="max-h-48 overflow-auto rounded-md border border-slate-200">
            {drugs.length === 0 ? (
              <div className="p-3 text-xs text-slate-400">no results</div>
            ) : (
              drugs.map((d) => (
                <button
                  key={d.ingredient_concept_id}
                  onClick={() => setSelectedDrug(d)}
                  className={`block w-full truncate px-3 py-1.5 text-left text-sm hover:bg-slate-50 ${
                    selectedDrug?.ingredient_concept_id === d.ingredient_concept_id
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

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Condition</label>
          <input
            className="mb-2 w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500"
            placeholder="search conditions…"
            value={conditionQuery}
            onChange={(e) => setConditionQuery(e.target.value)}
          />
          <div className="max-h-48 overflow-auto rounded-md border border-slate-200">
            {conditions.length === 0 ? (
              <div className="p-3 text-xs text-slate-400">no results</div>
            ) : (
              conditions.map((c) => (
                <button
                  key={c.condition_concept_id}
                  onClick={() => setSelectedCondition(c)}
                  className={`block w-full truncate px-3 py-1.5 text-left text-sm hover:bg-slate-50 ${
                    selectedCondition?.condition_concept_id === c.condition_concept_id
                      ? "bg-slate-100 font-medium"
                      : ""
                  }`}
                >
                  {c.condition_name}
                </button>
              ))
            )}
          </div>
        </div>

        <button
          disabled={!canPredict || loading}
          onClick={runPredict}
          className="w-full rounded-md bg-slate-900 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? "Predicting…" : "Predict"}
        </button>
      </section>

      <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
        {error && (
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
            {error}
          </div>
        )}
        {!prediction && !error && (
          <div className="flex h-64 items-center justify-center text-sm text-slate-400">
            pick a drug and a condition, then click Predict
          </div>
        )}
        {prediction && (
          <div className="space-y-5">
            <div>
              <div className="text-xs uppercase tracking-wide text-slate-500">
                {prediction.model ?? prediction.status}
              </div>
              <div className="mt-1 text-lg font-semibold">
                {prediction.ingredient_name}
                <span className="mx-2 text-slate-400">→</span>
                {prediction.condition_name}
              </div>
            </div>

            {prediction.status !== "ready" && (
              <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                No trained model yet — showing evidence only.{" "}
                <span className="font-mono text-xs">{prediction.status}</span>
              </div>
            )}

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <ScoreBar label="Treats" value={prediction.treats_score} tone="good" />
              <ScoreBar label="Causes" value={prediction.causes_score} tone="bad" />
            </div>
            {prediction.confidence !== null && (
              <div className="text-xs text-slate-500">
                confidence {(prediction.confidence * 100).toFixed(0)}%
              </div>
            )}

            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-700">Evidence</h3>
              {prediction.evidence.length === 0 ? (
                <div className="text-sm text-slate-400">no direct evidence rows found</div>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase text-slate-500">
                      <th className="py-1">Source</th>
                      <th className="py-1">Relationship</th>
                      <th className="py-1 text-right">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prediction.evidence.map((e, i) => (
                      <tr key={i} className="border-t border-slate-100">
                        <td className="py-1.5 font-mono text-xs">{e.source}</td>
                        <td className="py-1.5">{e.relationship ?? "—"}</td>
                        <td className="py-1.5 text-right font-mono text-xs">
                          {e.value !== null ? e.value.toFixed(2) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
