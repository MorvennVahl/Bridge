import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type TableMeta, type TablePage } from "../api";

type Edits = Record<number, Record<string, string>>;
type Adds = Record<string, string>[];
type TableCache = { edits: Edits; adds: Adds };

function cacheKey(name: string): string {
  return `bridge:tableCache:${name}`;
}

function loadCache(name: string): TableCache {
  try {
    const raw = localStorage.getItem(cacheKey(name));
    if (raw) return JSON.parse(raw) as TableCache;
  } catch {
    /* ignore */
  }
  return { edits: {}, adds: [] };
}

function saveCache(name: string, cache: TableCache) {
  localStorage.setItem(cacheKey(name), JSON.stringify(cache));
}

function clearCache(name: string) {
  localStorage.removeItem(cacheKey(name));
}

export default function DataPage() {
  const [tables, setTables] = useState<TableMeta[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [page, setPage] = useState<TablePage | null>(null);
  const [offset, setOffset] = useState(0);
  const [limit] = useState(50);
  const [cache, setCache] = useState<TableCache>({ edits: {}, adds: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.listTables()
      .then((list) => {
        setTables(list);
        if (!selected && list.length) setSelected(list[0].name);
      })
      .catch((e) => setError(String(e)));
  }, [selected]);

  useEffect(() => {
    if (!selected) return;
    setCache(loadCache(selected));
    setOffset(0);
  }, [selected]);

  const loadPage = useCallback(async () => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    try {
      const p = await api.readTable(selected, offset, limit);
      setPage(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [selected, offset, limit]);

  useEffect(() => {
    loadPage();
  }, [loadPage]);

  const pendingCount = useMemo(
    () =>
      Object.keys(cache.edits).length +
      cache.adds.length +
      Object.values(cache.edits).reduce((n, v) => n + Object.keys(v).length - 1, 0),
    [cache],
  );

  const dirtyCells =
    Object.keys(cache.edits).length +
    Object.values(cache.edits).reduce((n, v) => n + Math.max(0, Object.keys(v).length - 1), 0);

  function updateCache(next: TableCache) {
    setCache(next);
    if (selected) saveCache(selected, next);
  }

  function editCell(index: number, col: string, value: string) {
    const next: TableCache = {
      edits: { ...cache.edits, [index]: { ...(cache.edits[index] ?? {}), [col]: value } },
      adds: cache.adds,
    };
    updateCache(next);
  }

  function editAdded(row: number, col: string, value: string) {
    const nextAdds = cache.adds.slice();
    nextAdds[row] = { ...(nextAdds[row] ?? {}), [col]: value };
    updateCache({ edits: cache.edits, adds: nextAdds });
  }

  function newRow() {
    const empty: Record<string, string> = {};
    (page?.columns ?? []).forEach((c) => (empty[c] = ""));
    updateCache({ edits: cache.edits, adds: [...cache.adds, empty] });
  }

  function discard() {
    if (!selected) return;
    if (!confirm(`Discard ${pendingCount} pending change(s)?`)) return;
    clearCache(selected);
    setCache({ edits: {}, adds: [] });
  }

  async function saveAll() {
    if (!selected) return;
    setSaving(true);
    setError(null);
    try {
      const edits = Object.entries(cache.edits).map(([idx, values]) => ({
        index: Number(idx),
        values,
      }));
      const adds = cache.adds.map((values) => ({ values }));
      await api.commitTable(selected, { edits, adds });
      clearCache(selected);
      setCache({ edits: {}, adds: [] });
      await loadPage();
      // refresh table metadata (row counts)
      setTables(await api.listTables());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  const meta = tables.find((t) => t.name === selected);
  const isReadOnly = meta ? !meta.editable : false;
  const totalPages = page ? Math.max(1, Math.ceil(page.total / limit)) : 1;
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <div className="mx-auto max-w-[1400px] space-y-4 p-6">
      <header className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Data</h2>
          <p className="mt-1 text-sm text-slate-500">
            Edit any cell inline. Add rows with <span className="font-mono">+ Add row</span>.
            Pending changes are cached in your browser — click <b>Save all</b> to persist to disk.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex h-8 min-w-[96px] items-center justify-center rounded-md px-3 text-xs font-medium ${
              pendingCount > 0
                ? "bg-amber-100 text-amber-800"
                : "bg-slate-100 text-slate-400"
            }`}
          >
            {pendingCount} pending
          </span>
          <button
            onClick={discard}
            disabled={pendingCount === 0}
            className="inline-flex h-8 min-w-[96px] items-center justify-center rounded-md border border-slate-300 bg-white px-3 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-40"
          >
            Discard
          </button>
          <button
            onClick={saveAll}
            disabled={pendingCount === 0 || saving || isReadOnly}
            title={isReadOnly ? "This table is a source of truth — read only" : ""}
            className="inline-flex h-8 min-w-[96px] items-center justify-center rounded-md bg-emerald-600 px-3 text-xs font-medium text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-emerald-600/40"
          >
            {saving ? "Saving…" : "Save all"}
          </button>
        </div>
      </header>

      <section className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <label className="text-xs font-medium text-slate-600">Table:</label>
        <select
          value={selected ?? ""}
          onChange={(e) => setSelected(e.target.value || null)}
          className="min-w-[380px] rounded-md border border-slate-300 px-3 py-1.5 text-sm"
        >
          {tables.map((t) => {
            const label = t.name.replace(/__/g, " / ");
            return (
              <option key={t.name} value={t.name}>
                {label}  ·  {t.row_count.toLocaleString()} rows  ·  {t.fmt}
              </option>
            );
          })}
        </select>
        {isReadOnly && (
          <span className="rounded-full border border-rose-300 bg-rose-50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-rose-700">
            source · read only
          </span>
        )}
        {meta?.warn && (
          <span className="text-xs text-amber-700">⚠ {meta.warn}</span>
        )}
        <div className="ml-auto text-xs text-slate-500">
          {meta && (
            <>
              <span className="font-mono">{meta.path}</span> · {meta.columns.length} cols
            </>
          )}
        </div>
      </section>

      {error && (
        <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
          {error}
        </div>
      )}

      <section className="overflow-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        {loading && !page ? (
          <div className="p-8 text-center text-sm text-slate-400">loading…</div>
        ) : page ? (
          <table className="min-w-full text-sm">
            <thead className="sticky top-0 z-10 bg-slate-50">
              <tr>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wide text-slate-500">
                  #
                </th>
                {page.columns.map((c) => (
                  <th
                    key={c}
                    className="border-b border-slate-200 px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wide text-slate-500"
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {page.rows.map((row) => {
                const idx = row.__index__;
                const rowEdits = cache.edits[idx] ?? {};
                return (
                  <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50/50">
                    <td className="px-2 py-1 font-mono text-[10px] text-slate-400">{idx}</td>
                    {page.columns.map((c) => (
                      <EditableCell
                        key={c}
                        value={String(row[c] ?? "")}
                        edited={c in rowEdits}
                        editedValue={rowEdits[c] ?? ""}
                        onCommit={(v) => editCell(idx, c, v)}
                      />
                    ))}
                  </tr>
                );
              })}
              {cache.adds.map((row, i) => (
                <tr key={`add-${i}`} className="border-b border-emerald-200 bg-emerald-50/60">
                  <td className="px-2 py-1 text-[10px] font-medium text-emerald-700">NEW</td>
                  {page.columns.map((c) => (
                    <EditableCell
                      key={c}
                      value={row[c] ?? ""}
                      edited={false}
                      editedValue=""
                      onCommit={(v) => editAdded(i, c, v)}
                    />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="p-8 text-center text-sm text-slate-400">no data</div>
        )}
      </section>

      <footer className="flex items-center justify-between gap-3">
        <button
          onClick={newRow}
          disabled={!page || isReadOnly}
          title={isReadOnly ? "Read-only table" : ""}
          className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 disabled:opacity-40"
        >
          + Add row
        </button>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <button
            onClick={() => setOffset(Math.max(0, offset - limit))}
            disabled={offset === 0}
            className="rounded border border-slate-300 bg-white px-2 py-1 disabled:opacity-40"
          >
            ← prev
          </button>
          <span>
            page {currentPage} / {totalPages}
          </span>
          <button
            onClick={() => setOffset(offset + limit)}
            disabled={!page || offset + limit >= page.total}
            className="rounded border border-slate-300 bg-white px-2 py-1 disabled:opacity-40"
          >
            next →
          </button>
        </div>
        <div className="text-xs text-slate-400">
          {dirtyCells > 0 && `${dirtyCells} rows edited · `}
          {cache.adds.length > 0 && `${cache.adds.length} rows new`}
        </div>
      </footer>
    </div>
  );
}

function EditableCell({
  value,
  edited,
  editedValue,
  onCommit,
}: {
  value: string;
  edited: boolean;
  editedValue: string;
  onCommit: (v: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(edited ? editedValue : value);

  useEffect(() => {
    setDraft(edited ? editedValue : value);
  }, [value, edited, editedValue]);

  function commit() {
    if (draft !== (edited ? editedValue : value)) onCommit(draft);
    setEditing(false);
  }

  if (editing) {
    return (
      <td className="px-1 py-0.5">
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            else if (e.key === "Escape") setEditing(false);
          }}
          className="w-full rounded border border-slate-400 bg-white px-2 py-1 text-xs font-mono"
        />
      </td>
    );
  }

  const shown = edited ? editedValue : value;
  return (
    <td
      onClick={() => setEditing(true)}
      className={`cursor-text truncate px-2 py-1 font-mono text-xs ${
        edited ? "bg-amber-50 text-amber-900" : "text-slate-700"
      }`}
      style={{ maxWidth: 240 }}
      title={shown}
    >
      {edited && <span className="mr-1 text-amber-500">●</span>}
      {shown || <span className="text-slate-300">—</span>}
    </td>
  );
}
