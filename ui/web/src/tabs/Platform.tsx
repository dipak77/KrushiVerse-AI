import { useEffect, useState } from "react";
import { useDashboard } from "../context/DashboardContext";
import { api } from "../lib/api";
import { BarRows, Card, Reveal, SectionHead, StatusChip, Toggle } from "../lib/ui";

/* ================= RAG EXPLORER ================= */

export function RagExplorer() {
  const [q, setQ] = useState("Cotton pink bollworm organic control Maharashtra market price");
  const [cropHint, setCropHint] = useState("Cotton");
  const [topK, setTopK] = useState(8);
  const [forceWeb, setForceWeb] = useState(true);
  const [tools, setTools] = useState(true);
  const [running, setRunning] = useState(false);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setErr(null);
    try {
      const r = await api.rag({
        query: q,
        crop: cropHint,
        top_k: topK,
        enable_web: forceWeb,
        enable_tools: tools,
      });
      setData(r);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setRunning(false);
    }
  };

  const docs = data?.docs || [];
  const metrics = data?.metrics || {};
  const backends = data?.backends || [];
  const originTone: Record<string, string> = {
    KB: "chip-leaf",
    Graph: "chip-water",
    Agmarknet: "chip-gold",
    Web: "chip-marigold",
    Memory: "chip-soil",
    local_hybrid: "chip-leaf",
    GraphRAG: "chip-water",
  };

  return (
    <div>
      <SectionHead
        eyebrow="Advanced multi-source RAG · API"
        title={
          <>
            Retrieval <span className="text-[var(--leaf)]">Explorer</span>
          </>
        }
        sub="POST /api/ui/rag → advanced_rag multi-source fusion."
        right={<StatusChip tone="leaf">{data?.source || "backend"}</StatusChip>}
      />

      <Card ticks className="p-4">
        <span className="tick-b" />
        <div className="grid grid-cols-1 md:grid-cols-[1fr_150px_150px] gap-3">
          <div>
            <label className="text-[10.5px] font-mono2 text-[var(--dim)] block mb-1.5">RAG query</label>
            <input className="field" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <div>
            <label className="text-[10.5px] font-mono2 text-[var(--dim)] block mb-1.5">Crop hint</label>
            <input className="field" value={cropHint} onChange={(e) => setCropHint(e.target.value)} />
          </div>
          <div>
            <label className="text-[10.5px] font-mono2 text-[var(--dim)] block mb-1.5">Top-K · {topK}</label>
            <input type="range" min={3} max={12} value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full accent-[var(--gold)] mt-2.5" />
          </div>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-3 mt-3.5">
          <div className="flex items-center gap-5">
            <Toggle checked={forceWeb} onChange={setForceWeb} label="Enable web" />
            <Toggle checked={tools} onChange={setTools} label="Enable tools" />
          </div>
          <button className="btn btn-leaf" onClick={run} disabled={running || !q.trim()}>
            {running ? "Retrieving…" : "Run Advanced RAG (API)"}
          </button>
        </div>
      </Card>

      {err && <div className="mt-3 text-[12px] font-mono2 text-[var(--chili)]">{err}</div>}

      {data && (
        <>
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mt-4">
            {[
              { l: "Local hits", v: metrics.local, c: "var(--leaf)" },
              { l: "Fused docs", v: metrics.fused, c: "var(--gold)" },
              { l: "Web results", v: metrics.web, c: "var(--marigold)" },
              { l: "Tools used", v: metrics.tools, c: "var(--water)" },
            ].map((m) => (
              <Card key={m.l} hover className="p-3.5 text-center">
                <div className="font-display font-bold text-[26px]" style={{ color: m.c }}>
                  {m.v}
                </div>
                <div className="text-[10.5px] font-mono2 text-[var(--dim)] mt-1.5 uppercase">{m.l}</div>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-[300px_1fr] gap-4 mt-4">
            <Card className="p-4">
              <div className="eyebrow mb-3">Backends</div>
              {backends.map((b: any) => (
                <div key={b.name} className="rounded-[9px] border border-[var(--line)] px-3 py-2 mb-2">
                  <div className="flex justify-between text-[12px]">
                    <span className="font-semibold">{b.name}</span>
                    <span className="font-mono2 text-[var(--dim)]">{b.ms} ms</span>
                  </div>
                  <div className="text-[10.5px] font-mono2 text-[var(--dim)]">{b.docs} docs</div>
                </div>
              ))}
            </Card>
            <div className="space-y-3">
              {docs.map((d: any, i: number) => (
                <Card key={i} hover className="p-4">
                  <div className="flex justify-between gap-3">
                    <div>
                      <div className="font-display font-bold text-[14.5px]">{d.title}</div>
                      <div className="text-[11px] font-mono2 text-[var(--dim)] mt-1">
                        <span className={`chip ${originTone[d.origin] || "chip-leaf"} !text-[9px]`}>{d.origin}</span> {d.source}
                      </div>
                    </div>
                    <div className="font-display font-bold text-[18px] text-[var(--gold)]">{Number(d.score).toFixed(2)}</div>
                  </div>
                  <p className="text-[12.5px] text-[var(--mut)] mt-2">{d.snippet}</p>
                </Card>
              ))}
              {!docs.length && <div className="text-[var(--dim)] p-4">No fused docs returned.</div>}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/* ================= TAXONOMY ================= */

export function Taxonomy() {
  const [tax, setTax] = useState<any>(null);
  const [demo, setDemo] = useState("कापूस खत किती द्यावे Pune");
  const [resolved, setResolved] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setTax(await api.taxonomy());
      } catch (e: any) {
        setErr(e?.message || String(e));
      }
    })();
  }, []);

  const resolve = async () => {
    try {
      // use taxonomy resolve endpoint if available
      const r = await fetch(`/api/taxonomy/resolve?text=${encodeURIComponent(demo)}`).then((x) => x.json());
      setResolved(r);
    } catch (e: any) {
      setResolved({ error: e?.message || String(e) });
    }
  };

  const crops = tax?.crops || [];
  const categories = tax?.categories || [];
  const stages = tax?.stages || [];

  return (
    <div>
      <SectionHead
        eyebrow="Domain taxonomy · API"
        title={
          <>
            Taxonomy <span className="text-[var(--marigold)]">Browser</span>
          </>
        }
        sub="GET /api/ui/taxonomy (mini taxonomy + KB). Resolve via /api/taxonomy/resolve."
        right={<StatusChip tone="gold">v{tax?.version || "—"}</StatusChip>}
      />
      {err && <div className="text-[12px] font-mono2 text-[var(--chili)] mb-3">{err}</div>}

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mb-4">
        {[
          { l: "Version", v: tax?.version || "—", c: "var(--gold)" },
          { l: "Crops", v: crops.length, c: "var(--leaf)" },
          { l: "Categories", v: categories.length, c: "var(--water)" },
          { l: "Districts", v: (tax?.districts || []).length, c: "var(--marigold)" },
        ].map((m) => (
          <Card key={m.l} hover className="p-4 text-center">
            <div className="font-display font-bold text-[26px]" style={{ color: m.c }}>
              {m.v}
            </div>
            <div className="text-[10px] font-mono2 text-[var(--dim)] mt-1 uppercase">{m.l}</div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card ticks className="p-4 overflow-x-auto">
          <span className="tick-b" />
          <div className="eyebrow mb-3">Crops · API</div>
          <table className="dtable min-w-[420px]">
            <thead>
              <tr>
                <th>EN</th>
                <th>MR</th>
                <th>HI</th>
                <th>Group</th>
              </tr>
            </thead>
            <tbody>
              {crops.map((c: any) => (
                <tr key={c.en}>
                  <td className="font-semibold !text-[var(--ink)]">{c.en}</td>
                  <td className="devnagari">{c.mr}</td>
                  <td className="devnagari">{c.hi}</td>
                  <td>{c.group}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <div className="space-y-4">
          <Card className="p-4">
            <div className="eyebrow mb-3">Categories · API</div>
            <div className="grid grid-cols-2 gap-2">
              {categories.map((c: any) => (
                <div key={c.code} className="rounded-[9px] border border-[var(--line)] px-3 py-2">
                  <div className="text-[12.5px] font-semibold">{c.name}</div>
                  <div className="text-[11px] devnagari text-[var(--dim)]">{c.mr}</div>
                </div>
              ))}
            </div>
          </Card>
          <Card className="p-4">
            <div className="eyebrow mb-2">Resolve · API</div>
            <input className="field devnagari" value={demo} onChange={(e) => setDemo(e.target.value)} />
            <button className="btn btn-ghost mt-2" onClick={resolve}>
              Resolve text
            </button>
            {resolved && (
              <pre className="mt-2 text-[11px] font-mono2 text-[var(--mut)] overflow-x-auto">{JSON.stringify(resolved, null, 2)}</pre>
            )}
          </Card>
        </div>
      </div>

      {!!stages.length && (
        <Card className="p-4 mt-4">
          <div className="eyebrow mb-3">Stages · API</div>
          <div className="flex flex-wrap gap-2">
            {stages.map((s: any) => (
              <span key={s.code || s.en} className="chip">
                {s.en} · {s.mr}
              </span>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

/* ================= DATA FACTORY ================= */

export function DataFactory() {
  const [factory, setFactory] = useState<any>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    try {
      setFactory(await api.factory());
      setErr(null);
    } catch (e: any) {
      setErr(e?.message || String(e));
    }
  };

  useEffect(() => {
    load();
  }, []);

  const runWorker = async (id: string) => {
    setRunning(id);
    try {
      if (id === "W-ANALYZE") {
        await fetch("/api/lake/analyze?execute=false", { method: "POST" });
      }
      await load();
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setRunning(null);
    }
  };

  const workers = factory?.workers || [];
  const langs = factory?.langs || {};
  const cats = factory?.cats || {};
  const gaps = factory?.gaps || [];

  return (
    <div>
      <SectionHead
        eyebrow="Mini data factory · API"
        title={
          <>
            Data <span className="text-[var(--water)]">Factory</span>
          </>
        }
        sub="GET /api/ui/factory reads lake/tokenizer/pretrain reports when present."
        right={<StatusChip tone="water">{factory?.records ?? 0} records</StatusChip>}
      />
      {err && <div className="mb-3 text-[12px] font-mono2 text-[var(--chili)]">{err}</div>}
      {factory?.note && <div className="mb-3 text-[12px] font-mono2 text-[var(--marigold)]">{factory.note}</div>}

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mb-4">
        {[
          { l: "Records", v: factory?.records ?? "—", c: "var(--leaf)" },
          { l: "Dup %", v: factory?.dupPct ?? "—", c: "var(--gold)" },
          { l: "Gaps", v: gaps.length, c: "var(--chili)" },
          { l: "Reports", v: Object.keys(factory?.reports || {}).length, c: "var(--water)" },
        ].map((m) => (
          <Card key={m.l} hover className="p-3.5 text-center">
            <div className="font-display font-bold text-[24px]" style={{ color: m.c }}>
              {m.v}
            </div>
            <div className="text-[10px] font-mono2 text-[var(--dim)] mt-1 uppercase">{m.l}</div>
          </Card>
        ))}
      </div>

      <Card ticks className="p-4 mb-4">
        <span className="tick-b" />
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
          {workers.map((w: any) => (
            <div key={w.id} className="rounded-[11px] border border-[var(--line)] p-3">
              <div className="font-mono2 text-[11px] text-[var(--gold)]">{w.id}</div>
              <div className="text-[11.5px] text-[var(--mut)] mt-1">{w.desc}</div>
              <button className="btn btn-ghost !py-1 !px-2 !text-[10px] mt-2" disabled={!!running} onClick={() => runWorker(w.id)}>
                {running === w.id ? "…" : "refresh"}
              </button>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card className="p-4">
          <div className="eyebrow mb-3">Language mix · API</div>
          <BarRows
            rows={Object.entries(langs).map(([k, v]) => ({ label: k, value: Number(v) || 0 }))}
          />
        </Card>
        <Card className="p-4">
          <div className="eyebrow mb-3">Categories · API</div>
          <BarRows rows={Object.entries(cats).map(([k, v]) => ({ label: k, value: Number(v) || 0 }))} />
        </Card>
      </div>

      {!!gaps.length && (
        <Card className="p-4 mt-4">
          <div className="eyebrow mb-2">Gaps · API</div>
          {gaps.map((g: any, i: number) => (
            <div key={i} className="text-[12.5px] text-[var(--mut)] border-b border-[var(--line)] py-2">
              <b className="text-[var(--ink)]">{g.crop}</b> / {g.category}: {g.gap}
            </div>
          ))}
        </Card>
      )}

      <button className="btn btn-ghost mt-4" onClick={load}>
        Reload factory status
      </button>
    </div>
  );
}
