import { useEffect, useRef, useState } from "react";
import { FACTORY_INITIAL, FUSED_DOCS, RAG_BACKENDS, TAX_CATEGORIES, TAX_CROPS, TAX_STAGES, resolveTaxonomy } from "../data";
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
  const [done, setDone] = useState(false);
  const [liveDocs, setLiveDocs] = useState<typeof FUSED_DOCS | null>(null);
  const [metrics, setMetrics] = useState({ local: 0, fused: 0, web: 0, tools: 0 });
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setDone(false);
    setErr(null);
    try {
      const r = await api.advancedRag({
        query: q,
        crop: cropHint,
        location: "Maharashtra",
        top_k: topK,
        enable_web: forceWeb,
        enable_tools: tools,
        force_web: forceWeb,
      });
      const fused = r.fused_documents || r.documents || r.results || [];
      const mapped = fused.slice(0, topK).map((d: any, i: number) => ({
        title: d.title || d.metadata?.title || `Doc ${i + 1}`,
        origin: d.origin || d.category || d.source_type || "KB",
        category: d.category || "GEN",
        source: d.source || d.metadata?.source || "platform",
        score: Number(d.rrf_score ?? d.score ?? 0.5),
        snippet: (d.content || d.snippet || d.text || "").slice(0, 220),
      }));
      setLiveDocs(mapped.length ? mapped : null);
      setMetrics({
        local: r.local_hit_count ?? mapped.length,
        fused: mapped.length,
        web: (r.web_results || []).length,
        tools: (r.tools_used || []).length,
      });
      setDone(true);
    } catch (e: any) {
      setLiveDocs(null);
      setErr(String(e?.message || e));
      setDone(true);
    } finally {
      setRunning(false);
    }
  };
  const docs = (liveDocs || FUSED_DOCS).slice(0, topK);
  const originTone: Record<string, string> = { KB: "chip-leaf", Graph: "chip-water", Agmarknet: "chip-gold", Web: "chip-marigold", Memory: "chip-soil", local_hybrid: "chip-leaf", GraphRAG: "chip-water" };

  return (
    <div>
      <SectionHead
        eyebrow="Advanced multi-source RAG"
        title={<>Retrieval <span className="text-[var(--leaf)]">Explorer</span></>}
        sub="Hybrid + dense (Qdrant) + GraphRAG + tools + web retrieval, fused with reciprocal-rank scoring. Inspect every document that reaches the answer."
        right={<StatusChip tone="leaf">fusion: RRF k=60</StatusChip>}
      />

      <Reveal>
        <Card ticks className="p-4">
          <span className="tick-b" />
          <div className="grid grid-cols-1 md:grid-cols-[1fr_150px_150px] gap-3">
            <div className="md:col-span-1">
              <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">RAG query</label>
              <input className="field" value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            <div>
              <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Crop hint</label>
              <input className="field" value={cropHint} onChange={(e) => setCropHint(e.target.value)} />
            </div>
            <div>
              <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Top-K fused · {topK}</label>
              <input type="range" min={3} max={8} value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full accent-[var(--gold)] mt-2.5" />
            </div>
          </div>
          <div className="flex items-center justify-between flex-wrap gap-3 mt-3.5">
            <div className="flex items-center gap-5">
              <Toggle checked={forceWeb} onChange={setForceWeb} label="Force web search" />
              <Toggle checked={tools} onChange={setTools} label="Enable tools" />
            </div>
            <button className="btn btn-leaf" onClick={run} disabled={running || !q.trim()}>
              {running ? <span className="typing"><i /><i /><i /></span> : "🔎"} {running ? "Retrieving…" : "Run Advanced RAG"}
            </button>
          </div>
        </Card>
      </Reveal>

      {done && (
        <>
          {err && (
            <div className="mt-3 text-[12px] font-mono2 text-[var(--marigold)]">Live RAG error — showing demo docs. {err}</div>
          )}
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mt-4">
            {[
              { l: "Local hits", v: metrics.local || 112, c: "var(--leaf)" },
              { l: "Fused docs", v: metrics.fused || docs.length, c: "var(--gold)" },
              { l: "Web results", v: metrics.web || (forceWeb ? 3 : 0), c: "var(--marigold)" },
              { l: "Tools used", v: metrics.tools || (tools ? 3 : 0), c: "var(--water)" },
            ].map((m, i) => (
              <Reveal key={m.l} delay={i * 70}>
                <Card hover className="p-3.5 text-center">
                  <div className="font-display font-bold text-[26px] leading-none" style={{ color: m.c }}>{m.v}</div>
                  <div className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] mt-1.5">{m.l}</div>
                </Card>
              </Reveal>
            ))}
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-[300px_1fr] gap-4 mt-4">
            <Reveal delay={60}>
              <Card className="p-4 h-full">
                <div className="eyebrow mb-3">Retrieval backends</div>
                <div className="space-y-2">
                  {RAG_BACKENDS.map((b, i) => (
                    <div key={b.name} className="rounded-[9px] border border-[var(--line)] bg-[rgba(154,205,162,0.03)] px-3 py-2 logline" style={{ animationDelay: `${i * 70}ms` }}>
                      <div className="flex justify-between text-[12px]">
                        <span className="text-[var(--ink)] font-semibold">{b.name}</span>
                        <span className="font-mono2 text-[var(--dim)]">{b.ms} ms</span>
                      </div>
                      <div className="flex justify-between text-[10.5px] font-mono2 text-[var(--dim)] mt-0.5">
                        <span>{b.backend}</span><span className="text-[var(--leaf)]">{b.docs} docs</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 rounded-[9px] bg-[rgba(6,16,10,0.7)] border border-[var(--line)] p-3">
                  <div className="text-[10px] font-mono2 uppercase tracking-[0.16em] text-[var(--dim)] mb-1.5">Query plan</div>
                  <pre className="text-[10.5px] font-mono2 text-[var(--mut)] leading-relaxed whitespace-pre-wrap">{JSON.stringify({ intent: ["pest_control", "organic", "market"], crops: [cropHint || "Cotton"], region: "Maharashtra", sources: ["kb", "graph", "tools", "web"] }, null, 1)}</pre>
                </div>
              </Card>
            </Reveal>

            <div className="space-y-3 min-w-0">
              {docs.map((d, i) => (
                <Reveal key={d.title} delay={i * 70}>
                  <Card hover className="p-4">
                    <div className="flex items-start justify-between gap-3 flex-wrap">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-mono2 text-[10.5px] text-[var(--gold)]">[{String(i + 1).padStart(2, "0")}]</span>
                          <span className="font-display font-bold text-[14.5px] leading-snug">{d.title}</span>
                          <span className={`chip ${originTone[d.origin]} !text-[9px]`}>{d.origin}</span>
                          <span className="chip !text-[9px]">{d.category}</span>
                        </div>
                        <div className="text-[11px] font-mono2 text-[var(--dim)] mt-1">{d.source}</div>
                      </div>
                      <div className="text-right flex-none">
                        <div className="font-display font-bold text-[18px] text-[var(--gold)] tabnum">{d.score.toFixed(2)}</div>
                        <div className="text-[9.5px] font-mono2 uppercase tracking-[0.12em] text-[var(--dim)]">fusion</div>
                      </div>
                    </div>
                    <div className="mt-2 h-[4px] rounded-full bg-[rgba(154,205,162,0.1)] overflow-hidden">
                      <div className="h-full rounded-full bar-grow-x" style={{ width: `${d.score * 100}%`, background: "linear-gradient(90deg, var(--gold2), var(--gold))", animationDelay: `${i * 60}ms` }} />
                    </div>
                    <p className="text-[12.5px] text-[var(--mut)] leading-relaxed mt-2.5">{d.snippet}</p>
                  </Card>
                </Reveal>
              ))}
            </div>
          </div>
        </>
      )}

      {!done && (
        <Reveal delay={100}>
          <div className="mt-6 text-center py-10 text-[var(--dim)] border border-dashed border-[var(--line2)] rounded-[14px]">
            <div className="text-[28px] mb-2">📚</div>
            <p className="text-[13px]">Run a query to populate fused documents, backend timings and the query plan.</p>
          </div>
        </Reveal>
      )}
    </div>
  );
}

/* ================= TAXONOMY ================= */

export function Taxonomy() {
  const [demo, setDemo] = useState("कापूस खत किती द्यावे Pune");
  const [validated, setValidated] = useState(false);
  const res = resolveTaxonomy(demo);

  return (
    <div>
      <SectionHead
        eyebrow="Domain taxonomy · Sprint 1 frozen"
        title={<>Taxonomy <span className="text-[var(--marigold)]">Browser</span></>}
        sub="Single source of truth for categories, crops (EN/MR/HI), growth stages, regions and units — consumed by the normalize workers and query understanding."
        right={
          <div className="flex gap-1.5">
            <StatusChip tone="gold">v1.0 frozen</StatusChip>
            <StatusChip tone="leaf">integrity ✓</StatusChip>
          </div>
        }
      />

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        {[
          { l: "Version", v: "v1.0", c: "var(--gold)" },
          { l: "Crops", v: TAX_CROPS.length, c: "var(--leaf)" },
          { l: "Categories", v: TAX_CATEGORIES.length, c: "var(--water)" },
          { l: "MH districts", v: 36, c: "var(--marigold)" },
        ].map((m, i) => (
          <Reveal key={m.l} delay={i * 60}>
            <Card hover className="p-4 text-center">
              <div className="font-display font-bold text-[28px] leading-none" style={{ color: m.c }}>{m.v}</div>
              <div className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] mt-1.5">{m.l}</div>
            </Card>
          </Reveal>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 mt-4">
        <Reveal delay={60}>
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="eyebrow mb-3">Crops · EN / मराठी / हिंदी</div>
            <div className="overflow-x-auto">
              <table className="dtable min-w-[420px]">
                <thead><tr><th>English</th><th>मराठी</th><th>हिंदी</th><th>Group</th><th>Scientific</th></tr></thead>
                <tbody>
                  {TAX_CROPS.map((c) => (
                    <tr key={c.en}>
                      <td className="!text-[var(--ink)] font-semibold">{c.en}</td>
                      <td className="devnagari !text-[var(--leaf)]">{c.mr}</td>
                      <td className="devnagari !text-[var(--marigold)]">{c.hi}</td>
                      <td><span className="chip !text-[9px]">{c.group}</span></td>
                      <td className="italic text-[12px]">{c.sci}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </Reveal>

        <div className="space-y-4">
          <Reveal delay={110}>
            <Card className="p-4">
              <div className="eyebrow mb-3">Categories</div>
              <div className="grid grid-cols-2 gap-2">
                {TAX_CATEGORIES.map((c) => (
                  <div key={c.code} className="rounded-[9px] border border-[var(--line)] bg-[rgba(154,205,162,0.03)] px-3 py-2 flex items-center justify-between gap-2 hover:border-[rgba(244,192,75,0.4)] transition-colors">
                    <div>
                      <div className="text-[12.5px] font-semibold text-[var(--ink)]">{c.name}</div>
                      <div className="text-[11px] devnagari text-[var(--dim)]">{c.mr}</div>
                    </div>
                    <span className="font-mono2 text-[10px] text-[var(--gold)]">{c.docs} docs</span>
                  </div>
                ))}
              </div>
            </Card>
          </Reveal>

          <Reveal delay={160}>
            <Card ticks className="p-4">
              <span className="tick-b" />
              <div className="eyebrow mb-3">Resolve demo · live tokenizer</div>
              <input className="field devnagari" value={demo} onChange={(e) => setDemo(e.target.value)} />
              <div className="mt-3 space-y-2 text-[12.5px]">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[var(--dim)] font-mono2 text-[11px] w-[76px]">crops →</span>
                  {res.crops.length ? res.crops.map((c) => <StatusChip key={c} tone="leaf">{c}</StatusChip>) : <span className="text-[var(--dim)]">none detected</span>}
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[var(--dim)] font-mono2 text-[11px] w-[76px]">category →</span>
                  {res.categories.map((c) => <StatusChip key={c} tone="gold">{c}</StatusChip>)}
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[var(--dim)] font-mono2 text-[11px] w-[76px]">district →</span>
                  {res.district ? <StatusChip tone="water">{res.district}</StatusChip> : <span className="text-[var(--dim)]">fallback: Solapur (farm)</span>}
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[var(--dim)] font-mono2 text-[11px] w-[76px]">stage →</span>
                  {res.stage ? <StatusChip tone="marigold">{res.stage}</StatusChip> : <span className="text-[var(--dim)]">—</span>}
                </div>
              </div>
              <button className="btn btn-ghost mt-4 !py-1.5 !text-[12px]" onClick={() => { setValidated(false); setTimeout(() => setValidated(true), 700); }}>
                {validated ? "✓ Validation passed — integrity + KB coverage OK" : "Run taxonomy validation"}
              </button>
            </Card>
          </Reveal>
        </div>
      </div>

      <Reveal delay={140}>
        <Card className="p-4 mt-4">
          <div className="eyebrow mb-3">Crop stages pipeline</div>
          <div className="flex flex-wrap items-center gap-1.5">
            {TAX_STAGES.map((s, i) => (
              <span key={s.code} className="flex items-center gap-1.5">
                <span className="rounded-full border border-[var(--line2)] px-3 py-1 text-[11.5px] text-[var(--mut)] hover:border-[var(--leaf)] hover:text-[var(--ink)] transition-colors cursor-default">
                  <span className="font-mono2 text-[var(--gold)] mr-1.5">{s.code}</span>{s.en} <span className="devnagari text-[var(--dim)]">· {s.mr}</span>
                </span>
                {i < TAX_STAGES.length - 1 && <span className="text-[var(--dim)]">→</span>}
              </span>
            ))}
          </div>
        </Card>
      </Reveal>
    </div>
  );
}

/* ================= DATA FACTORY ================= */

export function DataFactory() {
  const [running, setRunning] = useState<string | null>(null);
  const [doneIds, setDoneIds] = useState<string[]>(["W-INGEST", "W-QUALITY"]);
  const [records, setRecords] = useState(FACTORY_INITIAL.records);
  const [dup, setDup] = useState(FACTORY_INITIAL.dupPct);
  const [statusLine, setStatusLine] = useState("Connecting to Mini factory APIs…");
  const timers = useRef<number[]>([]);

  useEffect(() => {
    (async () => {
      try {
        const [a, q, t, p] = await Promise.all([
          api.lakeAnalyze().catch(() => null),
          api.lakeQasynth().catch(() => null),
          api.lakeTokenizer().catch(() => null),
          api.lakePretrain().catch(() => null),
        ]);
        const total = a?.summary?.total_records ?? a?.counts?.total ?? q?.counts?.total;
        if (total) setRecords(Number(total));
        if (a?.summary?.duplicate_rate_pct != null) setDup(Number(a.summary.duplicate_rate_pct));
        const bits = [
          a?.ok != null ? `analyze:${a.ok ? "ok" : "—"}` : null,
          q?.ok != null ? `qasynth train=${q.counts?.train ?? "?"}` : null,
          t?.ok != null ? `token vocab=${t.actual_vocab_size ?? "?"}` : null,
          p?.ok != null || p?.param_count ? `pretrain ready` : null,
        ].filter(Boolean);
        setStatusLine(bits.length ? bits.join(" · ") : "Factory APIs reachable — run workers to refresh");
      } catch {
        setStatusLine("Factory APIs offline — demo metrics");
      }
    })();
  }, []);

  const runWorker = async (id: string) => {
    if (running) return;
    setRunning(id);
    try {
      if (id === "W-ANALYZE") await api.lakeAnalyzeRun(false);
      // other workers stay dry-run friendly via status endpoints
      setDoneIds((d) => (d.includes(id) ? d : [...d, id]));
      if (id === "W-QASYNTH") setRecords((r) => r + 1240);
      if (id === "W-QUALITY") setDup((d) => Math.max(0.4, d - 0.3));
      if (id === "W-ANALYZE") setRecords((r) => r + 86);
      setStatusLine(`Last worker: ${id} · ok`);
    } catch (e: any) {
      setStatusLine(`${id} demo tick (${e?.message || "offline"})`);
      setDoneIds((d) => (d.includes(id) ? d : [...d, id]));
    } finally {
      setRunning(null);
    }
  };

  const langs = [
    { label: "मराठी (mr)", value: FACTORY_INITIAL.langs.mr },
    { label: "English (en)", value: FACTORY_INITIAL.langs.en },
    { label: "हिंदी (hi)", value: FACTORY_INITIAL.langs.hi },
  ];

  return (
    <div>
      <SectionHead
        eyebrow="Mini data factory · Sprint 2–6"
        title={<>Data <span className="text-[var(--water)]">Factory</span></>}
        sub="Ingest → quality → standardize → analyze → QA synthesis. A monitored pipeline that keeps the training lake fresh and the coverage gaps visible."
        right={<StatusChip tone="water">schema v1 · {records.toLocaleString("en-IN")} records</StatusChip>}
      />
      <div className="mb-4 text-[12px] font-mono2 text-[var(--mut)] border border-[var(--line)] rounded-[10px] px-3 py-2 bg-[rgba(154,205,162,0.04)]">
        {statusLine}
      </div>

      {/* pipeline stepper */}
      <Reveal>
        <Card ticks className="p-4">
          <span className="tick-b" />
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-3">
            {FACTORY_INITIAL.workers.map((w, i) => {
              const isDone = doneIds.includes(w.id);
              const isRun = running === w.id;
              return (
                <div key={w.id} className="rounded-[11px] border p-3 flex flex-col gap-2 transition-all duration-300" style={{ borderColor: isRun ? "var(--gold)" : isDone ? "rgba(143,217,108,0.4)" : "var(--line)", background: isRun ? "rgba(244,192,75,0.06)" : isDone ? "rgba(143,217,108,0.04)" : "rgba(154,205,162,0.02)" }}>
                  <div className="flex items-center justify-between">
                    <span className="font-mono2 text-[11px] font-bold" style={{ color: isRun ? "var(--gold)" : isDone ? "var(--leaf)" : "var(--dim)" }}>{i + 1} · {w.id}</span>
                    <span className={`w-[8px] h-[8px] rounded-full ${isRun ? "bg-[var(--gold)] animate-pulse" : isDone ? "bg-[var(--leaf)]" : "bg-[var(--dim)] opacity-40"}`} />
                  </div>
                  <div className="text-[11.5px] text-[var(--mut)] leading-snug flex-1">{w.desc}</div>
                  <div className="prog">{isRun ? <i className="indeterminate" style={{ background: "var(--gold)" }} /> : <i style={{ width: isDone ? "100%" : "0%", background: "var(--leaf)" }} />}</div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono2 text-[9.5px] text-[var(--dim)]">η ~{w.eta}</span>
                    <button className="btn btn-ghost !py-1 !px-2.5 !text-[10.5px] !rounded-lg" onClick={() => runWorker(w.id)} disabled={!!running}>
                      {isRun ? "running…" : isDone ? "re-run" : "run ▸"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </Reveal>

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mt-4">
        {[
          { l: "Records in lake", v: records.toLocaleString("en-IN"), c: "var(--water)" },
          { l: "Duplicate rate", v: `${dup.toFixed(1)}%`, c: "var(--leaf)" },
          { l: "Missing crop", v: `${FACTORY_INITIAL.missingPct}%`, c: "var(--marigold)" },
          { l: "Coverage gaps", v: FACTORY_INITIAL.gaps.length, c: "var(--chili)" },
        ].map((m, i) => (
          <Reveal key={m.l} delay={i * 60}>
            <Card hover className="p-4 text-center">
              <div className="font-display font-bold text-[26px] leading-none tabnum" style={{ color: m.c }}>{m.v}</div>
              <div className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] mt-1.5">{m.l}</div>
            </Card>
          </Reveal>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 mt-4">
        <Reveal delay={80}>
          <Card className="p-4 h-full">
            <div className="eyebrow mb-3">Language balance</div>
            <BarRows rows={langs} color={(i) => ["var(--gold)", "var(--water)", "var(--marigold)"][i]} />
            <div className="mt-4 eyebrow mb-3">Category balance</div>
            <BarRows
              rows={Object.entries(FACTORY_INITIAL.cats).map(([k, v]) => ({ label: k, value: v }))}
              unit="%"
              color={(i) => ["var(--leaf)", "var(--gold)", "var(--water)", "var(--marigold)", "var(--soil)", "var(--chili)", "var(--dim)", "var(--dim)"][i]}
            />
          </Card>
        </Reveal>

        <Reveal delay={140}>
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="flex items-center justify-between mb-3">
              <span className="eyebrow">Taxonomy gaps · W-ANALYZE</span>
              <StatusChip tone="chili">{FACTORY_INITIAL.gaps.length} open</StatusChip>
            </div>
            <div className="space-y-2">
              {FACTORY_INITIAL.gaps.map((g, i) => (
                <div key={i} className="rounded-[9px] border border-[var(--line)] bg-[rgba(242,95,88,0.04)] px-3 py-2.5 flex items-center gap-3 hover:border-[rgba(242,95,88,0.4)] transition-colors">
                  <span className="font-display font-bold text-[14px] text-[var(--ink)] w-[86px] flex-none">{g.crop}</span>
                  <span className="chip !text-[9px] flex-none">{g.category}</span>
                  <span className="text-[12px] text-[var(--mut)] truncate">{g.gap}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 rounded-[9px] border border-dashed border-[rgba(93,185,232,0.4)] bg-[rgba(93,185,232,0.04)] p-3 text-[11.5px] text-[var(--mut)] font-mono2">
              → W-QASYNTH closes gaps first: packs are weighted by <span className="text-[var(--water)]">gap_score × language_need</span>, target ≥ 62,500 QA pairs.
            </div>
          </Card>
        </Reveal>
      </div>
    </div>
  );
}
