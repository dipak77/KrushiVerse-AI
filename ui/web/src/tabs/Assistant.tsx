import { useEffect, useRef, useState } from "react";
import { PIPELINE_STAGES, SAMPLE_QUERIES, matchQuery, type AsstResponse } from "../data";
import { api, mapQueryToAsst } from "../lib/api";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

type Phase = "idle" | "running" | "done";

export function Assistant({ lang, webRag = true }: { lang: "mr" | "en"; webRag?: boolean }) {
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [stageIdx, setStageIdx] = useState(0);
  const [result, setResult] = useState<(AsstResponse & { raw?: any }) | null>(null);
  const [showJson, setShowJson] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveMode, setLiveMode] = useState(true);
  const timers = useRef<number[]>([]);
  const outRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const run = async (q: string) => {
    if (!q.trim() || phase === "running") return;
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setQuery(q);
    setPhase("running");
    setStageIdx(0);
    setResult(null);
    setShowJson(false);
    setError(null);

    const t0 = performance.now();
    PIPELINE_STAGES.forEach((_, i) => {
      timers.current.push(window.setTimeout(() => setStageIdx(i + 1), 320 * (i + 1)));
    });

    try {
      let mapped: AsstResponse & { raw?: any };
      if (liveMode) {
        const resp = await api.query({
          query: q,
          farm_id: "FARM_101",
          language: lang,
          enable_web: webRag,
        });
        mapped = mapQueryToAsst(resp, lang);
        mapped.metrics.latencyMs = Math.round(performance.now() - t0);
      } else {
        await new Promise((r) => setTimeout(r, 420 * PIPELINE_STAGES.length + 200));
        mapped = matchQuery(q);
        mapped.metrics.latencyMs = Math.round(performance.now() - t0);
      }
      setStageIdx(PIPELINE_STAGES.length);
      setResult(mapped);
      setPhase("done");
    } catch (e: any) {
      // graceful fallback to offline demo responses
      const fallback = matchQuery(q);
      fallback.metrics.latencyMs = Math.round(performance.now() - t0);
      setResult(fallback);
      setPhase("done");
      setError(`Live API unavailable — showing demo advisory. (${e?.message || e})`);
    }
    setTimeout(() => outRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 200);
  };

  const primary = result ? (lang === "mr" ? result.mr : result.en) : [];
  const secondary = result ? (lang === "mr" ? result.en : result.mr) : [];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_330px] gap-5">
      <div className="min-w-0">
        <SectionHead
          eyebrow="Multi-agent advisory · संभाषण"
          title={
            <>
              AI Krushi <span className="text-[var(--gold)]">Assistant</span>{" "}
              <span className="devnagari text-[var(--leaf)] font-semibold text-[19px] ml-1">· एआय कृषी मित्र</span>
            </>
          }
          sub="Ask anything — crop management, disease treatment, market rates, schemes. Planner dispatches expert agents over the multi-source knowledge layer."
          right={
            <div className="flex gap-1.5 flex-wrap justify-end">
              <StatusChip tone="gold">planner v4</StatusChip>
              <StatusChip tone={liveMode ? "leaf" : "soil"}>{liveMode ? "live API" : "demo"}</StatusChip>
            </div>
          }
        />

        <Reveal delay={40}>
          <label className="flex items-center gap-2 mb-3 text-[12.5px] text-[var(--mut)] cursor-pointer select-none">
            <input type="checkbox" checked={liveMode} onChange={(e) => setLiveMode(e.target.checked)} />
            Call live FastAPI <code className="font-mono2 text-[var(--gold)]">/api/query</code>
          </label>
        </Reveal>

        <Reveal delay={60}>
          <div className="mb-3 text-[11px] font-mono2 uppercase tracking-[0.16em] text-[var(--dim)]">Sample queries · नमुना प्रश्न</div>
          <div className="flex flex-wrap gap-2 mb-4">
            {SAMPLE_QUERIES.map((s) => (
              <button
                key={s}
                onClick={() => setQuery(s)}
                className="text-left text-[12.5px] devnagari px-3 py-1.5 rounded-full border border-[var(--line2)] bg-[rgba(154,205,162,0.04)] text-[var(--mut)] hover:text-[var(--ink)] hover:border-[rgba(244,192,75,0.45)] hover:bg-[rgba(244,192,75,0.06)] transition-all duration-200 hover:-translate-y-0.5"
              >
                {s.length > 62 ? s.slice(0, 62) + "…" : s}
              </button>
            ))}
          </div>
        </Reveal>

        <Reveal delay={120}>
          <Card ticks className="p-4">
            <span className="tick-b" />
            <textarea
              className="field min-h-[92px]"
              placeholder={lang === "mr" ? "तुमचा प्रश्न येथे लिहा… (मराठी / English)" : "Type your query… (मराठी / English)"}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(query);
              }}
            />
            <div className="flex items-center justify-between mt-3 flex-wrap gap-2">
              <span className="text-[11px] font-mono2 text-[var(--dim)]">
                ⌘/Ctrl + ↵ · FARM_101 · {lang === "mr" ? "मराठी" : "EN"} · web {webRag ? "on" : "off"}
              </span>
              <button className="btn btn-primary" disabled={phase === "running" || !query.trim()} onClick={() => run(query)}>
                {phase === "running" ? (
                  <span className="typing">
                    <i />
                    <i />
                    <i />
                  </span>
                ) : (
                  "🚀"
                )}
                <span>{phase === "running" ? "Planning…" : "Ask Krushi Mitra"}</span>
              </button>
            </div>
          </Card>
        </Reveal>

        {error && (
          <div className="mt-3 text-[12px] font-mono2 text-[var(--marigold)] border border-[rgba(247,143,58,0.35)] rounded-[10px] px-3 py-2 bg-[rgba(247,143,58,0.08)]">
            {error}
          </div>
        )}

        {phase !== "idle" && (
          <Reveal delay={40}>
            <Card className="p-4 mt-4">
              <div className="flex items-center justify-between mb-3">
                <span className="eyebrow">Planner pipeline</span>
                <span className="font-mono2 text-[11px] text-[var(--dim)]">
                  {Math.min(stageIdx, PIPELINE_STAGES.length)}/{PIPELINE_STAGES.length} stages
                </span>
              </div>
              <div className="prog mb-3">
                <i
                  style={{
                    width: `${(Math.min(stageIdx, PIPELINE_STAGES.length) / PIPELINE_STAGES.length) * 100}%`,
                    background: "linear-gradient(90deg,var(--gold2),var(--gold))",
                  }}
                />
              </div>
              <ol className="space-y-2">
                {PIPELINE_STAGES.map((s, i) => {
                  const done = stageIdx > i;
                  const active = stageIdx === i;
                  return (
                    <li
                      key={s}
                      className="flex items-center gap-3 text-[13px] transition-opacity duration-300"
                      style={{ opacity: done || active ? 1 : 0.35 }}
                    >
                      <span
                        className="w-[18px] h-[18px] rounded-full flex items-center justify-center text-[10px] font-bold flex-none"
                        style={{
                          background: done ? "var(--leaf)" : active ? "rgba(244,192,75,0.2)" : "transparent",
                          border: `1.5px solid ${done ? "var(--leaf)" : active ? "var(--gold)" : "var(--line2)"}`,
                          color: done ? "#08130a" : "var(--dim)",
                        }}
                      >
                        {done ? "✓" : active ? <span className="w-[7px] h-[7px] rounded-full bg-[var(--gold)] animate-pulse" /> : i + 1}
                      </span>
                      <span className={done ? "text-[var(--mut)]" : active ? "text-[var(--ink)]" : "text-[var(--dim)]"}>{s}</span>
                    </li>
                  );
                })}
              </ol>
            </Card>
          </Reveal>
        )}

        {result && phase === "done" && (
          <div ref={outRef}>
            <Reveal>
              <Card ticks className="p-5 mt-4 border-[rgba(244,192,75,0.3)]">
                <span className="tick-b" />
                <div className="flex items-center justify-between flex-wrap gap-2 mb-4">
                  <div className="flex items-center gap-2">
                    <span className="text-[18px]">📝</span>
                    <span className="font-display font-bold text-[17px]">Synthesized advisory</span>
                    <span className="chip chip-gold">
                      {result.crop} · {result.cropMr}
                    </span>
                  </div>
                  <span className="font-mono2 text-[11px] text-[var(--dim)]">
                    {result.metrics.latencyMs} ms · fused {result.metrics.fused} docs
                  </span>
                </div>
                <div className="space-y-3 text-[14.5px] leading-relaxed">
                  {primary.map((p, i) => (
                    <p
                      key={i}
                      className="devnagari text-[var(--ink)]"
                      dangerouslySetInnerHTML={{
                        __html: p.replace(/\*\*(.+?)\*\*/g, "<strong class='text-[var(--gold)]'>$1</strong>"),
                      }}
                    />
                  ))}
                </div>
                {secondary.length > 0 && secondary !== primary && (
                  <details className="mt-4 group">
                    <summary className="cursor-pointer text-[12px] font-mono2 text-[var(--mut)] list-none flex items-center gap-2">
                      <span className="transition-transform group-open:rotate-90">▸</span>
                      {lang === "mr" ? "English version" : "मराठी आवृत्ती"}
                    </summary>
                    <div className="mt-3 pl-3 border-l-2 border-dashed border-[rgba(244,192,75,0.3)] space-y-2 text-[13.5px]">
                      {secondary.map((p, i) => (
                        <p key={i} className="devnagari text-[var(--mut)]" dangerouslySetInnerHTML={{ __html: p }} />
                      ))}
                    </div>
                  </details>
                )}
              </Card>
            </Reveal>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
              {[
                { l: "Crop resolved", v: result.crop, c: "var(--leaf)" },
                { l: "Fused docs", v: result.metrics.fused, c: "var(--gold)" },
                { l: "Web hits", v: result.metrics.web, c: "var(--water)" },
                { l: "Tools used", v: result.metrics.tools, c: "var(--marigold)" },
              ].map((m) => (
                <Card key={m.l} hover className="p-3.5 text-center">
                  <div className="font-display font-bold text-[22px] leading-none" style={{ color: m.c }}>
                    {m.v}
                  </div>
                  <div className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] mt-1.5">{m.l}</div>
                </Card>
              ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <Card className="p-4 h-full">
                <div className="eyebrow mb-3">Citations · sources</div>
                <ol className="space-y-2.5">
                  {result.citations.map((c, i) => (
                    <li key={i} className="flex gap-2.5 text-[13px]">
                      <span className="font-mono2 text-[var(--gold)] flex-none">{String(i + 1).padStart(2, "0")}</span>
                      <div>
                        <div className="text-[var(--ink)] leading-snug">{c.title}</div>
                        <div className="text-[11px] text-[var(--dim)] font-mono2 mt-0.5">
                          <span className="text-[var(--water)]">{c.origin}</span> · {c.src}
                        </div>
                      </div>
                    </li>
                  ))}
                </ol>
              </Card>
              <Card className="p-4 h-full">
                <div className="eyebrow mb-3">Agents · tools</div>
                <div className="flex flex-wrap gap-2 mb-3">
                  {result.agents.map((a) => (
                    <span key={a} className="chip chip-leaf">
                      {a}
                    </span>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  {result.tools.map((t) => (
                    <code
                      key={t}
                      className="text-[11.5px] font-mono2 px-2.5 py-1 rounded-lg bg-[rgba(93,185,232,0.08)] border border-[rgba(93,185,232,0.25)] text-[var(--water)]"
                    >
                      {t}()
                    </code>
                  ))}
                </div>
                <button className="btn btn-ghost !mt-4 !py-1.5 !px-3 !text-[12px] w-full" onClick={() => setShowJson(!showJson)}>
                  {showJson ? "▴ Hide" : "▾ Inspect"} JSON
                </button>
                {showJson && (
                  <pre className="mt-3 text-[10.5px] font-mono2 text-[var(--mut)] bg-[rgba(6,16,10,0.8)] border border-[var(--line)] rounded-[10px] p-3 overflow-x-auto max-h-[280px]">
                    {JSON.stringify(result.raw || result, null, 2)}
                  </pre>
                )}
              </Card>
            </div>
          </div>
        )}
      </div>

      <div className="space-y-4">
        <Card className="p-4">
          <div className="eyebrow mb-2">How it works</div>
          <ul className="text-[12.5px] text-[var(--mut)] space-y-2 list-disc pl-4">
            <li>Taxonomy resolves crop / category / district</li>
            <li>Planner dispatches soil, disease, market agents</li>
            <li>Multi-source RAG fuses KB + Graph + tools + web</li>
            <li>Bilingual synthesis for FARM_101 context</li>
          </ul>
        </Card>
        <Card className="p-4">
          <div className="eyebrow mb-2">API</div>
          <code className="text-[11px] font-mono2 text-[var(--gold)]">POST /api/query</code>
          <div className="text-[12px] text-[var(--dim)] mt-2">Body: query, farm_id, language, enable_web</div>
        </Card>
      </div>
    </div>
  );
}
