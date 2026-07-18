import { useEffect, useMemo, useState } from "react";
import { GRAPH_DATA, NODE_TYPE_META, type GraphEdge, type GraphNode, type NodeType } from "../data";
import { api } from "../lib/api";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

const W = 640, H = 500, CX = W / 2, CY = H / 2;

function ecoToGraph(crop: string, eco: any): { nodes: GraphNode[]; edges: GraphEdge[] } | null {
  if (!eco || eco.error) return null;
  const nodes: GraphNode[] = [{ id: crop, type: "crop" }];
  const edges: GraphEdge[] = [];
  const add = (id: string, type: NodeType, rel: string) => {
    if (!id) return;
    if (!nodes.find((n) => n.id === id)) nodes.push({ id, type });
    edges.push({ a: crop, b: id, rel });
  };
  (eco.pests_and_diseases || []).forEach((p: string) => add(p, "pest", "affected_by"));
  (eco.soil_types || []).forEach((s: string) => add(s, "fertilizer", "grows_in"));
  (eco.recommended_fertilizers || []).forEach((f: string) => add(f, "fertilizer", "fed_with"));
  (eco.applicable_schemes || []).forEach((s: string) => add(s, "scheme", "eligible_for"));
  if (nodes.length < 2) return null;
  return { nodes, edges };
}

export function Graph() {
  const [crop, setCrop] = useState("Pomegranate");
  const [hover, setHover] = useState<string | null>(null);
  const [live, setLive] = useState<Record<string, { nodes: GraphNode[]; edges: GraphEdge[] }>>({});
  const [mode, setMode] = useState<"live" | "demo">("demo");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const eco = await api.graph(crop);
        const g = ecoToGraph(crop, eco);
        if (!cancelled && g) {
          setLive((prev) => ({ ...prev, [crop]: g }));
          setMode("live");
        }
      } catch {
        if (!cancelled) setMode("demo");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [crop]);

  const data = live[crop] || GRAPH_DATA[crop] || GRAPH_DATA.Pomegranate;

  const layout = useMemo(() => {
    const outer = data.nodes.filter((n) => n.type !== "crop");
    const pos: Record<string, { x: number; y: number; n: GraphNode }> = {};
    pos[crop] = { x: CX, y: CY, n: data.nodes.find((n) => n.type === "crop")! };
    outer.forEach((n, i) => {
      const a = (i / outer.length) * Math.PI * 2 - Math.PI / 2 + 0.26;
      const r = 176 + (i % 2) * 26;
      pos[n.id] = { x: CX + Math.cos(a) * r, y: CY + Math.sin(a) * r, n };
    });
    return pos;
  }, [crop, data]);

  const neighbors = useMemo(() => {
    const m: Record<string, Set<string>> = {};
    data.edges.forEach((e) => {
      (m[e.a] ??= new Set()).add(e.b);
      (m[e.b] ??= new Set()).add(e.a);
    });
    return m;
  }, [data]);

  const isLit = (id: string) => !hover || id === hover || neighbors[hover]?.has(id);
  const edgeLit = (a: string, b: string) => !hover || a === hover || b === hover;

  const hoveredNode = hover ? layout[hover] : null;
  const rels = data.edges.filter((e) => hover && (e.a === hover || e.b === hover));

  return (
    <div>
      <SectionHead
        eyebrow="GraphRAG · knowledge graph explorer"
        title={<>Knowledge <span className="text-[var(--water)]">Graph</span> <span className="devnagari text-[18px] font-semibold text-[var(--mut)] ml-1">· ज्ञान जाल</span></>}
        sub="Explore how crops link to pests, diseases, treatments, fertilizers, schemes and markets. Hover any node to trace its relations — these walks feed the RAG fusion layer."
        right={
          <div className="flex gap-1.5">
            <StatusChip tone="water">{data.edges.length} edges</StatusChip>
            <StatusChip tone={mode === "live" ? "leaf" : "gold"}>{mode === "live" ? "API graph" : "demo graph"}</StatusChip>
          </div>
        }
      />

      <div className="flex flex-wrap gap-2 mb-4">
        {Object.keys(GRAPH_DATA).map((c) => (
          <button
            key={c}
            onClick={() => { setCrop(c); setHover(null); }}
            className={`px-4 py-1.5 rounded-full text-[13px] font-semibold border transition-all duration-200 ${
              crop === c
                ? "border-[var(--gold)] bg-[rgba(244,192,75,0.12)] text-[var(--gold)] shadow-[0_4px_14px_-6px_rgba(244,192,75,0.5)]"
                : "border-[var(--line2)] text-[var(--mut)] hover:border-[var(--leaf)] hover:text-[var(--ink)]"
            }`}
          >
            {c} <span className="devnagari font-normal opacity-70">· {GRAPH_DATA[c].nodes[0].mr}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_300px] gap-4">
        <Reveal>
          <Card ticks className="p-2 relative overflow-hidden">
            <span className="tick-b" />
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto select-none">
              <defs>
                <radialGradient id="gcenter" cx="50%" cy="50%">
                  <stop offset="0%" stopColor="rgba(143,217,108,0.28)" />
                  <stop offset="100%" stopColor="rgba(143,217,108,0)" />
                </radialGradient>
              </defs>
              <circle cx={CX} cy={CY} r={230} fill="url(#gcenter)" />

              {/* edges */}
              {data.edges.map((e) => {
                const a = layout[e.a], b = layout[e.b];
                if (!a || !b) return null;
                const lit = edgeLit(e.a, e.b);
                return (
                  <g key={e.a + e.b} className="glink" opacity={lit ? 1 : 0.12}>
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={lit && hover ? NODE_TYPE_META[b.n.type].color : "rgba(154,205,162,0.35)"} strokeWidth={lit && hover ? 2 : 1.2} strokeDasharray={e.rel === "sold_at" ? "5 4" : undefined} />
                    <circle r={2.4} fill={lit && hover ? NODE_TYPE_META[b.n.type].color : "rgba(154,205,162,0.5)"}>
                      <animateMotion dur="2.6s" repeatCount="indefinite" path={`M ${a.x} ${a.y} L ${b.x} ${b.y}`} />
                    </circle>
                  </g>
                );
              })}

              {/* nodes */}
              {Object.values(layout).map(({ x, y, n }) => {
                const meta = NODE_TYPE_META[n.type];
                const isCenter = n.type === "crop";
                const lit = isLit(n.id);
                const r = isCenter ? 34 : 26;
                return (
                  <g key={n.id} className="gnode" opacity={lit ? 1 : 0.16} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}>
                    {isCenter && <circle cx={x} cy={y} r={44} fill="none" stroke="rgba(244,192,75,0.35)" strokeDasharray="3 5" className="spin-slow" style={{ transformOrigin: `${x}px ${y}px` }} />}
                    {hover === n.id && <circle cx={x} cy={y} r={r + 7} fill="none" stroke={meta.color} strokeWidth={1.5} opacity={0.6} />}
                    <circle cx={x} cy={y} r={r} fill="#0c1d13" stroke={meta.color} strokeWidth={isCenter ? 2.4 : 1.6} style={{ filter: hover === n.id ? `drop-shadow(0 0 10px ${meta.color})` : undefined, transition: "filter .2s" }} />
                    <text x={x} y={y - 2} textAnchor="middle" fill={meta.color} fontSize={isCenter ? 15 : 13}>
                      {isCenter ? "🌱" : n.type === "disease" ? "🦠" : n.type === "pest" ? "🐛" : n.type === "treatment" ? "💊" : n.type === "fertilizer" ? "🧪" : n.type === "scheme" ? "🏛️" : "🏪"}
                    </text>
                    <text x={x} y={y + 14} textAnchor="middle" fill="var(--ink)" fontSize={9.5} fontWeight={600} fontFamily="var(--font-body)">
                      {n.id.length > 16 ? n.id.slice(0, 15) + "…" : n.id}
                    </text>
                    {n.mr && <text x={x} y={y + 25} textAnchor="middle" fill="var(--dim)" fontSize={8.5} fontFamily="Mukta">{n.mr}</text>}
                  </g>
                );
              })}
            </svg>

            {/* legend */}
            <div className="absolute bottom-3 left-3 flex flex-wrap gap-1.5">
              {Object.entries(NODE_TYPE_META).map(([t, m]) => (
                <span key={t} className="chip !text-[9px]" style={{ color: m.color, borderColor: `${m.color}66`, background: `${m.color}14` }}>● {m.label}</span>
              ))}
            </div>
          </Card>
        </Reveal>

        {/* relations panel */}
        <div className="space-y-4">
          <Reveal delay={80}>
            <Card className="p-4 h-full">
              <div className="eyebrow mb-3">{hover ? `Relations · ${hover}` : "Relations · hover a node"}</div>
              {hoveredNode ? (
                <>
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="w-[10px] h-[10px] rounded-full" style={{ background: NODE_TYPE_META[hoveredNode.n.type].color }} />
                    <div>
                      <div className="font-display font-bold text-[16px]">{hover}</div>
                      <div className="text-[11px] font-mono2 text-[var(--dim)]">{NODE_TYPE_META[hoveredNode.n.type].label}{hoveredNode.n.mr ? ` · ${hoveredNode.n.mr}` : ""}</div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {rels.map((e, i) => {
                      const other = e.a === hover ? e.b : e.a;
                      const otherNode = layout[other]?.n;
                      if (!otherNode) return null;
                      return (
                        <div key={i} className="flex items-center gap-2 text-[12px] rounded-[9px] border border-[var(--line)] bg-[rgba(154,205,162,0.03)] px-2.5 py-2 logline" style={{ animationDelay: `${i * 60}ms` }}>
                          <span className="text-[var(--gold)] font-mono2 text-[10.5px] flex-none">{e.a === hover ? "→" : "←"}</span>
                          <span className="flex-1 truncate text-[var(--ink)]">{other}</span>
                          <span className="font-mono2 text-[9.5px] px-1.5 py-0.5 rounded" style={{ color: NODE_TYPE_META[otherNode.type].color, background: `${NODE_TYPE_META[otherNode.type].color}14` }}>{e.rel}</span>
                        </div>
                      );
                    })}
                    {rels.length === 0 && <div className="text-[12px] text-[var(--dim)]">No edges recorded.</div>}
                  </div>
                </>
              ) : (
                <p className="text-[12.5px] text-[var(--mut)] leading-relaxed">
                  Every node carries Marathi aliases so queries like <span className="devnagari text-[var(--gold)]">“डाळिंब तेल्या रोग”</span> land directly on the right subgraph.
                </p>
              )}
            </Card>
          </Reveal>

          <Reveal delay={160}>
            <div className="rounded-[14px] border border-dashed border-[rgba(93,185,232,0.4)] bg-[rgba(93,185,232,0.05)] p-4">
              <div className="text-[12px] font-mono2 text-[var(--water)] tracking-[0.14em] uppercase mb-1.5">GraphRAG in the stack</div>
              <p className="text-[12.5px] text-[var(--mut)] leading-relaxed">
                <code className="text-[var(--water)] text-[11px]">graph_rag.get_crop_ecosystem()</code> walks 2-hops from the resolved crop and contributes ranked edges into the assistant's fused context — that's why treatments arrive with scheme eligibility attached.
              </p>
            </div>
          </Reveal>
        </div>
      </div>
    </div>
  );
}
