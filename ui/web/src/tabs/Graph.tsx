import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

const W = 640,
  H = 500,
  CX = W / 2,
  CY = H / 2;

const COLORS: Record<string, string> = {
  crop: "#8fd96c",
  disease: "#f25f58",
  pest: "#f78f3a",
  treatment: "#5db9e8",
  fertilizer: "#c09161",
  scheme: "#f4c04b",
  market: "#b48ce0",
};

const CROPS = ["Pomegranate", "Cotton", "Soybean", "Onion", "Wheat", "Rice", "Sugarcane"];

export function Graph() {
  const [crop, setCrop] = useState("Pomegranate");
  const [hover, setHover] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const g = await api.graph(crop);
        if (!cancelled) {
          setData(g);
          setErr(null);
        }
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [crop]);

  const nodes = data?.nodes || [];
  const edges = data?.edges || [];

  const layout = useMemo(() => {
    const pos: Record<string, { x: number; y: number; n: any }> = {};
    const outer = nodes.filter((n: any) => n.type !== "crop");
    const center = nodes.find((n: any) => n.type === "crop") || { id: crop, type: "crop" };
    pos[center.id] = { x: CX, y: CY, n: center };
    outer.forEach((n: any, i: number) => {
      const a = (i / Math.max(1, outer.length)) * Math.PI * 2 - Math.PI / 2;
      const r = 176 + (i % 2) * 26;
      pos[n.id] = { x: CX + Math.cos(a) * r, y: CY + Math.sin(a) * r, n };
    });
    return pos;
  }, [nodes, crop]);

  return (
    <div>
      <SectionHead
        eyebrow="GraphRAG · API"
        title={
          <>
            Knowledge <span className="text-[var(--water)]">Graph</span>
          </>
        }
        sub="Nodes/edges from GET /api/ui/graph/{crop} (GraphRAG + KB enrichment)."
        right={
          <div className="flex gap-1.5">
            <StatusChip tone="water">{edges.length} edges</StatusChip>
            <StatusChip tone="leaf">{data?.mode || data?.source || "backend"}</StatusChip>
          </div>
        }
      />

      {err && <div className="mb-3 text-[12px] font-mono2 text-[var(--chili)]">{err}</div>}

      <div className="flex flex-wrap gap-2 mb-4">
        {CROPS.map((c) => (
          <button
            key={c}
            onClick={() => {
              setCrop(c);
              setHover(null);
            }}
            className={`px-4 py-1.5 rounded-full text-[13px] font-semibold border transition-all ${
              crop === c ? "border-[var(--gold)] bg-[rgba(244,192,75,0.12)] text-[var(--gold)]" : "border-[var(--line2)] text-[var(--mut)]"
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_280px] gap-4">
        <Card ticks className="p-2">
          <span className="tick-b" />
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
            {edges.map((e: any, i: number) => {
              const a = layout[e.a];
              const b = layout[e.b];
              if (!a || !b) return null;
              const lit = !hover || e.a === hover || e.b === hover;
              return (
                <line
                  key={i}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={lit ? "rgba(244,192,75,0.55)" : "rgba(154,205,162,0.12)"}
                  strokeWidth={lit ? 1.6 : 1}
                />
              );
            })}
            {Object.values(layout).map((p: any) => {
              const lit = !hover || p.n.id === hover;
              const col = COLORS[p.n.type] || "#8fd96c";
              return (
                <g
                  key={p.n.id}
                  onMouseEnter={() => setHover(p.n.id)}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: "pointer", opacity: lit ? 1 : 0.25 }}
                >
                  <circle cx={p.x} cy={p.y} r={p.n.type === "crop" ? 28 : 18} fill={col} fillOpacity={0.25} stroke={col} strokeWidth={2} />
                  <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={p.n.type === "crop" ? 11 : 9} fill="var(--ink)">
                    {String(p.n.id).slice(0, 14)}
                  </text>
                </g>
              );
            })}
          </svg>
        </Card>
        <Card className="p-4">
          <div className="eyebrow mb-2">Relations</div>
          <div className="space-y-2 max-h-[420px] overflow-y-auto text-[12px]">
            {edges.map((e: any, i: number) => (
              <div key={i} className="border border-[var(--line)] rounded-[8px] px-2 py-1.5 font-mono2 text-[var(--mut)]">
                {e.a} <span className="text-[var(--gold)]">—[{e.rel}]→</span> {e.b}
              </div>
            ))}
            {!edges.length && <div className="text-[var(--dim)]">No edges yet for this crop.</div>}
          </div>
        </Card>
      </div>
    </div>
  );
}
