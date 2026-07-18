import { useMemo, useState } from "react";
import { FERT_CROPS, SOIL_DEFAULT, calcFertilizer, formatINR, parseSoilCard } from "../data";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

const STATUS_META: Record<string, { label: string; color: string }> = {
  low: { label: "LOW", color: "var(--chili)" },
  medium: { label: "MED", color: "var(--gold)" },
  high: { label: "HIGH", color: "var(--marigold)" },
  optimal: { label: "OPTIMAL", color: "var(--leaf)" },
};

export function Soil() {
  const [text, setText] = useState(SOIL_DEFAULT);
  const [crop, setCrop] = useState("Pomegranate");
  const [acres, setAcres] = useState(2.5);
  const [runId, setRunId] = useState(0);

  const soil = useMemo(() => parseSoilCard(text), [text]);
  const fert = useMemo(() => calcFertilizer(crop, acres, soil), [crop, acres, soil]);

  const bags = [
    { name: "Urea", sub: "46% N · 45 kg bag", count: fert.ureaBags, color: "var(--water)", price: 350 },
    { name: "DAP", sub: "18-46-0 · 50 kg bag", count: fert.dapBags, color: "var(--soil)", price: 1350 },
    { name: "MOP", sub: "60% K · 50 kg bag", count: fert.mopBags, color: "var(--marigold)", price: 1700 },
  ];
  const totalBags = bags.reduce((a, b) => a + b.count, 0);

  return (
    <div>
      <SectionHead
        eyebrow="Soil health card OCR · fertilizer engine"
        title={<>Soil & <span className="text-[var(--soil)]">Fertilizer Lab</span> <span className="devnagari text-[18px] font-semibold text-[var(--mut)] ml-1">· माती प्रयोगशाळा</span></>}
        sub="Parse a soil health card (or OCR text), then compute bag-level fertilizer quantities, cost and a Marathi application schedule for your crop and acreage."
        right={<StatusChip tone="soil">ocr-engine v1.8</StatusChip>}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* parse column */}
        <Reveal>
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="eyebrow mb-3">1 · Soil card parameters</div>
            <textarea
              className="field min-h-[88px] font-mono2 !text-[12.5px]"
              value={text}
              onChange={(e) => setText(e.target.value)}
              spellCheck={false}
            />
            <div className="text-[10.5px] font-mono2 text-[var(--dim)] mt-1.5">live parsing — edit values above, e.g. “Nitrogen: 320 kg/ha”</div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 mt-4">
              {soil.map((p, i) => {
                const meta = STATUS_META[p.status];
                const pos = isNaN(p.value) ? 0 : Math.min(100, Math.max(0, ((p.value - p.low) / (p.high - p.low)) * 100));
                const idealL = ((p.ideal[0] - p.low) / (p.high - p.low)) * 100;
                const idealW = ((p.ideal[1] - p.ideal[0]) / (p.high - p.low)) * 100;
                return (
                  <div key={p.key} className="rounded-[10px] border border-[var(--line)] bg-[rgba(154,205,162,0.03)] p-3 reveal" style={{ animationDelay: `${i * 60}ms` }}>
                    <div className="flex items-center justify-between">
                      <span className="text-[10.5px] font-mono2 uppercase tracking-[0.12em] text-[var(--dim)]">{p.label}</span>
                      <span className="font-mono2 text-[9px] px-1.5 py-0.5 rounded" style={{ color: meta.color, background: `${meta.color}1a`, border: `1px solid ${meta.color}55` }}>{meta.label}</span>
                    </div>
                    <div className="font-display font-bold text-[20px] mt-1 tabnum" style={{ color: meta.color }}>
                      {isNaN(p.value) ? "—" : p.value}<span className="text-[11px] text-[var(--dim)] ml-1 font-mono2">{p.unit}</span>
                    </div>
                    <div className="relative h-[5px] rounded-full bg-[rgba(154,205,162,0.12)] mt-2 overflow-visible">
                      <div className="absolute top-0 h-full rounded-full bg-[rgba(143,217,108,0.25)]" style={{ left: `${idealL}%`, width: `${idealW}%` }} />
                      <div className="absolute -top-[3px] w-[11px] h-[11px] rounded-full border-2 border-[var(--ink)] transition-all duration-500" style={{ left: `calc(${pos}% - 5px)`, background: meta.color }} />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="text-[11px] text-[var(--dim)] mt-3 font-mono2">▮ green band = target range for {crop.toLowerCase()}</div>
          </Card>
        </Reveal>

        {/* fertilizer column */}
        <Reveal delay={80}>
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="eyebrow mb-3">2 · Target fertilizer calculator</div>
            <div className="grid grid-cols-[1fr_auto] gap-3 items-end">
              <div>
                <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Crop for fertigation plan</label>
                <select className="field" value={crop} onChange={(e) => { setCrop(e.target.value); setRunId((r) => r + 1); }}>
                  {Object.keys(FERT_CROPS).map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Acreage</label>
                <div className="flex items-center gap-1">
                  <button className="btn btn-ghost !px-3 !py-2" onClick={() => setAcres(Math.max(0.5, acres - 0.5))}>−</button>
                  <span className="font-display font-bold text-[20px] w-[64px] text-center tabnum">{acres.toFixed(1)}</span>
                  <button className="btn btn-ghost !px-3 !py-2" onClick={() => setAcres(Math.min(20, acres + 0.5))}>+</button>
                </div>
              </div>
            </div>

            {/* bag visuals */}
            <div key={runId} className="grid grid-cols-3 gap-3 mt-5">
              {bags.map((b, i) => (
                <div key={b.name} className="rounded-[11px] border border-[var(--line)] bg-[rgba(6,16,10,0.5)] p-3 text-center reveal" style={{ animationDelay: `${i * 100}ms` }}>
                  <div className="text-[24px] leading-none mb-1">🧺</div>
                  <div className="font-display font-bold text-[30px] leading-none tabnum" style={{ color: b.color }}>{b.count}</div>
                  <div className="text-[12px] font-semibold mt-1">{b.name}</div>
                  <div className="text-[10px] font-mono2 text-[var(--dim)]">{b.sub}</div>
                  <div className="text-[10.5px] font-mono2 text-[var(--mut)] mt-1.5">{formatINR(b.count * b.price)}</div>
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between rounded-[10px] border border-[rgba(244,192,75,0.35)] bg-[rgba(244,192,75,0.06)] px-4 py-2.5 mt-4">
              <span className="text-[12px] font-mono2 uppercase tracking-[0.14em] text-[var(--mut)]">Season total · {totalBags} bags</span>
              <span className="font-display font-bold text-[20px] text-[var(--gold)] tabnum">{formatINR(fert.cost)}</span>
            </div>

            <div className="mt-4 rounded-[11px] border border-dashed border-[rgba(143,217,108,0.4)] bg-[rgba(143,217,108,0.05)] p-3.5">
              <div className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--leaf)] mb-1">🗓️ Application schedule · मराठी संदेश</div>
              <p className="text-[13px] devnagari text-[var(--ink)] leading-relaxed">{fert.schedule}</p>
            </div>
          </Card>
        </Reveal>
      </div>
    </div>
  );
}
