import { useMemo, useState } from "react";
import { FARM, TAX_CROPS } from "../data";
import { StatusChip } from "../lib/ui";

export function Sidebar({ lang, setLang, webRag, setWebRag }: {
  lang: "mr" | "en";
  setLang: (l: "mr" | "en") => void;
  webRag: boolean;
  setWebRag: (v: boolean) => void;
}) {
  const [taxCrop, setTaxCrop] = useState("Pomegranate");
  const rec = useMemo(() => TAX_CROPS.find((c) => c.en === taxCrop)!, [taxCrop]);

  return (
    <aside className="w-[300px] flex-none h-full overflow-y-auto border-r border-[var(--line)] bg-[rgba(8,19,12,0.55)] backdrop-blur-sm relative z-10">
      {/* farm hero */}
      <div className="relative h-[150px] overflow-hidden">
        <img src="/field-aerial.jpg" alt="FARM_101 aerial" className="absolute inset-0 w-full h-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#08130c] via-[#08130c88] to-transparent" />
        <div className="absolute bottom-3 left-4 right-4">
          <div className="flex items-center gap-2">
            <span className="livedot gold" />
            <span className="font-mono2 text-[10px] tracking-[0.2em] text-[var(--gold)]">FARM_101 · LIVE TELEMETRY</span>
          </div>
          <h3 className="font-display font-bold text-[20px] leading-tight mt-1">
            {FARM.farmerMr} <span className="text-[var(--mut)] text-[15px] font-medium">· {FARM.farmer}</span>
          </h3>
          <div className="text-[12px] text-[var(--mut)]">{FARM.village}, {FARM.district} · {FARM.state}</div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* farm facts */}
        <div className="grid grid-cols-2 gap-2">
          {[
            { k: "Current crop", v: `${FARM.cropMr} · ${FARM.crop}`, cls: "text-[var(--leaf)]" },
            { k: "Acreage", v: `${FARM.acres} acres`, cls: "text-[var(--ink)]" },
            { k: "Soil", v: FARM.soil, cls: "text-[var(--ink)]" },
            { k: "Season", v: FARM.memory.seasonDay, cls: "text-[var(--gold)]" },
          ].map((f) => (
            <div key={f.k} className="rounded-[10px] border border-[var(--line)] bg-[rgba(154,205,162,0.04)] px-3 py-2">
              <div className="text-[9.5px] font-mono2 uppercase tracking-[0.16em] text-[var(--dim)]">{f.k}</div>
              <div className={`text-[12.5px] font-semibold mt-0.5 leading-snug ${f.cls}`}>{f.v}</div>
            </div>
          ))}
        </div>

        {/* memory */}
        <div>
          <div className="eyebrow mb-2">Farm memory</div>
          <div className="space-y-1.5 text-[12.5px]">
            <div className="flex justify-between gap-2"><span className="text-[var(--dim)]">Last audit</span><span className="text-[var(--mut)] text-right">{FARM.memory.lastVisit}</span></div>
            <div className="flex justify-between gap-2"><span className="text-[var(--dim)]">Last diagnosis</span><span className="text-[var(--mut)] text-right">{FARM.memory.lastDiagnosis}</span></div>
            <div className="flex justify-between gap-2 items-center">
              <span className="text-[var(--dim)]">Active alerts</span>
              <StatusChip tone="chili">{FARM.memory.activeAlerts} open</StatusChip>
            </div>
          </div>
        </div>

        <hr className="stitch-soft" />

        {/* controls */}
        <div className="space-y-3">
          <div>
            <div className="text-[11px] font-mono2 uppercase tracking-[0.16em] text-[var(--dim)] mb-1.5">Response language · भाषा</div>
            <div className="seg w-full">
              <button className={lang === "mr" ? "on flex-1" : "flex-1"} onClick={() => setLang("mr")}>मराठी</button>
              <button className={lang === "en" ? "on flex-1" : "flex-1"} onClick={() => setLang("en")}>ENGLISH</button>
            </div>
          </div>
          <label className="flex items-center justify-between gap-2 cursor-pointer select-none">
            <span className="text-[13px] text-[var(--mut)]">Enable Web RAG</span>
            <span className="switch">
              <input type="checkbox" checked={webRag} onChange={(e) => setWebRag(e.target.checked)} />
              <span className="track" /><span className="knob" />
            </span>
          </label>
        </div>

        <hr className="stitch-soft" />

        {/* knowledge layer */}
        <div>
          <div className="eyebrow mb-2">Knowledge layer</div>
          <div className="space-y-1.5 font-mono2 text-[11.5px]">
            {[
              ["Indexed docs", "1,284"],
              ["Embedding", "bge-m3 · 1024d"],
              ["Dense store", "qdrant (local)"],
              ["Graph", "v3.1 · 8,412 edges"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-[var(--dim)]">{k}</span>
                <span className="text-[var(--mut)]">{v}</span>
              </div>
            ))}
            <div className="flex justify-between items-center pt-0.5">
              <span className="text-[var(--dim)]">Agmarknet</span>
              <span className="text-[var(--leaf)]">✅ key set · live</span>
            </div>
          </div>
        </div>

        <hr className="stitch-soft" />

        {/* taxonomy quick browser */}
        <div>
          <div className="eyebrow mb-2">Taxonomy · S1 frozen</div>
          <div className="flex items-center gap-1.5 mb-2 font-mono2 text-[10.5px] text-[var(--dim)]">
            <span className="chip chip-gold !text-[9px]">v1.0</span>
            <span>frozen · 12 crops · 8 cats</span>
          </div>
          <select className="field !py-2 !text-[13px]" value={taxCrop} onChange={(e) => setTaxCrop(e.target.value)}>
            {TAX_CROPS.map((c) => (
              <option key={c.en} value={c.en}>{c.en} — {c.mr}</option>
            ))}
          </select>
          <div className="mt-2.5 rounded-[10px] border border-[var(--line)] bg-[rgba(154,205,162,0.04)] p-3">
            <div className="font-semibold text-[14px] leading-snug">
              {rec.en} <span className="text-[var(--leaf)] devnagari">/ {rec.mr}</span> <span className="text-[var(--marigold)] devnagari">/ {rec.hi}</span>
            </div>
            <div className="text-[11px] text-[var(--dim)] mt-1 font-mono2">{rec.group} · <i className="not-italic text-[var(--mut)]">{rec.sci}</i></div>
            <div className="flex flex-wrap gap-1 mt-2">
              {rec.aliases.map((a) => (
                <span key={a} className="text-[10px] font-mono2 px-1.5 py-0.5 rounded bg-[rgba(154,205,162,0.08)] text-[var(--mut)]">{a}</span>
              ))}
            </div>
          </div>
        </div>

        <div className="pt-1 pb-3 text-center">
          <span className="text-[10px] font-mono2 text-[var(--dim)] tracking-widest">KRV-OS v10.2 · GEN 10.2 + SPRINT 10</span>
        </div>
      </div>
    </aside>
  );
}
