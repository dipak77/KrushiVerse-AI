import { useMemo, useState } from "react";
import { useDashboard } from "../context/DashboardContext";
import { StatusChip } from "../lib/ui";

export function Sidebar({
  lang,
  setLang,
  webRag,
  setWebRag,
}: {
  lang: "mr" | "en";
  setLang: (l: "mr" | "en") => void;
  webRag: boolean;
  setWebRag: (v: boolean) => void;
}) {
  const { bootstrap, loading, error } = useDashboard();
  const farm = bootstrap?.farm;
  const taxCrops = bootstrap?.taxonomy_crops || [];
  const kl = bootstrap?.knowledge_layer || {};
  const [taxCrop, setTaxCrop] = useState("Pomegranate");

  const rec = useMemo(() => {
    const found = taxCrops.find((c: any) => c.en === taxCrop);
    return found || taxCrops[0] || { en: taxCrop, mr: taxCrop, hi: taxCrop, group: "—", sci: "", aliases: [] };
  }, [taxCrop, taxCrops]);

  return (
    <aside className="w-[300px] flex-none h-full overflow-y-auto border-r border-[var(--line)] bg-[rgba(8,19,12,0.55)] backdrop-blur-sm relative z-10">
      <div className="relative h-[150px] overflow-hidden">
        <img src="/field-aerial.jpg" alt="Farm aerial" className="absolute inset-0 w-full h-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#08130c] via-[#08130c88] to-transparent" />
        <div className="absolute bottom-3 left-4 right-4">
          <div className="flex items-center gap-2">
            <span className="livedot gold" />
            <span className="font-mono2 text-[10px] tracking-[0.2em] text-[var(--gold)]">
              {farm?.id || "FARM_101"} · {loading ? "LOADING" : "API"}
            </span>
          </div>
          <h3 className="font-display font-bold text-[20px] leading-tight mt-1">
            {farm?.farmerMr || "—"} <span className="text-[var(--mut)] text-[15px] font-medium">· {farm?.farmer}</span>
          </h3>
          <div className="text-[12px] text-[var(--mut)]">
            {farm?.village}, {farm?.district} · {farm?.state}
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {error && <div className="text-[11px] font-mono2 text-[var(--chili)]">Bootstrap API: {error}</div>}

        <div className="grid grid-cols-2 gap-2">
          {[
            { k: "Current crop", v: `${farm?.cropMr || "—"} · ${farm?.crop || "—"}`, cls: "text-[var(--leaf)]" },
            { k: "Acreage", v: `${farm?.acres ?? "—"} acres`, cls: "text-[var(--ink)]" },
            { k: "Soil", v: farm?.soil || "—", cls: "text-[var(--ink)]" },
            { k: "Season", v: farm?.memory?.seasonDay || "—", cls: "text-[var(--gold)]" },
          ].map((f) => (
            <div key={f.k} className="rounded-[10px] border border-[var(--line)] bg-[rgba(154,205,162,0.04)] px-3 py-2">
              <div className="text-[9.5px] font-mono2 uppercase tracking-[0.16em] text-[var(--dim)]">{f.k}</div>
              <div className={`text-[12.5px] font-semibold mt-0.5 leading-snug ${f.cls}`}>{f.v}</div>
            </div>
          ))}
        </div>

        <div>
          <div className="eyebrow mb-2">Farm memory · API</div>
          <div className="space-y-1.5 text-[12.5px]">
            <div className="flex justify-between gap-2">
              <span className="text-[var(--dim)]">Last audit</span>
              <span className="text-[var(--mut)] text-right">{farm?.memory?.lastVisit || "—"}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-[var(--dim)]">Last diagnosis</span>
              <span className="text-[var(--mut)] text-right">{farm?.memory?.lastDiagnosis || "—"}</span>
            </div>
            <div className="flex justify-between gap-2 items-center">
              <span className="text-[var(--dim)]">Active alerts</span>
              <StatusChip tone="chili">{farm?.memory?.activeAlerts ?? 0} open</StatusChip>
            </div>
          </div>
        </div>

        <hr className="stitch-soft" />

        <div className="space-y-3">
          <div>
            <div className="text-[11px] font-mono2 uppercase tracking-[0.16em] text-[var(--dim)] mb-1.5">Response language · भाषा</div>
            <div className="seg w-full">
              <button className={lang === "mr" ? "on flex-1" : "flex-1"} onClick={() => setLang("mr")}>
                मराठी
              </button>
              <button className={lang === "en" ? "on flex-1" : "flex-1"} onClick={() => setLang("en")}>
                ENGLISH
              </button>
            </div>
          </div>
          <label className="flex items-center justify-between gap-2 cursor-pointer select-none">
            <span className="text-[13px] text-[var(--mut)]">Enable Web RAG</span>
            <span className="switch">
              <input type="checkbox" checked={webRag} onChange={(e) => setWebRag(e.target.checked)} />
              <span className="track" />
              <span className="knob" />
            </span>
          </label>
        </div>

        <hr className="stitch-soft" />

        <div>
          <div className="eyebrow mb-2">Knowledge layer · API</div>
          <div className="space-y-1.5 font-mono2 text-[11.5px]">
            {[
              ["Indexed docs", kl.indexed_docs ?? "—"],
              ["Embedding", kl.embedding ?? "—"],
              ["Dense store", kl.dense_store ?? "—"],
              ["Graph nodes", kl.graph_nodes ?? "—"],
              ["Graph edges", kl.graph_edges ?? "—"],
            ].map(([k, v]) => (
              <div key={String(k)} className="flex justify-between gap-2">
                <span className="text-[var(--dim)]">{k}</span>
                <span className="text-[var(--mut)] text-right truncate max-w-[140px]">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>

        <hr className="stitch-soft" />

        <div>
          <div className="eyebrow mb-2">Taxonomy · API</div>
          <select
            className="field !py-2 !text-[13px]"
            value={rec.en}
            onChange={(e) => setTaxCrop(e.target.value)}
          >
            {taxCrops.map((c: any) => (
              <option key={c.en} value={c.en}>
                {c.en} — {c.mr}
              </option>
            ))}
          </select>
          <div className="mt-2.5 rounded-[10px] border border-[var(--line)] bg-[rgba(154,205,162,0.04)] p-3">
            <div className="font-semibold text-[14px] leading-snug">
              {rec.en} <span className="text-[var(--leaf)] devnagari">/ {rec.mr}</span>
            </div>
            <div className="text-[11px] text-[var(--dim)] mt-1 font-mono2">
              {rec.group} · <i className="not-italic text-[var(--mut)]">{rec.sci}</i>
            </div>
          </div>
        </div>

        <div className="pt-1 pb-3 text-center">
          <span className="text-[10px] font-mono2 text-[var(--dim)] tracking-widest">
            KRV-OS · data via /api/ui/*
          </span>
        </div>
      </div>
    </aside>
  );
}
