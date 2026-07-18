import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

export function Soil() {
  const [crop, setCrop] = useState("Pomegranate");
  const [acres, setAcres] = useState(2.5);
  const [text, setText] = useState(
    "pH: 7.2, EC: 0.45, Organic Carbon: 0.52%, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha"
  );
  const [plan, setPlan] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await api.soil({ crop, acreage: acres, soil_text: text, farm_id: "FARM_101" });
      setPlan(r);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const params = plan?.params || [];
  const fert = plan?.fertilizer || {};

  return (
    <div>
      <SectionHead
        eyebrow="Soil health · fertilizer · API"
        title={
          <>
            Soil & <span className="text-[var(--soil)]">Fertilizer Lab</span>
          </>
        }
        sub="All soil params and bag calculations from POST /api/ui/soil (farm memory + fertilizer planner)."
        right={<StatusChip tone="soil">{plan?.source || "backend"}</StatusChip>}
      />

      {err && <div className="mb-3 text-[12px] font-mono2 text-[var(--chili)]">{err}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card ticks className="p-4">
          <span className="tick-b" />
          <div className="eyebrow mb-3">Input · sent to backend</div>
          <label className="text-[11px] font-mono2 text-[var(--dim)]">Crop</label>
          <input className="field mb-2" value={crop} onChange={(e) => setCrop(e.target.value)} />
          <label className="text-[11px] font-mono2 text-[var(--dim)]">Acres</label>
          <input
            className="field mb-2"
            type="number"
            step="0.1"
            value={acres}
            onChange={(e) => setAcres(parseFloat(e.target.value) || 0)}
          />
          <label className="text-[11px] font-mono2 text-[var(--dim)]">Soil card text</label>
          <textarea className="field min-h-[100px] font-mono2 !text-[12px]" value={text} onChange={(e) => setText(e.target.value)} />
          <button className="btn btn-primary mt-3" onClick={run} disabled={loading}>
            {loading ? "Computing…" : "Run soil plan (API)"}
          </button>
        </Card>

        <Card className="p-4">
          <div className="eyebrow mb-3">Parsed parameters · API</div>
          <div className="grid grid-cols-2 gap-2">
            {params.map((p: any) => (
              <div key={p.key} className="rounded-[10px] border border-[var(--line)] p-3">
                <div className="text-[10px] font-mono2 text-[var(--dim)] uppercase">{p.label}</div>
                <div className="font-display font-bold text-[20px] text-[var(--gold)]">
                  {p.value}
                  <span className="text-[11px] text-[var(--dim)] ml-1">{p.unit}</span>
                </div>
                <div className="text-[10px] font-mono2 text-[var(--mut)]">{p.status}</div>
              </div>
            ))}
            {!params.length && <div className="text-[var(--dim)] text-[13px]">No data yet — run plan.</div>}
          </div>
        </Card>
      </div>

      <Card ticks className="p-4 mt-4">
        <span className="tick-b" />
        <div className="eyebrow mb-2">Fertilizer planner · API response</div>
        <pre className="text-[11.5px] font-mono2 text-[var(--mut)] bg-[rgba(6,16,10,0.75)] border border-[var(--line)] rounded-[10px] p-3 overflow-x-auto max-h-[320px]">
          {JSON.stringify(fert, null, 2)}
        </pre>
        <div className="mt-3 text-[13px] text-[var(--ink)]">{plan?.schedule_en}</div>
        <div className="text-[13px] devnagari text-[var(--mut)]">{plan?.schedule_mr}</div>
      </Card>
    </div>
  );
}
