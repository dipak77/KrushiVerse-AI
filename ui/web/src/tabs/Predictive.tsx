import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

export function Predictive() {
  const [crop, setCrop] = useState("Pomegranate");
  const [acres, setAcres] = useState(2.5);
  const [temp, setTemp] = useState(30);
  const [rh, setRh] = useState(75);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await api.predict({
        crop,
        acreage: acres,
        temperature_c: temp,
        humidity_pct: rh,
        farm_id: "FARM_101",
      });
      setData(r);
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

  const y = data?.yield || {};
  const irr = data?.irrigation || {};
  const alerts = data?.alerts || [];
  const chart = data?.chart || { months: [], dist: [] };
  const maxBar = Math.max(...(chart.dist || [1]), 1);

  return (
    <div>
      <SectionHead
        eyebrow="Predictive models · workflows · API"
        title={
          <>
            Predictive <span className="text-[var(--gold)]">AI</span>
          </>
        }
        sub="Yield, irrigation and farm audit from POST /api/ui/predict (yield_model, irrigation_model, workflow_engine)."
        right={<StatusChip tone="gold">{data?.source || "backend"}</StatusChip>}
      />

      {err && <div className="mb-3 text-[12px] font-mono2 text-[var(--chili)]">{err}</div>}

      <Card className="p-4 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
          <div>
            <label className="text-[10px] font-mono2 text-[var(--dim)]">Crop</label>
            <input className="field" value={crop} onChange={(e) => setCrop(e.target.value)} />
          </div>
          <div>
            <label className="text-[10px] font-mono2 text-[var(--dim)]">Acres</label>
            <input className="field" type="number" value={acres} onChange={(e) => setAcres(parseFloat(e.target.value) || 0)} />
          </div>
          <div>
            <label className="text-[10px] font-mono2 text-[var(--dim)]">Temp °C</label>
            <input className="field" type="number" value={temp} onChange={(e) => setTemp(parseFloat(e.target.value) || 0)} />
          </div>
          <div>
            <label className="text-[10px] font-mono2 text-[var(--dim)]">RH %</label>
            <input className="field" type="number" value={rh} onChange={(e) => setRh(parseFloat(e.target.value) || 0)} />
          </div>
        </div>
        <button className="btn btn-primary mt-3" onClick={run} disabled={loading}>
          {loading ? "Running models…" : "Run predictive suite (API)"}
        </button>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="eyebrow mb-2">Yield model</div>
          <div className="font-display font-bold text-[28px] text-[var(--leaf)]">{y.total_predicted_yield ?? "—"}</div>
          <div className="text-[12px] text-[var(--mut)]">{y.unit} · per acre {y.predicted_yield_per_acre}</div>
          <div className="text-[11px] font-mono2 text-[var(--dim)] mt-2">conf {y.yield_confidence_score}</div>
        </Card>
        <Card className="p-4">
          <div className="eyebrow mb-2">Irrigation model</div>
          <pre className="text-[11px] font-mono2 text-[var(--mut)] overflow-x-auto max-h-[160px]">{JSON.stringify(irr, null, 1)}</pre>
        </Card>
        <Card className="p-4">
          <div className="eyebrow mb-2">Workflow alerts</div>
          <div className="space-y-2 max-h-[180px] overflow-y-auto">
            {alerts.map((a: any, i: number) => (
              <div key={i} className="text-[12px] border border-[var(--line)] rounded-[8px] p-2">
                <span className="font-mono2 text-[var(--gold)]">{a.sev}</span> · {a.msgEn || a.msgMr}
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card className="p-4 mt-4">
        <div className="eyebrow mb-3">Distribution chart · backend managed</div>
        <div className="flex items-end gap-1 h-[120px]">
          {(chart.dist || []).map((v: number, i: number) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div className="w-full rounded-t bg-[var(--gold)] opacity-80" style={{ height: `${(v / maxBar) * 100}%` }} />
              <span className="text-[9px] font-mono2 text-[var(--dim)]">{(chart.months || [])[i]}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
