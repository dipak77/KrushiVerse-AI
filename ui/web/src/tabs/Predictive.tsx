import { useRef, useState } from "react";
import { CROP_ETC, WORKFLOW_ALERTS, YIELD_CROPS, formatINR, irrigation } from "../data";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

const SEV_META: Record<string, { color: string; label: string; icon: string }> = {
  critical: { color: "var(--chili)", label: "CRITICAL", icon: "🚨" },
  warning: { color: "var(--marigold)", label: "WARNING", icon: "⚠️" },
  info: { color: "var(--water)", label: "INFO", icon: "ℹ️" },
  opportunity: { color: "var(--gold)", label: "OPPORTUNITY", icon: "💰" },
};

export function Predictive() {
  const [crop, setCrop] = useState("Pomegranate");
  const [acres, setAcres] = useState(2.5);
  const [temp, setTemp] = useState(30);
  const [rh, setRh] = useState(75);
  const [audit, setAudit] = useState<number>(0); // number of revealed alerts
  const [auditing, setAuditing] = useState(false);
  const timers = useRef<number[]>([]);

  const y = YIELD_CROPS[crop];
  const totalQ = Math.round(y.perAcre * acres * 10) / 10;
  const maxBar = Math.max(...y.dist);
  const ir = irrigation(crop, acres, temp, rh);
  const revenue = totalQ * (crop === "Pomegranate" ? 11850 : crop === "Cotton" ? 7420 : crop === "Soybean" ? 4685 : crop === "Onion" ? 2340 : crop === "Sugarcane" ? 315 : 2275);

  const runAudit = () => {
    if (auditing) return;
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setAuditing(true);
    setAudit(0);
    WORKFLOW_ALERTS.forEach((_, i) => {
      timers.current.push(window.setTimeout(() => {
        setAudit(i + 1);
        if (i === WORKFLOW_ALERTS.length - 1) setAuditing(false);
      }, 700 * (i + 1)));
    });
  };

  return (
    <div>
      <SectionHead
        eyebrow="Predictive models · workflows"
        title={<>Predictive <span className="text-[var(--gold)]">AI</span> & Automation <span className="devnagari text-[18px] font-semibold text-[var(--mut)] ml-1">· पूर्वानुमान इंजिन</span></>}
        sub="Yield forecasting, smart irrigation sizing and an automated farm-health audit that triggers workflow actions on your behalf."
        right={<StatusChip tone="gold">yield-model v5 · MAPE 8.2%</StatusChip>}
      />

      {/* controls */}
      <Reveal>
        <Card className="p-4 mb-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Forecast crop</label>
              <select className="field" value={crop} onChange={(e) => setCrop(e.target.value)}>
                {Object.keys(YIELD_CROPS).map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Land · {acres.toFixed(1)} acres</label>
              <input type="range" min={0.5} max={10} step={0.5} value={acres} onChange={(e) => setAcres(parseFloat(e.target.value))} className="w-full accent-[var(--gold)]" />
            </div>
            <div>
              <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Temp · {temp}°C</label>
              <input type="range" min={20} max={42} value={temp} onChange={(e) => setTemp(parseInt(e.target.value))} className="w-full accent-[var(--marigold)]" />
            </div>
            <div>
              <label className="text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] block mb-1.5">Humidity · {rh}%</label>
              <input type="range" min={30} max={95} value={rh} onChange={(e) => setRh(parseInt(e.target.value))} className="w-full accent-[var(--water)]" />
            </div>
          </div>
        </Card>
      </Reveal>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* yield forecast */}
        <Reveal delay={60}>
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="flex items-center justify-between flex-wrap gap-2 mb-1">
              <div className="eyebrow">🔮 Yield forecasting</div>
              <span className="font-mono2 text-[10.5px] text-[var(--dim)]">±12% confidence band</span>
            </div>
            <div className="flex items-end gap-6 flex-wrap mb-4">
              <div>
                <div className="font-display font-bold text-[38px] leading-none text-[var(--gold)] tabnum">
                  {totalQ.toLocaleString("en-IN")} <span className="text-[15px] text-[var(--dim)]">qtl</span>
                </div>
                <div className="text-[11.5px] text-[var(--mut)] mt-1 font-mono2">{(totalQ / acres).toFixed(1)} qtl/acre · season total</div>
              </div>
              <div>
                <div className="font-display font-bold text-[24px] leading-none text-[var(--leaf)] tabnum">{formatINR(revenue)}</div>
                <div className="text-[11.5px] text-[var(--mut)] mt-1 font-mono2">projected revenue @ modal price</div>
              </div>
            </div>
            <div className="relative h-[150px] flex items-end gap-[5px] px-1">
              {/* band */}
              <div className="absolute inset-x-0 bottom-0 top-0 pointer-events-none">
                {y.dist.map((d, i) => (
                  <div key={i} className="absolute bottom-0 rounded-t-sm bg-[rgba(244,192,75,0.07)]" style={{ left: `calc(${(i / y.dist.length) * 100}% + 2px)`, width: `calc(${100 / y.dist.length}% - 7px)`, height: `${(d / maxBar) * 96 * 1.12}%` }} />
                ))}
              </div>
              {y.dist.map((d, i) => (
                <div key={i} className="flex-1 flex flex-col items-center gap-1 relative group">
                  <div className="text-[9px] font-mono2 text-[var(--gold)] opacity-0 group-hover:opacity-100 transition-opacity">{Math.round((d / 100) * totalQ)}</div>
                  <div
                    className="w-full rounded-t-[4px] bar-grow"
                    style={{
                      height: `${(d / maxBar) * 96}px`,
                      background: `linear-gradient(180deg, var(--gold), var(--gold2) 90%)`,
                      animationDelay: `${i * 55}ms`,
                      opacity: 0.92,
                    }}
                  />
                  <span className="text-[9px] font-mono2 text-[var(--dim)]">{y.months[i]}</span>
                </div>
              ))}
            </div>
          </Card>
        </Reveal>

        {/* irrigation */}
        <Reveal delay={120}>
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="flex items-center justify-between flex-wrap gap-2 mb-1">
              <div className="eyebrow">💧 Smart irrigation runtime</div>
              <span className="font-mono2 text-[10.5px] text-[var(--dim)]">ETc = {(CROP_ETC[crop] ?? 5.2).toFixed(1)} mm/day base</span>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="rounded-[11px] border border-[rgba(93,185,232,0.35)] bg-[rgba(93,185,232,0.06)] p-3.5">
                <div className="font-display font-bold text-[30px] leading-none text-[var(--water)] tabnum">{ir.litersDay.toLocaleString("en-IN")}</div>
                <div className="text-[11px] font-mono2 uppercase tracking-[0.12em] text-[var(--dim)] mt-1.5">liters / day demand</div>
              </div>
              <div className="rounded-[11px] border border-[rgba(93,185,232,0.35)] bg-[rgba(93,185,232,0.06)] p-3.5">
                <div className="font-display font-bold text-[30px] leading-none text-[var(--water)] tabnum">{ir.dripHours} <span className="text-[14px] text-[var(--dim)]">h</span></div>
                <div className="text-[11px] font-mono2 uppercase tracking-[0.12em] text-[var(--dim)] mt-1.5">drip runtime / day</div>
              </div>
            </div>
            <div className="space-y-2.5 text-[12.5px]">
              {[
                ["ET adjusted (temp + RH)", `${ir.etAdj.toFixed(2)} mm/day`, "var(--marigold)"],
                ["Drip discharge capacity", `${ir.discharge.toLocaleString("en-IN")} L/h`, "var(--water)"],
                ["Suggested slot", "18:00 – 21:00 (low evaporation)", "var(--leaf)"],
              ].map(([k, v, c]) => (
                <div key={k as string} className="flex justify-between items-center">
                  <span className="text-[var(--mut)]">{k}</span>
                  <span className="font-mono2 tabnum" style={{ color: c as string }}>{v}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center gap-2 text-[11.5px] text-[var(--dim)]">
              <span className="livedot" style={{}} /> coupled to IoT soil-moisture — runtime auto-trims when band ≥ 40%
            </div>
          </Card>
        </Reveal>
      </div>

      {/* workflow audit */}
      <Reveal delay={100}>
        <Card ticks className="p-4 mt-4">
          <span className="tick-b" />
          <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
            <div>
              <div className="eyebrow">⚡ Automated farm health audit</div>
              <div className="font-display font-bold text-[17px] mt-1">Workflow engine · FARM_101</div>
            </div>
            <button className="btn btn-primary" onClick={runAudit} disabled={auditing}>
              {auditing ? <span className="typing"><i /><i /><i /></span> : "▶"} {auditing ? "Auditing…" : "Run farm health audit"}
            </button>
          </div>

          {audit === 0 && !auditing && (
            <div className="text-[12.5px] text-[var(--dim)] py-5 text-center border border-dashed border-[var(--line2)] rounded-[10px]">
              The engine cross-checks soil telemetry, NDVI zones, pest forecasts and mandi trends, then books irrigation slots & scouting tasks automatically.
            </div>
          )}

          <div className="space-y-2.5">
            {WORKFLOW_ALERTS.slice(0, audit).map((a, i) => {
              const m = SEV_META[a.sev];
              return (
                <div key={a.trigger} className="flex gap-3 rounded-[11px] border p-3.5 logline" style={{ borderColor: `${m.color}44`, background: `${m.color}0a`, animationDelay: `${i * 60}ms` }}>
                  <span className="text-[18px] flex-none">{m.icon}</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono2 text-[10px] tracking-[0.12em] px-1.5 py-0.5 rounded" style={{ color: m.color, background: `${m.color}1a` }}>{m.label}</span>
                      <span className="font-mono2 text-[11px] text-[var(--mut)]">{a.trigger}</span>
                    </div>
                    <p className="text-[13.5px] devnagari text-[var(--ink)] mt-1 leading-relaxed">{a.msgMr}</p>
                    <div className="text-[11px] font-mono2 mt-1.5" style={{ color: m.color }}>{a.action}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </Reveal>
    </div>
  );
}
