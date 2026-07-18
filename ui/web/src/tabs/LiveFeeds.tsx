import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Card, DeltaTag, Gauge, Reveal, SectionHead, StatTile, StatusChip, useLiveJitter } from "../lib/ui";

function ndviColor(v: number): string {
  if (v < 0.5) return "#92683a";
  if (v < 0.62) return "#ba9e4a";
  if (v < 0.75) return "#7a9c42";
  return "#4a9438";
}

export function LiveFeeds() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [cell, setCell] = useState<{ r: number; c: number; v: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const d = await api.live({ farm_id: "FARM_101", location: "Solapur", crop: "Pomegranate" });
        if (!cancelled) {
          setData(d);
          setErr(null);
        }
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const wx = data?.weather || {};
  const iot = data?.iot || {};
  const sat = data?.satellite || {};
  const mandi = data?.mandi || [];
  const grid: number[][] = data?.ndvi_grid || [];

  const temp = useLiveJitter(Number(wx.temp) || 30, 0.3);
  const humidity = useLiveJitter(Number(wx.humidity) || 60, 1);
  const moisture = useLiveJitter(Number(iot.moisture) || 30, 0.5);
  const ndvi = useLiveJitter(Number(sat.ndvi) || 0.7, 0.005);

  return (
    <div>
      <SectionHead
        eyebrow="Live RAG · weather / market / IoT / satellite"
        title={
          <>
            Intelligence <span className="text-[var(--leaf)]">Center</span>
          </>
        }
        sub="All feed values from backend /api/ui/live (weather, IoT, satellite, mandi, NDVI grid)."
        right={
          <div className="flex gap-1.5 items-center">
            <span className="livedot" />
            <span className="font-mono2 text-[10.5px] text-[var(--leaf)] tracking-[0.16em]">
              {data ? "BACKEND LIVE" : err ? "API ERROR" : "LOADING…"}
            </span>
          </div>
        }
      />

      {err && <div className="mb-3 text-[12px] font-mono2 text-[var(--chili)]">{err}</div>}

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile
          label="Live temperature"
          tone="marigold"
          live
          value={
            <>
              {temp.toFixed(1)}
              <span className="text-[16px] text-[var(--dim)]">°C</span>
            </>
          }
          sub={
            <span>
              Humidity <b className="text-[var(--ink)]">{humidity.toFixed(0)}%</b> · wind {wx.wind ?? "—"} km/h · {wx.station}
            </span>
          }
          spark={wx.series || [temp]}
        />
        <StatTile
          label="APMC modal"
          tone="gold"
          live
          value={
            <>
              ₹{Number(mandi[0]?.modal || 0).toLocaleString("en-IN")}
              <span className="text-[14px] text-[var(--dim)]">/qtl</span>
            </>
          }
          sub={
            <span>
              {mandi[0]?.crop || "—"} · <DeltaTag d={Number(mandi[0]?.d7) || 0} />
            </span>
          }
          spark={mandi.slice(0, 7).map((m: any) => Number(m.modal) || 0)}
        />
        <StatTile
          label="IoT soil moisture"
          tone="water"
          live
          value={
            <>
              {moisture.toFixed(1)}
              <span className="text-[16px] text-[var(--dim)]">%</span>
            </>
          }
          sub={<span>EC {iot.ec ?? "—"} · soil {iot.soilTemp ?? "—"}°C</span>}
          spark={iot.series || [moisture]}
        />
        <StatTile
          label="Sentinel NDVI"
          tone="leaf"
          live
          value={ndvi.toFixed(3)}
          sub={<span>{sat.vigor} · NDWI {sat.ndwi}</span>}
          spark={[0.6, 0.65, 0.68, 0.7, ndvi]}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mt-4">
        <Reveal className="xl:col-span-2">
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="eyebrow mb-2">NDVI grid · backend</div>
            <div className="font-display font-bold text-[17px] mb-3">{sat.pass || "Satellite pass"}</div>
            <div className="grid gap-[3px]" style={{ gridTemplateColumns: "repeat(14, 1fr)" }} onMouseLeave={() => setCell(null)}>
              {grid.flatMap((row, r) =>
                row.map((v, c) => (
                  <div
                    key={`${r}-${c}`}
                    className="ndvi-cell aspect-square rounded-[3px]"
                    style={{ background: ndviColor(v) }}
                    onMouseEnter={() => setCell({ r, c, v })}
                  />
                ))
              )}
            </div>
            <div className="mt-2 text-[12px] font-mono2 text-[var(--mut)]">
              {cell ? `zone ${cell.r + 1},${cell.c + 1} · NDVI ${cell.v.toFixed(2)}` : "hover a zone"}
            </div>
          </Card>
        </Reveal>
        <Card className="p-4">
          <div className="eyebrow mb-2">IoT gateway</div>
          <div className="text-[13px] text-[var(--mut)] mb-3">{iot.gateway}</div>
          <div className="space-y-2 text-[12px]">
            <div className="flex justify-between">
              <span className="text-[var(--dim)]">Moisture</span>
              <span>{moisture.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--dim)]">Soil temp</span>
              <span>{iot.soilTemp ?? "—"}°C</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--dim)]">Battery</span>
              <span>{iot.battery ?? "—"}%</span>
            </div>
          </div>
          <div className="mt-4">
            <Gauge pct={ndvi * 100} label="NDVI" color="var(--leaf)" size={90} />
          </div>
        </Card>
      </div>

      <Card ticks className="p-4 mt-4">
        <span className="tick-b" />
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="eyebrow">Mandi · backend market feed</div>
            <div className="font-display font-bold text-[17px] mt-1">APMC prices</div>
          </div>
          <StatusChip tone="leaf">{wx.advisory ? "advisory on" : "ok"}</StatusChip>
        </div>
        <div className="text-[13px] text-[var(--mut)] mb-3">{wx.advisory}</div>
        <div className="overflow-x-auto">
          <table className="dtable min-w-[680px]">
            <thead>
              <tr>
                <th>Commodity</th>
                <th>Variety</th>
                <th>Mandi</th>
                <th className="!text-right">Modal</th>
                <th className="!text-right">Min–Max</th>
                <th>Mode</th>
              </tr>
            </thead>
            <tbody>
              {mandi.map((r: any, i: number) => (
                <tr key={i}>
                  <td className="!text-[var(--ink)] font-semibold">{r.crop}</td>
                  <td>{r.variety}</td>
                  <td>{r.mandi}</td>
                  <td className="!text-right font-mono2">₹{Number(r.modal).toLocaleString("en-IN")}</td>
                  <td className="!text-right font-mono2">
                    {r.min}–{r.max}
                  </td>
                  <td className="font-mono2 text-[var(--dim)]">{r.mode}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
