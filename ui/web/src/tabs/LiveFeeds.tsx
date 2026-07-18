import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Card, DeltaTag, Gauge, Reveal, SectionHead, StatTile, StatusChip, useLiveJitter } from "../lib/ui";

function ndviColor(v: number): string {
  if (v < 0.5) return "#92683a";
  if (v < 0.62) return "#ba9e4a";
  if (v < 0.75) return "#7a9c42";
  return "#4a9438";
}

function n(v: unknown, fallback = 0): number {
  const x = Number(v);
  return Number.isFinite(x) ? x : fallback;
}

function sparkOf(arr: unknown, fallback: number[]): number[] {
  if (Array.isArray(arr) && arr.length) {
    const out = arr.map((x) => n(x, 0));
    return out.length >= 2 ? out : [out[0], out[0]];
  }
  return fallback.length >= 2 ? fallback : [fallback[0] ?? 0, fallback[0] ?? 0];
}

/** Default payload so the page always renders even before / while API loads */
const EMPTY = {
  weather: {
    station: "Loading…",
    temp: 30,
    humidity: 60,
    wind: 10,
    series: [28, 29, 30, 31, 30, 29, 30],
    advisory: "Loading live feeds from backend…",
  },
  iot: {
    gateway: "LoRa gateway…",
    moisture: 30,
    soilTemp: 26,
    ec: 0.5,
    battery: 85,
    series: [28, 29, 30, 31, 30, 29, 30, 31],
  },
  satellite: {
    pass: "Satellite…",
    ndvi: 0.7,
    evi: 0.4,
    ndwi: 0.2,
    vigor: "—",
  },
  mandi: [] as any[],
  ndvi_grid: [] as number[][],
};

export function LiveFeeds() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [cell, setCell] = useState<{ r: number; c: number; v: number } | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const d = await api.live({ farm_id: "FARM_101", location: "Solapur", crop: "Pomegranate" });
      setData(d);
    } catch (e: any) {
      setErr(e?.message || String(e));
      // keep last good data if any; otherwise EMPTY shows skeleton values
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const src = data || EMPTY;
  const wx = src.weather || EMPTY.weather;
  const iot = src.iot || EMPTY.iot;
  const sat = src.satellite || EMPTY.satellite;
  const mandi = Array.isArray(src.mandi) ? src.mandi : [];
  const grid: number[][] = Array.isArray(src.ndvi_grid) ? src.ndvi_grid : [];

  const baseTemp = n(wx.temp, 30);
  const baseHum = n(wx.humidity, 60);
  const baseMoist = n(iot.moisture, 30);
  const baseNdvi = n(sat.ndvi, 0.7);

  const temp = useLiveJitter(baseTemp, 0.3);
  const humidity = useLiveJitter(baseHum, 1);
  const moisture = useLiveJitter(baseMoist, 0.5);
  const ndvi = useLiveJitter(baseNdvi, 0.005);

  const mandiSpark = sparkOf(
    mandi.map((m: any) => n(m.modal, 0)).filter((x: number) => x > 0),
    [10000, 10500, 11000, 10800, 11200, 11500, n(mandi[0]?.modal, 11000)]
  );

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
          <div className="flex gap-1.5 items-center flex-wrap justify-end">
            <span className="livedot" />
            <span className="font-mono2 text-[10.5px] text-[var(--leaf)] tracking-[0.16em]">
              {loading ? "LOADING…" : data ? "BACKEND LIVE" : "RETRY"}
            </span>
            <button className="btn btn-ghost !py-1 !px-2.5 !text-[11px]" onClick={load} disabled={loading}>
              ⟳ Refresh
            </button>
          </div>
        }
      />

      {err && (
        <div className="mb-3 text-[12px] font-mono2 text-[var(--chili)] border border-[rgba(242,95,88,0.35)] rounded-[10px] px-3 py-2 bg-[rgba(242,95,88,0.08)]">
          Live API error: {err}
          <div className="mt-1 text-[var(--mut)]">Showing last/placeholder values. Check API is running (e.g. :8002).</div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile
          label="Live temperature"
          tone="marigold"
          live
          value={
            <>
              {n(temp, baseTemp).toFixed(1)}
              <span className="text-[16px] text-[var(--dim)]">°C</span>
            </>
          }
          sub={
            <span>
              Humidity <b className="text-[var(--ink)]">{n(humidity, baseHum).toFixed(0)}%</b> · wind {wx.wind ?? "—"} km/h ·{" "}
              {wx.station || "—"}
            </span>
          }
          spark={sparkOf(wx.series, [baseTemp - 1, baseTemp, baseTemp + 1, baseTemp])}
        />
        <StatTile
          label="APMC modal"
          tone="gold"
          live
          value={
            <>
              ₹{n(mandi[0]?.modal, 0).toLocaleString("en-IN")}
              <span className="text-[14px] text-[var(--dim)]">/qtl</span>
            </>
          }
          sub={
            <span>
              {mandi[0]?.crop || "—"} · <DeltaTag d={n(mandi[0]?.d7, 0)} />
            </span>
          }
          spark={mandiSpark}
        />
        <StatTile
          label="IoT soil moisture"
          tone="water"
          live
          value={
            <>
              {n(moisture, baseMoist).toFixed(1)}
              <span className="text-[16px] text-[var(--dim)]">%</span>
            </>
          }
          sub={
            <span>
              EC {iot.ec ?? "—"} · soil {iot.soilTemp ?? "—"}°C
            </span>
          }
          spark={sparkOf(iot.series, [baseMoist - 2, baseMoist, baseMoist + 1, baseMoist])}
        />
        <StatTile
          label="Sentinel NDVI"
          tone="leaf"
          live
          value={n(ndvi, baseNdvi).toFixed(3)}
          sub={
            <span>
              {sat.vigor || "—"} · NDWI {sat.ndwi ?? "—"}
            </span>
          }
          spark={sparkOf([0.6, 0.65, 0.68, 0.7, baseNdvi], [0.6, 0.7])}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mt-4">
        <Reveal className="xl:col-span-2">
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="eyebrow mb-2">NDVI grid · backend</div>
            <div className="font-display font-bold text-[17px] mb-3">{sat.pass || "Satellite pass"}</div>
            {grid.length > 0 ? (
              <div
                className="grid gap-[3px]"
                style={{ gridTemplateColumns: "repeat(14, 1fr)" }}
                onMouseLeave={() => setCell(null)}
              >
                {grid.flatMap((row, r) =>
                  (row || []).map((v, c) => (
                    <div
                      key={`${r}-${c}`}
                      className="ndvi-cell aspect-square rounded-[3px]"
                      style={{ background: ndviColor(n(v, 0.5)) }}
                      onMouseEnter={() => setCell({ r, c, v: n(v, 0.5) })}
                    />
                  ))
                )}
              </div>
            ) : (
              <div className="h-[180px] flex items-center justify-center text-[var(--dim)] font-mono2 text-[12px] border border-dashed border-[var(--line2)] rounded-[10px]">
                {loading ? "Loading NDVI grid…" : "No NDVI grid in API response"}
              </div>
            )}
            <div className="mt-2 text-[12px] font-mono2 text-[var(--mut)]">
              {cell ? `zone ${cell.r + 1},${cell.c + 1} · NDVI ${cell.v.toFixed(2)}` : "hover a zone"}
            </div>
          </Card>
        </Reveal>
        <Card className="p-4">
          <div className="eyebrow mb-2">IoT gateway</div>
          <div className="text-[13px] text-[var(--mut)] mb-3">{iot.gateway || "—"}</div>
          <div className="space-y-2 text-[12px]">
            <div className="flex justify-between">
              <span className="text-[var(--dim)]">Moisture</span>
              <span>{n(moisture, baseMoist).toFixed(1)}%</span>
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
            <Gauge pct={n(ndvi, baseNdvi) * 100} label="NDVI" color="var(--leaf)" size={90} />
          </div>
        </Card>
      </div>

      <Card ticks className="p-4 mt-4">
        <span className="tick-b" />
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div>
            <div className="eyebrow">Mandi · backend market feed</div>
            <div className="font-display font-bold text-[17px] mt-1">APMC prices</div>
          </div>
          <StatusChip tone="leaf">{wx.advisory ? "advisory on" : "ok"}</StatusChip>
        </div>
        <div className="text-[13px] text-[var(--mut)] mb-3">{wx.advisory || "—"}</div>
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
              {mandi.length === 0 && (
                <tr>
                  <td colSpan={6} className="!text-[var(--dim)]">
                    {loading ? "Loading mandi prices…" : "No mandi rows from API"}
                  </td>
                </tr>
              )}
              {mandi.map((r: any, i: number) => (
                <tr key={i}>
                  <td className="!text-[var(--ink)] font-semibold">{r.crop}</td>
                  <td>{r.variety}</td>
                  <td>{r.mandi}</td>
                  <td className="!text-right font-mono2">₹{n(r.modal, 0).toLocaleString("en-IN")}</td>
                  <td className="!text-right font-mono2">
                    {n(r.min, 0)}–{n(r.max, 0)}
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
