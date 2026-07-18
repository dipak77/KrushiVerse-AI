import { useEffect, useState } from "react";
import { IOT, MANDI_TABLE, NDVI_GRID, SAT, WEATHER, ndviColor } from "../data";
import { api } from "../lib/api";
import { Card, DeltaTag, Gauge, Reveal, SectionHead, StatTile, StatusChip, useLiveJitter } from "../lib/ui";

export function LiveFeeds() {
  const [wx, setWx] = useState(WEATHER);
  const [iot, setIot] = useState(IOT);
  const [sat, setSat] = useState(SAT);
  const [mandi, setMandi] = useState(MANDI_TABLE);
  const [apiStatus, setApiStatus] = useState<"live" | "demo">("demo");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [w, i, s, m] = await Promise.all([
          api.weather("Solapur"),
          api.iot("FARM_101"),
          api.satellite("FARM_101", "Pomegranate"),
          api.market("Pomegranate", "Solapur").catch(() => null),
        ]);
        if (cancelled) return;
        setApiStatus("live");
        if (w) {
          setWx((prev) => ({
            ...prev,
            temp: w.temperature_c ?? w.temp ?? prev.temp,
            humidity: w.relative_humidity_pct ?? w.humidity_pct ?? w.humidity ?? prev.humidity,
            wind: w.wind_speed_kmh ?? w.wind_kmh ?? w.wind ?? prev.wind,
            condition: w.condition || prev.condition,
            station: w.station || w.location || prev.station,
            advisory: w.advisory || w.farming_advice || prev.advisory,
          }));
        }
        if (i) {
          setIot((prev) => ({
            ...prev,
            moisture: i.soil_moisture_pct ?? i.moisture ?? prev.moisture,
            soilTemp: i.soil_temperature_c ?? i.soil_temp_c ?? i.soil_temperature ?? prev.soilTemp,
            ec: i.ec_ds_m ?? i.ec ?? prev.ec,
            battery: i.battery_pct ?? prev.battery,
            gateway: i.gateway || i.gateway_id || prev.gateway,
          }));
        }
        if (s) {
          setSat((prev) => ({
            ...prev,
            ndvi: s.ndvi ?? prev.ndvi,
            evi: s.evi ?? prev.evi,
            ndwi: s.ndwi ?? prev.ndwi,
            vigor: s.vigor || prev.vigor,
            pass: s.pass || s.source || prev.pass,
          }));
        }
        if (m && Array.isArray(m.prices || m.records || m.data)) {
          const rows = m.prices || m.records || m.data;
          if (rows.length) {
            setMandi(
              rows.slice(0, 8).map((r: any) => ({
                crop: r.commodity || r.crop || "Crop",
                variety: r.variety || "—",
                mandi: r.market || r.mandi || "APMC",
                modal: Number(r.modal_price || r.modal || r.price || 0),
                min: Number(r.min_price || r.min || 0),
                max: Number(r.max_price || r.max || 0),
                d7: Number(r.change_pct || 0),
                mode: "API live",
              }))
            );
          }
        }
      } catch {
        setApiStatus("demo");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const temp = useLiveJitter(wx.temp, 0.4);
  const humidity = useLiveJitter(wx.humidity, 1.6);
  const moisture = useLiveJitter(iot.moisture, 0.7);
  const ndvi = useLiveJitter(sat.ndvi, 0.008);
  const [cell, setCell] = useState<{ r: number; c: number; v: number } | null>(null);

  return (
    <div>
      <SectionHead
        eyebrow="Live RAG · weather / market / IoT / satellite"
        title={<>Intelligence <span className="text-[var(--leaf)]">Center</span> <span className="devnagari text-[18px] font-semibold text-[var(--mut)] ml-1">· थेट माहिती केंद्र</span></>}
        sub="Four live feeds fused into one picture: AWS weather, APMC market prices, LoRa soil telemetry and Sentinel-2 vegetation indices — refreshed continuously."
        right={
          <div className="flex gap-1.5 items-center">
            <span className="livedot" />
            <span className="font-mono2 text-[10.5px] text-[var(--leaf)] tracking-[0.16em]">
              {apiStatus === "live" ? "LIVE API · 4 FEEDS" : "DEMO FEEDS"}
            </span>
          </div>
        }
      />

      {/* stat tiles */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile
          label="Live temperature" tone="marigold" live delay={0}
          value={<>{temp.toFixed(1)}<span className="text-[16px] text-[var(--dim)]">°C</span></>}
          sub={<span>Humidity <b className="text-[var(--ink)]">{humidity.toFixed(0)}%</b> · wind {wx.wind} km/h · {wx.station}</span>}
          spark={wx.series}
        />
        <StatTile
          label="APMC Solapur · modal" tone="gold" live delay={70}
          value={<>₹{Math.round((mandi[0]?.modal || 11850) + (temp - wx.temp) * 12).toLocaleString("en-IN")}<span className="text-[14px] text-[var(--dim)]">/qtl</span></>}
          sub={<span>{mandi[0]?.crop || "Pomegranate"} · <DeltaTag d={mandi[0]?.d7 ?? 1.2} /> 7d</span>}
          spark={[11400, 11550, 11480, 11620, 11700, 11780, mandi[0]?.modal || 11850]}
        />
        <StatTile
          label="IoT soil moisture" tone="water" live delay={140}
          value={<>{moisture.toFixed(1)}<span className="text-[16px] text-[var(--dim)]">%</span></>}
          sub={<span>vol % · band 30–45 <StatusChip tone="leaf">OPTIMAL</StatusChip></span>}
          spark={iot.series}
        />
        <StatTile
          label="Sentinel-2 NDVI" tone="leaf" live delay={210}
          value={ndvi.toFixed(3)}
          sub={<span>{sat.vigor} · EVI {sat.evi} · NDWI {sat.ndwi}</span>}
          spark={[0.61, 0.64, 0.66, 0.68, 0.7, 0.71, sat.ndvi]}
        />
      </div>

      {/* NDVI field + weather / satellite column */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mt-4">
        <Reveal delay={100} className="xl:col-span-2">
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
              <div>
                <div className="eyebrow">Vegetation map · 14 × 9 zones</div>
                <div className="font-display font-bold text-[17px] mt-1">Field NDVI grid — {sat.pass}</div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono2 text-[var(--dim)]">bare</span>
                <div className="h-[8px] w-[110px] rounded-full" style={{ background: "linear-gradient(90deg,#92683a,#ba9e4a,#7a9c42,#4a9438,#2a7630)" }} />
                <span className="text-[10px] font-mono2 text-[var(--dim)]">vigor</span>
              </div>
            </div>
            <div
              className="grid gap-[3px] select-none"
              style={{ gridTemplateColumns: "repeat(14, 1fr)" }}
              onMouseLeave={() => setCell(null)}
            >
              {NDVI_GRID.flatMap((row, r) =>
                row.map((v, c) => (
                  <div
                    key={`${r}-${c}`}
                    className="ndvi-cell aspect-square rounded-[3px]"
                    style={{ background: ndviColor(v), animationDelay: `${(r + c) * 18}ms` }}
                    onMouseEnter={() => setCell({ r, c, v })}
                  />
                ))
              )}
            </div>
            <div className="flex items-center justify-between mt-3 flex-wrap gap-2">
              <span className="text-[12px] text-[var(--mut)] font-mono2">
                {cell
                  ? <>zone <b className="text-[var(--ink)]">{String.fromCharCode(65 + cell.r)}{cell.c + 1}</b> · NDVI <b style={{ color: ndviColor(cell.v) }}>{cell.v.toFixed(2)}</b> {cell.v < 0.5 ? "· ⚠ low vigor — scout zone" : cell.v > 0.75 ? "· ✓ healthy canopy" : "· nominal"}</>
                  : "hover a zone for NDVI detail"}
              </span>
              <span className="text-[11px] text-[var(--dim)] font-mono2">⚠ Zone E3–E4 depression matches diagnosis #D-2214</span>
            </div>
          </Card>
        </Reveal>

        <div className="space-y-4">
          <Reveal delay={160}>
            <div className="relative rounded-[14px] overflow-hidden border border-[var(--line2)] h-[190px]">
              <img src="/field-aerial.jpg" alt="Satellite pass" className="absolute inset-0 w-full h-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-[#08130cdd] to-transparent" />
              <div className="sweep" />
              <div className="absolute top-3 left-3 flex gap-1.5">
                <StatusChip tone="water">SENTINEL-2B</StatusChip>
                <StatusChip tone="leaf">cloud 8%</StatusChip>
              </div>
              <div className="absolute bottom-3 left-4 right-4 flex items-end justify-between">
                <div>
                  <div className="font-mono2 text-[10px] tracking-[0.18em] text-[var(--water)]">LAST PASS 06:12 IST</div>
                  <div className="font-display font-bold text-[17px]">Orbital coverage OK</div>
                </div>
                <Gauge pct={ndvi * 100} label="NDVI" color="var(--leaf)" size={86} />
              </div>
            </div>
          </Reveal>

          <Reveal delay={220}>
            <Card className="p-4">
              <div className="flex items-center justify-between mb-2.5">
                <span className="eyebrow">LoRa gateway</span>
                <span className="font-mono2 text-[10.5px] text-[var(--dim)]">🔋 {iot.battery}%</span>
              </div>
              <div className="text-[13px] text-[var(--mut)] mb-3">{iot.gateway}</div>
              <div className="space-y-2.5">
                {[
                  { l: "Soil moisture", v: `${moisture.toFixed(1)}%`, p: moisture / 60, c: "var(--water)" },
                  { l: "Soil temp", v: `${iot.soilTemp}°C`, p: iot.soilTemp / 45, c: "var(--marigold)" },
                  { l: "EC", v: `${iot.ec} dS/m`, p: iot.ec / 4, c: "var(--leaf)" },
                ].map((s) => (
                  <div key={s.l} className="flex items-center gap-3">
                    <span className="text-[12px] text-[var(--dim)] w-[92px] flex-none">{s.l}</span>
                    <div className="flex-1 h-[7px] rounded-full bg-[rgba(154,205,162,0.1)] overflow-hidden">
                      <div className="h-full rounded-full bar-grow-x" style={{ width: `${Math.min(100, s.p * 100)}%`, background: s.c, animationDelay: "200ms" }} />
                    </div>
                    <span className="font-mono2 text-[12px] text-[var(--ink)] w-[64px] text-right">{s.v}</span>
                  </div>
                ))}
              </div>
            </Card>
          </Reveal>
        </div>
      </div>

      {/* weather advisory strip */}
      <Reveal delay={140}>
        <div className="mt-4 rounded-[12px] border border-[rgba(244,192,75,0.3)] bg-gradient-to-r from-[rgba(244,192,75,0.09)] to-transparent px-4 py-3 flex items-center gap-3 flex-wrap">
          <span className="text-[18px]">🌤️</span>
          <div className="text-[13.5px] flex-1 min-w-[240px]">
            <b className="text-[var(--gold)]">Spray window:</b> <span className="text-[var(--ink)]">{wx.advisory}</span>
            <span className="devnagari text-[var(--mut)] block text-[12.5px]">{wx.advisoryMr}</span>
          </div>
          <StatusChip tone="gold">next 72h clear</StatusChip>
        </div>
      </Reveal>

      {/* mandi table */}
      <Reveal delay={200}>
        <Card ticks className="p-4 mt-4 overflow-hidden">
          <span className="tick-b" />
          <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
            <div>
              <div className="eyebrow">Open data · data.gov.in</div>
              <div className="font-display font-bold text-[17px] mt-1">APMC Mandi price feeds</div>
            </div>
            <div className="flex gap-1.5">
              <StatusChip tone="leaf">Agmarknet ✅ key set</StatusChip>
              <StatusChip tone="water">e-NAM mirror</StatusChip>
            </div>
          </div>
          <div className="overflow-x-auto -mx-1">
            <table className="dtable min-w-[680px]">
              <thead>
                <tr>
                  <th>Commodity</th><th>Variety</th><th>Mandi</th>
                  <th className="!text-right">Modal ₹/qtl</th><th className="!text-right">Min – Max</th>
                  <th className="!text-right">Δ 7d</th><th>Source mode</th>
                </tr>
              </thead>
              <tbody>
                {/* live or demo rows */}
                {mandi.map((r) => (
                  <tr key={r.crop + r.mandi}>
                    <td className="!text-[var(--ink)] font-semibold">{r.crop}</td>
                    <td className="text-[12px]">{r.variety}</td>
                    <td className="text-[12px]">{r.mandi}</td>
                    <td className="!text-right font-mono2 !text-[var(--gold)] font-semibold">₹{r.modal.toLocaleString("en-IN")}</td>
                    <td className="!text-right font-mono2 text-[11.5px]">₹{r.min.toLocaleString("en-IN")} – ₹{r.max.toLocaleString("en-IN")}</td>
                    <td className="!text-right"><DeltaTag d={r.d7} /></td>
                    <td><span className={`chip ${r.mode.includes("live") ? "chip-leaf" : "chip-soil"} !text-[9px]`}>{r.mode}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Reveal>
    </div>
  );
}
