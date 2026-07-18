import { useEffect, useRef, useState, type ReactNode, type CSSProperties } from "react";

/* ---------------- hooks ---------------- */

export function useCountUp(target: number, duration = 900, decimals = 0): string {
  const [val, setVal] = useState(0);
  const raf = useRef(0);
  useEffect(() => {
    const start = performance.now();
    const from = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const e = 1 - Math.pow(1 - t, 3);
      setVal(from + (target - from) * e);
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);
  return val.toLocaleString("en-IN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export function useLiveJitter(base: number, amp: number, ms = 3200): number {
  const [v, setV] = useState(base);
  useEffect(() => {
    const id = setInterval(() => {
      setV(Math.round((base + (Math.random() - 0.5) * 2 * amp) * 100) / 100);
    }, ms);
    return () => clearInterval(id);
  }, [base, amp, ms]);
  return v;
}

export function useClock(): string {
  const [t, setT] = useState(() =>
    new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date())
  );
  useEffect(() => {
    const id = setInterval(
      () => setT(new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date())),
      1000
    );
    return () => clearInterval(id);
  }, []);
  return t;
}

/* ---------------- primitives ---------------- */

export function Reveal({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  return (
    <div className={`reveal ${className}`} style={{ animationDelay: `${delay}ms` } as CSSProperties}>
      {children}
    </div>
  );
}

export function Card({ children, className = "", ticks = false, hover = false }: { children: ReactNode; className?: string; ticks?: boolean; hover?: boolean }) {
  return (
    <div className={`card ${ticks ? "ticks" : ""} ${hover ? "hoverable" : ""} ${className}`}>
      {ticks && <span className="tick-b" />}
      {children}
    </div>
  );
}

export function SectionHead({ eyebrow, title, sub, right }: { eyebrow: string; title: ReactNode; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3 mb-5">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2 className="font-display text-[26px] leading-tight font-bold mt-1.5">{title}</h2>
        {sub && <p className="text-[13px] text-[var(--mut)] mt-1 max-w-xl">{sub}</p>}
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}

export function StatTile({
  label, value, sub, tone = "leaf", spark, delay = 0, live = false,
}: {
  label: string; value: ReactNode; sub?: ReactNode; tone?: "leaf" | "gold" | "water" | "chili" | "marigold";
  spark?: number[]; delay?: number; live?: boolean;
}) {
  const toneMap: Record<string, string> = {
    leaf: "var(--leaf)", gold: "var(--gold)", water: "var(--water)", chili: "var(--chili)", marigold: "var(--marigold)",
  };
  const c = toneMap[tone];
  return (
    <Reveal delay={delay}>
      <Card ticks className="p-4 h-full">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)]">{label}</span>
          {live && (
            <span className="flex items-center gap-1.5 text-[10px] font-mono2 text-[var(--leaf)] tracking-widest">
              <span className="livedot" /> LIVE
            </span>
          )}
        </div>
        <div className="mt-2 flex items-end justify-between gap-2">
          <div className="font-display font-bold text-[30px] leading-none tabnum" style={{ color: c }}>
            {value}
          </div>
          {spark && <Sparkline data={spark} color={c} />}
        </div>
        {sub && <div className="mt-2 text-[12px] text-[var(--mut)]">{sub}</div>}
      </Card>
    </Reveal>
  );
}

export function Sparkline({ data, color = "var(--leaf)", w = 84, h = 30 }: { data: number[]; color?: string; w?: number; h?: number }) {
  const min = Math.min(...data), max = Math.max(...data);
  const pts = data.map((v, i) => [ (i / (data.length - 1)) * w, h - 3 - ((v - min) / (max - min || 1)) * (h - 8) ]);
  const line = pts.map((p) => p.join(",")).join(" ");
  const area = `0,${h} ` + line + ` ${w},${h}`;
  return (
    <svg width={w} height={h} className="overflow-visible flex-none">
      <polygon points={area} fill={color} opacity={0.12} />
      <polyline points={line} fill="none" stroke={color} strokeWidth={1.8} strokeLinecap="round" className="draw-line" />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r={2.6} fill={color}>
        <animate attributeName="opacity" values="1;0.3;1" dur="1.6s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

export function Gauge({ pct, label, color = "var(--leaf)", size = 120 }: { pct: number; label: string; color?: string; size?: number }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const angle = -90 + (clamped / 100) * 180;
  const r = 46;
  const cx = 60, cy = 60;
  const arc = (from: number, to: number) => {
    const p = (a: number) => [cx + r * Math.cos((a * Math.PI) / 180), cy + r * Math.sin((a * Math.PI) / 180)];
    const [x1, y1] = p(from), [x2, y2] = p(to);
    return `M ${x1} ${y1} A ${r} ${r} 0 ${to - from > 180 ? 1 : 0} 1 ${x2} ${y2}`;
  };
  return (
    <svg width={size} height={size * 0.62} viewBox="0 0 120 74" className="overflow-visible">
      <path d={arc(180, 360)} fill="none" stroke="rgba(154,205,162,0.14)" strokeWidth={9} strokeLinecap="round" />
      <path d={arc(180, 180 + (clamped / 100) * 180)} fill="none" stroke={color} strokeWidth={9} strokeLinecap="round" style={{ transition: "all 1s cubic-bezier(0.3,1.2,0.4,1)" }} />
      <g className="needle" style={{ transform: `rotate(${angle + 90}deg)` }}>
        <line x1={60} y1={60} x2={60} y2={26} stroke="var(--ink)" strokeWidth={2} strokeLinecap="round" />
        <circle cx={60} cy={60} r={4} fill="var(--ink)" />
      </g>
      <text x={60} y={72} textAnchor="middle" fill="var(--mut)" fontSize={9} fontFamily="var(--font-mono)" letterSpacing={1.5}>
        {label}
      </text>
    </svg>
  );
}

export function RadialPct({ pct, size = 92, color = "var(--gold)", track = "rgba(154,205,162,0.14)", label }: { pct: number; size?: number; color?: string; track?: string; label?: string }) {
  const r = 40, c = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" className="flex-none">
      <circle cx={50} cy={50} r={r} fill="none" stroke={track} strokeWidth={8} />
      <circle
        cx={50} cy={50} r={r} fill="none" stroke={color} strokeWidth={8} strokeLinecap="round"
        strokeDasharray={c} strokeDashoffset={c * (1 - pct)} transform="rotate(-90 50 50)"
        style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(0.3,1.2,0.4,1)" }}
      />
      <text x={50} y={50} textAnchor="middle" dominantBaseline="central" fill="var(--ink)" fontSize={22} fontWeight={700} fontFamily="var(--font-display)">
        {Math.round(pct * 100)}
        <tspan fontSize={11} fill="var(--dim)">%</tspan>
      </text>
      {label && (
        <text x={50} y={94} textAnchor="middle" fontSize={8} fill="var(--dim)" fontFamily="var(--font-mono)" letterSpacing={1}>
          {label}
        </text>
      )}
    </svg>
  );
}

export function BarRows({ rows, unit = "%", color = "var(--leaf)" }: { rows: { label: string; value: number }[]; unit?: string; color?: string | ((i: number) => string) }) {
  const max = Math.max(...rows.map((r) => r.value), 1);
  return (
    <div className="space-y-2.5">
      {rows.map((r, i) => (
        <div key={r.label} className="group">
          <div className="flex justify-between text-[11.5px] mb-1">
            <span className="text-[var(--mut)] group-hover:text-[var(--ink)] transition-colors">{r.label}</span>
            <span className="font-mono2 text-[var(--dim)]">{r.value}{unit}</span>
          </div>
          <div className="h-[7px] rounded-full bg-[rgba(154,205,162,0.1)] overflow-hidden">
            <div
              className="h-full rounded-full bar-grow-x"
              style={{
                width: `${(r.value / max) * 100}%`,
                background: typeof color === "function" ? color(i) : color,
                animationDelay: `${i * 70}ms`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function DeltaTag({ d, suffix = "%" }: { d: number; suffix?: string }) {
  const up = d > 0;
  const flat = d === 0;
  return (
    <span
      className="font-mono2 text-[11px] px-1.5 py-0.5 rounded-md"
      style={{
        color: flat ? "var(--dim)" : up ? "var(--leaf)" : "var(--chili)",
        background: flat ? "rgba(154,205,162,0.07)" : up ? "rgba(143,217,108,0.1)" : "rgba(242,95,88,0.1)",
      }}
    >
      {flat ? "•" : up ? "▲" : "▼"} {Math.abs(d).toFixed(1)}{suffix}
    </span>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <label className="flex items-center gap-2.5 cursor-pointer select-none">
      <span className="switch">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="track" />
        <span className="knob" />
      </span>
      {label && <span className="text-[13px] text-[var(--mut)]">{label}</span>}
    </label>
  );
}

export function StatusChip({ tone = "leaf", children }: { tone?: "leaf" | "gold" | "water" | "chili" | "marigold" | "soil" | "muted"; children: ReactNode }) {
  const cls =
    tone === "muted" ? "chip" :
    tone === "gold" ? "chip chip-gold" :
    tone === "water" ? "chip chip-water" :
    tone === "chili" ? "chip chip-chili" :
    tone === "marigold" ? "chip chip-marigold" :
    tone === "soil" ? "chip chip-soil" : "chip chip-leaf";
  return <span className={cls}>{children}</span>;
}

export function Stitch() {
  return <hr className="stitch my-6" />;
}
