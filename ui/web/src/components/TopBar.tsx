import { useState } from "react";
import { useDashboard } from "../context/DashboardContext";
import { useClock } from "../lib/ui";

export function TopBar({ tabLabel }: { tabLabel: string }) {
  const clock = useClock();
  const { bootstrap, refresh } = useDashboard();
  const [syncing, setSyncing] = useState(false);
  const ticker = bootstrap?.ticker || [];

  const sync = async () => {
    setSyncing(true);
    try {
      await refresh();
    } finally {
      setTimeout(() => setSyncing(false), 600);
    }
  };

  const items = ticker.length ? [...ticker, ...ticker] : [];

  return (
    <header className="h-[58px] flex-none flex items-center gap-4 px-5 border-b border-[var(--line)] bg-[rgba(8,19,12,0.6)] backdrop-blur-sm relative z-10">
      <div className="flex items-center gap-3 min-w-0">
        <div>
          <div className="text-[10px] font-mono2 tracking-[0.2em] text-[var(--dim)] uppercase leading-none">KrushiVerse OS</div>
          <div className="font-display font-bold text-[15px] leading-tight truncate">{tabLabel}</div>
        </div>
      </div>

      <div className="flex-1 marquee-wrap hidden md:block">
        <div className="marquee">
          {items.map((t: any, i: number) => (
            <span key={i} className="inline-flex items-center gap-2 text-[12px]">
              <span>{t.e}</span>
              <span className="text-[var(--mut)]">{t.name}</span>
              <span className="text-[var(--dim)]">{t.mandi}</span>
              <span className="font-mono2 text-[var(--ink)]">₹{Number(t.price || 0).toLocaleString("en-IN")}</span>
              <span
                className="font-mono2"
                style={{ color: t.d > 0 ? "var(--leaf)" : t.d < 0 ? "var(--chili)" : "var(--dim)" }}
              >
                {t.d > 0 ? "▲" : t.d < 0 ? "▼" : "•"}
                {Math.abs(Number(t.d) || 0).toFixed(1)}%
              </span>
            </span>
          ))}
          {!items.length && <span className="text-[var(--dim)] font-mono2 text-[12px]">Loading market ticker from API…</span>}
        </div>
      </div>

      <div className="flex items-center gap-3 flex-none">
        <div className="hidden sm:flex items-center gap-1.5 chip chip-leaf !normal-case">
          <span className="livedot" /> API data
        </div>
        <div className="font-mono2 text-[12px] text-[var(--mut)] tabnum hidden sm:block">
          {clock} <span className="text-[var(--dim)]">IST</span>
        </div>
        <button onClick={sync} className="btn btn-ghost !px-3 !py-1.5 !text-[12px] !rounded-[9px]" title="Refresh bootstrap">
          <span className={syncing ? "inline-block animate-spin" : "inline-block"}>⟳</span>
          {syncing ? "syncing…" : "sync"}
        </button>
      </div>
    </header>
  );
}
