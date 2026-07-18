import { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { Assistant } from "./tabs/Assistant";
import { LiveFeeds } from "./tabs/LiveFeeds";
import { Vision } from "./tabs/Vision";
import { Soil } from "./tabs/Soil";
import { Graph } from "./tabs/Graph";
import { Predictive } from "./tabs/Predictive";
import { DataFactory, RagExplorer, Taxonomy } from "./tabs/Platform";

const TABS = [
  { id: "assistant", icon: "💬", label: "AI Assistant", full: "AI Krushi Assistant" },
  { id: "live", icon: "📡", label: "Live Feeds", full: "Live RAG Intelligence Center" },
  { id: "vision", icon: "🔬", label: "Vision Lab", full: "Vision Disease Diagnostic" },
  { id: "soil", icon: "🧪", label: "Soil Lab", full: "Soil & Fertilizer Planner" },
  { id: "graph", icon: "🕸️", label: "GraphRAG", full: "GraphRAG Knowledge Explorer" },
  { id: "predict", icon: "📊", label: "Predictive AI", full: "Predictive AI & Workflows" },
  { id: "rag", icon: "📚", label: "RAG Explorer", full: "Advanced Multi-Source RAG" },
  { id: "taxonomy", icon: "🗂️", label: "Taxonomy", full: "Domain Taxonomy Browser" },
  { id: "factory", icon: "🏭", label: "Data Factory", full: "Mini Data Factory" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>("assistant");
  const [lang, setLang] = useState<"mr" | "en">("mr");
  const [webRag, setWebRag] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const active = TABS.find((t) => t.id === tab)!;

  return (
    <div className="h-full relative">
      <div className="app-bg" />

      {/* icon rail */}
      <nav className="hidden lg:flex fixed left-0 top-0 bottom-0 w-[72px] z-30 border-r border-[var(--line)] bg-[rgba(7,16,10,0.72)] backdrop-blur-md flex-col items-center py-4 gap-1.5">
        {/* brand */}
        <div className="relative mb-4 flex-none">
          <span className="sun-ring spin-slow" />
          <div className="w-[44px] h-[44px] rounded-[13px] bg-gradient-to-b from-[#f8d070] to-[#d99a26] flex items-center justify-center text-[22px] shadow-[0_6px_18px_-6px_rgba(244,192,75,0.7)]">
            🌾
          </div>
        </div>

        {TABS.map((t) => (
          <button key={t.id} className={`rail-btn ${tab === t.id ? "on" : ""}`} onClick={() => setTab(t.id)} aria-label={t.label}>
            <span>{t.icon}</span>
            <span className="tip">{t.label}</span>
          </button>
        ))}

        <div className="mt-auto flex flex-col items-center gap-3">
          <button
            className="rail-btn !w-[38px] !h-[38px] !text-[16px]"
            onClick={() => setSidebarOpen((s) => !s)}
            aria-label="Toggle farm context"
          >
            <span>{sidebarOpen ? "⟨" : "⟩"}</span>
            <span className="tip">Farm context</span>
          </button>
          <div className="flex flex-col items-center gap-1">
            <span className="livedot gold" />
            <span className="font-mono2 text-[8.5px] text-[var(--dim)] tracking-widest">v10.2·S10</span>
          </div>
        </div>
      </nav>

      {/* main area */}
      <div className="lg:ml-[72px] h-full flex relative z-10">
        {sidebarOpen && (
          <div className="hidden lg:block h-full">
            <Sidebar lang={lang} setLang={setLang} webRag={webRag} setWebRag={setWebRag} />
          </div>
        )}

        <div className="flex-1 min-w-0 h-full flex flex-col">
          <TopBar tabLabel={active.full} />
          <main className="flex-1 overflow-y-auto">
            <div className="max-w-[1280px] mx-auto px-5 lg:px-7 py-6 pb-24 lg:pb-6">
              {/* tab headline strip */}
              <div className="mb-5 flex items-center gap-3 flex-wrap reveal" key={`head-${tab}`}>
                <span className="text-[26px] leading-none">{active.icon}</span>
                <span className="font-mono2 text-[11px] uppercase tracking-[0.22em] text-[var(--dim)]">
                  KrushiVerse OS <span className="text-[var(--gold)]">/</span> {active.label}
                </span>
                <span className="flex-1 h-px bg-gradient-to-r from-[var(--line2)] to-transparent min-w-[60px]" />
                <span className="chip chip-gold">data: /api/ui/* · FARM_101</span>
              </div>

              <div key={tab}>
                {tab === "assistant" && <Assistant lang={lang} webRag={webRag} />}
                {tab === "live" && <LiveFeeds />}
                {tab === "vision" && <Vision />}
                {tab === "soil" && <Soil />}
                {tab === "graph" && <Graph />}
                {tab === "predict" && <Predictive />}
                {tab === "rag" && <RagExplorer />}
                {tab === "taxonomy" && <Taxonomy />}
                {tab === "factory" && <DataFactory />}
              </div>

              <footer className="mt-10 mb-4 flex items-center gap-3 flex-wrap text-[11px] font-mono2 text-[var(--dim)]">
                <hr className="stitch flex-1 min-w-[120px] !my-0" />
                <span>KrushiVerse-AI · Gen 10.2 + Mini S10</span>
                <span>·</span>
                <span>React OS UI · multi-source RAG · GraphRAG · live FastAPI</span>
                <span>·</span>
                <span className="devnagari text-[var(--mut)]">शेतकऱ्यासाठी, मातीसाठी 🌾</span>
              </footer>
            </div>
          </main>
        </div>
      </div>

      {/* mobile tab bar */}
      <nav className="lg:hidden fixed bottom-0 inset-x-0 z-30 bg-[rgba(7,16,10,0.92)] backdrop-blur-md border-t border-[var(--line)] flex justify-around py-2 px-1 ml-0">
        {TABS.slice(0, 6).map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} className={`flex flex-col items-center gap-0.5 px-2 py-1 rounded-lg transition-all ${tab === t.id ? "bg-[rgba(244,192,75,0.12)]" : "opacity-55"}`}>
            <span className="text-[18px]">{t.icon}</span>
            <span className="text-[8.5px] font-mono2 tracking-wider" style={{ color: tab === t.id ? "var(--gold)" : "var(--dim)" }}>{t.label.split(" ")[0].toUpperCase()}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
