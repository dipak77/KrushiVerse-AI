import streamlit as st
import pandas as pd
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.planner import planner_agent
from app.live_feeds.weather_feed import weather_feed
from app.live_feeds.market_feed import market_feed
from app.live_feeds.iot_feed import iot_feed
from app.live_feeds.satellite_feed import satellite_feed
from app.live_feeds.opendata_client import opendata_client
from app.vision.disease_classifier import vision_classifier
from app.vision.ocr_processor import ocr_processor
from app.predictive.yield_model import yield_model
from app.predictive.irrigation_model import irrigation_model
from app.predictive.fertilizer_planner import fertilizer_planner
from app.knowledge.graph_rag import graph_rag
from app.knowledge.advanced_rag import advanced_rag
from app.knowledge.dataset_loader import kb_loader
from app.knowledge.hybrid_search import hybrid_retriever
from app.knowledge.embeddings import embedding_provider
from app.memory.farm_memory import farm_memory_store
from app.workflows.automation import workflow_engine
from app.knowledge.tools.registry import tool_registry
from mini.taxonomy.service import taxonomy_service

st.set_page_config(
    page_title="AI Krushi Mitra — KrushiVerse-AI Platform",
    page_icon="🌾",
    layout="wide",
)

# Primary OS UI is the React app in ui/web (reference design).
# Streamlit remains available for factory/dev operators.
st.sidebar.markdown(
    """
**🖥️ Primary UI (new design)**  
Build: `cd ui/web && npm install && npm run build`  
Open: [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)  
Dev: `npm run dev` in `ui/web` (proxies `/api` → :8000)
"""
)

# ============================================================================
# DESIGN SYSTEM
# Palette: deep indigo (--ink/--indigo), marigold saffron (--saffron), and
# vegetation leaf-green (--leaf) — a nod to dawn-to-dusk over a plowed field.
# The "furrow" gradient (indigo -> saffron -> leaf) is the page's signature
# motif and recurs as card borders, the hero footer strip, and section
# dividers. Note: this styles Streamlit's generated DOM via data-testid
# selectors, which can shift slightly between Streamlit versions — if a
# style doesn't seem to apply after an upgrade, inspect the element and
# adjust the matching selector below.
# ============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');

:root {
    --ink: #1E1B4B;
    --ink-soft: #57534E;
    --indigo: #4F46E5;
    --indigo-dark: #3730A3;
    --saffron: #F59E0B;
    --saffron-dark: #D97706;
    --leaf: #16A34A;
    --terracotta: #C2410C;
    --cream: #FBF7EF;
    --card: #FFFFFF;
    --line: #E7E1D3;
    --furrow: linear-gradient(90deg, var(--indigo) 0%, var(--saffron) 55%, var(--leaf) 100%);
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
}

.stApp {
    background:
        repeating-linear-gradient(180deg, rgba(79,70,229,0.025) 0px, rgba(79,70,229,0.025) 1px, transparent 1px, transparent 34px),
        var(--cream);
}

h1, h2, h3 {
    font-family: 'Baloo 2', 'Noto Sans Devanagari', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}

/* ---------- Thinker Mode Banner ---------- */
.thinker-banner {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: linear-gradient(90deg, rgba(79, 70, 229, 0.12) 0%, rgba(245, 158, 11, 0.08) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-left: 4px solid var(--saffron);
    border-radius: 14px;
    padding: 0.85rem 1.2rem;
    margin: 0.8rem 0;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.12);
}
.thinker-pulse {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: var(--saffron);
    box-shadow: 0 0 0 rgba(245, 158, 11, 0.7);
    animation: thinkerPulse 1.4s infinite;
    flex-shrink: 0;
}
@keyframes thinkerPulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}
.thinker-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink);
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(120deg, var(--ink) 0%, var(--indigo-dark) 55%, var(--indigo) 100%);
    border-radius: 20px;
    padding: 2rem 2.25rem 1.6rem 2.25rem;
    margin-bottom: 0.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px -12px rgba(30,27,75,0.45);
}
.hero::after {
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 6px;
    background: var(--furrow);
}
.hero-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.09em;
    color: var(--saffron);
    background: rgba(245,158,11,0.14);
    border: 1px solid rgba(245,158,11,0.35);
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.7rem;
}
.hero-title {
    font-family: 'Baloo 2', 'Noto Sans Devanagari', sans-serif !important;
    color: #FFFFFF !important;
    font-size: 2.3rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.35rem 0 !important;
}
.hero-sub {
    color: #D8D6F5;
    font-size: 0.98rem;
    max-width: 780px;
    margin: 0;
    line-height: 1.5;
}

/* ---------- Field divider (signature motif) ---------- */
.field-divider {
    height: 4px;
    border-radius: 999px;
    background: var(--furrow);
    opacity: 0.55;
    margin: 1.1rem 0 1.4rem 0;
}

/* ---------- Stat cards ---------- */
.stat-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.1rem 0.9rem 1.1rem;
    box-shadow: 0 1px 2px rgba(30,27,75,0.04);
    border-top: 4px solid var(--indigo);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    min-height: 104px;
    margin-bottom: 0.4rem;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px -8px rgba(30,27,75,0.25);
}
.stat-card.accent-indigo { border-top-color: var(--indigo); }
.stat-card.accent-saffron { border-top-color: var(--saffron); }
.stat-card.accent-leaf { border-top-color: var(--leaf); }
.stat-card.accent-terracotta { border-top-color: var(--terracotta); }
.stat-icon { font-size: 1.3rem; margin-bottom: 0.15rem; }
.stat-value {
    font-family: 'Baloo 2', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.15;
    word-break: break-word;
}
.stat-label {
    font-size: 0.78rem;
    color: var(--ink-soft);
    margin-top: 0.15rem;
}

/* ---------- Badges ---------- */
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-weight: 500;
}
.badge-success { background: rgba(22,163,74,0.12); color: var(--leaf); border: 1px solid rgba(22,163,74,0.3); }
.badge-warn { background: rgba(217,119,6,0.12); color: var(--saffron-dark); border: 1px solid rgba(217,119,6,0.3); }
.badge-neutral { background: rgba(79,70,229,0.1); color: var(--indigo); border: 1px solid rgba(79,70,229,0.25); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--ink) 0%, #241F5E 100%);
}
section[data-testid="stSidebar"] * {
    color: #EDEBFB !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
.id-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-left: 4px solid var(--saffron);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}
.id-card-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.65rem; }
.id-avatar {
    width: 40px; height: 40px; border-radius: 50%;
    background: var(--furrow);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Baloo 2', sans-serif; font-weight: 700; color: #1E1B4B;
    flex-shrink: 0;
}
.id-name { font-weight: 600; font-size: 0.95rem; }
.id-sub { font-size: 0.72rem; color: #B9B6E8; font-family: 'JetBrains Mono', monospace; }
.id-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.chip {
    font-size: 0.68rem;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.14);
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
}
.sys-row {
    font-size: 0.8rem;
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.28rem 0;
    color: #D8D6F5;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(79,70,229,0.06);
    padding: 6px;
    border-radius: 14px;
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: var(--ink-soft);
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--ink) !important;
    box-shadow: 0 2px 8px rgba(30,27,75,0.12);
    font-weight: 600;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(120deg, var(--indigo) 0%, var(--indigo-dark) 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79,70,229,0.35);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, var(--saffron-dark) 0%, var(--saffron) 100%);
    box-shadow: 0 2px 8px rgba(245,158,11,0.35);
}

/* ---------- Inputs ---------- */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border: 1px solid var(--line) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.2) !important;
}

/* ---------- Native metrics (kept for less-prominent tabs) ---------- */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-top: 4px solid var(--indigo);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 2px rgba(30,27,75,0.04);
}
[data-testid="stMetricLabel"] { color: var(--ink-soft) !important; }
[data-testid="stMetricValue"] { font-family: 'Baloo 2', sans-serif !important; color: var(--ink) !important; }

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    background: var(--card);
}

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--line);
}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- Result / diagnostic cards ---------- */
.diag-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 5px solid var(--leaf);
    border-radius: 12px;
    padding: 0.85rem 1.05rem;
    margin-bottom: 0.6rem;
}
.diag-card.chem { border-left-color: var(--terracotta); }
.diag-card.neutral { border-left-color: var(--indigo); }
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# ============================================================================
# REUSABLE COMPONENTS
# ============================================================================

def render_stat_cards(items):
    """Render a row of stat cards. items: list of dicts with icon/label/value/accent."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        accent = item.get("accent", "indigo")
        col.markdown(
            f'<div class="stat-card accent-{accent}">'
            f'<div class="stat-icon">{item["icon"]}</div>'
            f'<div class="stat-value">{item["value"]}</div>'
            f'<div class="stat-label">{item["label"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def badge(text, kind="neutral"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def field_divider():
    st.markdown('<div class="field-divider"></div>', unsafe_allow_html=True)


def diag_card(title, body_html, variant=""):
    css_class = f"diag-card {variant}".strip()
    st.markdown(
        f'<div class="{css_class}"><b>{title}</b><br>{body_html}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# HERO
# ============================================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">GEN 10.2 &nbsp;·&nbsp; MINI SPRINT 9</div>
        <h1 class="hero-title">🌾 AI Krushi Mitra</h1>
        <p class="hero-sub">Autonomous AI Agriculture Platform — domain tokenizer · multi-source RAG ·
        GraphRAG · open data · agents · predictive models</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar - Farm Profile
farm_id = st.sidebar.text_input("Farm ID", value="FARM_101")
farm_info = farm_memory_store.get_farm(farm_id) or farm_memory_store.get_farm("FARM_101")

_farmer_name = farm_info["farmer_name"]
_village = farm_info["location"]["village"]
_district = farm_info["location"]["district"]
_crop_mr = farm_info["current_crop"]["crop_name_mr"]
_crop_en = farm_info["current_crop"]["crop_name"]
_acres = farm_info["land_area_acres"]
_initials = "".join([w[0] for w in _farmer_name.split()[:2]]).upper() or "🌾"

st.sidebar.markdown(
    f"""
    <div class="id-card">
        <div class="id-card-row">
            <div class="id-avatar">{_initials}</div>
            <div>
                <div class="id-name">{_farmer_name}</div>
                <div class="id-sub">🆔 {farm_id}</div>
            </div>
        </div>
        <div class="id-chips">
            <span class="chip">📍 {_village}, {_district}</span>
            <span class="chip">🌱 {_crop_en} / {_crop_mr}</span>
            <span class="chip">📐 {_acres} acres</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

lang_option = st.sidebar.radio("Response Language / भाषा", ["Marathi (मराठी)", "English"])
lang_code = "mr" if "Marathi" in lang_option else "en"

enable_web = st.sidebar.checkbox("Enable Web RAG", value=True)
from app.config import settings as _app_settings

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 KrushiVerse-AI LLM (v2-12M-fixed)")
use_local_model = st.sidebar.checkbox(
    "Use Local KrushiVerse-AI Model",
    value=True,
    help="Primary local LLM (v2-12M-fixed: 10 layers, 8192 vocab, 512 block). Auto failover if online LLM call fails.",
)
st.sidebar.caption(
    f"Model variant: **v2-12M-fixed** · Env flag USE_MINI_LLM = **{_app_settings.USE_MINI_LLM}**"
)
use_mini_panel = st.sidebar.checkbox(
    "Prefer Mini panel chat (POST /api/mini/chat path)",
    value=bool(_app_settings.USE_MINI_LLM),
    help="Uses Mini+RAG grounded chain. Planner uses Local Model for synthesis.",
)
st.sidebar.markdown("---")
st.sidebar.subheader("KB / Embeddings")
_stats = kb_loader.knowledge_stats()
st.sidebar.metric("Indexed docs", _stats["total_documents"])

_emb = embedding_provider.info()
_emb_backend = _emb["backend"]
_emb_dim = _emb["dim"]
st.sidebar.markdown(
    f'<div class="sys-row"><span>Embedding</span>{badge(f"{_emb_backend} · dim={_emb_dim}", "neutral")}</div>',
    unsafe_allow_html=True,
)

_hyb = hybrid_retriever.backend_info()
_dense_backend = _hyb.get("dense_backend") or "—"
st.sidebar.markdown(
    f'<div class="sys-row"><span>Dense backend</span>{badge(_dense_backend, "neutral")}</div>',
    unsafe_allow_html=True,
)

_od = opendata_client.status()
_od_kind = "success" if _od.get("configured") else "warn"
_od_text = "✅ key set" if _od.get("configured") else "⚠️ local fallback"
st.sidebar.markdown(
    f'<div class="sys-row"><span>Agmarknet live</span>{badge(_od_text, _od_kind)}</div>',
    unsafe_allow_html=True,
)
if not _od.get("configured"):
    st.sidebar.caption("Set DATA_GOV_IN_API_KEY to enable live prices.")

# Sidebar taxonomy browser (read-only) — Sprint 1 demo
st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Taxonomy (S1 frozen)")
_tax = taxonomy_service.summary()
_tax_version_label = "v" + str(_tax["version"])
_tax_status_label = _tax["status"]
_tax_crops_label = str(_tax["crops"])
st.sidebar.markdown(
    f'<div class="sys-row"><span>Version</span>{badge(_tax_version_label, "neutral")}</div>'
    f'<div class="sys-row"><span>Status</span>{badge(_tax_status_label, "success")}</div>'
    f'<div class="sys-row"><span>Crops</span>{badge(_tax_crops_label, "neutral")}</div>',
    unsafe_allow_html=True,
)
_tax_cat = st.sidebar.selectbox("Category", taxonomy_service.categories())
_tax_crop = st.sidebar.selectbox("Crop", taxonomy_service.crop_names())
_crop_rec = taxonomy_service.get_crop_record(_tax_crop)
if _crop_rec:
    st.sidebar.markdown(
        f"**{_crop_rec['name_en']}** / {_crop_rec.get('name_mr')} / {_crop_rec.get('name_hi')}"
    )
    st.sidebar.caption(f"Group: {_crop_rec.get('group')} · {_crop_rec.get('scientific', '')}")
_aliases = taxonomy_service.crop_aliases().get(_tax_crop, [])
if _aliases:
    st.sidebar.caption("Aliases: " + ", ".join(_aliases[:8]) + ("…" if len(_aliases) > 8 else ""))

tabs = st.tabs([
    "🤖 AI Krushi Assistant",
    "🧠 Mini LLM + Citations",
    "📚 Advanced RAG & Sources",
    "🗂️ Domain Taxonomy",
    "🏭 Data Factory",
    "📸 Vision Disease Diagnostic",
    "📡 Live RAG (Weather, Market, IoT, Satellite)",
    "🧪 Soil & Fertilizer Planner",
    "🕸️ GraphRAG Knowledge Explorer",
    "📊 Predictive AI & Workflows",
])

# TAB 1: AI Krushi Assistant
with tabs[0]:
    st.header("💬 AI Krushi Assistant / एआय कृषी मित्र संभाषण")
    st.markdown("##### 💡 Quick Suggestions / नमुना प्रश्न सुचवणी (1-Click Pill Selection)")

    sample_options = [
        "🔴 डाळिंब तेल्या रोगासाठी कोणते औषध फवारावे?",
        "🌾 ऊसासाठी हवामान व सेंद्रिय खत सल्ला",
        "🧅 कांदा जांभळा करपा नियंत्रण उपाय",
        "🌿 कपाशीवरील गुलाबी बोंड अळी नियंत्रण व पीक विमा",
        "💧 ठिबक सिंचन शासकीय अनुदान व योजना लाभ",
    ]

    selected_pill = st.pills(
        "नमुना प्रश्न निवड:",
        options=sample_options,
        selection_mode="single",
        label_visibility="collapsed",
    )

    query_input = st.text_area(
        "Your Query / तुमचा प्रश्न:",
        value=selected_pill if selected_pill else "",
        placeholder="उदा. डाळिंबावरील तेल्या रोगासाठी कोणते औषध फवारावे?...",
        height=90,
    )

    if st.button("🚀 Ask AI Krushi Assistant / उत्तर मिळवा", type="primary"):
        if query_input.strip():
            thinker_placeholder = st.empty()
            thinker_placeholder.markdown(
                """
                <div class="thinker-banner">
                    <span class="thinker-pulse"></span>
                    <span class="thinker-title">🧠 Thinker Mode: Multi-Source RAG & Agent Network Coordinating...</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            res = planner_agent.plan_and_execute(
                query=query_input,
                farm_id=farm_id,
                language=lang_code,
                enable_web=enable_web,
                use_local_llm=use_local_model,
            )
            st.session_state["last_assistant_result"] = res
            thinker_placeholder.empty()

            field_divider()
            st.markdown(res["synthesized_answer"])

            kl = res.get("knowledge_layer") or {}
            render_stat_cards([
                {"icon": "🌾", "label": "Crop resolved", "value": res.get("crop", "—"), "accent": "leaf"},
                {"icon": "📚", "label": "Fused docs", "value": kl.get("fused_document_count", 0), "accent": "indigo"},
                {"icon": "🌐", "label": "Web hits", "value": kl.get("web_result_count", 0), "accent": "saffron"},
                {"icon": "🛠️", "label": "Tools used", "value": len(kl.get("tools_used") or []), "accent": "terracotta"},
            ])

            st.markdown("#### 📎 Citations / Sources")
            cites = kl.get("citations") or []
            if cites:
                for i, c in enumerate(cites[:10], 1):
                    url = c.get("url") or ""
                    title = c.get("title") or c.get("source")
                    origin = c.get("origin")
                    source = c.get("source")
                    link_html = f' · <a href="{url}" target="_blank">Open source ↗</a>' if url else ""
                    diag_card(
                        f"{i}. {title}",
                        f'<span class="stat-label">{origin} / {source}</span>{link_html}',
                        variant="neutral",
                    )
            else:
                st.info("No citations attached for this answer.")

            if res.get("use_mini_llm") or res.get("synthesizer") == "mini_llm":
                st.success(f"Synthesizer: Mini LLM (USE_MINI_LLM={res.get('use_mini_llm')})")
            else:
                st.caption(f"Synthesizer: {res.get('synthesizer') or 'template'} (flag off = classic)")

            with st.expander("🔍 Multi-Agent plan & knowledge layer JSON"):
                st.json({
                    "active_agents": res["active_agent_names"],
                    "knowledge_layer": res["knowledge_layer"],
                    "raw_agent_outputs": res["agent_outputs"],
                    "synthesizer": res.get("synthesizer"),
                    "use_mini_llm": res.get("use_mini_llm"),
                })
        else:
            st.warning("Please type a query or select a sample query above.")

# TAB 2: Mini LLM + Citations (Sprint 16)
with tabs[1]:
    st.header("🧠 Mini LLM Assistant (Sprint 16 / FP-9)")
    st.write(
        "Grounded Mini+RAG chat with citations. Uses `run_mini_chat` / `POST /api/mini/chat`. "
        "Planner synthesizer still respects env `USE_MINI_LLM` (default off)."
    )
    st.info(
        f"USE_MINI_LLM={_app_settings.USE_MINI_LLM} · mode={_app_settings.MINI_DEFAULT_MODE} · "
        f"model={_app_settings.MINI_MODEL_VERSION}"
    )
    mini_q = st.text_area(
        "Mini query",
        value="How do I manage pink bollworm in cotton with IPM in Maharashtra?",
        height=90,
        key="mini_llm_query",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        mini_mode = st.selectbox("Mode", ["grounded", "open"], index=0, key="mini_mode")
    with c2:
        mini_crop = st.text_input("Crop hint", value="Cotton", key="mini_crop")
    with c3:
        mini_agents = st.checkbox("Enable agent notes", value=True, key="mini_agents")

    if st.button("🚀 Ask Mini", type="primary", key="mini_ask_btn"):
        if mini_q.strip():
            with st.spinner("Mini inference: intent → RAG → generate → validate..."):
                from app.llm.mini_bridge import run_mini_chat

                mini_res = run_mini_chat(
                    mini_q.strip(),
                    language=lang_code,
                    crop=mini_crop or None,
                    location=farm_info["location"].get("district", "Pune"),
                    mode=mini_mode,
                    enable_web=enable_web if use_mini_panel else False,
                    enable_agents=mini_agents,
                )
                st.session_state["last_mini_result"] = mini_res

            mini_res = st.session_state.get("last_mini_result") or {}
            st.markdown("### Answer")
            st.markdown(mini_res.get("answer") or mini_res.get("synthesized_answer") or "—")
            render_stat_cards(
                [
                    {
                        "icon": "⚙️",
                        "label": "Engine",
                        "value": mini_res.get("engine") or "—",
                        "accent": "indigo",
                    },
                    {
                        "icon": "📚",
                        "label": "Sources",
                        "value": mini_res.get("n_sources") or 0,
                        "accent": "leaf",
                    },
                    {
                        "icon": "🔁",
                        "label": "Fallback",
                        "value": str(mini_res.get("used_fallback")),
                        "accent": "saffron",
                    },
                    {
                        "icon": "✅",
                        "label": "OK",
                        "value": str(mini_res.get("ok")),
                        "accent": "terracotta",
                    },
                ]
            )
            st.markdown("#### Citations")
            cites = mini_res.get("citations") or []
            if cites:
                for i, c in enumerate(cites[:10], 1):
                    st.markdown(
                        f"**{c.get('marker') or f'[{i}]'}** {c.get('title')} — "
                        f"`{c.get('origin')}`"
                    )
                    if c.get("text"):
                        st.caption((c.get("text") or "")[:240])
            else:
                st.warning("No citations (grounded mode should refuse empty sources).")
            with st.expander("Mini JSON"):
                st.json(mini_res)
        else:
            st.warning("Enter a query for Mini.")

# TAB 3: Advanced RAG & Sources
with tabs[2]:
    st.header("📚 Advanced Multi-Source RAG Explorer")
    st.write(
        "Run hybrid + dense (Qdrant/local) + GraphRAG + tools + web retrieval. "
        "Inspect fused documents, citations, embedding backends, and Agmarknet open data."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        rag_query = st.text_input(
            "RAG query",
            value="Cotton pink bollworm organic control Maharashtra market price",
        )
        rag_crop = st.text_input("Crop hint (optional)", value="Cotton")
    with col_b:
        force_web = st.checkbox("Force web search", value=True)
        enable_tools = st.checkbox("Enable tools", value=True)
        top_k = st.slider("Top-K fused docs", 3, 15, 8)

    if st.button("🔎 Run Advanced RAG", type="primary"):
        with st.spinner("Multi-source retrieval in progress..."):
            rag = advanced_rag.retrieve(
                rag_query,
                crop=rag_crop or None,
                location=farm_info["location"].get("district", "Pune"),
                top_k=top_k,
                enable_web=enable_web or force_web,
                enable_tools=enable_tools,
                force_web=force_web,
            )
            st.session_state["last_rag_result"] = rag

    rag = st.session_state.get("last_rag_result")
    if rag:
        field_divider()
        render_stat_cards([
            {"icon": "📂", "label": "Local hits", "value": rag.get("local_hit_count", 0), "accent": "indigo"},
            {"icon": "🧬", "label": "Fused docs", "value": len(rag.get("fused_documents") or []), "accent": "leaf"},
            {"icon": "🌐", "label": "Web results", "value": len(rag.get("web_results") or []), "accent": "saffron"},
            {"icon": "🛠️", "label": "Tools used", "value": len(rag.get("tools_used") or []), "accent": "terracotta"},
        ])

        st.subheader("Query plan")
        st.json(rag.get("query_plan"))

        st.subheader("Retrieval backends")
        st.json(rag.get("retrieval_backends"))

        st.subheader("Fused documents (ranked)")
        for i, d in enumerate(rag.get("fused_documents") or [], 1):
            with st.expander(f"[{i}] {d.get('title')} · score={d.get('fusion_score')} · {d.get('origin')}"):
                st.markdown(f"**Category:** {d.get('category')}  \n**Source:** {d.get('source')}")
                if d.get("url"):
                    st.markdown(f"**URL:** {d.get('url')}")
                st.write(d.get("content"))

        st.subheader("Citations")
        cite_rows = rag.get("citations") or []
        if cite_rows:
            st.dataframe(pd.DataFrame(cite_rows), use_container_width=True)
        else:
            st.info("No citations.")

        st.subheader("Web results")
        web_rows = rag.get("web_results") or []
        if web_rows:
            st.dataframe(pd.DataFrame(web_rows), use_container_width=True)
        else:
            st.caption("No web hits (disabled, offline, or empty).")

        st.subheader("Tools used")
        st.write(", ".join(rag.get("tools_used") or []) or "—")

        with st.expander("Raw RAG JSON"):
            st.json(rag)
    else:
        st.info("Run a query above to populate fused docs and citations.")

    field_divider()
    st.subheader("🏛 data.gov.in / Agmarknet open data")
    od1, od2, od3 = st.columns(3)
    commodity = od1.text_input("Commodity", value="Cotton")
    district = od2.text_input("District (optional)", value="")
    state = od3.text_input("State", value="Maharashtra")
    if st.button("Fetch Agmarknet prices"):
        prices = opendata_client.fetch_commodity_prices(
            state=state,
            district=district or None,
            commodity=commodity or None,
            limit=30,
        )
        st.json({"status": opendata_client.status(), "mode": prices.get("mode"), "count": prices.get("count"), "fallback_reason": prices.get("fallback_reason")})
        st.dataframe(pd.DataFrame(prices.get("records") or []), use_container_width=True)

    st.subheader("Knowledge base stats")
    st.json(_stats)
    st.subheader("Registered tools")
    st.dataframe(pd.DataFrame(tool_registry.list_tools()), use_container_width=True)

    field_divider()
    st.subheader("🗂 Data lake ingest (Sprint 2)")
    from mini.lake.registry import load_source_registry
    from mini.lake.ingest import lake_tree_summary
    from mini.workers.base import get_worker

    st.json(load_source_registry().summary())
    if st.button("Run W-INGEST (write to lake/raw)"):
        with st.spinner("Ingesting sources..."):
            res = get_worker("W-INGEST").run(dry_run=False, include_http=False)
        st.success(res.message if res.ok else "Ingest failed")
        st.json(res.metrics)
    st.caption("Lake raw inventory")
    st.json(lake_tree_summary())

    st.subheader("Quality pipeline (Sprint 3)")
    if st.button("Run validate → clean → dedup"):
        with st.spinner("Running quality pipeline..."):
            qres = get_worker("W-QUALITY").run(dry_run=False)
        st.success(qres.message if qres.ok else "Quality run finished with issues")
        st.json(
            {
                "validation": (qres.metrics or {}).get("validation", {}),
                "cleaning": {
                    k: (qres.metrics or {}).get("cleaning", {}).get(k)
                    for k in ("cleaned", "skipped", "failed")
                },
                "dedup": {
                    k: (qres.metrics or {}).get("dedup", {}).get(k)
                    for k in ("exact_removed", "near_removed", "records_kept", "files")
                },
            }
        )

    st.subheader("Standardize Schema v1 (Sprint 4)")
    if st.button("Export train/val/test StandardRecords"):
        with st.spinner("Standardizing..."):
            sres = get_worker("W-STANDARDIZE").run(dry_run=False)
        st.success(sres.message if sres.ok else "Standardize finished with issues")
        st.json(
            {
                "counts": (sres.metrics or {}).get("export", {}).get("counts"),
                "coverage": (sres.metrics or {}).get("coverage"),
                "version": (sres.metrics or {}).get("export", {}).get("version"),
            }
        )

    st.subheader("QA Synthesis (Sprint 6)")
    if st.button("Run W-QASYNTH (≥50k train target)"):
        with st.spinner("Synthesizing expert QA packs (may take 1–2 minutes)..."):
            qsyn = get_worker("W-QASYNTH").run(dry_run=False, target_min_total=62500)
        st.session_state["factory_qasynth"] = qsyn
        st.success(qsyn.message if qsyn.ok else "Synth finished — check targets")
        st.json(
            {
                "counts": (qsyn.metrics or {}).get("counts"),
                "targets_met": (qsyn.metrics or {}).get("targets_met"),
                "by_category": (qsyn.metrics or {}).get("by_category"),
                "by_language": (qsyn.metrics or {}).get("by_language"),
                "by_pack": (qsyn.metrics or {}).get("by_pack"),
                "version": (qsyn.metrics or {}).get("version"),
            }
        )

# TAB 3: Domain Taxonomy browser (Sprint 1)
with tabs[3]:
    st.header("🗂️ Domain Taxonomy Browser (Sprint 1 — Frozen v1.0)")
    st.write(
        "Single source of truth for categories, crops (EN/MR/HI), stages, regions, and units. "
        "Used by normalize workers and query understanding."
    )
    summ = taxonomy_service.summary()
    render_stat_cards([
        {"icon": "🏷️", "label": "Version", "value": summ["version"], "accent": "indigo"},
        {"icon": "🌾", "label": "Crops", "value": summ["crops"], "accent": "leaf"},
        {"icon": "📁", "label": "Categories", "value": summ["categories"], "accent": "saffron"},
        {"icon": "🗺️", "label": "MH districts", "value": summ["mh_districts"], "accent": "terracotta"},
    ])
    field_divider()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Categories")
        st.dataframe(pd.DataFrame(taxonomy_service.category_details()), use_container_width=True)
        st.subheader("Crop stages")
        st.dataframe(pd.DataFrame(taxonomy_service.stages()), use_container_width=True)
    with col_t2:
        st.subheader("Crops (EN / MR / HI)")
        st.dataframe(pd.DataFrame(taxonomy_service.crops()), use_container_width=True)
        st.subheader("Resolve demo")
        demo_q = st.text_input("Text to resolve", value="कापूस खत किती द्यावे Pune")
        st.json(
            {
                "crops": taxonomy_service.extract_crops(demo_q),
                "categories": taxonomy_service.detect_category(demo_q),
                "region": taxonomy_service.resolve_district("Pune"),
            }
        )

    st.subheader("Validation report")
    if st.button("Run taxonomy validation"):
        report = taxonomy_service.validate()
        if report.get("ok"):
            st.success("Taxonomy integrity + platform KB coverage OK")
        else:
            st.error("Validation failed — see report")
        st.json(report)

    with st.expander("Unit dimensions"):
        from mini.taxonomy.units import UNITS

        st.json(UNITS["preferred_display"])
        st.caption("Dimensions: " + ", ".join(UNITS["dimensions"].keys()))

# TAB 4: Data Factory (Sprint 5)
with tabs[4]:
    st.header("🏭 Mini Data Factory — Coverage Analysis (Sprint 5)")
    st.write(
        "Run W-ANALYZE after standardize to inspect missingness, language/crop balance, "
        "duplicates, and taxonomy coverage gaps."
    )
    from mini.paths import LAKE_ROOT
    import json as _json

    c1, c2, c3 = st.columns(3)
    if c1.button("1. Standardize export"):
        with st.spinner("Standardizing..."):
            sres = get_worker("W-STANDARDIZE").run(dry_run=False)
        st.session_state["factory_standard"] = sres
    if c2.button("2. Run W-ANALYZE", type="primary"):
        with st.spinner("Analyzing dataset..."):
            ares = get_worker("W-ANALYZE").run(dry_run=False)
        st.session_state["factory_analyze"] = ares
    if c3.button("Reload ANALYZE_LATEST"):
        p = LAKE_ROOT / "ANALYZE_LATEST.json"
        if p.exists():
            st.session_state["factory_report"] = _json.loads(p.read_text(encoding="utf-8"))
        else:
            st.warning("No ANALYZE_LATEST.json yet")

    ares = st.session_state.get("factory_analyze")
    report = st.session_state.get("factory_report")
    if ares and ares.metrics:
        report = ares.metrics
    if report:
        field_divider()
        summ = report.get("summary") or {}
        render_stat_cards([
            {"icon": "🧾", "label": "Records", "value": summ.get("total_records", 0), "accent": "indigo"},
            {"icon": "🕳️", "label": "Gaps", "value": summ.get("gap_count", 0), "accent": "terracotta"},
            {"icon": "🪞", "label": "Dup rate %", "value": summ.get("duplicate_rate_pct", 0), "accent": "saffron"},
            {"icon": "🌱", "label": "Missing crop %", "value": summ.get("missing_crop_pct", 0), "accent": "leaf"},
        ])

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Language balance")
            st.bar_chart(summ.get("language_balance") or {})
            st.subheader("Category balance")
            st.bar_chart(summ.get("category_balance") or {})
        with col_b:
            st.subheader("Length histograms (question)")
            qh = ((report.get("records") or {}).get("length") or {}).get("question", {}).get("histogram") or {}
            if qh:
                st.bar_chart(qh)
            st.subheader("Taxonomy gaps")
            gaps = (report.get("taxonomy_gaps") or {}).get("gaps") or []
            if gaps:
                st.dataframe(pd.DataFrame(gaps), use_container_width=True)
            else:
                st.info("No gaps listed")
            miss = (report.get("taxonomy_gaps") or {}).get("missing_crops") or []
            if miss:
                st.caption("Missing crops vs taxonomy: " + ", ".join(miss[:20]))

        with st.expander("Full analysis JSON"):
            st.json(report)
        html_p = LAKE_ROOT / "ANALYZE_LATEST.html"
        if html_p.exists():
            st.caption(f"HTML report: `{html_p}`")
    else:
        st.info("Run standardize then W-ANALYZE to populate the dashboard.")

# TAB 5: Vision
with tabs[5]:
    st.header("🔬 Computer Vision Plant Disease Classifier")
    st.write("Upload a crop leaf photo or select sample leaf image to perform diagnostic analysis.")

    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        uploaded_file = st.file_uploader("Upload Leaf Photo (JPG/PNG)", type=["jpg", "png", "jpeg"])
        crop_hint = st.selectbox("Crop Type", ["Pomegranate", "Cotton", "Soybean", "Onion", "Sugarcane", "Rice", "Wheat", "Tomato", "Chilli", "Grapes"])
        sample_diag_btn = st.button("Analyze Sample Leaf Photo")

    with col_v2:
        if uploaded_file or sample_diag_btn:
            filename = uploaded_file.name if uploaded_file else f"{crop_hint.lower()}_leaf.jpg"
            img_bytes = uploaded_file.read() if uploaded_file else None

            result = vision_classifier.diagnose_image(image_bytes=img_bytes, filename=filename, crop_hint=crop_hint)

            st.success("Analysis Complete!")
            _disease_mr = result["disease_identified_mr"]
            _disease_en = result["disease_identified_en"]
            _confidence_pct = int(result["confidence_score"] * 100)
            render_stat_cards([
                {"icon": "🩺", "label": "Detected disease", "value": f"{_disease_mr} ({_disease_en})", "accent": "terracotta"},
                {"icon": "📈", "label": "Confidence", "value": f"{_confidence_pct}%", "accent": "leaf"},
            ])

            diag_card("Symptoms", result["symptoms_mr"], variant="neutral")
            diag_card("🌿 Organic Control", result["organic_treatment"]["mr"], variant="")
            diag_card("🧪 Chemical Control", result["chemical_treatment"]["mr"], variant="chem")

# TAB 6: Live RAG
with tabs[6]:
    st.header("📡 Live RAG Intelligence Center")

    w_data = weather_feed.get_weather("Pune")
    m_data = market_feed.get_market_summary_for_crop("Pomegranate")
    iot_data = iot_feed.get_sensor_telemetry(farm_id)
    sat_data = satellite_feed.get_satellite_indices(farm_id)

    _temp = w_data["temperature_c"]
    _humidity = w_data["relative_humidity_pct"]
    _apmc_price = m_data.get("average_modal_price_rs_quintal", 10500)
    _apmc_mode = m_data.get("source_mode", "local")
    _soil_moisture = iot_data["sensors"]["soil_moisture_vol_pct"]
    _soil_status = iot_data["status"]["soil_moisture_status"]
    _ndvi = sat_data["indices"]["NDVI_normalized_difference_vegetation_index"]

    render_stat_cards([
        {"icon": "🌡️", "label": f"Live temp · humidity {_humidity}%", "value": f"{_temp}°C", "accent": "saffron"},
        {"icon": "🏷️", "label": f"APMC Solapur modal · {_apmc_mode}", "value": f"₹{_apmc_price}/qtl", "accent": "indigo"},
        {"icon": "💧", "label": f"IoT soil moisture · {_soil_status}", "value": f"{_soil_moisture}%", "accent": "leaf"},
        {"icon": "🛰️", "label": "Sentinel-2 NDVI · high vigor", "value": f"{_ndvi}", "accent": "terracotta"},
    ])
    field_divider()

    st.subheader("📊 APMC Mandi Price Feeds")
    st.dataframe(pd.DataFrame(market_feed.get_market_prices()), use_container_width=True)
    st.caption(f"Open data status: {opendata_client.status()}")

# TAB 7: Soil & Fertilizer
with tabs[7]:
    st.header("🧪 Soil Health Card OCR & Fertilizer Planner")

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.subheader("1. Soil Health Card Parameters")
        ocr_text = st.text_area(
            "Soil Card Text or Scanned Output:",
            value="pH: 7.2, EC: 0.45, Organic Carbon: 0.52%, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha",
        )
        extracted_soil = ocr_processor.process_soil_card(ocr_text)
        st.json(extracted_soil)

    with col_s2:
        st.subheader("2. Target Fertilizer Calculator")
        fert_crop = st.selectbox(
            "Crop for Fertigation Plan",
            ["Pomegranate", "Cotton", "Soybean", "Onion", "Sugarcane", "Rice", "Wheat", "Grapes", "Tomato"],
        )
        fert_acres = st.number_input("Acreage", value=2.5, step=0.5)

        if st.button("Calculate Fertilizer Bags"):
            p_res = fertilizer_planner.calculate_fertilizer_bags(
                crop=fert_crop,
                acreage=fert_acres,
                N_kg_ha=extracted_soil["extracted_parameters"]["nitrogen_kg_ha"],
                P_kg_ha=extracted_soil["extracted_parameters"]["phosphorus_kg_ha"],
                K_kg_ha=extracted_soil["extracted_parameters"]["potassium_kg_ha"],
            )

            st.success("Fertilizer Recommendation Generated!")
            f_bags = p_res["recommended_fertilizer_bags"]
            render_stat_cards([
                {"icon": "🧪", "label": "Urea (45kg bags)", "value": f_bags["Urea_45kg_bags"], "accent": "indigo"},
                {"icon": "🧪", "label": "DAP (50kg bags)", "value": f_bags["DAP_50kg_bags"], "accent": "saffron"},
                {"icon": "🧪", "label": "MOP (50kg bags)", "value": f_bags["MOP_50kg_bags"], "accent": "leaf"},
            ])

            diag_card("मराठी संदेश", p_res["application_schedule_mr"], variant="neutral")

# TAB 8: GraphRAG
with tabs[8]:
    st.header("🕸️ GraphRAG Agricultural Knowledge Graph")
    st.write("Explore graph relations between crops, pests, diseases, fertilizers, and government schemes.")

    crop_nodes = [n["id"] for n in kb_loader.graph_data.get("nodes", []) if n.get("label") == "Crop"]
    if not crop_nodes:
        crop_nodes = ["Pomegranate", "Cotton", "Soybean", "Onion"]
    selected_graph_crop = st.selectbox("Select Crop Entity", crop_nodes)
    ecosystem = graph_rag.get_crop_ecosystem(selected_graph_crop)
    st.json(ecosystem)

# TAB 9: Predictive
with tabs[9]:
    st.header("📊 Predictive AI Models & Automated Workflows")

    p1, p2 = st.columns(2)

    with p1:
        st.subheader("🔮 Crop Yield Forecasting")
        y_crop = st.selectbox("Yield Crop", ["Pomegranate", "Cotton", "Soybean", "Sugarcane", "Onion", "Rice", "Wheat"])
        y_acres = st.number_input("Land Acres", value=3.0)
        yp = yield_model.predict_yield(crop=y_crop, acreage=y_acres)
        render_stat_cards([
            {"icon": "🌾", "label": "Predicted total yield (qtl)", "value": yp["total_predicted_yield"], "accent": "leaf"},
            {"icon": "📏", "label": "Per acre (qtl)", "value": yp["predicted_yield_per_acre"], "accent": "indigo"},
        ])

    with p2:
        st.subheader("💧 Smart Irrigation Runtime Calculator")
        ir = irrigation_model.calculate_water_requirement(
            crop=y_crop, acreage=y_acres, temperature_c=30.0, humidity_pct=75.0
        )
        render_stat_cards([
            {"icon": "🚰", "label": "Daily water requirement (L/day)", "value": ir["total_farm_water_required_liters_day"], "accent": "indigo"},
            {"icon": "⏱️", "label": "Drip runtime (hrs/day)", "value": ir["drip_irrigation_schedule"]["drip_runtime_hours_per_day"], "accent": "saffron"},
        ])

    field_divider()
    st.subheader("⚡ Automated Farm Health Audit & Workflow Actions")
    if st.button("Run Automated Farm Health Audit"):
        wf_res = workflow_engine.run_farm_health_checks(farm_id)
        st.write("Automated Actions Triggered:")
        for act in wf_res["triggered_workflows"]:
            st.warning(f"🚨 **{act['trigger']}** — {act['message_mr']}")