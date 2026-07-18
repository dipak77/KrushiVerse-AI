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
    layout="wide"
)

st.title("🌾 AI Krushi Mitra — Autonomous AI Agriculture Platform")
st.caption(
    "Gen 10.2 + Mini Sprint 1 — Frozen domain taxonomy · multi-source RAG · "
    "GraphRAG · open data · agents · predictive models"
)

# Sidebar - Farm Profile
st.sidebar.header("🏡 Farm Context & Memory")
farm_id = st.sidebar.text_input("Farm ID", value="FARM_101")
farm_info = farm_memory_store.get_farm(farm_id) or farm_memory_store.get_farm("FARM_101")

st.sidebar.markdown(f"**Farmer:** {farm_info['farmer_name']}")
st.sidebar.markdown(f"**Location:** {farm_info['location']['village']}, {farm_info['location']['district']}")
st.sidebar.markdown(f"**Current Crop:** {farm_info['current_crop']['crop_name_mr']} ({farm_info['current_crop']['crop_name']})")
st.sidebar.markdown(f"**Acreage:** {farm_info['land_area_acres']} Acres")

lang_option = st.sidebar.radio("Response Language / भाषा", ["Marathi (मराठी)", "English"])
lang_code = "mr" if "Marathi" in lang_option else "en"

enable_web = st.sidebar.checkbox("Enable Web RAG", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("KB / Embeddings")
_stats = kb_loader.knowledge_stats()
st.sidebar.metric("Indexed docs", _stats["total_documents"])
_emb = embedding_provider.info()
st.sidebar.caption(f"Embedding: `{_emb['backend']}` · dim={_emb['dim']}")
_hyb = hybrid_retriever.backend_info()
st.sidebar.caption(f"Dense: `{_hyb.get('dense_backend')}`")
_od = opendata_client.status()
st.sidebar.caption(
    "Agmarknet live: " + ("✅ key set" if _od.get("configured") else "⚠️ local fallback (set DATA_GOV_IN_API_KEY)")
)

# Sidebar taxonomy browser (read-only) — Sprint 1 demo
st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Taxonomy (S1 frozen)")
_tax = taxonomy_service.summary()
st.sidebar.caption(f"v{_tax['version']} · {_tax['status']} · {_tax['crops']} crops")
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
    "📚 Advanced RAG & Sources",
    "🗂️ Domain Taxonomy",
    "📸 Vision Disease Diagnostic",
    "📡 Live RAG (Weather, Market, IoT, Satellite)",
    "🧪 Soil & Fertilizer Planner",
    "🕸️ GraphRAG Knowledge Explorer",
    "📊 Predictive AI & Workflows",
])

# TAB 1: AI Krushi Assistant
with tabs[0]:
    st.header("💬 AI Krushi Assistant / एआय कृषी मित्र संभाषण")
    st.write("Ask any question regarding crop management, disease treatment, market rates, or government schemes.")

    sample_queries = [
        "डाळिंबावरील तेल्या रोगासाठी कोणते औषध फवारावे? बाजारात काय भाव चालू आहे?",
        "What fertilizers should I apply for Cotton in black soil?",
        "कपाशीवरील गुलाबी बोंड अळीचे नियंत्रण कसे करावे आणि पीक विमा कसा मिळेल?",
        "What are the top government schemes for drip irrigation subsidy in Maharashtra?",
        "Latest soybean mandi price in Maharashtra",
    ]

    selected_sample = st.selectbox("Sample Queries / नमुना प्रश्न:", ["-- Select a sample query --"] + sample_queries)
    query_input = st.text_area(
        "Your Query / तुमचा प्रश्न:",
        value="" if selected_sample == "-- Select a sample query --" else selected_sample,
        height=90,
    )

    if st.button("🚀 Ask AI Krushi Assistant / उत्तर मिळवा", type="primary"):
        if query_input.strip():
            with st.spinner("Planner + multi-agent network + multi-source RAG..."):
                res = planner_agent.plan_and_execute(
                    query=query_input,
                    farm_id=farm_id,
                    language=lang_code,
                    enable_web=enable_web,
                )
                st.session_state["last_assistant_result"] = res

            st.markdown("### 📝 Synthesized Marathi/English Response:")
            st.markdown(res["synthesized_answer"])

            kl = res.get("knowledge_layer") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Crop resolved", res.get("crop", "—"))
            c2.metric("Fused docs", kl.get("fused_document_count", 0))
            c3.metric("Web hits", kl.get("web_result_count", 0))
            c4.metric("Tools", len(kl.get("tools_used") or []))

            st.markdown("#### 📎 Citations / Sources")
            cites = kl.get("citations") or []
            if cites:
                for i, c in enumerate(cites[:10], 1):
                    url = c.get("url") or ""
                    line = f"**{i}. {c.get('title') or c.get('source')}** — `{c.get('origin')}` / {c.get('source')}"
                    if url:
                        line += f" · [link]({url})"
                    st.markdown(line)
            else:
                st.info("No citations attached for this answer.")

            with st.expander("🔍 Multi-Agent plan & knowledge layer JSON"):
                st.json({
                    "active_agents": res["active_agent_names"],
                    "knowledge_layer": res["knowledge_layer"],
                    "raw_agent_outputs": res["agent_outputs"],
                })
        else:
            st.warning("Please type a query or select a sample query above.")

# TAB 2: Advanced RAG & Sources
with tabs[1]:
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
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Local hits", rag.get("local_hit_count", 0))
        b2.metric("Fused docs", len(rag.get("fused_documents") or []))
        b3.metric("Web results", len(rag.get("web_results") or []))
        b4.metric("Tools used", len(rag.get("tools_used") or []))

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

    st.markdown("---")
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

    st.markdown("---")
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

# TAB 3: Domain Taxonomy browser (Sprint 1)
with tabs[2]:
    st.header("🗂️ Domain Taxonomy Browser (Sprint 1 — Frozen v1.0)")
    st.write(
        "Single source of truth for categories, crops (EN/MR/HI), stages, regions, and units. "
        "Used by normalize workers and query understanding."
    )
    s1, s2, s3, s4 = st.columns(4)
    summ = taxonomy_service.summary()
    s1.metric("Version", summ["version"])
    s2.metric("Crops", summ["crops"])
    s3.metric("Categories", summ["categories"])
    s4.metric("MH districts", summ["mh_districts"])

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

# TAB 4: Vision
with tabs[3]:
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
            st.metric("Detected Disease", f"{result['disease_identified_mr']} ({result['disease_identified_en']})")
            st.metric("Confidence", f"{int(result['confidence_score'] * 100)}%")

            st.markdown(f"**Symptoms:** {result['symptoms_mr']}")
            st.markdown(f"**🌿 Organic Control:** {result['organic_treatment']['mr']}")
            st.markdown(f"**🧪 Chemical Control:** {result['chemical_treatment']['mr']}")

# TAB 5: Live RAG
with tabs[4]:
    st.header("📡 Live RAG Intelligence Center")

    m1, m2, m3, m4 = st.columns(4)

    w_data = weather_feed.get_weather("Pune")
    m1.metric("Live Temp (°C)", f"{w_data['temperature_c']}°C", f"Humidity: {w_data['relative_humidity_pct']}%")

    m_data = market_feed.get_market_summary_for_crop("Pomegranate")
    m2.metric("APMC Solapur Modal", f"₹{m_data.get('average_modal_price_rs_quintal', 10500)} / qtl", m_data.get("source_mode", "local"))

    iot_data = iot_feed.get_sensor_telemetry(farm_id)
    m3.metric("IoT Soil Moisture", f"{iot_data['sensors']['soil_moisture_vol_pct']}%", iot_data['status']['soil_moisture_status'])

    sat_data = satellite_feed.get_satellite_indices(farm_id)
    m4.metric("Sentinel-2 NDVI", f"{sat_data['indices']['NDVI_normalized_difference_vegetation_index']}", "High Chlorophyll Vigor")

    st.subheader("📊 APMC Mandi Price Feeds")
    st.dataframe(pd.DataFrame(market_feed.get_market_prices()), use_container_width=True)
    st.caption(f"Open data status: {opendata_client.status()}")

# TAB 6: Soil & Fertilizer
with tabs[5]:
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
            b1, b2, b3 = st.columns(3)
            b1.metric("Urea (45kg Bags)", f_bags["Urea_45kg_bags"])
            b2.metric("DAP (50kg Bags)", f_bags["DAP_50kg_bags"])
            b3.metric("MOP (50kg Bags)", f_bags["MOP_50kg_bags"])

            st.info(f"**मराठी संदेश:** {p_res['application_schedule_mr']}")

# TAB 7: GraphRAG
with tabs[6]:
    st.header("🕸️ GraphRAG Agricultural Knowledge Graph")
    st.write("Explore graph relations between crops, pests, diseases, fertilizers, and government schemes.")

    crop_nodes = [n["id"] for n in kb_loader.graph_data.get("nodes", []) if n.get("label") == "Crop"]
    if not crop_nodes:
        crop_nodes = ["Pomegranate", "Cotton", "Soybean", "Onion"]
    selected_graph_crop = st.selectbox("Select Crop Entity", crop_nodes)
    ecosystem = graph_rag.get_crop_ecosystem(selected_graph_crop)
    st.json(ecosystem)

# TAB 8: Predictive
with tabs[7]:
    st.header("📊 Predictive AI Models & Automated Workflows")

    p1, p2 = st.columns(2)

    with p1:
        st.subheader("🔮 Crop Yield Forecasting")
        y_crop = st.selectbox("Yield Crop", ["Pomegranate", "Cotton", "Soybean", "Sugarcane", "Onion", "Rice", "Wheat"])
        y_acres = st.number_input("Land Acres", value=3.0)
        yp = yield_model.predict_yield(crop=y_crop, acreage=y_acres)
        st.metric(
            "Predicted Total Yield",
            f"{yp['total_predicted_yield']} Quintals",
            f"Per Acre: {yp['predicted_yield_per_acre']} Quintals",
        )

    with p2:
        st.subheader("💧 Smart Irrigation Runtime Calculator")
        ir = irrigation_model.calculate_water_requirement(
            crop=y_crop, acreage=y_acres, temperature_c=30.0, humidity_pct=75.0
        )
        st.metric("Daily Water Requirement", f"{ir['total_farm_water_required_liters_day']} Liters/day")
        st.metric("Drip Runtime", f"{ir['drip_irrigation_schedule']['drip_runtime_hours_per_day']} Hours/day")

    st.markdown("---")
    st.subheader("⚡ Automated Farm Health Audit & Workflow Actions")
    if st.button("Run Automated Farm Health Audit"):
        wf_res = workflow_engine.run_farm_health_checks(farm_id)
        st.write("Automated Actions Triggered:")
        for act in wf_res["triggered_workflows"]:
            st.warning(f"🚨 **{act['trigger']}** — {act['message_mr']}")
