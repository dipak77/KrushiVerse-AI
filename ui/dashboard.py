import streamlit as st
import pandas as pd
import requests
import json
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.planner import planner_agent
from app.live_feeds.weather_feed import weather_feed
from app.live_feeds.market_feed import market_feed
from app.live_feeds.iot_feed import iot_feed
from app.live_feeds.satellite_feed import satellite_feed
from app.vision.disease_classifier import vision_classifier
from app.vision.ocr_processor import ocr_processor
from app.predictive.yield_model import yield_model
from app.predictive.irrigation_model import irrigation_model
from app.predictive.fertilizer_planner import fertilizer_planner
from app.knowledge.graph_rag import graph_rag
from app.memory.farm_memory import farm_memory_store
from app.workflows.automation import workflow_engine

st.set_page_config(
    page_title="AI Krushi Mitra — KrushiVerse-AI Platform",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AI Krushi Mitra — Autonomous AI Agriculture Platform")
st.caption("Generation 10 Platform combining RAG, GraphRAG, AI Agents, Computer Vision, Live Feeds, IoT, Predictive Models & Farm Memory")

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

tabs = st.tabs([
    "🤖 AI Krushi Assistant",
    "📸 Vision Disease Diagnostic",
    "📡 Live RAG (Weather, Market, IoT, Satellite)",
    "🧪 Soil & Fertilizer Planner",
    "🕸️ GraphRAG Knowledge Explorer",
    "📊 Predictive AI & Workflows"
])

# TAB 1: AI Krushi Assistant
with tabs[0]:
    st.header("💬 AI Krushi Assistant / एआय कृषी मित्र संभाषण")
    st.write("Ask any question regarding crop management, disease treatment, market rates, or government schemes.")

    sample_queries = [
        "डाळिंबावरील तेल्या रोगासाठी कोणते औषध फवारावे? बाजारात काय भाव चालू आहे?",
        "What fertilizers should I apply for Cotton in black soil?",
        "कपाशीवरील गुलाबी बोंड अळीचे नियंत्रण कसे करावे आणि पीक विमा कसा मिळेल?",
        "What are the top government schemes for drip irrigation subsidy in Maharashtra?"
    ]

    selected_sample = st.selectbox("Sample Queries / नमुना प्रश्न:", ["-- Select a sample query --"] + sample_queries)
    query_input = st.text_area("Your Query / तुमचा प्रश्न:", value="" if selected_sample == "-- Select a sample query --" else selected_sample, height=90)

    if st.button("🚀 Ask AI Krushi Assistant / उत्तर मिळवा", type="primary"):
        if query_input.strip():
            with st.spinner("Executing Planner Agent & Specialized Multi-Agent Network..."):
                res = planner_agent.plan_and_execute(
                    query=query_input,
                    farm_id=farm_id,
                    language=lang_code
                )
                st.markdown("### 📝 Synthesized Marathi/English Response:")
                st.markdown(res["synthesized_answer"])

                with st.expander("🔍 Inspect Multi-Agent Execution Plan & Knowledge Graph Hits"):
                    st.json({
                        "active_agents": res["active_agent_names"],
                        "knowledge_layer": res["knowledge_layer"],
                        "raw_agent_outputs": res["agent_outputs"]
                    })
        else:
            st.warning("Please type a query or select a sample query above.")

# TAB 2: Vision Disease Diagnostic
with tabs[1]:
    st.header("🔬 Computer Vision Plant Disease Classifier")
    st.write("Upload a crop leaf photo or select sample leaf image to perform diagnostic analysis.")

    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        uploaded_file = st.file_uploader("Upload Leaf Photo (JPG/PNG)", type=["jpg", "png", "jpeg"])
        crop_hint = st.selectbox("Crop Type", ["Pomegranate", "Cotton", "Soybean", "Onion", "Sugarcane"])
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

# TAB 3: Live RAG
with tabs[2]:
    st.header("📡 Live RAG Intelligence Center")

    m1, m2, m3, m4 = st.columns(4)

    # Weather
    w_data = weather_feed.get_weather("Pune")
    m1.metric("Live Temp (°C)", f"{w_data['temperature_c']}°C", f"Humidity: {w_data['relative_humidity_pct']}%")

    # Market
    m_data = market_feed.get_market_summary_for_crop("Pomegranate")
    m2.metric("APMC Solapur Modal", f"₹{m_data.get('average_modal_price_rs_quintal', 10500)} / qtl", "Bullish (+5%)")

    # IoT
    iot_data = iot_feed.get_sensor_telemetry(farm_id)
    m3.metric("IoT Soil Moisture", f"{iot_data['sensors']['soil_moisture_vol_pct']}%", iot_data['status']['soil_moisture_status'])

    # Satellite
    sat_data = satellite_feed.get_satellite_indices(farm_id)
    m4.metric("Sentinel-2 NDVI", f"{sat_data['indices']['NDVI_normalized_difference_vegetation_index']}", "High Chlorophyll Vigor")

    st.subheader("📊 APMC Mandi Price Feeds")
    st.dataframe(pd.DataFrame(market_feed.get_market_prices()))

# TAB 4: Soil & Fertilizer Planner
with tabs[3]:
    st.header("🧪 Soil Health Card OCR & Fertilizer Planner")

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.subheader("1. Soil Health Card Parameters")
        ocr_text = st.text_area("Soil Card Text or Scanned Output:", value="pH: 7.2, EC: 0.45, Organic Carbon: 0.52%, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha")
        extracted_soil = ocr_processor.process_soil_card(ocr_text)
        st.json(extracted_soil)

    with col_s2:
        st.subheader("2. Target Fertilizer Calculator")
        fert_crop = st.selectbox("Crop for Fertigation Plan", ["Pomegranate", "Cotton", "Soybean", "Onion", "Sugarcane"])
        fert_acres = st.number_input("Acreage", value=2.5, step=0.5)

        if st.button("Calculate Fertilizer Bags"):
            p_res = fertilizer_planner.calculate_fertilizer_bags(
                crop=fert_crop,
                acreage=fert_acres,
                N_kg_ha=extracted_soil['extracted_parameters']['nitrogen_kg_ha'],
                P_kg_ha=extracted_soil['extracted_parameters']['phosphorus_kg_ha'],
                K_kg_ha=extracted_soil['extracted_parameters']['potassium_kg_ha']
            )

            st.success("Fertilizer Recommendation Generated!")
            f_bags = p_res["recommended_fertilizer_bags"]
            b1, b2, b3 = st.columns(3)
            b1.metric("Urea (45kg Bags)", f_bags["Urea_45kg_bags"])
            b2.metric("DAP (50kg Bags)", f_bags["DAP_50kg_bags"])
            b3.metric("MOP (50kg Bags)", f_bags["MOP_50kg_bags"])

            st.info(f"**मराठी संदेश:** {p_res['application_schedule_mr']}")

# TAB 5: GraphRAG Knowledge Explorer
with tabs[4]:
    st.header("🕸️ GraphRAG Agricultural Knowledge Graph")
    st.write("Explore graph relations between crops, pests, diseases, fertilizers, and government schemes.")

    selected_graph_crop = st.selectbox("Select Crop Entity", ["Pomegranate", "Cotton", "Soybean", "Onion"])
    ecosystem = graph_rag.get_crop_ecosystem(selected_graph_crop)

    st.json(ecosystem)

# TAB 6: Predictive AI & Workflows
with tabs[5]:
    st.header("📊 Predictive AI Models & Automated Workflows")

    p1, p2 = st.columns(2)

    with p1:
        st.subheader("🔮 Crop Yield Forecasting")
        y_crop = st.selectbox("Yield Crop", ["Pomegranate", "Cotton", "Soybean", "Sugarcane", "Onion"])
        y_acres = st.number_input("Land Acres", value=3.0)
        yp = yield_model.predict_yield(crop=y_crop, acreage=y_acres)
        st.metric("Predicted Total Yield", f"{yp['total_predicted_yield']} Quintals", f"Per Acre: {yp['predicted_yield_per_acre']} Quintals")

    with p2:
        st.subheader("💧 Smart Irrigation Runtime Calculator")
        ir = irrigation_model.calculate_water_requirement(crop=y_crop, acreage=y_acres, temperature_c=30.0, humidity_pct=75.0)
        st.metric("Daily Water Requirement", f"{ir['total_farm_water_required_liters_day']} Liters/day")
        st.metric("Drip Runtime", f"{ir['drip_irrigation_schedule']['drip_runtime_hours_per_day']} Hours/day")

    st.markdown("---")
    st.subheader("⚡ Automated Farm Health Audit & Workflow Actions")
    if st.button("Run Automated Farm Health Audit"):
        wf_res = workflow_engine.run_farm_health_checks(farm_id)
        st.write("Automated Actions Triggered:")
        for act in wf_res["triggered_workflows"]:
            st.warning(f"🚨 **{act['trigger']}** — {act['message_mr']}")
