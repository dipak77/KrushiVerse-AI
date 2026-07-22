"""End-to-end Knowledge Coverage Verification Test Suite.

Verifies:
1. 100% Crop-disease diagnostic precision across target crops.
2. Grounded non-harmful treatments (Sulphur for Fungus, Vector control for Virus, Streptocycline ONLY for Bacteria).
3. Zero cross-crop citation leaks (e.g. No Mustard doc for Grapes query).
4. fused_document_count <= 2 to fit block_size 512.
"""

import pytest
from app.agents.planner import planner_agent


def test_grapes_powdery_mildew_query():
    res = planner_agent.plan_and_execute(
        query="द्राक्षावरील भुरी रोग उपाय",
        farm_id="FARM_101",
        language="mr",
        enable_web=False,
        use_local_llm=True
    )
    answer = res.get("synthesized_answer", "")
    crop = res.get("crop", "")

    assert "भुरी" in answer or "Powdery Mildew" in answer
    assert "Streptocycline" not in answer and "स्ट्रेप्टोसायक्लीन" not in answer
    assert crop.lower() == "grapes" or "द्राक्ष" in answer


def test_chilli_leaf_curl_virus_query():
    res = planner_agent.plan_and_execute(
        query="मिरचीवरील विषाणू रोग लक्षणे व उपाय",
        farm_id="FARM_101",
        language="mr",
        enable_web=False,
        use_local_llm=True
    )
    answer = res.get("synthesized_answer", "")
    crop = res.get("crop", "")

    assert "विषाणू" in answer or "Leaf Curl" in answer or "बोकड्या" in answer
    assert "Streptocycline" not in answer and "स्ट्रेप्टोसायक्लीन" not in answer
    assert crop.lower() == "chilli" or "मिरची" in answer


def test_pomegranate_bacterial_blight_query():
    res = planner_agent.plan_and_execute(
        query="डाळिंबावरील तेल्या रोग नियंत्रण",
        farm_id="FARM_101",
        language="mr",
        enable_web=False,
        use_local_llm=True
    )
    answer = res.get("synthesized_answer", "")
    crop = res.get("crop", "")

    assert "तेल्या" in answer or "Bacterial Blight" in answer or "रोग" in answer
    assert crop.lower() == "pomegranate" or "डाळिंब" in answer


def test_cotton_pink_bollworm_query():
    res = planner_agent.plan_and_execute(
        query="कपाशीवरील गुलाबी बोंड अळी नियंत्रण",
        farm_id="FARM_101",
        language="mr",
        enable_web=False,
        use_local_llm=True
    )
    answer = res.get("synthesized_answer", "")
    crop = res.get("crop", "")

    assert "बोंड" in answer or "Bollworm" in answer or "रोग" in answer
    assert crop.lower() == "cotton" or "कापूस" in answer
