"""Unit & Integration Tests for CoT Reasoning Engine, Predictive Feeds, and Memory Store."""

import pytest
from app.knowledge.reasoning_engine import agri_reasoning_engine
from app.live_feeds.predictive_feed import predictive_agri_feed
from app.llm.memory_store import user_memory_store


def test_fertilizer_dosage_math():
    res = agri_reasoning_engine.calculate_fertilizer_dosage("2.5 एकर कापूस साठी युरिया किती लागेल?")
    assert res.crop == "Cotton"
    assert res.acres == 2.5
    assert res.urea_kg == 250.0
    assert res.ssp_kg == 375.0
    assert res.mop_kg == 100.0


def test_irrigation_math():
    # Dry warm day
    res = agri_reasoning_engine.calculate_irrigation_schedule("Cotton", temp_c=36.0, rainfall_mm=0.0)
    assert res.status == "ACTIVE_DRIP"
    assert res.drip_hours_per_day >= 3.0

    # Rainy day
    res_rain = agri_reasoning_engine.calculate_irrigation_schedule("Cotton", temp_c=25.0, rainfall_mm=15.0)
    assert res_rain.status == "OFF_RAINFALL"
    assert res_rain.drip_hours_per_day == 0.0


def test_predictive_market_feed():
    trend = predictive_agri_feed.predict_market_trend("Soybean", "Latur")
    assert trend.crop == "Soybean"
    assert trend.predicted_trend in ("BULLISH", "STABLE", "BEARISH")
    assert trend.confidence_pct >= 70


def test_predictive_weather_outbreak():
    outbreak = predictive_agri_feed.predict_weather_outbreak_risk("Pune", "Grapes")
    assert outbreak.crop == "Grapes"
    assert outbreak.risk_level in ("HIGH", "MEDIUM", "LOW")


def test_working_memory_store():
    mem = user_memory_store.update_session(
        session_id="user_test_123",
        query="2.5 एकर कापूस साठी",
        crop="Cotton",
        location="Nashik",
        acres=2.5,
    )
    assert mem.location == "Nashik"
    assert mem.acres == 2.5
    assert "Cotton" in mem.primary_crops

    ctx_prompt = user_memory_store.build_memory_context_prompt("user_test_123")
    assert "Location: Nashik" in ctx_prompt
    assert "2.5 acre(s)" in ctx_prompt
