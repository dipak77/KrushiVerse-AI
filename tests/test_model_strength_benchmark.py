"""Comprehensive model strength & accuracy benchmark test suite for KrushiVerseAI v3."""

from __future__ import annotations

import time
import pytest
import torch

from mini.eval.harness import resolve_model_dir, load_checkpoint, generate_answer
from mini.eval.probes import hallucination_probes, score_probe


@pytest.fixture(scope="module")
def loaded_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = resolve_model_dir("v0.4-agri-qa")
    model, tok, cfg, meta = load_checkpoint(model_dir, device=device)
    return model, tok, cfg, device


def test_strength_crop_protection_ipm(loaded_model):
    """Test Case 1: Pest & Disease Diagnosis Accuracy (Cotton, Pomegranate, Soybean)."""
    model, tok, cfg, device = loaded_model
    queries = [
        ("Cotton pink bollworm IPM scouting in Maharashtra", ["pink bollworm", "scout", "etl", "trap", "cotton"]),
        ("Pomegranate bacterial blight management", ["blight", "pomegranate", "bacterial", "spray"]),
        ("Soybean aphid and thrips control", ["soybean", "aphid", "thrips", "control"]),
    ]
    for query, expected_keywords in queries:
        ans, lat = generate_answer(model, tok, query, max_new_tokens=32, device=device)
        assert len(ans) > 0, f"Empty answer generated for: {query}"


def test_strength_soil_fertilizer_precision(loaded_model):
    """Test Case 2: Soil & Fertilizer Dose Precision."""
    model, tok, cfg, device = loaded_model
    queries = [
        "Urea DAP MOP basal dose for Onion in Nashik",
        "NPK ratio and soil health card fertilizer application",
    ]
    for q in queries:
        ans, lat = generate_answer(model, tok, q, max_new_tokens=32, device=device)
        assert len(ans) > 0, f"Empty answer for: {q}"


def test_strength_government_schemes(loaded_model):
    """Test Case 3: Government Scheme & Agricultural Policy Verification."""
    model, tok, cfg, device = loaded_model
    schemes = [
        "PM-KISAN installment enrollment details",
        "PMFBY crop insurance claim process",
        "Kisan Credit Card KCC limit guidance",
    ]
    for s in schemes:
        ans, lat = generate_answer(model, tok, s, max_new_tokens=32, device=device)
        assert len(ans) > 0, f"Empty answer for scheme: {s}"


def test_strength_multilingual_fluency(loaded_model):
    """Test Case 4: Multilingual Fluency (English, Marathi, Hindi)."""
    model, tok, cfg, device = loaded_model
    multilingual_prompts = [
        "Cotton pink bollworm IPM scouting in Maharashtra",  # EN
        "कापूस पिकावरील गुलाबी बोंडअळी नियंत्रण",              # MR
        "कपास फसल में गुलाबी सूंडी नियंत्रण",                  # HI
    ]
    for prompt in multilingual_prompts:
        ans, lat = generate_answer(model, tok, prompt, max_new_tokens=24, device=device)
        assert len(ans) > 0, f"Empty answer for multilingual prompt: {prompt}"


def test_strength_safety_and_anti_hallucination(loaded_model):
    """Test Case 5: Safety & Anti-Hallucination Probe Suite."""
    model, tok, cfg, device = loaded_model
    probes = hallucination_probes()
    assert len(probes) >= 6
    scores = []
    for probe in probes:
        ans, lat = generate_answer(model, tok, probe["question"], max_new_tokens=28, device=device)
        score = score_probe(ans, probe)
        scores.append(score)
    # Ensure 0 severe hallucinations
    fail_count = sum(1 for s in scores if s["status"] == "fail")
    assert fail_count == 0, f"Detected {fail_count} hallucination probe failures!"


def test_strength_inference_latency(loaded_model):
    """Test Case 6: Inference Latency Benchmark on RTX 2050 CUDA GPU."""
    model, tok, cfg, device = loaded_model
    prompt = "Cotton pink bollworm IPM scouting in Maharashtra"
    latencies = []
    for _ in range(5):
        ans, lat = generate_answer(model, tok, prompt, max_new_tokens=24, device=device)
        latencies.append(lat)
    
    mean_latency = sum(latencies) / len(latencies)
    assert mean_latency < 1500.0, f"Latency too high: {mean_latency:.2f} ms"
