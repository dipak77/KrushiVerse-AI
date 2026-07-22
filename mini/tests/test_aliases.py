"""Unit tests for smart Marathi stemmer and multi-word crop resolution."""

import pytest
from mini.taxonomy.aliases import resolve_crops_smart, stem_mr_token


def test_marathi_inflection():
    """Verify Marathi inflected words resolve to base canonical crop names."""
    assert resolve_crops_smart("सोयाबीनला खत किती?") == ["Soybean"]
    assert resolve_crops_smart("डाळिंबाला पाणी कधी?") == ["Pomegranate"]
    assert resolve_crops_smart("कापसावरील बोंड अळी") == ["Cotton"]
    assert resolve_crops_smart("वांग्यावरील शेंडा अळी") == ["Brinjal"]
    assert resolve_crops_smart("सोयाबीनसाठी ठिबक किती?") == ["Soybean"]
    assert resolve_crops_smart("तूर डाळीला किती पाणी?") == ["Tur"]


def test_multiword():
    """Verify multi-word English and Marathi crop phrases map accurately."""
    assert resolve_crops_smart("pigeon pea price in Latur") == ["Tur"]
    assert resolve_crops_smart("nagpur santra market rate") == ["Orange"]
    assert resolve_crops_smart("red gram fertilizer schedule") == ["Tur"]
    assert resolve_crops_smart("gossypium cotton pest") == ["Cotton"]
