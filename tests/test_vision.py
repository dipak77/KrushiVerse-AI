from app.vision.disease_classifier import vision_classifier
from app.vision.ocr_processor import ocr_processor

def test_vision_classifier():
    diag = vision_classifier.diagnose_image(filename="pomegranate_leaf.jpg", crop_hint="Pomegranate")
    assert diag["detected_crop"] == "Pomegranate"
    assert "disease_identified_en" in diag
    assert "disease_identified_mr" in diag
    assert "organic_treatment" in diag

def test_soil_ocr_processor():
    sample_ocr = "pH: 7.2, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha"
    parsed = ocr_processor.process_soil_card(sample_ocr)
    assert parsed["extracted_parameters"]["pH"] == 7.2
    assert parsed["extracted_parameters"]["nitrogen_kg_ha"] == 180.0
    assert parsed["evaluations"]["nitrogen_status"] == "Deficient"
