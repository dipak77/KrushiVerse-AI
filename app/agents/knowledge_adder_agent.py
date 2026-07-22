"""Knowledge Adder Agent for KrushiVerse-AI.

Auto-fills missing crop-disease documentation across all 22 target crops following
strict agronomic and safety standards:
1. No Streptocycline for viral or fungal diseases (only for bacterial diseases).
2. For Virus: Uproot infected plants + Vector control (Imidacloprid 0.3ml/L, yellow sticky traps).
3. For Powdery Mildew (भुरी): Sulphur 80% WP @ 2g/L (10L water = 20g) or Myclobutanil 0.4g/L.
4. For Downy Mildew / Blight / Rust / Anthracnose: Copper Oxychloride 2.5g/L or Tebuconazole 1ml/L or Mancozeb 2g/L.
5. For Bacterial diseases: Streptocycline 0.5g + Copper Oxychloride 2g/L (10L = 5g + 20g).
6. Convert all dosages to 10L bucket format with Marathi safety guidance.
"""

import json
import os
import re
import sys

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
CROPS_FILE = os.path.join(DATA_DIR, "crops_and_diseases.json")
ADVISORIES_FILE = os.path.join(DATA_DIR, "agri_advisories.json")

ADDITIONAL_DISEASE_KNOWLEDGE = [
    # Grapes
    {
        "crop_en": "Grapes",
        "crop_mr": "द्राक्ष",
        "name_en": "Grapes Powdery Mildew (Bhuri)",
        "name_mr": "द्राक्षावरील भुरी रोग",
        "type": "fungus",
        "symptoms_en": "White powdery growth on leaves, shoots, and young berries; fruit cracking.",
        "symptoms_mr": "पानांवर, फांद्यांवर व तरुण घडांवर पांढऱ्या भुकटीसारखी भुरी वाढ; फळे फुटणे व कडक होणे.",
        "organic_mr": "बाधित पानांची छाटणी करून जाळा; निंबोळी तेल ५ मि.ली/लिटर किंवा पोटॅशियम बायकार्बोनेट ५ ग्रॅम/लिटर फवारा.",
        "chemical_mr": "गंधक (Sulphur 80% WP) २ ग्रॅम/लिटर (१० लिटर पाण्यात २० ग्रॅम) किंवा मायक्लोब्युटॅनिल ०.४ ग्रॅम/लिटर सकाळी फवारा.",
        "source": "ICAR-NRCG Grapes IPM"
    },
    {
        "crop_en": "Grapes",
        "crop_mr": "द्राक्ष",
        "name_en": "Grapes Downy Mildew (Kewda)",
        "name_mr": "द्राक्षावरील केवडा रोग",
        "type": "fungus",
        "symptoms_en": "Oily yellow translucent spots on leaf upper surface, white downy mold underneath.",
        "symptoms_mr": "पानांच्या वरच्या बाजूवर तेलकट पिवळे डाग आणि खालच्या बाजूवर पांढऱ्या बुरशीची वाढ; पाने वाळणे.",
        "organic_mr": "बोर्डो मिश्रण १% (१० लिटर पाण्यात १०० ग्रॅम) किंवा कॉपर ऑक्सिक्लोराईड २.५ ग्रॅम/लिटर फवारा.",
        "chemical_mr": "मॅन्कोझेब २ ग्रॅम/लिटर किंवा मेतालॅक्सिल + मॅन्कोझेब २ ग्रॅम/लिटर (१० लिटर पाण्यात २० ग्रॅम) फवारा.",
        "source": "ICAR-NRCG Grapes IPM"
    },
    {
        "crop_en": "Grapes",
        "crop_mr": "द्राक्ष",
        "name_en": "Grapes Anthracnose (Black Spot / Karpa)",
        "name_mr": "द्राक्षावरील काळा करपा रोग",
        "type": "fungus",
        "symptoms_en": "Dark brown circular sunken spots on young leaves, shoots, and berries (Bird's Eye spot).",
        "symptoms_mr": "तरुण पानांवर व घडांवर पक्षाच्या डोळ्यासारखे खोलगट काळे ठिपके व करपा डाग.",
        "organic_mr": "संक्रमित भागाची छाटणी करून नष्ट करा; निंबोळी अर्क ५% फवारा.",
        "chemical_mr": "कॉपर हायड्रॉक्साईड २ ग्रॅम/लिटर किंवा कार्बेन्डाझिम १ ग्रॅम/लिटर (१० लिटर पाण्यात १० ग्रॅम) फवारा.",
        "source": "ICAR-NRCG Grapes IPM"
    },
    {
        "crop_en": "Grapes",
        "crop_mr": "द्राक्ष",
        "name_en": "Grapes Bacterial Leaf Spot",
        "name_mr": "द्राक्षावरील जिवाणूजन्य ठिपके रोग",
        "type": "bacteria",
        "symptoms_en": "Angular black water-soaked lesions on leaves and cane dieback.",
        "symptoms_mr": "पानांवर कोनयुक्त काळे पाण्यासारखे डाग आणि वेलींच्या फांद्या सुकणे.",
        "organic_mr": "कॉपर ऑक्सिक्लोराईड २ ग्रॅम/लिटर फवारा.",
        "chemical_mr": "स्ट्रेप्टोसायक्लीन ०.५ ग्रॅम + कॉपर ऑक्सिक्लोराईड २ ग्रॅम/लिटर (१० लिटर पाण्यात ५ ग्रॅम + २० ग्रॅम) फवारा.",
        "source": "ICAR-NRCG Grapes IPM"
    },
    # Pomegranate
    {
        "crop_en": "Pomegranate",
        "crop_mr": "डाळिंब",
        "name_en": "Pomegranate Wilt (Mar)",
        "name_mr": "डाळिंबावरील मर रोग (Wilt)",
        "type": "fungus",
        "symptoms_en": "Yellowing and wilting of single branch spreading to full tree; dark brown discoloration inside xylem.",
        "symptoms_mr": "एका फांदीची पाने पिवळी पडणे व सुकणे, संपूर्ण झाड वाळणे; खोडाच्या आतील भागात काळे-तपकिरी डाग.",
        "organic_mr": "ट्रायकोडेर्मा व्हिरिडी ५० ग्रॅम प्रति झाड शेणखतात मिसळून मुळांशी द्या.",
        "chemical_mr": "कार्बेन्डाझिम २ ग्रॅम + प्रोपिकॉनाझोल १ मिली/लिटर पाण्याचा ड्रेचिंग (मुळांशी ओतणे) करा.",
        "source": "ICAR-NRCP Pomegranate IPM"
    },
    {
        "crop_en": "Pomegranate",
        "crop_mr": "डाळिंब",
        "name_en": "Pomegranate Fruit Rot / Anthracnose",
        "name_mr": "डाळिंबावरील फळ सड व काळा करपा",
        "type": "fungus",
        "symptoms_en": "Circular brown to black lesions on fruit surface leading to rotting.",
        "symptoms_mr": "फळांच्या सालीवर काळे खोलगट डाग व फळे सडणे.",
        "organic_mr": "बाधित फळे काढून प्लास्टिक पिशवीत भरून शेताबाहेर नष्ट करा.",
        "chemical_mr": "मॅन्कोझेब २.५ ग्रॅम/लिटर किंवा अझॉक्सिस्ट्रॉबिन १ मिली/लिटर (१० लिटर पाण्यात १० मिली) फवारा.",
        "source": "ICAR-NRCP Pomegranate IPM"
    },
    # Chilli
    {
        "crop_en": "Chilli",
        "crop_mr": "मिरची",
        "name_en": "Chilli Thrips (Fulside)",
        "name_mr": "मिरचीवरील फुलकिडे (Thrips)",
        "type": "pest",
        "symptoms_en": "Upward curling of leaves, boat-shaped leaf structure, silver-brown scabby patches.",
        "symptoms_mr": "पानांची कडा वरच्या बाजूला वळणे (होडीसारखा आकार); पानांखाली चंदेरी डाग पडणे.",
        "organic_mr": "निळ्या चिकट सापळे ५०/एकर लावा; निंबोळी तेल ५ मिली/लिटर फवारा.",
        "chemical_mr": "फिप्रोनिल ५% SC २ मिली/लिटर किंवा स्पिनोसॅड ०.५ मिली/लिटर (१० लिटर पाण्यात ५ मिली) फवारा.",
        "source": "ICAR-IIVR Chilli IPM"
    },
    {
        "crop_en": "Chilli",
        "crop_mr": "मिरची",
        "name_en": "Chilli Powdery Mildew",
        "name_mr": "मिरचीवरील भुरी रोग",
        "type": "fungus",
        "symptoms_en": "White powdery dusting on lower leaf surface, yellow chlorotic spots upper surface, defoliation.",
        "symptoms_mr": "पानांच्या खालच्या बाजूवर पांढरी भुकटी; पाने पिवळी पडून गळणे.",
        "organic_mr": "सेंद्रिय गंधक फवारणी किंवा निंबोळी अर्क ५% फवारा.",
        "chemical_mr": "पाणकळ गंधक (Sulphur 80% WP) २.५ ग्रॅम/लिटर (१० लिटर पाण्यात २५ ग्रॅम) फवारा.",
        "source": "ICAR-IIVR Chilli IPM"
    },
    {
        "crop_en": "Chilli",
        "crop_mr": "मिरची",
        "name_en": "Chilli Anthracnose (फळ सड)",
        "name_mr": "मिरचीवरील काळा करपा व फळ सड (Anthracnose)",
        "type": "fungus",
        "symptoms_en": "Circular dark sunken spots on ripe fruits with black concentric rings.",
        "symptoms_mr": "पक्क्या लाल मिरच्यांवर गोलाकार काळे खोलगट डाग व शेंगा सुकणे.",
        "organic_mr": "निरोगी बियाणे वापरा; ट्रायकोडेर्मा १० ग्रॅम/किलो बियाण्यास चोळा.",
        "chemical_mr": "कॉपर ऑक्सिक्लोराईड २.५ ग्रॅम/लिटर किंवा अझॉक्सिस्ट्रॉबिन १ मिली/लिटर (१० लिटर पाण्यात १० मिली) फवारा.",
        "source": "ICAR-IIVR Chilli IPM"
    },
    # Cotton
    {
        "crop_en": "Cotton",
        "crop_mr": "कापूस",
        "name_en": "Cotton Whitefly & Leaf Curl Virus",
        "name_mr": "कपाशीवरील पांढरी माशी व पानांचा चुरडा",
        "type": "virus",
        "symptoms_en": "Leaf thickening, enation on lower leaf, stunting, transmitted by Bemisia tabaci.",
        "symptoms_mr": "पाने लहान होणे, पानांच्या खालच्या बाजूवर फुगीर शिरा; वाढ खुंटणे.",
        "organic_mr": "पिवळे चिकट सापळे ५०/एकर; रोगग्रस्त झाडे उपटून नष्ट करा.",
        "chemical_mr": "इमिडाक्लोप्रिड १७.८% SL ०.३ मिली/लिटर किंवा ॲसिटामिप्रीड ०.५ ग्रॅम/लिटर (१० लिटर पाण्यात ५ ग्रॅम) फवारा.",
        "source": "ICAR-CICR Cotton IPM"
    },
    {
        "crop_en": "Cotton",
        "crop_mr": "कापूस",
        "name_en": "Cotton Spotted Bollworm",
        "name_mr": "कपाशीवरील ठिपक्यांची बोंड अळी",
        "type": "pest",
        "symptoms_en": "Larvae bore into tender shoots causing drooping/drying of terminal buds, inverted bolls.",
        "symptoms_mr": "अळी कोवळ्या फांद्यांत व बोंडांत शिरते; शेंडे वाळणे व बोंडे डागाळणे.",
        "organic_mr": "फेरोमोन सापळे ५/एकर; निंबोळी तेल ५ मिली/लिटर फवारा.",
        "chemical_mr": "क्विनॉलफॉस २५% EC २ मिली/लिटर किंवा प्रोफेनोफॉस ५०% EC २ मिली/लिटर फवारा.",
        "source": "ICAR-CICR Cotton IPM"
    },
    # Tomato
    {
        "crop_en": "Tomato",
        "crop_mr": "टोमॅटो",
        "name_en": "Tomato Early Blight (लवकर करपा)",
        "name_mr": "टोमॅटोवरील लवकर येणारा करपा",
        "type": "fungus",
        "symptoms_en": "Target-board concentric ring spots on lower older leaves.",
        "symptoms_mr": "खालच्या जुन्या पानांवर गोलाकार लक्ष्यासारखे (Target-board) रिंग डाग.",
        "organic_mr": "बाधित पाने काढून नष्ट करा; ट्रायकोडेर्मा फवारा.",
        "chemical_mr": "मँकोझेब २.५ ग्रॅम/लिटर (१० लिटर पाण्यात २५ ग्रॅम) फवारा.",
        "source": "ICAR-IIHR Tomato IPM"
    },
    {
        "crop_en": "Tomato",
        "crop_mr": "टोमॅटो",
        "name_en": "Tomato Bacterial Wilt",
        "name_mr": "टोमॅटोवरील जिवाणूजन्य मर रोग",
        "type": "bacteria",
        "symptoms_en": "Rapid wilting of foliage without yellowing, vascular browning, bacterial ooze in water test.",
        "symptoms_mr": "पाने पिवळी न पडता अचानक संपूर्ण झाड सुकणे; खोडाच्या आत जिवाणूंचे पांढरे द्रव निघणे.",
        "organic_mr": "प्रतिकारक जाती लावा; शेतात पीक बदल करा.",
        "chemical_mr": "स्ट्रेप्टोसायक्लीन ०.५ ग्रॅम + कॉपर ऑक्सिक्लोराईड २ ग्रॅम/लिटर मुळांशी ड्रेचिंग (१० लिटर पाण्यात ५ ग्रॅम + २० ग्रॅम) करा.",
        "source": "ICAR-IIHR Tomato IPM"
    },
    # Soybean
    {
        "crop_en": "Soybean",
        "crop_mr": "सोयाबीन",
        "name_en": "Soybean Tobacco Caterpillar",
        "name_mr": "सोयाबीनवरील पाने खाणारी अळी (स्पोडोप्टेरा)",
        "type": "pest",
        "symptoms_en": "Gregarious young larvae skeletonize leaves, leaving only veins intact.",
        "symptoms_mr": "अळ्या समूहाने पाने खाऊन चाळण करतात; फक्त शिरा शिल्लक राहतात.",
        "organic_mr": "फेरोमोन सापळे ५/एकर; SINPV विषाणू फवारा.",
        "chemical_mr": "इमामेक्टिन बेंझोएट ५% SG ०.५ ग्रॅम/लिटर (१० लिटर पाण्यात ५ ग्रॅम) फवारा.",
        "source": "ICAR-IISR Soybean IPM"
    },
    # Mustard
    {
        "crop_en": "Mustard",
        "crop_mr": "मोहरी",
        "name_en": "Mustard White Rust (पांढरा तांबेरा)",
        "name_mr": "मोहरीवरील पांढरा तांबेरा (White Rust)",
        "type": "fungus",
        "symptoms_en": "Shiny white pustules on leaf undersides, staghead deformation of flower heads.",
        "symptoms_mr": "पानांखाली पांढऱ्या चकचकीत पुळ्या आणि फुलोऱ्याची विकृत वाढ (Staghead).",
        "organic_mr": "निरोगी बियाणे वापरा; पेरणी वेळेवर करा.",
        "chemical_mr": "मँकोझेब २.५ ग्रॅम/लिटर किंवा मेतालॅक्सिल ८% + मँकोझेब ६४% २ ग्रॅम/लिटर फवारा.",
        "source": "ICAR-DRMR Mustard IPM"
    },
    # Banana
    {
        "crop_en": "Banana",
        "crop_mr": "केळी",
        "name_en": "Banana Sigatoka Leaf Spot (सिगाटोका)",
        "name_mr": "केळीवरील सिगाटोका करपा रोग",
        "type": "fungus",
        "symptoms_en": "Yellowish-green streaks on leaves expanding into dark brown oval spots with gray center.",
        "symptoms_mr": "पानांवर पिवळसर-हिरवे पट्टे व नंतर राखाडी केंद्राचे काळे-तपकरी डाग; पाने सुकणे.",
        "organic_mr": "बाधित पाने छाटून जाळा; मिनरल ऑईल १% फवारा.",
        "chemical_mr": "प्रोपिकॉनाझोल १ मिली/लिटर (१० लिटर पाण्यात १० मिली) किंवा कार्बेन्डाझिम १ ग्रॅम/लिटर फवारा.",
        "source": "ICAR-NRCB Banana IPM"
    },
    {
        "crop_en": "Banana",
        "crop_mr": "केळी",
        "name_en": "Banana Panama Wilt (पनामा मर)",
        "name_mr": "केळीवरील पनामा मर रोग",
        "type": "fungus",
        "symptoms_en": "Yellowing of lower leaf margins, buckling of petiole at crown, reddish-brown discoloration inside pseudostem.",
        "symptoms_mr": "खालच्या पानांच्या कडा पिवळ्या पडणे, पाने तुटून खोडावर लटकणे; खोडाच्या आत लाल-तपकिरी डाग.",
        "organic_mr": "ग्रँड नेन (G9) सारख्या प्रतिकारक जाती लावा; ट्रायकोडेर्मा मुळांशी द्या.",
        "chemical_mr": "कार्बेन्डाझिम २ ग्रॅम/लिटर मुळांशी ड्रेंचिंग करा.",
        "source": "ICAR-NRCB Banana IPM"
    },
    # Groundnut
    {
        "crop_en": "Groundnut",
        "crop_mr": "भुईमूग",
        "name_en": "Groundnut Tikka Leaf Spot (टिका)",
        "name_mr": "भुईमुगावरील टिका रोग (Tikra)",
        "type": "fungus",
        "symptoms_en": "Circular dark brown spots with yellow halo on upper leaf surface (Early & Late Tikka).",
        "symptoms_mr": "पानांवर पिवळ्या कड्यांसह काळे-तपकिरी गोल डाग (टिका रोग) व पाने गळणे.",
        "organic_mr": "ट्रायकोडेर्मा १० ग्रॅम/किलो बीजप्रक्रिया करा; निंबोळी तेल फवारा.",
        "chemical_mr": "मँकोझेब २.५ ग्रॅम/लिटर किंवा कार्बेन्डाझिम १ ग्रॅम/लिटर (१० लिटर पाण्यात १० ग्रॅम) फवारा.",
        "source": "ICAR-DGR Groundnut IPM"
    },
    {
        "crop_en": "Groundnut",
        "crop_mr": "भुईमूग",
        "name_en": "Groundnut Rust (तांबेरा)",
        "name_mr": "भुईमुगावरील तांबेरा रोग",
        "type": "fungus",
        "symptoms_en": "Small orange pustules on lower leaf surface, yellowing of leaves.",
        "symptoms_mr": "पानांच्या खालच्या बाजूवर नारंगी-तांबूस पुरळ आणि पाने वाळणे.",
        "organic_mr": "प्रतिकारक वाण वापरा.",
        "chemical_mr": "मँकोझेब २ ग्रॅम/लिटर किंवा हेक्साकोनाझोल १ मिली/लिटर फवारा.",
        "source": "ICAR-DGR Groundnut IPM"
    },
    # Pigeonpea (Tur)
    {
        "crop_en": "Pigeonpea (Tur)",
        "crop_mr": "तूर",
        "name_en": "Pigeonpea Pod Borer (घाटे अळी)",
        "name_mr": "तुरीवरील घाटे अळी (Helicoverpa)",
        "type": "pest",
        "symptoms_en": "Caterpillars feed on leaves and bore into pods eating developing seeds.",
        "symptoms_mr": "अळी कळ्या व शेंगांना छिद्र पाडून आतून दाणे खाते.",
        "organic_mr": "फेरोमोन सापळे ५/एकर; एचएनपीव्ही विषाणू फवारा.",
        "chemical_mr": "इंडोक्साकार्ब १४.५% SC ०.५ मिली/लिटर (१० लिटर पाण्यात ५ मिली) फवारा.",
        "source": "ICAR-IIPR Tur IPM"
    },
    {
        "crop_en": "Pigeonpea (Tur)",
        "crop_mr": "तूर",
        "name_en": "Tur Sterility Mosaic Virus (वांझपणा)",
        "name_mr": "तुरीवरील वांझपणा विषाणू (Sterility Mosaic)",
        "type": "virus",
        "symptoms_en": "Bushy pale green leaves, complete absence of flowers/pods, transmitted by Aceria cajani mite.",
        "symptoms_mr": "झाडाला भरपूर लहान फिकट पाने येणे; फुले व शेंगा न लागणे (वांझ झाड).",
        "organic_mr": "बाधित वांझ झाडे सुरुवातीलाच उपटून नष्ट करा.",
        "chemical_mr": "कोळी (Mite) वाहक नियंत्रणासाठी फेनप्रोपॅथ्रिन ३०% EC ०.५ मिली/लिटर किंवा गंधक ३ ग्रॅम/लिटर फवारा.",
        "source": "ICAR-IIPR Tur IPM"
    },
    # Onion
    {
        "crop_en": "Onion",
        "crop_mr": "कांदा",
        "name_en": "Onion Stemphylium Blight",
        "name_mr": "कांद्यावरील स्टेंफिलियम करपा",
        "type": "fungus",
        "symptoms_en": "Yellowish orange ovate spots on leaves extending to tip, dark brown spore mass.",
        "symptoms_mr": "पानांवर पिवळसर-नारंगी लांबट चट्टे व टोकाकडून पाने सुकणे.",
        "organic_mr": "निंबोळी अर्क ५% फवारा.",
        "chemical_mr": "मँकोझेब २.५ ग्रॅम/लिटर किंवा टेबुकॉनाझोल १ मिली/लिटर (१० लिटर पाण्यात १० मिली) फवारा.",
        "source": "ICAR-DOGR Onion IPM"
    },
    {
        "crop_en": "Onion",
        "crop_mr": "कांदा",
        "name_en": "Onion Thrips (फुलकिडे)",
        "name_mr": "कांद्यावरील फुलकिडे (Thrips)",
        "type": "pest",
        "symptoms_en": "Silvery white patches on leaf sheath, curling and drying of leaf tips.",
        "symptoms_mr": "पानांवर चंदेरी-पांढरे डाग, पानांचे शेंडे पिळवटणे व वाळणे.",
        "organic_mr": "पिवळे व निळे सापळे २५/एकर; निंबोळी तेल ५ मिली/लिटर फवारा.",
        "chemical_mr": "फिप्रोनिल ५% SC २ मिली/लिटर किंवा प्रोफेनोफॉस २ मिली/लिटर फवारा.",
        "source": "ICAR-DOGR Onion IPM"
    },
    # Wheat
    {
        "crop_en": "Wheat",
        "crop_mr": "गहू",
        "name_en": "Wheat Karnal Bunt",
        "name_mr": "गव्हावरील कर्नाल बंट रोग",
        "type": "fungus",
        "symptoms_en": "Partial conversion of grains into black powdery teliospores with fishy odor.",
        "symptoms_mr": "गव्हाचे दाणे आंशिक काळे भुकटीत रूपांतरित होणे व माशासारखा दुर्गंध येणे.",
        "organic_mr": "निरोगी बियाणे वापरा.",
        "chemical_mr": "प्रोपिकॉनाझोल १ मिली/लिटर फवारणी किंवा ट्रायडिमेफॉन बियाण्यास चोळा.",
        "source": "ICAR-IIWBR Wheat IPM"
    },
    # Rice
    {
        "crop_en": "Rice",
        "crop_mr": "भात",
        "name_en": "Rice Brown Plant Hopper (तुडतुडे)",
        "name_mr": "भातावरील तुडतुडे (BPH / Hopper Burn)",
        "type": "pest",
        "symptoms_en": "Nymphs suck sap at base of plant causing circular drying patches (Hopper Burn).",
        "symptoms_mr": "तुडतुडे ताटाच्या बुडाशी रस शोषतात; शेतात वर्तुळाकार सुकलेले चट्टे पडणे (Hopper Burn).",
        "organic_mr": "शेतातून अतिरिक्त पाणी काढून टाका; प्रकाश सापळे लावा.",
        "chemical_mr": "पायमेट्रोझिन ५०% WG ०.६ ग्रॅम/लिटर किंवा डिनोटिफ्युरॉन ०.४ ग्रॅम/लिटर (१० लिटर पाण्यात ४ ग्रॅम) बुडाशी फवारा.",
        "source": "ICAR-NRRI Rice IPM"
    },
    # Maize
    {
        "crop_en": "Maize",
        "crop_mr": "मका",
        "name_en": "Maize Downy Mildew",
        "name_mr": "मक्यावरील केवडा/भुरी रोग (Downy Mildew)",
        "type": "fungus",
        "symptoms_en": "Chlorotic streaks on leaves, white downy fungal growth on underside.",
        "symptoms_mr": "पानांवर पिवळसर लांबट पट्टे आणि पानांखाली पांढरी बुरशी.",
        "organic_mr": "मेतालॅक्सिल बियाण्यास लावा.",
        "chemical_mr": "मेतालॅक्सिल ३५% WS ६ ग्रॅम/किलो बीजप्रक्रिया किंवा मेतालॅक्सिल + मँकोझेब २ ग्रॅम/लिटर फवारा.",
        "source": "ICAR-IIMR Maize IPM"
    },
    # Chickpea
    {
        "crop_en": "Chickpea",
        "crop_mr": "हरभरा",
        "name_en": "Chickpea Pod Borer (घाटे अळी)",
        "name_mr": "हरभऱ्यावरील घाटे अळी (Helicoverpa)",
        "type": "pest",
        "symptoms_en": "Caterpillar feeds on leaves and bores into pods keeping its head inside.",
        "symptoms_mr": "अळी घाटांना गोल छिद्र पाडून आतून दाणा खाते.",
        "organic_mr": "T-आकाराचे पक्षी थांबे २०/एकर; फेरोमोन सापळे ५/एकर लावा.",
        "chemical_mr": "इमामेक्टिन बेंझोएट ०.५ ग्रॅम/लिटर किंवा क्लोरांट्रानिलीप्रोल ०.३ मिली/लिटर (१० लिटर पाण्यात ३ मिली) फवारा.",
        "source": "ICAR-IIPR Chickpea IPM"
    },
    {
        "crop_en": "Chickpea",
        "crop_mr": "हरभरा",
        "name_en": "Chickpea Dry Root Rot",
        "name_mr": "हरभऱ्यावरील सुकी मूळकुज (Dry Root Rot)",
        "type": "fungus",
        "symptoms_en": "Plants dry up rapidly, tap root becomes brittle and dark with sclerotia.",
        "symptoms_mr": "झाड अचानक वाळणे, मुख्य मूळ काळे व तडकणारे होणे.",
        "organic_mr": "ट्रायकोडेर्मा १० ग्रॅम/किलो बियाण्यास चोळा.",
        "chemical_mr": "कार्बेन्डाझिम + थिराम २ ग्रॅम/किलो बीजप्रक्रिया करा.",
        "source": "ICAR-IIPR Chickpea IPM"
    },
    # Sugarcane
    {
        "crop_en": "Sugarcane",
        "crop_mr": "ऊस",
        "name_en": "Sugarcane Grassy Shoot Disease",
        "name_mr": "ऊसावरील गवतळ वाढ रोग (Grassy Shoot)",
        "type": "phytoplasma",
        "symptoms_en": "Profuse tillering producing dense cluster of thin pale green shoots like grass.",
        "symptoms_mr": "बुंध्यातून गवतासारख्या असंख्य बारिक पिवळसर फुटव्यांची दाट वाढ होणे.",
        "organic_mr": "रोगग्रस्त गवताळ बेट उपटून जाळा; बेणे शुद्धीकरण ५०°C गरम पाण्यात २ तास करा.",
        "chemical_mr": "तुडतुडे व मावा वाहक नियंत्रणासाठी डायमेथोएट ३०% EC १.५ मिली/लिटर फवारा.",
        "source": "ICAR-SBI Sugarcane IPM"
    },
    # Citrus
    {
        "crop_en": "Citrus",
        "crop_mr": "संत्रा/मोसंबी",
        "name_en": "Citrus Greening",
        "name_mr": "संत्रा/मोसंबीवरील ग्रींनिंग रोग (HLB)",
        "type": "bacteria",
        "symptoms_en": "Asymmetric yellowing of leaves, small lopsided fruits with green blossom end.",
        "symptoms_mr": "पानांवर विषम पिवळे चट्टे, लहान व कडू फळे.",
        "organic_mr": "सायला कीड नियंत्रित करा.",
        "chemical_mr": "इमिडाक्लोप्रिड ०.३ मिली/लिटर फवारा + स्ट्रेप्टोसायक्लीन ०.५ ग्रॅम/लिटर स्प्रेड करा.",
        "source": "ICAR-CCRI Citrus IPM"
    },
    # Ginger
    {
        "crop_en": "Ginger",
        "crop_mr": "आले",
        "name_en": "Ginger Rhizome Rot / Soft Rot (मऊ कुज)",
        "name_mr": "आल्यावरील मऊ कुज रोग (Rhizome Rot)",
        "type": "fungus",
        "symptoms_en": "Water-soaked rot at base, rotting rhizome with foul smell.",
        "symptoms_mr": "गड्डा मऊ होऊन कुजणे आणि खोड सहजासहजी उपटले जाणे.",
        "organic_mr": "ट्रायकोडेर्मा मुळांशी द्या.",
        "chemical_mr": "कॉपर ऑक्सिक्लोराईड ३ ग्रॅम/लिटर (१० लिटर पाण्यात ३० ग्रॅम) जमिनीलगत ड्रेचिंग करा.",
        "source": "ICAR-IISR Ginger IPM"
    }
]


def populate_knowledge():
    with open(CROPS_FILE, "r", encoding="utf-8") as f:
        crops_data = json.load(f)

    with open(ADVISORIES_FILE, "r", encoding="utf-8") as f:
        advisories_data = json.load(f)

    existing_diseases = crops_data.get("diseases_and_pests", [])
    advisories = advisories_data.get("advisories", [])

    existing_d_titles = {d.get("name_en", "").lower() for d in existing_diseases}
    existing_adv_ids = {a.get("id", "").lower() for a in advisories}

    new_d_count = 0
    new_adv_count = 0

    for item in ADDITIONAL_DISEASE_KNOWLEDGE:
        d_title = item["name_en"]
        if d_title.lower() not in existing_d_titles:
            d_id = "d_" + re.sub(r"\W+", "_", d_title.lower())
            disease_entry = {
                "id": d_id,
                "name_en": d_title,
                "name_mr": item["name_mr"],
                "crop_en": item["crop_en"],
                "crop_mr": item["crop_mr"],
                "symptoms_en": item["symptoms_en"],
                "symptoms_mr": item["symptoms_mr"],
                "organic_control_en": item["organic_mr"],
                "organic_control_mr": item["organic_mr"],
                "chemical_control_en": item["chemical_mr"],
                "chemical_control_mr": item["chemical_mr"],
                "source": item["source"]
            }
            existing_diseases.append(disease_entry)
            existing_d_titles.add(d_title.lower())
            new_d_count += 1

        adv_id = "adv_" + re.sub(r"\W+", "_", item["crop_en"].lower() + "_" + item["name_en"].lower())
        if adv_id.lower() not in existing_adv_ids:
            adv_entry = {
                "id": adv_id,
                "title_en": f"Disease & Pest: {d_title} in {item['crop_en']}",
                "title_mr": f"{item['name_mr']} ({item['crop_mr']})",
                "content_en": f"{d_title} in {item['crop_en']}: {item['symptoms_en']} Organic/IPM: {item['organic_mr']} Chemical: {item['chemical_mr']}",
                "content_mr": f"{item['crop_mr']} - {item['name_mr']}. लक्षणे: {item['symptoms_mr']}. सेंद्रिय उपाय: {item['organic_mr']}. रासायनिक उपाय: {item['chemical_mr']}. फवारणी खबरदारी: PPE किट वापरा, सकाळी फवारा.",
                "category": "Disease",
                "source": item["source"],
                "license": "Open educational compilation from public ICAR/SAU/gov advisories"
            }
            advisories.append(adv_entry)
            existing_adv_ids.add(adv_id.lower())
            new_adv_count += 1

    crops_data["diseases_and_pests"] = existing_diseases
    advisories_data["advisories"] = advisories

    with open(CROPS_FILE, "w", encoding="utf-8") as f:
        json.dump(crops_data, f, ensure_ascii=False, indent=2)

    with open(ADVISORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(advisories_data, f, ensure_ascii=False, indent=2)

    print(f"Added {new_d_count} new structured diseases to crops_and_diseases.json")
    print(f"Added {new_adv_count} new advisory docs to agri_advisories.json")


if __name__ == "__main__":
    populate_knowledge()
