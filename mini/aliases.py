import re
import functools
from typing import List

# Marathi postpositions to strip from end of crop tokens
MR_POST = ["ला", "साठी", "वरील", "वर", "चा", "ची", "चे", "ने", "च्या", "मधून", "ंमध्ये"]

# Oblique rules: source → base (singular)
MR_OBLIQUE = {
    "कापसाला": "कापूस", "कापस": "कापूस", "कपाशी": "कापूस", "कापसावरील": "कापूस",
    "वांग्यांना": "वांगी", "वांग्या": "वांगी", "वांग्यावरील": "वांगी",
    "सोयाबीनला": "सोयाबीन", "सोयाबीनचे": "सोयाबीन", "सोयाबीनसाठी": "सोयाबीन",
    "तुरीचे": "तुर", "तुरीला": "तुर", "तुरी": "तुर",
    "ज्वारीला": "ज्वारी", "बाजरीला": "बाजरी", "ज्वारी": "ज्वारी", "ज्वार": "ज्वारी",
    "गहूंचा": "गहू", "गव्हाचे": "गहू", "गव्हावरील": "गहू",
    "डाळिंबाला": "डाळिंब", "डाळिंबावरील": "डाळिंब",
    "द्राक्षावरील": "द्राक्ष", "द्राक्षाला": "द्राक्ष",
    "कांद्याला": "कांदा", "कांद्यावरील": "कांदा",
    "मुगावर": "मूग", "मुगाला": "मूग", "मूग": "मूग",
    "तीळ": "तिळ", "तिळाला": "तिळ", "तिळावर": "तिळ", "तिळाची": "तिळ",
    "आल्याला": "आले", "आल्यावरील": "आले", "आले": "आले",
    "सूर्यफूल": "सूर्यफूल", "सूरजमुखी": "सूर्यफूल",
}

EN_ALIAS = {
    "pigeon pea": "Tur", "pigeonpea": "Tur", "arhar": "Tur", "toor": "Tur", "red gram": "Tur",
    "green gram": "Green Gram", "mung bean": "Green Gram", "moong": "Green Gram",
    "black gram": "Black Gram", "urad": "Black Gram",
    "cotton": "Cotton", "nagpur santra": "Orange", "santra": "Orange",
    "soybean": "Soybean", "soya": "Soybean",
    "chilli": "Chilli", "chili": "Chilli",
    "groundnut": "Groundnut", "peanut": "Groundnut",
    "sesame": "Sesame", "til": "Sesame",
    "jowar": "Sorghum", "sorghum": "Sorghum",
    "ginger": "Ginger", "sunflower": "Sunflower",
}

INNOVATION_HINTS = {
    "drone", "sri", "intercrop", "biofertilizer", "drip",
    "precision", "sensor", "iot", "aerosol", "spray drone",
    "मेडामा", "शेती", "सर्वसमावेशक", "द्राक्षे", "छाटणी", "प्रुनिंग",
    "अंतर", "साठवणूक", "फेरपालट", "आंतरपीक", "आंतरपीकात", "sowing time", "spacing", "storage", "rotation",
    "बोर्डो", "पेस्ट", "अंतर किती", "तापमान किती", "ठेवावे", "हवा खेळती"
}

CROP_NAME_MAP = {
    "Jowar": "Sorghum",
    "Sorghum": "Sorghum",
    "Green Gram": "Green Gram",
    "Mung": "Green Gram",
    "Moong": "Green Gram",
    "Tur": "Tur", "Toor": "Tur", "Pigeon Pea": "Tur",
    "Sesame": "Sesame", "Til": "Sesame",
    "Ginger": "Ginger", "Ale": "Ginger",
    "Sunflower": "Sunflower",
    "Cotton": "Cotton", "Soybean": "Soybean", "Sugarcane": "Sugarcane",
    "Pomegranate": "Pomegranate", "Onion": "Onion", "Rice": "Rice",
    "Wheat": "Wheat", "Maize": "Maize", "Gram": "Gram", "Groundnut": "Groundnut",
    "Turmeric": "Turmeric", "Grapes": "Grapes", "Banana": "Banana",
    "Mango": "Mango", "Tomato": "Tomato", "Chilli": "Chilli", "Bajra": "Bajra",
    "Orange": "Orange", "Brinjal": "Brinjal", "Potato": "Potato", "Mustard": "Mustard",
}

def resolve_crop_name(crop: str) -> str:
    if not crop:
        return ""
    c = crop.strip()
    return CROP_NAME_MAP.get(c, CROP_NAME_MAP.get(c.capitalize(), c))

def _strip_mr_post(tok: str) -> str:
    for p in sorted(MR_POST, key=len, reverse=True):
        if tok.endswith(p) and len(tok) > len(p) + 2:
            return tok[:-len(p)]
    return tok

@functools.lru_cache(maxsize=4096)
def normalize(text: str) -> str:
    text = text.lower().strip()
    for src, tgt in EN_ALIAS.items():
        if src in text:
            text = text.replace(src, tgt.lower())
    for src, tgt in MR_OBLIQUE.items():
        if src in text:
            text = text.replace(src, tgt)
    return text

@functools.lru_cache(maxsize=4096)
def stem_token(tok: str) -> str:
    if tok in MR_OBLIQUE:
        return MR_OBLIQUE[tok]
    tok = _strip_mr_post(tok)
    return tok

@functools.lru_cache(maxsize=4096)
def _split_tokens_tuple(text: str) -> tuple[str, ...]:
    """Longest-first multiword alias matching returning cached tuple."""
    norm = normalize(text)
    tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", norm)
    out = []
    i = 0
    keys = sorted(EN_ALIAS.keys(), key=len, reverse=True)
    while i < len(tokens):
        matched = False
        for k in keys:
            ks = k.split()
            if tokens[i:i+len(ks)] == ks:
                out.append(EN_ALIAS[k])
                i += len(ks)
                matched = True
                break
        if not matched:
            out.append(stem_token(tokens[i]))
            i += 1
    return tuple(out)

def split_tokens(text: str) -> List[str]:
    return list(_split_tokens_tuple(text))

@functools.lru_cache(maxsize=4096)
def detect_innovation(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in INNOVATION_HINTS)

@functools.lru_cache(maxsize=4096)
def detect_intent(text: str) -> str:
    t = text.lower()
    # Explicit intent keyword routing hierarchy
    if any(k in t for k in ["market", "mandi", "price", "bhav", "भाव", "बाजार", "दर", "बाजारभाव"]):
        return "market"
    if any(k in t for k in ["scheme", "subsidy", "yojana", "योजना", "सबसिडी", "pmfby", "विमा", "मागेल", "प्रीमियम", "योजनेत", "अनुदान", "पोर्टल", "शेततळे", "apeda", "नोंदणी"]):
        return "scheme"
    if any(k in t for k in ["fertilizer", "खत", "npk", "urea", "dap", "mop", "19:19:19", "जिप्सम", "शेणखत", "डोस", "अन्नद्रव्ये"]):
        return "fertilizer"
    if detect_innovation(text):
        return "innovation"
    if any(k in t for k in ["irrigation", "drip", "water", "ठिबक", "पाणी", "सिंचन", "दिवसांनी", "तास", "खर्च"]):
        return "irrigation"
    if any(k in t for k in ["soil test", "माती तपासणी", "soil health", "माती"]):
        return "soil"
    if any(k in t for k in ["pest", "disease", "रोग", "कीड", "symptom", "अळी", "करपा", "भुंगा", "नेक्रोसिस", "कुज", "विषाणू", "तुडतुडे", "बोंड अळी", "सापळा"]):
        return "disease"
    return "general"

if __name__ == "__main__":
    print(split_tokens("सोयाबीनला खत द्या"))
    print(split_tokens("pigeon pea market price nagpur"))
    print(detect_intent("drone spray for cotton"))
