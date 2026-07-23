import re
from typing import List

# Marathi postpositions to strip from end of crop tokens
MR_POST = ["ला", "साठी", "वरील", "वर", "चा", "ची", "चे", "ने", "च्या", "मधून", "मध्ये"]

# Oblique rules: source → base (singular)
MR_OBLIQUE = {
    "कापसाला": "कापूस", "कापस": "कापूस", "कपाशी": "कापूस", "कापसावरील": "कापूस",
    "वांग्यांना": "वांगी", "वांग्या": "वांगी", "वांग्यावरील": "वांगी",
    "सोयाबीनला": "सोयाबीन", "सोयाबीनचे": "सोयाबीन", "सोयाबीनसाठी": "सोयाबीन",
    "तुरीचे": "तुर", "तुरीला": "तुर", "तुरी": "तुर",
    "ज्वारीला": "ज्वारी", "बाजरीला": "बाजरी",
    "गहूंचा": "गहू", "गव्हाचे": "गहू", "गव्हावरील": "गहू",
    "डाळिंबाला": "डाळिंब", "डाळिंबावरील": "डाळिंब",
    "द्राक्षावरील": "द्राक्ष", "द्राक्षाला": "द्राक्ष",
    "कांद्याला": "कांदा", "कांद्यावरील": "कांदा",
    "मुगावर": "मूग", "मुगाला": "मूग",
}

EN_ALIAS = {
    "pigeon pea": "Tur", "pigeonpea": "Tur", "arhar": "Tur", "toor": "Tur", "red gram": "Tur",
    "green gram": "Green Gram", "mung bean": "Green Gram", "moong": "Green Gram",
    "black gram": "Black Gram", "urad": "Black Gram",
    "cotton": "Cotton", "nagpur santra": "Orange", "santra": "Orange",
    "soybean": "Soybean", "soya": "Soybean",
    "chilli": "Chilli", "chili": "Chilli",
    "groundnut": "Groundnut", "peanut": "Groundnut",
}

INNOVATION_HINTS = {
    "drone", "sri", "intercrop", "biofertilizer", "drip",
    "precision", "sensor", "iot", "aerosol", "spray drone",
    "मेडामा", "शेती", "सर्वसमावेशक", "द्राक्षे", "छाटणी", "प्रुनिंग",
    "अंतर", "साठवणूक", "फेरपालट", "आंतरपीक", "sowing time", "spacing", "storage", "rotation"
}

def _strip_mr_post(tok: str) -> str:
    for p in sorted(MR_POST, key=len, reverse=True):
        if tok.endswith(p) and len(tok) > len(p) + 2:
            return tok[:-len(p)]
    return tok

def normalize(text: str) -> str:
    text = text.lower().strip()
    for src, tgt in EN_ALIAS.items():
        if src in text:
            text = text.replace(src, tgt.lower())
    for src, tgt in MR_OBLIQUE.items():
        if src in text:
            text = text.replace(src, tgt)
    return text

def stem_token(tok: str) -> str:
    if tok in MR_OBLIQUE:
        return MR_OBLIQUE[tok]
    tok = _strip_mr_post(tok)
    return tok

def split_tokens(text: str) -> List[str]:
    """Longest-first multiword alias matching."""
    text = normalize(text)
    tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]+", text)
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
    return out

def detect_innovation(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in INNOVATION_HINTS)

def detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["market", "mandi", "price", "भाव", "बाजार", "दर"]):
        return "market"
    if any(k in t for k in ["scheme", "subsidy", "yojana", "योजना", "सबसिडी", "pmfby", "विमा", "मागेल"]):
        return "scheme"
    if detect_innovation(text):
        return "innovation"
    if any(k in t for k in ["soil test", "माती तपासणी", "soil health", "माती"]):
        return "soil"
    if any(k in t for k in ["pest", "disease", "रोग", "कीड", "symptom", "अळी", "करपा"]):
        return "pest"
    if any(k in t for k in ["variety", "जात", "cultivar"]):
        return "variety"
    if any(k in t for k in ["weather", "हवामान", "rain", "forecast"]):
        return "weather"
    if any(k in t for k in ["fertilizer", "खत", "npk", "urea", "dap", "mop", "19:19:19"]):
        return "fertilizer"
    return "general"

if __name__ == "__main__":
    print(split_tokens("सोयाबीनला खत द्या"))
    print(split_tokens("pigeon pea market price nagpur"))
    print(detect_intent("drone spray for cotton"))
