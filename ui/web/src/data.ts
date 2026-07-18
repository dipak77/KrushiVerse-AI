/* ------------------------------------------------------------------ */
/*  KrushiVerse AI — domain data & model simulations                   */
/* ------------------------------------------------------------------ */

export const FARM = {
  id: "FARM_101",
  farmer: "Sunil Patil",
  farmerMr: "सुनील पाटील",
  village: "Mangalwedha",
  district: "Solapur",
  state: "Maharashtra",
  crop: "Pomegranate",
  cropMr: "डाळिंब",
  cropHi: "अनार",
  acres: 2.5,
  soil: "Deep black (Regur) · काळी कसदार",
  zone: "Agro-zone: Western Maharashtra · Rainfed + drip",
  memory: {
    lastVisit: "Field audit · 2 days ago",
    lastDiagnosis: "Bacterial blight — Zone C (treated)",
    activeAlerts: 2,
    seasonDay: "Day 128 · Fruit development",
  },
};

export const TICKER = [
  { e: "🧅", name: "Onion", mandi: "Lasalgaon", price: 2340, d: 2.4 },
  { e: "🔴", name: "Pomegranate", mandi: "Solapur", price: 11850, d: 1.2 },
  { e: "🌱", name: "Soybean", mandi: "Latur", price: 4685, d: -0.8 },
  { e: "☁️", name: "Cotton", mandi: "Akola", price: 7420, d: 0.6 },
  { e: "🌾", name: "Wheat", mandi: "Nagpur", price: 2275, d: 0.2 },
  { e: "🎋", name: "Sugarcane", mandi: "Kolhapur", price: 315, d: 0.0 },
  { e: "🌶️", name: "Chilli", mandi: "Guntur", price: 14200, d: 3.1 },
  { e: "🍅", name: "Tomato", mandi: "Pune", price: 1860, d: -1.6 },
];

export const MANDI_TABLE = [
  { crop: "Pomegranate", variety: "Bhagwa", mandi: "Solapur APMC", modal: 11850, min: 9200, max: 13400, d7: 1.2, mode: "Agmarknet live" },
  { crop: "Onion", variety: "Red", mandi: "Lasalgaon APMC", modal: 2340, min: 1450, max: 3150, d7: 2.4, mode: "Agmarknet live" },
  { crop: "Cotton", variety: "Shankar-6", mandi: "Akola APMC", modal: 7420, min: 6900, max: 7850, d7: 0.6, mode: "Agmarknet live" },
  { crop: "Soybean", variety: "Yellow", mandi: "Latur APMC", modal: 4685, min: 4310, max: 4990, d7: -0.8, mode: "Agmarknet live" },
  { crop: "Turmeric", variety: "Rajapuri", mandi: "Sangli APMC", modal: 9870, min: 8400, max: 11200, d7: 4.1, mode: "Agmarknet live" },
  { crop: "Grapes", variety: "Sharad Seedless", mandi: "Nashik APMC", modal: 5240, min: 3800, max: 6900, d7: -2.2, mode: "cached 3h" },
  { crop: "Wheat", variety: "Lokwan", mandi: "Nagpur APMC", modal: 2275, min: 2150, max: 2420, d7: 0.2, mode: "Agmarknet live" },
  { crop: "Chilli", variety: "Teja", mandi: "Guntur APMC", modal: 14200, min: 12600, max: 15800, d7: 3.1, mode: "cached 6h" },
];

export const WEATHER = {
  station: "Solapur AWS-7",
  temp: 31.4,
  humidity: 62,
  wind: 14,
  condition: "Partly cloudy · monsoon break",
  conditionMr: "अंशतः ढगाळ · पावसाचा खंड",
  rain7d: 34,
  series: [29.2, 30.1, 31.8, 31.2, 32.9, 32.1, 31.4],
  advisory: "No rain next 72h — safe window for copper-based sprays.",
  advisoryMr: "पुढील ७२ तास पाऊस नाही — तांब्या-आधारित फवारणीसाठी योग्य संधी.",
};

export const IOT = {
  gateway: "LoRa gateway FARM_101 · 4 nodes online",
  moisture: 34.2,
  soilTemp: 27.8,
  ec: 0.62,
  battery: 87,
  series: [31, 32.4, 33.1, 32.2, 33.8, 34.6, 33.9, 34.2],
};

export const SAT = {
  pass: "Sentinel-2B · last pass 06:12 IST · cloud 8%",
  ndvi: 0.72,
  evi: 0.41,
  ndwi: 0.18,
  vigor: "High chlorophyll vigor",
  vigorMr: "उच्च हरितद्रव्य सक्षमता",
};

/* 14 × 9 NDVI grid (rows of the field) */
export const NDVI_GRID: number[][] = [
  [0.62, 0.66, 0.71, 0.74, 0.72, 0.69, 0.66, 0.61, 0.58, 0.63, 0.7, 0.74, 0.71, 0.68],
  [0.58, 0.64, 0.7, 0.76, 0.78, 0.74, 0.7, 0.64, 0.6, 0.66, 0.73, 0.78, 0.75, 0.7],
  [0.55, 0.61, 0.68, 0.74, 0.79, 0.81, 0.77, 0.71, 0.65, 0.69, 0.76, 0.8, 0.78, 0.72],
  [0.52, 0.58, 0.65, 0.71, 0.77, 0.82, 0.84, 0.78, 0.7, 0.66, 0.72, 0.77, 0.74, 0.69],
  [0.5, 0.55, 0.6, 0.66, 0.72, 0.77, 0.74, 0.62, 0.54, 0.6, 0.67, 0.72, 0.7, 0.66],
  [0.53, 0.58, 0.63, 0.68, 0.73, 0.75, 0.7, 0.58, 0.47, 0.52, 0.61, 0.67, 0.65, 0.62],
  [0.56, 0.61, 0.66, 0.7, 0.74, 0.72, 0.66, 0.55, 0.42, 0.48, 0.57, 0.63, 0.61, 0.58],
  [0.6, 0.64, 0.69, 0.72, 0.75, 0.7, 0.62, 0.52, 0.45, 0.5, 0.55, 0.6, 0.58, 0.55],
  [0.63, 0.67, 0.71, 0.73, 0.72, 0.67, 0.6, 0.54, 0.5, 0.53, 0.57, 0.61, 0.6, 0.57],
];

export function ndviColor(v: number): string {
  const stops: [number, [number, number, number]][] = [
    [0.3, [146, 104, 58]],
    [0.5, [186, 158, 74]],
    [0.62, [122, 156, 66]],
    [0.75, [74, 148, 56]],
    [0.9, [42, 118, 48]],
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (v >= stops[i][0] && v <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break; }
  }
  const t = Math.min(1, Math.max(0, (v - lo[0]) / (hi[0] - lo[0] || 1)));
  const c = lo[1].map((a, i) => Math.round(a + (hi[1][i] - a) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/* ---------------- assistant ---------------- */

export const SAMPLE_QUERIES = [
  "डाळिंबावरील तेल्या रोगासाठी कोणते औषध फवारावे? बाजारात काय भाव चालू आहे?",
  "What fertilizers should I apply for Cotton in black soil?",
  "Top government schemes for drip irrigation subsidy in Maharashtra?",
  "Latest soybean mandi price in Maharashtra",
  "कपाशीवरील गुलाबी बोंड अळीचे नियंत्रण कसे करावे?",
];

export interface AsstResponse {
  key: string;
  crop: string;
  cropMr: string;
  agents: string[];
  metrics: { fused: number; web: number; tools: number; latencyMs: number };
  mr: string[];
  en: string[];
  citations: { title: string; origin: string; src: string; url?: string }[];
  tools: string[];
}

export const ASST_RESPONSES: Record<string, AsstResponse> = {
  disease_pomegranate: {
    key: "disease_pomegranate",
    crop: "Pomegranate",
    cropMr: "डाळिंब",
    agents: ["Disease Agent", "Market Agent", "Advisory Agent"],
    metrics: { fused: 8, web: 3, tools: 4, latencyMs: 2840 },
    mr: [
      "🌿 **रोग निदान — तेल्या रोग** (*Bacterial Blight · Xanthomonas axonopodis pv. punicae*)",
      "**लक्षणे:** पानांवर तेलकट, पाण्यासारखे डाग; फांद्यांवर भेगा; फळांवर काळे ठिपके. ओलसर हवामानात वेगाने पसरतो.",
      "**उपाययोजना:** ① बाधित फांद्या छाटून मळ्याबाहेर नष्ट करा. ② **फवारणी:** कॉपर ऑक्सिक्लोराइड 50% WP @ 2.5 ग्रॅ./लि. + स्ट्रेप्टोसायक्लिन 100 ppm — 10–12 दिवसांच्या अंतराने 2–3 फवारण्या. ③ ठिबक सिंचन वापरा; तुषार सिंचन टाळा. ④ पावसापूर्वी खोडावर 1% बोर्डो लेप.",
      "**बाजारभाव (सोलापूर APMC):** डाळिंब भगवा ₹11,850/क्वि. (मॉडल) · किमान ₹9,200 – कमाल ₹13,400 · ७ दिवस +1.2%. निर्यात मागणी मजबूत — प्रतवारी करून विक्री करा.",
    ],
    en: [
      "🌿 **Diagnosis — Bacterial Blight** (*Xanthomonas axonopodis pv. punicae*)",
      "**Action:** prune & destroy infected branches → spray Copper Oxychloride 50% WP @ 2.5 g/L + Streptocycline 100 ppm, 2–3 rounds at 10–12 day intervals → switch to drip (avoid overhead irrigation) → 1% Bordeaux paste on trunks pre-monsoon.",
      "**Market (Solapur APMC):** Bhagwa modal ₹11,850/qtl · range ₹9,200–13,400 · +1.2% (7d). Export demand (Dubai, Russia) firm — grade & sell.",
    ],
    citations: [
      { title: "Pomegranate disease management package", origin: "KB · CropWiki", src: "ICAR-NRCP advisory" },
      { title: "Xanthomonas → copper-based treatment edge", origin: "GraphRAG", src: "knowledge graph v3.1" },
      { title: "Solapur APMC — Pomegranate modal price", origin: "Agmarknet", src: "data.gov.in feed", url: "https://agmarknet.gov.in" },
      { title: "Bacterial blight of pomegranate: epidemiology", origin: "Web", src: "agricoop.gov.in" },
    ],
    tools: ["mandi.price_lookup", "disease.classifier_hint", "spray.dosage_calc", "weather.window_check"],
  },
  fertilizer_cotton: {
    key: "fertilizer_cotton",
    crop: "Cotton",
    cropMr: "कापूस",
    agents: ["Soil Agent", "Agronomy Agent"],
    metrics: { fused: 6, web: 2, tools: 3, latencyMs: 2110 },
    mr: [
      "🧪 **कापूस — काळी माती खत नियोजन (1 हेक्टरसाठी):** NPK 100:50:50 कि./हे. + ZnSO₄ 25 कि. + शेणखत 5 टन.",
      "**मात्रा क्रम:** ① पेरणीवेळी: पूर्ण P + K + अर्धा N (खोलवर, ओळीबाजूला). ② 30–35 दिवसांनी: उरलेला अर्धा N — विरळणीनंतर ओलवाणीत. ③ फुलोऱ्यापूर्वी ZnSO₄ 25 कि. जमिनीत.",
      "**टीपा:** काळी मातीत पोटॅश नैसर्गिक भरपूर — मृदुपरीक्षणाशिवाय K वाढवू नका. पावसात पेरणा-मात्रा टाळा; ठिबकद्वारे फर्टिगेशन केल्यास 20% बचत.",
    ],
    en: [
      "🧪 **Cotton in black soil — NPK 100:50:50 kg/ha** + ZnSO₄ 25 kg + FYM 5 t.",
      "**Schedule:** ① basal: full P+K, half N placed deep beside rows · ② 30–35 DAS: remaining N after thinning · ③ pre-flowering: ZnSO₄ incorporated.",
      "**Notes:** Regur soils are K-rich — don't add potash without a soil test. Avoid broadcast application before rain; drip fertigation saves ~20%.",
    ],
    citations: [
      { title: "Cotton nutrient package — black soil zone", origin: "KB · PDIC", src: "Vasantrao Naik KRUSHI data book" },
      { title: "FARM_101 soil health card (pH 7.2, OC 0.52%)", origin: "Memory", src: "farm_memory_store" },
      { title: "Drip fertigation response in Bt cotton", origin: "Web", src: "ICAR-CICR bulletin" },
    ],
    tools: ["soil.card_parser", "fert.bag_calculator", "ndvi.zone_advice"],
  },
  scheme_drip: {
    key: "scheme_drip",
    crop: "Pomegranate",
    cropMr: "डाळिंब",
    agents: ["Scheme Agent", "Advisory Agent"],
    metrics: { fused: 5, web: 4, tools: 2, latencyMs: 1930 },
    mr: [
      "💧 **ठिबक सिंचन अनुदान — मुख्य योजना:**",
      "① **PMKSY (सूक्ष्म सिंचन):** लहान/सीमांत व SC-ST शेतकऱ्यांना 55%, इतरांना 45% अनुदान — फळबागांसाठी कमाल ₹1.05 लाख/हे. ② **महा-ड्रीप (राज्य):** PMKSY च्या अतिरिक्त 15–20% टॉप-अप काही जिल्ह्यांत. ③ **अटी:** 7/12 उतारा, बँक पासबुक, आधार; अर्ज **MahaDBT पोर्टलवर** ऑनलाईन.",
      "**सल्ला:** तुमच्या 2.5 एकर डाळिंबासाठी अपेक्षित अनुदान ≈ ₹1.1 लाख. फळधारणा-पूर्व काळात बसवणी पूर्ण करा — प्रतीक्षा यादी लांब असते.",
    ],
    en: [
      "💧 **Drip subsidy schemes:** PMKSY — 55% for small/marginal & SC-ST, 45% others (max ₹1.05 L/ha for orchards); some MH districts add a 15–20% state top-up.",
      "**How:** apply online on **MahaDBT** with 7/12 extract, bank passbook & Aadhaar. For your 2.5 ac pomegranate, expected benefit ≈ ₹1.1 L. Install before fruit-set — waitlists run long.",
    ],
    citations: [
      { title: "PMKSY guidelines 2025-26 — micro irrigation", origin: "Web", src: "agricoop.gov.in", url: "https://agricoop.gov.in" },
      { title: "MahaDBT scheme catalogue — drip", origin: "KB · Schemes", src: "mahadbt.maharashtra.gov.in" },
      { title: "FARM_101 eligibility profile", origin: "Memory", src: "farm_memory_store" },
    ],
    tools: ["scheme.eligibility_check", "docs.checklist_builder"],
  },
  market_soybean: {
    key: "market_soybean",
    crop: "Soybean",
    cropMr: "सोयाबीन",
    agents: ["Market Agent"],
    metrics: { fused: 4, web: 2, tools: 2, latencyMs: 1480 },
    mr: [
      "📈 **सोयाबीन बाजारभाव (मराठवाडा):** लातूर APMC मॉडल ₹4,685/क्वि. · किमान ₹4,310 – कमाल ₹4,990 · ७ दिवस −0.8%.",
      "MSP ₹4,892 (2025–26) — बाजार MSP च्या किंचित खाली. e-NAM वर बोली तपासा; पावसानंतर खरिपाच्या मागणीत सुधारणा अपेक्षित. तूर-सोया आंतरपीक असल्यास वेगळे ग्रेडिंग करा.",
    ],
    en: [
      "📈 **Soybean (Marathwada):** Latur APMC modal ₹4,685/qtl · range ₹4,310–4,990 · −0.8% (7d). MSP ₹4,892 — market marginally below MSP.",
      "Check e-NAM bids before local sale; post-monsoon crushing demand typically lifts arrivals-week prices 2–3%.",
    ],
    citations: [
      { title: "Latur APMC — Soybean daily arrivals", origin: "Agmarknet", src: "data.gov.in feed", url: "https://agmarknet.gov.in" },
      { title: "MSP 2025-26 kharif notification", origin: "Web", src: "dfpd.gov.in" },
    ],
    tools: ["mandi.price_lookup", "mandi.trend_forecast"],
  },
  default: {
    key: "default",
    crop: "Pomegranate",
    cropMr: "डाळिंब",
    agents: ["Advisory Agent", "Agronomy Agent"],
    metrics: { fused: 5, web: 1, tools: 2, latencyMs: 1720 },
    mr: [
      "🌾 **सारांश:** तुमच्या प्रश्नाचे विश्लेषण FARM_101 च्या संदर्भात (डाळिंब, 2.5 ए, सोलापूर) करण्यात आले. स्थानिक KB, ज्ञान-ग्राफ आणि वेब स्रोतांचे एकत्रीकरण करून वरील शिफारस तयार केली आहे.",
      "अधिक अचूकतेसाठी पीक, रोग/कीड किंवा बाजारपेठेचा उल्लेख करून विचारा — उदा. “डाळिंब तेल्या रोग” किंवा “कांदा भाव नाशिक”.",
    ],
    en: [
      "🌾 **Summary:** your query was resolved against FARM_101 context (pomegranate · 2.5 ac · Solapur) and answered by fusing the local KB, knowledge graph and web sources.",
      "For sharper answers, name the crop, pest or mandi — e.g. “pomegranate bacterial blight” or “onion price Nashik”.",
    ],
    citations: [
      { title: "General crop advisory — western Maharashtra", origin: "KB · CropWiki", src: "platform knowledge base" },
      { title: "FARM_101 profile & history", origin: "Memory", src: "farm_memory_store" },
    ],
    tools: ["query.understanding", "kb.fusion_search"],
  },
};

export function matchQuery(q: string): AsstResponse {
  const s = q.toLowerCase();
  if (s.includes("तेल्या") || (s.includes("pomegranate") && s.includes("disease")) || (s.includes("डाळिंब") && s.includes("रोग"))) return ASST_RESPONSES.disease_pomegranate;
  if ((s.includes("fertilizer") || s.includes("खत")) && (s.includes("cotton") || s.includes("कापूस"))) return ASST_RESPONSES.fertilizer_cotton;
  if ((s.includes("scheme") || s.includes("योजना") || s.includes("drip") || s.includes("सिंचन") || s.includes("subsidy"))) return ASST_RESPONSES.scheme_drip;
  if ((s.includes("soybean") || s.includes("सोयाबीन")) && (s.includes("price") || s.includes("भाव") || s.includes("mandi"))) return ASST_RESPONSES.market_soybean;
  return ASST_RESPONSES.default;
}

export const PIPELINE_STAGES = [
  "Taxonomy resolve — crop · category · district",
  "Planner decomposes query → sub-tasks",
  "Dispatch agents across expert network",
  "Multi-source RAG fusion (KB + Graph + Web + Agmarknet)",
  "Synthesizing bilingual advisory (मराठी + EN)",
];

/* ---------------- vision ---------------- */

export interface LeafSample {
  id: string; name: string; nameMr: string; img: string;
  disease: string; diseaseMr: string; scientific: string;
  conf: number; severity: number; zone: string;
  diff: { name: string; pct: number }[];
  symptomsMr: string; symptomsEn: string;
  organicMr: string; organicEn: string;
  chemMr: string; chemEn: string;
}

export const LEAF_SAMPLES: LeafSample[] = [
  {
    id: "pomegranate", name: "Pomegranate", nameMr: "डाळिंब", img: "/leaves/pomegranate.jpg",
    disease: "Bacterial Blight", diseaseMr: "तेल्या रोग", scientific: "Xanthomonas axonopodis pv. punicae",
    conf: 0.94, severity: 62, zone: "Zone C · NE quadrant",
    diff: [{ name: "Bacterial Blight", pct: 94 }, { name: "Anthracnose", pct: 4 }, { name: "Fruit borer injury", pct: 2 }],
    symptomsMr: "पानांवर तेलकट पाण्यासारखे डाग, पिवळ्या कडा; फांद्यांवर भेगा; फळांवर काळे ठिपके.",
    symptomsEn: "Water-soaked oily leaf spots with yellow halos; bark cracks on twigs; black lesions on fruit.",
    organicMr: "बाधित भाग छाटून जाळा · 5% गोमूत्र + निंबोळी अर्क फवारणी (आठवड्यातून 1) · ठिबक सिंचनाकडे वळा.",
    organicEn: "Prune & burn infected parts · spray 5% cow-urine + neem kernel extract weekly · shift to drip irrigation.",
    chemMr: "कॉपर ऑक्सिक्लोराइड 50% WP @ 2.5 ग्रॅ./लि. + स्ट्रेप्टोसायक्लिन 100 ppm · 2–3 फवारण्या, 10–12 दिवसांच्या अंतराने.",
    chemEn: "Copper Oxychloride 50% WP @ 2.5 g/L + Streptocycline 100 ppm · 2–3 sprays at 10–12 day intervals.",
  },
  {
    id: "cotton", name: "Cotton", nameMr: "कापूस", img: "/leaves/cotton.jpg",
    disease: "Pink Bollworm damage", diseaseMr: "गुलाबी बोंड अळी", scientific: "Pectinophora gossypiella",
    conf: 0.91, severity: 48, zone: "Zone A · rows 4–9",
    diff: [{ name: "Pink bollworm", pct: 91 }, { name: "American bollworm", pct: 6 }, { name: "Leaf roller", pct: 3 }],
    symptomsMr: "पानांत छिद्रे; बोंडांवर अळीचे प्रवेशद्वार; रोझेट फुले; अकाली बोंड उघडणे.",
    symptomsEn: "Shot-holes in leaves; larval entry points on bolls; rosette flowers; premature boll opening.",
    organicMr: "फेरोमोन सापळे 8/एकर · Trichogramma ब्रासीडे 50,000/एकर प्रत्येक 10 दिवसांनी · 5% निंबोळी अर्क.",
    organicEn: "Pheromone traps @ 8/ac · release Trichogramma brassicae 50k/ac every 10 days · 5% neem extract.",
    chemMr: "इमामेक्टिन बेंझोएट 5% SG @ 0.4 ग्रॅ./लि. — बोंडधारणेनंतरच; Bt-आश्रित क्षेत्रात विष-बाधित क्षेत्र तपासा.",
    chemEn: "Emamectin benzoate 5% SG @ 0.4 g/L — only post boll-set; monitor refuge compliance in Bt plots.",
  },
  {
    id: "soybean", name: "Soybean", nameMr: "सोयाबीन", img: "/leaves/soybean.jpg",
    disease: "Yellow Mosaic Virus", diseaseMr: "पिवळा करपा रोग", scientific: "Begomovirus (whitefly vector)",
    conf: 0.89, severity: 38, zone: "Zone B · west margin",
    diff: [{ name: "Yellow mosaic virus", pct: 89 }, { name: "Zn deficiency", pct: 7 }, { name: "Herbicide drift", pct: 4 }],
    symptomsMr: "पानांवर पिवळे-हिरवे ठिपके, मosaic पॅटर्न; वाढ खुंटणे; शेंगा कमी भरणे.",
    symptomsEn: "Yellow-green mosaic mottling; stunted growth; poor pod filling near field margins.",
    organicMr: "बाधित रोटे उपटून नष्ट · पिवळे चिकट सापळे 10–12/एकर · निंबोळी अर्काने पांढरी माशी नियंत्रण.",
    organicEn: "Uproot infected plants · yellow sticky traps 10–12/ac · neem-based whitefly suppression.",
    chemMr: "थायोमेथोक्झाम 25% WG @ 100 ग्रॅ./हे. — वेक्टर (पांढरी माशी) नियंत्रणासाठी; विषाणूवर थेट औषध नाही.",
    chemEn: "Thiamethoxam 25% WG @ 100 g/ha against the whitefly vector; no direct viricide exists.",
  },
  {
    id: "healthy", name: "Healthy leaf", nameMr: "निरोगी पान", img: "/leaves/healthy.jpg",
    disease: "No disease detected", diseaseMr: "रोग आढळला नाही", scientific: "—",
    conf: 0.97, severity: 0, zone: "All zones nominal",
    diff: [{ name: "Healthy", pct: 97 }, { name: "Early leaf spot", pct: 2 }, { name: "Abiotic stress", pct: 1 }],
    symptomsMr: "पान निरोगी — गडद हिरवा रंग, स्वच्छ पृष्ठभाग. निबंधित तेलकट डाग आढळले नाहीत.",
    symptomsEn: "Leaf is healthy — deep green, clean lamina. No water-soaked lesions recorded.",
    organicMr: "प्रतिबंधक: 15 दिवसांनी निंबोळी अर्क; संतुलित NPK; पाण्याचा ताण टाळा.",
    organicEn: "Preventive: neem spray fortnightly; balanced NPK; avoid water stress.",
    chemMr: "रासायनिक फवारणीची गरज नाही. निरीक्षण सुरू ठेवा.",
    chemEn: "No chemical intervention required. Continue weekly scouting.",
  },
];

/* ---------------- soil & fertilizer ---------------- */

export const SOIL_DEFAULT = "pH: 7.2, EC: 0.45, Organic Carbon: 0.52%, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha";

export interface SoilParam { key: string; label: string; unit: string; value: number; low: number; high: number; ideal: [number, number]; status: "low" | "medium" | "high" | "optimal"; }

export function parseSoilCard(text: string): SoilParam[] {
  const num = (re: RegExp) => { const m = text.match(re); return m ? parseFloat(m[1]) : NaN; };
  const ph = num(/ph[:\s]*([0-9.]+)/i);
  const ec = num(/ec[:\s]*([0-9.]+)/i);
  const oc = num(/(?:organic carbon|oc)[:\s]*([0-9.]+)/i);
  const n = num(/nitrogen[^0-9]*([0-9.]+)/i);
  const p = num(/phosphorus[^0-9]*([0-9.]+)/i);
  const k = num(/potassium[^0-9]*([0-9.]+)/i);
  const band = (v: number, lo: number, hi: number): SoilParam["status"] => (isNaN(v) ? "medium" : v < lo ? "low" : v > hi ? "high" : "medium");
  return [
    { key: "ph", label: "pH", unit: "", value: ph, low: 5, high: 9.5, ideal: [6.5, 7.8], status: isNaN(ph) ? "medium" : ph < 6.5 ? "low" : ph > 7.8 ? "high" : "optimal" },
    { key: "ec", label: "EC", unit: "dS/m", value: ec, low: 0, high: 4, ideal: [0, 2], status: isNaN(ec) ? "medium" : ec <= 2 ? "optimal" : ec <= 4 ? "medium" : "high" },
    { key: "oc", label: "Org. Carbon", unit: "%", value: oc, low: 0, high: 1.5, ideal: [0.4, 0.75], status: band(oc, 0.4, 0.75) },
    { key: "n", label: "Nitrogen", unit: "kg/ha", value: n, low: 0, high: 800, ideal: [280, 560], status: band(n, 280, 560) },
    { key: "p", label: "Phosphorus", unit: "kg/ha", value: p, low: 0, high: 90, ideal: [25, 50], status: band(p, 25, 50) },
    { key: "k", label: "Potassium", unit: "kg/ha", value: k, low: 0, high: 900, ideal: [141, 337], status: band(k, 141, 337) },
  ];
}

export const FERT_CROPS: Record<string, { npk: [number, number, number]; price: number; schedule: string }> = {
  Pomegranate: { npk: [110, 55, 75], price: 11850, schedule: "छेदणीनंतर 15 दिवसांनी पहिली मात्रा · फुलोऱ्यापूर्वी दुसरी · फळधारणेनंतर तिसरी — प्रत्येक वेळी ओलवाणीसोबत." },
  Cotton: { npk: [100, 50, 50], price: 7420, schedule: "पेरणीवेळी पूर्ण P+K व अर्धा N · 30–35 दिवसांनी उरलेला N विरळणीनंतर." },
  Soybean: { npk: [40, 60, 40], price: 4685, schedule: "पेरणीवेळी संपूर्ण मात्रा बी-रोवणात / ओळीबाजूला; रायझोबियम + PSB बीजप्रक्रिया करा." },
  Onion: { npk: [120, 60, 60], price: 2340, schedule: "रोपणपूर्वी शेणखतासोबत बेसल · वाढीच्या 30 व 45 व्या दिवशी टॉप-ड्रेसिंग." },
  Sugarcane: { npk: [300, 100, 120], price: 315, schedule: "बेसल + 30/60/90 दिवसांनी N विभागून; गळू नियंत्रणासह खोल सरीत मातीत द्या." },
  Rice: { npk: [120, 60, 60], price: 2275, schedule: "बेसल + टिलरिंग + पॅनिकल-इनिसिएशन अशा 3 मात्रांमध्ये N विभागा." },
  Wheat: { npk: [120, 60, 40], price: 2275, schedule: "बेसल पूर्ण P+K, अर्धा N · क्राउन-रूट स्टेजवर उरलेला N." },
  Grapes: { npk: [140, 70, 90], price: 5240, schedule: "फाउंडेशन डोस छाटणीनंतर · बेरी-सेट व व्हेरेझॉन टप्प्यांत विभागून, फर्टिगेशनद्वारे." },
  Tomato: { npk: [120, 60, 80], price: 1860, schedule: "बेसल + रोपणानंतर 20/40/60 दिवसांनी टॉप-ड्रेसिंग; फळधारणेदरम्यान K वाढवा." },
};

export function calcFertilizer(crop: string, acres: number, soil: SoilParam[]) {
  const t = FERT_CROPS[crop] ?? FERT_CROPS.Pomegranate;
  const [tN, tP, tK] = t.npk;
  const ha = acres * 0.4047;
  const avail = (k: string) => soil.find((s) => s.key === k)?.value || 0;
  const nNeed = Math.max(0, tN - avail("n") * 0.5) * ha;
  const pNeed = Math.max(0, tP - avail("p")) * ha;
  const kNeed = Math.max(0, tK - avail("k") * 0.8) * ha;
  const dapBags = Math.max(0, Math.ceil((pNeed / 0.46) / 50 * 2) / 2);
  const dapN = dapBags * 50 * 0.18;
  const ureaBags = Math.max(0, Math.ceil((Math.max(0, nNeed - dapN) / 0.46) / 45 * 2) / 2);
  const mopBags = Math.max(0, Math.ceil((kNeed / 0.6) / 50 * 2) / 2);
  const cost = ureaBags * 350 + dapBags * 1350 + mopBags * 1700;
  return { ureaBags, dapBags, mopBags, cost, revenue: t.price, schedule: t.schedule };
}

/* ---------------- predictive ---------------- */

export const YIELD_CROPS: Record<string, { perAcre: number; months: string[]; dist: number[]; monthsMr: string[] }> = {
  Pomegranate: { perAcre: 46, months: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], monthsMr: ["जाने", "फेब्रु", "मार्च", "एप्रि", "मे", "जून", "जुलै", "ऑग", "सप्टें", "ऑक्टो", "नोव्हें", "डिसें"], dist: [4, 6, 8, 10, 12, 13, 11, 9, 8, 7, 5, 3] },
  Cotton: { perAcre: 12.5, months: ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan"], monthsMr: ["जून", "जुलै", "ऑग", "सप्टें", "ऑक्टो", "नोव्हें", "डिसें", "जाने"], dist: [2, 4, 8, 16, 28, 24, 12, 6] },
  Soybean: { perAcre: 10.5, months: ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"], monthsMr: ["जून", "जुलै", "ऑग", "सप्टें", "ऑक्टो", "नोव्हें"], dist: [4, 10, 18, 30, 28, 10] },
  Sugarcane: { perAcre: 420, months: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], monthsMr: ["जाने", "फेब्रु", "मार्च", "एप्रि", "मे", "जून", "जुलै", "ऑग", "सप्टें", "ऑक्टो", "नोव्हें", "डिसें"], dist: [9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 8, 8] },
  Onion: { perAcre: 115, months: ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"], monthsMr: ["नोव्हें", "डिसें", "जाने", "फेब्रु", "मार्च", "एप्रि"], dist: [5, 12, 22, 30, 24, 7] },
  Rice: { perAcre: 22, months: ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"], monthsMr: ["जून", "जुलै", "ऑग", "सप्टें", "ऑक्टो", "नोव्हें"], dist: [3, 8, 18, 32, 30, 9] },
  Wheat: { perAcre: 18, months: ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"], monthsMr: ["नोव्हें", "डिसें", "जाने", "फेब्रु", "मार्च", "एप्रि"], dist: [4, 10, 20, 30, 27, 9] },
};

export const CROP_ETC: Record<string, number> = { Pomegranate: 5.2, Cotton: 6.1, Soybean: 5.4, Onion: 4.6, Sugarcane: 7.8, Rice: 6.6, Wheat: 4.8, Grapes: 4.2, Tomato: 5.0 };

export function irrigation(crop: string, acres: number, tempC: number, rh: number) {
  const etc = CROP_ETC[crop] ?? 5.2;
  const etAdj = etc * (0.75 + tempC / 120) * (1.12 - rh / 400);
  const litersDay = Math.round(acres * 4046.86 * etAdj);
  const dripHours = Math.round((litersDay / (acres * 7000)) * 10) / 10;
  return { etAdj: Math.round(etAdj * 100) / 100, litersDay, dripHours, discharge: Math.round(acres * 7000) };
}

export const WORKFLOW_ALERTS = [
  { sev: "critical", trigger: "SOIL_MOISTURE_LOW · Zone E", msgMr: "झोन E मध्ये माती ओलावा 21% — ड्रिप लाइन तपासा, आज सायंकाळी 90 मि. पाणी द्या.", action: "AUTO · irrigation slot booked 18:00" },
  { sev: "warning", trigger: "PEST_FORECAST · Pink bollworm", msgMr: "पुढील 6 दिवस बोंड अळी जोखीम मध्यम — फेरोमोन सापळे तपासा, 2 नवे सापळे झोन A मध्ये.", action: "TASK · scouting visit added" },
  { sev: "info", trigger: "NDVI_DECLINE · Zone C", msgMr: "झोन C मध्ये NDVI 0.05 घसरण — गेल्या आठवड्यातील तेल्या रोग क्षेत्राशी सुसंगत; पुनर्भेटीची नोंद.", action: "NOTE · linked to diagnosis #D-2214" },
  { sev: "opportunity", trigger: "PRICE_ALERT · Pomegranate", msgMr: "सोलापूर APMC मध्ये भाव 7 दिवसांत +1.2% — 40% फळे पोचण्याच्या उंबरठ्यावर, ग्रेडिंग बुक करा.", action: "AUTO · mandi watch enabled" },
];

/* ---------------- GraphRAG ---------------- */

export type NodeType = "crop" | "disease" | "pest" | "treatment" | "fertilizer" | "scheme" | "market";

export const NODE_TYPE_META: Record<NodeType, { color: string; label: string }> = {
  crop: { color: "#8fd96c", label: "Crop" },
  disease: { color: "#f25f58", label: "Disease" },
  pest: { color: "#f78f3a", label: "Pest" },
  treatment: { color: "#5db9e8", label: "Treatment" },
  fertilizer: { color: "#c09161", label: "Fertilizer" },
  scheme: { color: "#f4c04b", label: "Scheme" },
  market: { color: "#b48ce0", label: "Market" },
};

export interface GraphNode { id: string; type: NodeType; mr?: string }
export interface GraphEdge { a: string; b: string; rel: string }

export const GRAPH_DATA: Record<string, { nodes: GraphNode[]; edges: GraphEdge[] }> = {
  Pomegranate: {
    nodes: [
      { id: "Pomegranate", type: "crop", mr: "डाळिंब" },
      { id: "Bacterial blight", type: "disease", mr: "तेल्या रोग" },
      { id: "Fruit borer", type: "pest", mr: "फळ अळी" },
      { id: "Thrips", type: "pest", mr: "करड्या" },
      { id: "Copper oxychloride", type: "treatment" },
      { id: "Neem extract", type: "treatment", mr: "निंबोळी अर्क" },
      { id: "NPK 110-55-75", type: "fertilizer" },
      { id: "PMFBY insurance", type: "scheme", mr: "पीक विमा" },
      { id: "MIDH orchard subsidy", type: "scheme" },
      { id: "Solapur APMC", type: "market", mr: "सोलापूर बाजार" },
    ],
    edges: [
      { a: "Pomegranate", b: "Bacterial blight", rel: "affected_by" },
      { a: "Pomegranate", b: "Fruit borer", rel: "attacked_by" },
      { a: "Pomegranate", b: "Thrips", rel: "attacked_by" },
      { a: "Bacterial blight", b: "Copper oxychloride", rel: "treated_by" },
      { a: "Thrips", b: "Neem extract", rel: "treated_by" },
      { a: "Pomegranate", b: "NPK 110-55-75", rel: "fed_with" },
      { a: "Pomegranate", b: "PMFBY insurance", rel: "eligible_for" },
      { a: "Pomegranate", b: "MIDH orchard subsidy", rel: "eligible_for" },
      { a: "Pomegranate", b: "Solapur APMC", rel: "sold_at" },
    ],
  },
  Cotton: {
    nodes: [
      { id: "Cotton", type: "crop", mr: "कापूस" },
      { id: "Pink bollworm", type: "pest", mr: "गुलाबी बोंड अळी" },
      { id: "Aphids", type: "pest", mr: "मावा" },
      { id: "Leaf curl virus", type: "disease", mr: "पाने गुंडाळी" },
      { id: "Emamectin benzoate", type: "treatment" },
      { id: "Trichogramma release", type: "treatment" },
      { id: "NPK 100-50-50", type: "fertilizer" },
      { id: "PM-KISAN", type: "scheme" },
      { id: "CIP procurement", type: "scheme", mr: "कापूस हमीभाव" },
      { id: "Akola APMC", type: "market", mr: "अकोला बाजार" },
    ],
    edges: [
      { a: "Cotton", b: "Pink bollworm", rel: "attacked_by" },
      { a: "Cotton", b: "Aphids", rel: "attacked_by" },
      { a: "Cotton", b: "Leaf curl virus", rel: "affected_by" },
      { a: "Pink bollworm", b: "Emamectin benzoate", rel: "treated_by" },
      { a: "Pink bollworm", b: "Trichogramma release", rel: "treated_by" },
      { a: "Cotton", b: "NPK 100-50-50", rel: "fed_with" },
      { a: "Cotton", b: "PM-KISAN", rel: "eligible_for" },
      { a: "Cotton", b: "CIP procurement", rel: "eligible_for" },
      { a: "Cotton", b: "Akola APMC", rel: "sold_at" },
    ],
  },
  Soybean: {
    nodes: [
      { id: "Soybean", type: "crop", mr: "सोयाबीन" },
      { id: "Yellow mosaic", type: "disease", mr: "पिवळा करपा" },
      { id: "Stem fly", type: "pest", mr: "खोड माशी" },
      { id: "Girdle beetle", type: "pest" },
      { id: "Thiamethoxam", type: "treatment" },
      { id: "Rhizobium seed treat", type: "treatment" },
      { id: "NPK 40-60-40", type: "fertilizer" },
      { id: "PMFBY insurance", type: "scheme", mr: "पीक विमा" },
      { id: "e-NAM", type: "market" },
      { id: "Latur APMC", type: "market", mr: "लातूर बाजार" },
    ],
    edges: [
      { a: "Soybean", b: "Yellow mosaic", rel: "affected_by" },
      { a: "Soybean", b: "Stem fly", rel: "attacked_by" },
      { a: "Soybean", b: "Girdle beetle", rel: "attacked_by" },
      { a: "Yellow mosaic", b: "Thiamethoxam", rel: "treated_by" },
      { a: "Soybean", b: "Rhizobium seed treat", rel: "treated_by" },
      { a: "Soybean", b: "NPK 40-60-40", rel: "fed_with" },
      { a: "Soybean", b: "PMFBY insurance", rel: "eligible_for" },
      { a: "Soybean", b: "Latur APMC", rel: "sold_at" },
      { a: "Soybean", b: "e-NAM", rel: "sold_at" },
    ],
  },
  Onion: {
    nodes: [
      { id: "Onion", type: "crop", mr: "कांदा" },
      { id: "Purple blotch", type: "disease", mr: "करपा रोग" },
      { id: "Thrips", type: "pest", mr: "करड्या" },
      { id: "Mancozeb", type: "treatment" },
      { id: "Blue sticky traps", type: "treatment" },
      { id: "NPK 120-60-60", type: "fertilizer" },
      { id: "Onion buffer scheme", type: "scheme", mr: "कांदा साठवण योजना" },
      { id: "PMFBY insurance", type: "scheme", mr: "पीक विमा" },
      { id: "Lasalgaon APMC", type: "market", mr: "लासलगाव बाजार" },
    ],
    edges: [
      { a: "Onion", b: "Purple blotch", rel: "affected_by" },
      { a: "Onion", b: "Thrips", rel: "attacked_by" },
      { a: "Purple blotch", b: "Mancozeb", rel: "treated_by" },
      { a: "Thrips", b: "Blue sticky traps", rel: "treated_by" },
      { a: "Onion", b: "NPK 120-60-60", rel: "fed_with" },
      { a: "Onion", b: "Onion buffer scheme", rel: "eligible_for" },
      { a: "Onion", b: "PMFBY insurance", rel: "eligible_for" },
      { a: "Onion", b: "Lasalgaon APMC", rel: "sold_at" },
    ],
  },
};

/* ---------------- taxonomy ---------------- */

export const TAX_CATEGORIES = [
  { code: "NUT", name: "Nutrient management", mr: "खत व्यवस्थापन", docs: 214 },
  { code: "PDM", name: "Pest & disease", mr: "रोग व कीड नियंत्रण", docs: 342 },
  { code: "MKT", name: "Market & prices", mr: "बाजारभाव", docs: 158 },
  { code: "SCH", name: "Schemes & insurance", mr: "योजना व विमा", docs: 121 },
  { code: "IRR", name: "Irrigation", mr: "सिंचन", docs: 96 },
  { code: "SOIL", name: "Soil health", mr: "माती आरोग्य", docs: 173 },
  { code: "SEED", name: "Seed & sowing", mr: "बीज व पेरणी", docs: 88 },
  { code: "PHD", name: "Post-harvest", mr: "काढणी-पश्चात", docs: 92 },
];

export const TAX_CROPS = [
  { en: "Pomegranate", mr: "डाळिंब", hi: "अनार", group: "Horticulture", sci: "Punica granatum", aliases: ["anar", "bhagwa", "डाळिंबे"] },
  { en: "Cotton", mr: "कापूस", hi: "कपास", group: "Fibre", sci: "Gossypium hirsutum", aliases: ["kapus", "white gold"] },
  { en: "Soybean", mr: "सोयाबीन", hi: "सोयाबीन", group: "Oilseed", sci: "Glycine max", aliases: ["soyabean", "soya"] },
  { en: "Onion", mr: "कांदा", hi: "प्याज", group: "Vegetable", sci: "Allium cepa", aliases: ["kanda", "pyaaz", "lasalgaon"] },
  { en: "Sugarcane", mr: "ऊस", hi: "गन्ना", group: "Cash", sci: "Saccharum officinarum", aliases: ["oos", "ganna"] },
  { en: "Rice", mr: "भात", hi: "धान", group: "Cereal", sci: "Oryza sativa", aliases: ["bhaat", "paddy", "dhan"] },
  { en: "Wheat", mr: "गहू", hi: "गेहूं", group: "Cereal", sci: "Triticum aestivum", aliases: ["gahu", "gehu"] },
  { en: "Grapes", mr: "द्राक्षे", hi: "अंगूर", group: "Horticulture", sci: "Vitis vinifera", aliases: ["draksha", "angoor"] },
  { en: "Tomato", mr: "टोमॅटो", hi: "टमाटर", group: "Vegetable", sci: "Solanum lycopersicum", aliases: ["tamatat"] },
  { en: "Chilli", mr: "मिरची", hi: "मिर्ची", group: "Spice", sci: "Capsicum annuum", aliases: ["mirchi", "mirchi"] },
  { en: "Maize", mr: "मका", hi: "मक्का", group: "Cereal", sci: "Zea mays", aliases: ["maka", "corn", "makka"] },
  { en: "Sorghum", mr: "ज्वारी", hi: "ज्वार", group: "Cereal", sci: "Sorghum bicolor", aliases: ["jwari", "jowar"] },
];

export const TAX_STAGES = [
  { code: "SOW", en: "Sowing", mr: "पेरणी" },
  { code: "GER", en: "Germination", mr: "उगवण" },
  { code: "VEG", en: "Vegetative growth", mr: "वाढ" },
  { code: "FLW", en: "Flowering", mr: "फुलोरा" },
  { code: "FRT", en: "Fruit set", mr: "फळधारणा" },
  { code: "MAT", en: "Maturation", mr: "पोचणे" },
  { code: "HRV", en: "Harvest", mr: "काढणी" },
];

const DISTRICTS = ["Pune", "Solapur", "Nashik", "Kolhapur", "Latur", "Akola", "Nagpur", "Ahmednagar", "Sangli", "Jalgaon", "Aurangabad", "Amravati"];

const CATEGORY_HINTS: Record<string, string[]> = {
  NUT: ["fertilizer", "खत", "npk", "urea", "डॅप", "zinc", "manure", "शेणखत", "मात्रा"],
  PDM: ["disease", "रोग", "pest", "कीड", "अळी", "bollworm", "blight", "virus", "फवारणी", "spray", "telya", "तेल्या", "करपा"],
  MKT: ["price", "भाव", "mandi", "बाजार", "rate", "apmc", "market"],
  SCH: ["scheme", "योजना", "subsidy", "अनुदान", "insurance", "विमा", "drip", "ठिबक", "loan", "कर्ज"],
  IRR: ["irrigation", "सिंचन", "drip", "ठिबक", "water", "पाणी", "sprinkler"],
  SOIL: ["soil", "माती", "ph", "ec", "organic carbon", "मृदुपरीक्षण"],
  SEED: ["seed", "बी", "बीज", "sowing", "पेरणी", "variety", "जात"],
  PHD: ["harvest", "काढणी", "storage", "साठवण", "grading", "प्रतवारी", "packaging"],
};

export function resolveTaxonomy(text: string) {
  const s = text.toLowerCase();
  const crops = TAX_CROPS.filter((c) =>
    s.includes(c.en.toLowerCase()) || s.includes(c.mr) || s.includes(c.hi) || c.aliases.some((a) => s.includes(a.toLowerCase()))
  ).map((c) => c.en);
  const categories = Object.entries(CATEGORY_HINTS)
    .filter(([, hints]) => hints.some((h) => s.includes(h)))
    .map(([code]) => code);
  const district = DISTRICTS.find((d) => s.includes(d.toLowerCase())) || null;
  const stage = TAX_STAGES.find((st) => s.includes(st.en.toLowerCase()) || s.includes(st.mr)) || null;
  return { crops, categories: categories.length ? categories : ["GEN"], district, stage: stage?.en ?? null };
}

/* ---------------- RAG explorer ---------------- */

export const FUSED_DOCS = [
  { title: "Pink bollworm IPM module for Bt cotton", origin: "KB", category: "PDM", source: "CICR Nagpur bulletin", score: 0.94, snippet: "Pheromone traps @ 8/ac from 45 DAS; Emamectin benzoate 5% SG @ 0.4 g/L only after boll formation; maintain 20% non-Bt refuge…" },
  { title: "Xanthomonas → copper treatment pathway", origin: "Graph", category: "PDM", source: "knowledge graph v3.1", score: 0.91, snippet: "edge: bacterial_blight --treated_by--> copper_oxychloride (conf 0.97, 3 sources) + streptocycline adjuvant 100 ppm…" },
  { title: "Cotton modal prices — Akola APMC week 41", origin: "Agmarknet", category: "MKT", source: "data.gov.in feed", score: 0.88, snippet: "Shankar-6 modal ₹7,420/qtl, arrivals 1,240 qtl, +0.6% WoW. Kapas grade-A premium ₹320/qtl over base…" },
  { title: "Organic transition guide — Vidarbha cluster", origin: "KB", category: "NUT", source: "platform knowledge base", score: 0.83, snippet: "Year-1 organic cotton: FYM 10 t/ha + vermicompost; neem cake 250 kg/ha in furrows; expect 12–15% yield dip then premium recovery…" },
  { title: "Maharashtra pink bollworm resistance alert 2025", origin: "Web", category: "PDM", source: "agricoop.gov.in", score: 0.79, snippet: "Resistance hotspots flagged in Akola, Yavatmal, Wardha. Rotate MoA groups; avoid consecutive diamide sprays…" },
  { title: "FARM_101 history — last 3 advisories", origin: "Memory", category: "GEN", source: "farm_memory_store", score: 0.74, snippet: "Aug 12: blight advisory (zone C, treated) · Jul 30: N top-dressing · Jul 02: drip schedule revised to 5.2 mm/day…" },
  { title: "Bio-control economics: Trichogramma vs spray", origin: "KB", category: "PDM", source: "ICAR-NBAIM study", score: 0.71, snippet: "Trichogramma releases cut bollworm damage 61% at ₹1,850/season/ha vs ₹4,200 chemical baseline across 3 districts…" },
  { title: "Cotton futures — NCDEX commentary", origin: "Web", category: "MKT", source: "nseindia research", score: 0.66, snippet: "Dec cotton futures +1.1% on export enquiry uptick; monsoon-deficit districts may tighten Nov arrivals…" },
];

export const RAG_BACKENDS = [
  { name: "Hybrid (BM25 + dense)", backend: "qdrant · bge-m3", ms: 214, docs: 5 },
  { name: "Dense vector", backend: "qdrant local", ms: 96, docs: 4 },
  { name: "GraphRAG walk", backend: "networkx · v3.1", ms: 188, docs: 3 },
  { name: "Tools layer", backend: "mandi · weather · calc", ms: 322, docs: 2 },
  { name: "Web retrieval", backend: "serper-lite", ms: 1240, docs: 3 },
];

/* ---------------- data factory ---------------- */

export const FACTORY_INITIAL = {
  records: 62412,
  langs: { mr: 46, en: 38, hi: 16 },
  cats: { PDM: 24, MKT: 18, NUT: 16, SCH: 12, IRR: 10, SOIL: 9, SEED: 6, PHD: 5 },
  dupPct: 1.8,
  missingPct: 2.1,
  gaps: [
    { crop: "Maize", category: "PDM", gap: "no stem-borer QA pairs in mr" },
    { crop: "Sorghum", category: "NUT", gap: "no fertilizer plan in hi" },
    { crop: "Grapes", category: "PHD", gap: "cold-storage QA sparse (<40)" },
    { crop: "Chilli", category: "MKT", gap: "Guntur vs Byadgi price QA missing" },
  ],
  workers: [
    { id: "W-INGEST", desc: "Pull 14 sources → lake/raw", eta: "8s" },
    { id: "W-QUALITY", desc: "Validate → clean → dedup", eta: "12s" },
    { id: "W-STANDARDIZE", desc: "Schema v1 → train/val/test", eta: "6s" },
    { id: "W-ANALYZE", desc: "Coverage & gap report", eta: "9s" },
    { id: "W-QASYNTH", desc: "Expert QA packs ≥ 62.5k", eta: "74s" },
  ],
};

/* ---------------- misc helpers ---------------- */

export function formatINR(n: number): string {
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(1)} L`;
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

export function istClock(): string {
  return new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date());
}
