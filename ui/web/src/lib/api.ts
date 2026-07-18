/**
 * KrushiVerse API client — talks to FastAPI (app.main).
 * Uses relative URLs in production (same origin) and Vite proxy in dev.
 */

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "";

async function request<T = any>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${path}: ${detail.slice(0, 240)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request("/"),

  query: (body: {
    query: string;
    farm_id?: string;
    language?: string;
    enable_web?: boolean;
  }) =>
    request("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  advancedRag: (body: {
    query: string;
    crop?: string;
    location?: string;
    top_k?: number;
    enable_web?: boolean;
    enable_tools?: boolean;
    force_web?: boolean;
  }) =>
    request("/api/rag/advanced", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  knowledgeStats: () => request("/api/knowledge/stats"),
  ragBackends: () => request("/api/rag/backends"),

  weather: (location = "Solapur") =>
    request(`/api/live/weather?location=${encodeURIComponent(location)}`),
  market: (crop?: string, district?: string) => {
    const q = new URLSearchParams();
    if (crop) q.set("crop", crop);
    if (district) q.set("district", district);
    const s = q.toString();
    return request(`/api/live/market${s ? `?${s}` : ""}`);
  },
  iot: (farmId = "FARM_101") =>
    request(`/api/live/iot?farm_id=${encodeURIComponent(farmId)}`),
  satellite: (farmId = "FARM_101", crop = "Pomegranate") =>
    request(
      `/api/live/satellite?farm_id=${encodeURIComponent(farmId)}&crop=${encodeURIComponent(crop)}`
    ),

  graph: (crop: string) =>
    request(`/api/knowledge/graph/${encodeURIComponent(crop)}`),

  visionDiagnose: async (file?: File, cropHint = "Pomegranate") => {
    const fd = new FormData();
    if (file) fd.append("file", file);
    fd.append("crop_hint", cropHint);
    return request("/api/vision/diagnose", { method: "POST", body: fd });
  },

  soilOcr: async (rawText: string) => {
    const fd = new FormData();
    fd.append("raw_ocr_text", rawText);
    return request("/api/soil/ocr", { method: "POST", body: fd });
  },

  yieldPredict: (body: Record<string, unknown>) =>
    request("/api/predict/yield", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  irrigationPredict: (body: Record<string, unknown>) =>
    request("/api/predict/irrigation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  fertilizerPredict: (body: Record<string, unknown>) =>
    request("/api/predict/fertilizer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  memory: (farmId = "FARM_101") =>
    request(`/api/memory/${encodeURIComponent(farmId)}`),

  workflowAudit: (farmId = "FARM_101") =>
    request(`/api/workflows/audit?farm_id=${encodeURIComponent(farmId)}`, {
      method: "POST",
    }),

  taxonomy: () => request("/api/taxonomy"),
  taxonomyResolve: (text: string) =>
    request(`/api/taxonomy/resolve?text=${encodeURIComponent(text)}`),

  lakeStatus: () => request("/api/lake/status"),
  lakeAnalyze: () => request("/api/lake/analyze"),
  lakeAnalyzeRun: (execute = false) =>
    request(`/api/lake/analyze?execute=${execute}`, { method: "POST" }),
  lakeQasynth: () => request("/api/lake/qasynth"),
  lakeKg: () => request("/api/lake/kg"),
  lakeTokenizer: () => request("/api/lake/tokenizer"),
  lakePretrain: () => request("/api/lake/pretrain"),

  agmarknet: (commodity?: string, state = "Maharashtra") => {
    const q = new URLSearchParams({ state });
    if (commodity) q.set("commodity", commodity);
    return request(`/api/opendata/agmarknet?${q}`);
  },
};

/** Map planner /api/query response into Assistant UI shape */
export function mapQueryToAsst(resp: any, lang: "mr" | "en") {
  const kl = resp?.knowledge_layer || {};
  const answer = String(resp?.synthesized_answer || "No advisory generated.");
  const paragraphs = answer
    .split(/\n+/)
    .map((s: string) => s.trim())
    .filter(Boolean);
  const citations = (kl.citations || []).map((c: any) => ({
    title: c.title || c.source || "Source",
    origin: c.origin || "KB",
    src: c.source || c.url || "local",
    url: c.url || undefined,
  }));
  const tools = kl.tools_used || [];
  return {
    key: "live_api",
    crop: resp?.crop || "Crop",
    cropMr: resp?.crop || "पीक",
    agents: resp?.active_agent_names || ["Advisory Agent"],
    metrics: {
      fused: kl.fused_document_count ?? 0,
      web: kl.web_result_count ?? 0,
      tools: Array.isArray(tools) ? tools.length : 0,
      latencyMs: 0,
    },
    mr: lang === "mr" ? paragraphs : paragraphs,
    en: lang === "en" ? paragraphs : paragraphs,
    citations:
      citations.length > 0
        ? citations
        : [{ title: "Local knowledge layer", origin: "KB", src: "platform" }],
    tools: Array.isArray(tools) ? tools : ["query.pipeline"],
    raw: resp,
  };
}
