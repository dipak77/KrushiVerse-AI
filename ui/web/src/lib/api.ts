/**
 * KrushiVerse API client — all dashboard data via backend.
 * Prefer /api/ui/* unified endpoints; fall back only if network fails.
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

function jsonPost<T = any>(path: string, body: Record<string, unknown>) {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  health: () => request("/api/health"),

  // Unified dashboard
  bootstrap: (farmId = "FARM_101") =>
    request(`/api/ui/bootstrap?farm_id=${encodeURIComponent(farmId)}`),
  live: (opts?: { farm_id?: string; location?: string; crop?: string }) => {
    const q = new URLSearchParams({
      farm_id: opts?.farm_id || "FARM_101",
      location: opts?.location || "Solapur",
      crop: opts?.crop || "Pomegranate",
    });
    return request(`/api/ui/live?${q}`);
  },
  visionSamples: () => request("/api/ui/vision/samples"),
  graph: (crop: string) => request(`/api/ui/graph/${encodeURIComponent(crop)}`),
  soil: (body: { crop?: string; acreage?: number; soil_text?: string; farm_id?: string }) =>
    jsonPost("/api/ui/soil", body),
  predict: (body: {
    crop?: string;
    acreage?: number;
    temperature_c?: number;
    humidity_pct?: number;
    farm_id?: string;
  }) => jsonPost("/api/ui/predict", body),
  taxonomy: () => request("/api/ui/taxonomy"),
  factory: () => request("/api/ui/factory"),
  rag: (body: {
    query: string;
    crop?: string;
    top_k?: number;
    enable_web?: boolean;
    enable_tools?: boolean;
  }) => jsonPost("/api/ui/rag", body),

  // Core agent / vision
  query: (body: {
    query: string;
    farm_id?: string;
    language?: string;
    enable_web?: boolean;
    use_local_llm?: boolean;
  }) => jsonPost("/api/query", body),

  visionDiagnose: async (file?: File, cropHint = "Pomegranate") => {
    const fd = new FormData();
    if (file) fd.append("file", file);
    fd.append("crop_hint", cropHint);
    return request("/api/vision/diagnose", { method: "POST", body: fd });
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
    mr: paragraphs,
    en: paragraphs,
    citations:
      citations.length > 0
        ? citations
        : [{ title: "Local knowledge layer", origin: "KB", src: "platform" }],
    tools: Array.isArray(tools) ? tools : ["query.pipeline"],
    raw: resp,
  };
}
