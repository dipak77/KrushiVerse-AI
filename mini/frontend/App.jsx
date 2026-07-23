import React, { useState, useEffect } from "react";

const API = "http://localhost:8000";

export default function App() {
  const [jobId, setJobId] = useState(null);
  const [results, setResults] = useState([]);
  const [progress, setProgress] = useState({ done: 0, total: 0, status: "idle" });
  const [stats, setStats] = useState({ total: 0, pass: 0, latency: 0, crop: 0, intent: 0 });

  const max = (a, b) => (a > b ? a : b);

  const runBulk = async () => {
    try {
      const res = await fetch(`${API}/api/test/bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([
          { id: "q001", query: "उन्हाळ्यात कापसाला ठिबक किती तास?", crop: "Cotton", intent: "irrigation" },
          { id: "q002", query: "सोयाबीनला खत कोणते द्यावे?", crop: "Soybean", intent: "fertilizer" },
          { id: "q006", query: "pigeon pea price in Latur mandi", crop: "Tur", intent: "market" },
          { id: "q030", query: "माती परीक्षण प्रयोगशाळा नोंदणी?", crop: "Generic", intent: "soil" }
        ])
      });
      const data = await res.json();
      const job_id = data.job_id;
      setJobId(job_id);

      const es = new EventSource(`${API}/stream?job_id=${job_id}`);
      es.onmessage = async (e) => {
        const p = JSON.parse(e.data);
        setProgress(p);
        if (p.status === "done") {
          const rr = await fetch(`${API}/results/${job_id}`).then((r) => r.json());
          const resList = rr.results || [];
          setResults(resList);
          const n = resList.length;
          const pass = resList.filter((x) => x.status === "PASS").length;
          setStats({
            total: n,
            pass: pass,
            latency: Math.round(resList.reduce((a, b) => a + (b.latency_ms || 0), 0) / max(1, n)),
            crop: Math.round((resList.filter((x) => x.crop_match).length / max(1, n)) * 100),
            intent: Math.round((resList.filter((x) => x.intent_match).length / max(1, n)) * 100)
          });
          es.close();
        }
      };
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui", padding: 20 }}>
      <h1>KrushiVerseAI Mini v3 — 18M Pro (1024 block)</h1>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 10, marginBottom: 20 }}>
        {[
          ["Total", stats.total],
          ["Pass", stats.pass],
          ["Latency(ms)", stats.latency],
          ["Crop %", stats.crop],
          ["Intent %", stats.intent]
        ].map(([k, v]) => (
          <div key={k} style={{ padding: 14, border: "1px solid #ccc", borderRadius: 8, background: "#f9f9f9" }}>
            <div style={{ fontSize: 11, color: "#666" }}>{k}</div>
            <div style={{ fontSize: 22, fontWeight: 600 }}>{v}</div>
          </div>
        ))}
      </div>
      <button onClick={() => runBulk()} style={{ padding: "8px 16px", cursor: "pointer", marginRight: 10 }}>
        Run bulk_30 Benchmark
      </button>
      <button onClick={() => window.open(`${API}/report/${jobId}/html`)} disabled={!jobId} style={{ marginRight: 10 }}>
        HTML Report
      </button>
      <button onClick={() => window.open(`${API}/report/${jobId}/csv`)} disabled={!jobId}>
        CSV Export
      </button>
      <div style={{ margin: "16px 0" }}>
        {progress.done}/{progress.total} — {progress.status}
      </div>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
        <thead>
          <tr>
            {["ID", "Crop", "Intent", "Latency", "Crop", "Intent", "KW", "Final", "Status"].map((h) => (
              <th key={h} style={{ border: "1px solid #ccc", padding: 6, background: "#eee" }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr
              key={i}
              style={{
                background: r.final_score >= 0.7 ? "#d4edda" : r.final_score >= 0.4 ? "#fff3cd" : "#f8d7da"
              }}
            >
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.id}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.crop}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.intent}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.latency_ms}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.crop_match}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.intent_match}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.keyword_hit}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.final_score}</td>
              <td style={{ padding: 6, border: "1px solid #ccc" }}>{r.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
