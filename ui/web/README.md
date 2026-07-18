# KrushiVerse OS — React UI

Reference front-end design (dark agri OS: icon rail, farm sidebar, mandi ticker, 9 workspaces).

## Dev

```bash
# Terminal 1 — FastAPI
cd ../..
.\venv\Scripts\uvicorn.exe app.main:app --reload --port 8000

# Terminal 2 — Vite (proxies /api → :8000)
cd ui/web
npm install
npm run dev
```

Open http://127.0.0.1:5173

## Production build (served by FastAPI)

```bash
cd ui/web
npm install
npm run build
```

Then start the API and open:

- http://127.0.0.1:8000/  (serves SPA when `dist/` exists)
- http://127.0.0.1:8000/ui
- http://127.0.0.1:8000/dashboard
- http://127.0.0.1:8000/api/health  (JSON health)

## Tabs ↔ APIs

| Tab | Primary API |
|---|---|
| AI Assistant | `POST /api/query` |
| Live Feeds | `/api/live/*` |
| Vision Lab | `POST /api/vision/diagnose` |
| Soil Lab | local engine + optional OCR |
| GraphRAG | `GET /api/knowledge/graph/{crop}` |
| Predictive | `/api/predict/*`, workflows |
| RAG Explorer | `POST /api/rag/advanced` |
| Taxonomy | `/api/taxonomy` |
| Data Factory | `/api/lake/*` |

Offline-safe: each tab falls back to rich demo data if the API is down.
