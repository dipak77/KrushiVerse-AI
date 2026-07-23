import io
import csv
import json
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mini.workers.tester import KrushiBulkTester

app = FastAPI(title="KrushiVerseAI Mini v3-18M Pro")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}
TESTER = None

def get_tester():
    global TESTER
    if TESTER is None:
        TESTER = KrushiBulkTester()
    return TESTER

@app.post("/api/test/bulk")
async def bulk(items: list[dict], bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "queued", "results": [], "total": len(items), "done": 0}
    
    async def run():
        JOBS[job_id]["status"] = "running"
        t = get_tester()
        for q in items:
            r = t.score_one(q)
            JOBS[job_id]["results"].append(r)
            JOBS[job_id]["done"] += 1
            await asyncio.sleep(0.01)
        JOBS[job_id]["status"] = "done"

    bg.add_task(run)
    return {"job_id": job_id, "total": len(items)}

@app.get("/results/{job_id}")
def results(job_id: str):
    return JOBS.get(job_id, {"error": "not found"})

@app.get("/stream")
async def stream(job_id: str):
    async def ev():
        while True:
            j = JOBS.get(job_id, {})
            yield f"data: {json.dumps({'done': j.get('done',0), 'total': j.get('total',0), 'status': j.get('status')})}\n\n"
            if j.get("status") == "done":
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(ev(), media_type="text/event-stream")

@app.get("/report/{job_id}/html", response_class=HTMLResponse)
def html(job_id: str):
    j = JOBS.get(job_id, {})
    rows = j.get("results", [])
    head = """<!doctype html><html><head><meta charset='utf-8'><style>
    body{font-family:system-ui;margin:20px} table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #ccc;padding:6px;text-align:left;font-size:13px}
    .PASS{background:#d4edda}.FAIL{background:#f8d7da}.mid{background:#fff3cd}
    </style></head><body>"""
    body = [f"<h2>KrushiVerse v3-18M Pro — {job_id}</h2>",
            f"<table><tr><th>Q</th><th>Crop</th><th>Intent</th><th>Latency(ms)</th><th>Crop</th><th>Intent</th><th>KW</th><th>Ground</th><th>Final</th><th>Status</th><th>Response</th></tr>"]
    for r in rows:
        st = r.get("status","")
        cls = "PASS" if st=="PASS" else "mid" if r.get("final_score",0)>=0.4 else "FAIL"
        body.append(f"<tr class='{cls}'><td>{r.get('id','')}</td><td>{r.get('crop','')}</td><td>{r.get('intent','')}</td><td>{r.get('latency_ms',0)}</td><td>{r.get('crop_match',0)}</td><td>{r.get('intent_match',0)}</td><td>{r.get('keyword_hit',0)}</td><td>{r.get('grounding_ok',0)}</td><td>{r.get('final_score',0)}</td><td>{st}</td><td>{r.get('response','')[:300]}</td></tr>")
    body.append("</table></body></html>")
    return head + "".join(body)

@app.get("/report/{job_id}/csv")
def csv_report(job_id: str):
    j = JOBS.get(job_id, {})
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["id","crop","intent","latency_ms","crop_match","intent_match","keyword_hit","checklist_ok","section_ok","grounding_ok","final_score","status","response"])
    for r in j.get("results", []):
        w.writerow([r.get("id"),r.get("crop"),r.get("intent"),r.get("latency_ms"),
                    r.get("crop_match"),r.get("intent_match"),r.get("keyword_hit"),
                    r.get("checklist_ok"),r.get("section_ok"),r.get("grounding_ok"),
                    r.get("final_score"),r.get("status"),r.get("response","")])
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=bulk_{job_id}.csv"})
