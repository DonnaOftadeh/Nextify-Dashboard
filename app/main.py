# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import asyncio
import uuid
import time
from fpdf import FPDF
import os

# NEW: agent pipeline
from .agents import run_multi_agent

app = FastAPI(title="Nextify Backend MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory job store
# -----------------------------
class Submission(BaseModel):
    # pydantic v2: regex->pattern
    journey_type: str = Field(..., pattern="^(company|industry|product|idea)$")
    payload: Dict[str, Any] = Field(..., description="Raw form fields from the page")

JOBS: Dict[str, Dict[str, Any]] = {}
PDF_DIR = os.path.join("data", "pdf")
os.makedirs(PDF_DIR, exist_ok=True)

# -----------------------------
# PDF helper
# -----------------------------
def _pdf_path(job_id: str) -> str:
    return os.path.join(PDF_DIR, f"{job_id}.pdf")

def generate_pdf(job_id: str, submission: Submission, final_markdown: Optional[str] = None) -> str:
    """Simple PDF writer. If final_markdown is provided, we dump it verbatim."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Nextify — Innovation Brief", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Job ID: {job_id}", ln=True)
    pdf.cell(0, 8, f"Journey Type: {submission.journey_type}", ln=True)
    pdf.ln(4)

    if not final_markdown:
        # fallback to raw fields (older behavior)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Submitted Fields:", ln=True)
        pdf.set_font("Arial", "", 12)
        for k, v in submission.payload.items():
            pdf.multi_cell(0, 7, f"- {k}: {v}")
        pdf.ln(6)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Notes:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, "No agent output was generated.")
    else:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Report:", ln=True)
        pdf.set_font("Arial", "", 12)
        # FPDF has no markdown, so print plain text
        for line in final_markdown.splitlines():
            pdf.multi_cell(0, 6, line)

    path = _pdf_path(job_id)
    pdf.output(path)
    return path

# -----------------------------
# Background worker
# -----------------------------
async def run_job(job_id: str, submission: Submission):
    """Run the multi-agent pipeline and stream coarse progress to JOBS."""
    job = JOBS[job_id]
    job["status"] = "running"
    job["step"] = "Starting"
    job["progress"] = 2
    job["message"] = "Initializing…"

    try:
        # call the agent pipeline
        events, final_md = run_multi_agent(submission.journey_type, submission.payload)

        # stream events to our in-memory job
        for ev in events:
            job["step"] = ev["step"]
            job["message"] = ev["message"]
            job["progress"] = max(job["progress"], int(ev["progress"]))
            await asyncio.sleep(0.05)  # tiny yield for UX

        job["message"] = "Writing PDF…"
        job["step"] = "Compose"
        await asyncio.sleep(0.05)

        pdf_path = generate_pdf(job_id, submission, final_markdown=final_md)
        job["pdf_path"] = pdf_path
        job["progress"] = 100
        job["step"] = "Complete"
        job["status"] = "done"
        job["message"] = "Report ready."

    except Exception as e:
        job["status"] = "error"
        job["step"] = "Error"
        job["message"] = f"Failed: {e}"

# -----------------------------
# API endpoints
# -----------------------------
@app.post("/api/submit")
async def submit(submission: Submission):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "created_at": time.time(),
        "status": "queued",
        "step": "Queued",
        "progress": 0,
        "message": "Job queued.",
        "pdf_path": None,
        "journey_type": submission.journey_type,
    }
    asyncio.create_task(run_job(job_id, submission))
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "step": job["step"],
        "progress": job["progress"],
        "message": job["message"],
        "ready": job["status"] == "done",
    }

@app.get("/api/result/{job_id}")
async def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job["pdf_path"]:
        return JSONResponse({"error": "Result not ready"}, status_code=202)
    return FileResponse(job["pdf_path"], media_type="application/pdf", filename=f"{job_id}.pdf")

@app.get("/")
async def root():
    return {"ok": True, "service": "Nextify Backend MVP"}
