# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
import uuid
import time
from fpdf import FPDF
import os

# NEW: import the agent runner
from app.agents import run_journey

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Nextify Backend MVP")

# Allow your GH Pages origin while testing locally (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://donnaoftadeh.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory job store (MVP)
# -----------------------------
class Submission(BaseModel):
    # pydantic v2: use 'pattern' instead of 'regex'
    journey_type: str = Field(..., pattern="^(company|industry|product|idea)$")
    payload: Dict[str, Any] = Field(
        ...,
        description="Raw form fields from the page"
    )

JOBS: Dict[str, Dict[str, Any]] = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
os.makedirs(PDF_DIR, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def _pdf_path(job_id: str) -> str:
    return os.path.join(PDF_DIR, f"{job_id}.pdf")

def _safe_get(d: Dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    return str(v) if v is not None else default

def generate_pdf_from_result(job_id: str, submission: Submission, result: Dict[str, Any]) -> str:
    """
    Create a simple PDF from agent result.
    result structure (from agents.run_journey):
        {
          "journey_type": ...,
          "payload": {...},
          "provider": "gemini" | "openai" | "stub",
          "outputs": {"context": "...", "research": "...", "strategy": "...", "mvp": "...", "summary": "..."},
          "brief": "...",
        }
    """
    outputs = result.get("outputs", {})
    provider = result.get("provider", "unknown")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Nextify — Innovation Brief", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Job ID: {job_id}", ln=True)
    pdf.cell(0, 8, f"Journey Type: {submission.journey_type}", ln=True)
    pdf.cell(0, 8, f"LLM Provider: {provider}", ln=True)
    pdf.ln(4)

    # Submitted fields
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Submitted Fields:", ln=True)
    pdf.set_font("Arial", "", 12)
    for k, v in submission.payload.items():
        pdf.multi_cell(0, 7, f"- {k}: {v}")
    pdf.ln(3)

    # Sections (if present)
    def section(title: str, key: str):
        txt = _safe_get(outputs, key, "")
        if not txt:
            return
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, txt)
        pdf.ln(2)

    section("Context", "context")
    section("Research", "research")
    section("Strategy", "strategy")
    section("MVP Plan", "mvp")

    # Summary / brief
    brief = _safe_get(result, "brief", "")
    if brief:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Executive Summary", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, brief)
        pdf.ln(2)

    # Footer note
    pdf.set_font("Arial", "I", 11)
    pdf.multi_cell(
        0, 6,
        "Note: This is an automatically generated brief. When you plug in your "
        "full multi-agent graph, these sections will reflect the parallel agent outputs."
    )

    path = _pdf_path(job_id)
    pdf.output(path)
    return path

# -----------------------------
# Background worker
# -----------------------------
async def run_job(job_id: str, submission: Submission):
    """
    Run the multi-step agent journey and generate the PDF.
    Uses the 'update' function so the UI can poll /api/status and show progress.
    """

    job = JOBS[job_id]
    job["status"] = "running"
    job["step"] = "Starting"
    job["progress"] = 1
    job["message"] = "Initializing…"

    # progress callback used by agents.py
    def update(status: str, pct: int, meta: Dict[str, Any] | None = None):
        job["status"] = "running"
        job["step"] = status
        job["progress"] = max(0, min(100, int(pct)))
        job["message"] = status
        if meta:
            job["meta"] = meta

    try:
        # Call the agent pipeline (Gemini/OpenAI/stub depending on env)
        result = await run_journey(submission.journey_type, submission.payload, update)

        # Create PDF from result
        update("Writing PDF…", 98, None)
        pdf_path = generate_pdf_from_result(job_id, submission, result)
        job["pdf_path"] = pdf_path
        job["result"] = result

        job["progress"] = 100
        job["step"] = "Complete"
        job["status"] = "done"
        job["message"] = "Report ready."
    except Exception as e:
        job["progress"] = 100
        job["step"] = "Error"
        job["status"] = "error"
        job["message"] = f"Error: {e}"

# -----------------------------
# API endpoints
# -----------------------------
@app.post("/api/submit")
async def submit(submission: Submission):
    """Accepts a submission, starts a background task, returns a job_id."""
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
    # Fire and forget
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
        "meta": job.get("meta"),
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
