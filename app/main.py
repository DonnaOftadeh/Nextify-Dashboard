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
from .agents import run_multi_agent
from dotenv import load_dotenv
load_dotenv("app/.env")
# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Nextify Backend (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory job store (MVP)
# -----------------------------
class Submission(BaseModel):
    # pydantic v2 uses 'pattern' instead of 'regex'
    journey_type: str = Field(..., pattern="^(company|industry|product|idea)$")
    payload: Dict[str, Any] = Field(..., description="Raw form fields from the page")

JOBS: Dict[str, Dict[str, Any]] = {}
PDF_DIR = os.path.join("data", "pdf")
os.makedirs(PDF_DIR, exist_ok=True)

# phases (UI progress)
UI_STEPS = [
    "Parse Submission",
    "Agent Orchestration",
    "Howler Whisperer",
    "The Marauder",
    "The Legilimens",
    "The Seer",
    "Room of Requirement (R1)",
    "The Pensive (v1)",
    "The Headmaster",
    "Room of Requirement (R2)",
    "The Pensive (v2)",
    "The Sorting Hat",
    "The Story Weaver",
    "Write Report (PDF)"
]

def _pdf_path(job_id: str) -> str:
    return os.path.join(PDF_DIR, f"{job_id}.pdf")

def generate_pdf(job_id: str, title: str, report_text: str) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 9, title)
    pdf.ln(3)
    pdf.set_font("Arial", "", 11)

    # split long content into lines to avoid encoding issues
    for line in report_text.splitlines():
        pdf.multi_cell(0, 6, line)

    path = _pdf_path(job_id)
    pdf.output(path)
    return path

async def _run_pipeline(job_id: str, submission: Submission):
    job = JOBS[job_id]

    # Step 1: parse
    job["status"] = "running"
    job["step"] = UI_STEPS[0]
    job["progress"] = 3
    job["message"] = "Validating input…"
    await asyncio.sleep(0.2)

    # Step 2: orchestration
    job["step"] = UI_STEPS[1]
    job["progress"] = 6
    job["message"] = "Spinning up agents…"
    await asyncio.sleep(0.2)

    # progress callback from agents.py
    def cb(idx: int, section_title: str, message: str):
        # map agent idx to UI step index (starting after 'Agent Orchestration')
        ui_index = min(1 + idx, len(UI_STEPS) - 1)
        job["step"] = section_title
        # spread progress across 10 agent phases
        job["progress"] = min(6 + int(idx * (90 / 11)), 95)
        job["message"] = message

    # Run agents (sequential for now)
    report_text = await run_multi_agent(submission.model_dump(), cb)

    # Final step: write PDF
    job["step"] = UI_STEPS[-1]
    job["message"] = "Writing PDF…"
    job["progress"] = 97
    await asyncio.sleep(0.3)

    # pick a nice title
    subject = (submission.payload.get("bench_company")
               or submission.payload.get("company_name")
               or submission.payload.get("industry")
               or submission.payload.get("product_name")
               or submission.payload.get("idea_text")
               or "Nextify Report")
    title = f"Nextify — {submission.journey_type.capitalize()} Report — {subject}"

    pdf_path = generate_pdf(job_id, title, report_text)
    job["pdf_path"] = pdf_path
    job["progress"] = 100
    job["step"] = "Complete"
    job["status"] = "done"
    job["message"] = "Report ready."

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
    asyncio.create_task(_run_pipeline(job_id, submission))
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
    return {"ok": True, "service": "Nextify Backend (Gemini)"}
