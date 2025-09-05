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

from .agents import run_agents

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Nextify Backend MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for prod
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
    payload: Dict[str, Any] = Field(..., description="Raw form fields from the page")


JOBS: Dict[str, Dict[str, Any]] = {}
DATA_DIR = os.path.join("data")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
os.makedirs(PDF_DIR, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------
def _pdf_path(job_id: str) -> str:
    return os.path.join(PDF_DIR, f"{job_id}.pdf")


def generate_pdf_from_markdown(job_id: str, title: str, markdown_text: str) -> str:
    """
    Simple PDF output of the final report markdown.
    For now, render as plain text with section breaks. (You can beautify later.)
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 9, f"Nextify — Innovation Brief\n{title}")
    pdf.ln(2)

    pdf.set_font("Arial", "", 11)
    for line in markdown_text.splitlines():
        # basic handling for bullets/headers
        if line.startswith("#") or line.endswith(":"):
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 6, line.strip("# ").strip())
            pdf.set_font("Arial", "", 11)
        else:
            pdf.multi_cell(0, 6, line)

    path = _pdf_path(job_id)
    pdf.output(path)
    return path


# -----------------------------
# Background job
# -----------------------------
async def run_job(job_id: str, submission: Submission):
    """
    Execute the multi-agent pipeline and create a PDF.
    """

    # helper to update job status
    def update(step: str, progress: int | None, message: str):
        job = JOBS[job_id]
        job["status"] = "running"
        job["step"] = step
        if progress is not None:
            job["progress"] = progress
        job["message"] = message

    job = JOBS[job_id]
    job["status"] = "running"
    job["step"] = "Initialize"
    job["progress"] = 2
    job["message"] = "Initializing…"

    # Run agents (parallel) and assemble markdown
    try:
        pieces, final_md = await run_agents(
            submission.journey_type,
            submission.payload,
            status_cb=update
        )
    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Agent error: {e}"
        return

    update("Write PDF", 92, "Writing PDF…")
    # Title is included as the first line by templates renderer; extract a concise title:
    title_line = (final_md.splitlines()[0] if final_md else "Nextify v4 Report")[:120]
    pdf_path = generate_pdf_from_markdown(job_id, title_line, final_md)

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
        "payload": submission.payload,
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
