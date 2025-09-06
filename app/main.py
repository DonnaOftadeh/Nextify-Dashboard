# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
import uuid
import time
from fpdf import FPDF
import os
from pathlib import Path
import unicodedata

from dotenv import load_dotenv
load_dotenv("app/.env")

from .agents import run_multi_agent

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

FONT_DIR = Path(__file__).resolve().parent / "fonts"
DEJAVU_REG = FONT_DIR / "DejaVuSans.ttf"
DEJAVU_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

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

# -----------------------------
# Helpers (PDF + wrapping)
# -----------------------------
def _pdf_path(job_id: str) -> str:
    return os.path.join(PDF_DIR, f"{job_id}.pdf")

def _ascii_sanitize(text: str) -> str:
    """Fallback: strip non-ASCII so FPDF core fonts won't choke (when no DejaVu)."""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def _chunk_long_token(tok: str, limit: int = 120) -> str:
    """Insert soft breaks into very long unbroken tokens so FPDF can wrap them."""
    if len(tok) <= limit:
        return tok
    return " ".join(tok[i:i+limit] for i in range(0, len(tok), limit))

def _soft_wrap_line(s: str, limit: int = 120) -> str:
    """Break any very-long 'word' (no spaces) into chunks; keep normal words as-is."""
    if not s:
        return s
    tokens = s.split(" ")
    tokens = [_chunk_long_token(t, limit) for t in tokens]
    return " ".join(tokens)

def _use_unicode_fonts(pdf: FPDF) -> bool:
    """
    Try to register DejaVu fonts for full Unicode support.
    Returns True if fonts loaded, else False.
    """
    try:
        if DEJAVU_REG.exists() and DEJAVU_BOLD.exists():
            pdf.add_font("DejaVu", "", str(DEJAVU_REG), uni=True)
            pdf.add_font("DejaVu", "B", str(DEJAVU_BOLD), uni=True)
            return True
        return False
    except Exception:
        return False

def generate_pdf(job_id: str, title: str, report_text: str) -> str:
    """
    Unicode-safe PDF writer. Uses DejaVu (if present). Otherwise falls back
    to ASCII-sanitized text with core fonts.
    """
    if not isinstance(report_text, str) or not report_text.strip():
        report_text = (
            "⚠️ No content was produced by the agent pipeline.\n\n"
            "This can happen if the LLM call failed or returned nothing.\n"
            f"Please check /api/debug/{job_id}/raw for the raw output, or the server logs for errors."
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    has_unicode = _use_unicode_fonts(pdf)

    if has_unicode:
        pdf.set_font("DejaVu", "B", 16)
        pdf.multi_cell(0, 9, _soft_wrap_line(title, 120))
        pdf.ln(3)
        pdf.set_font("DejaVu", "", 11)
        for line in report_text.splitlines():
            line = line if isinstance(line, str) else str(line)
            if len(line) > 8000:
                line = line[:8000] + " …(truncated)"
            line = _soft_wrap_line(line, 120)
            pdf.multi_cell(0, 6, line)
    else:
        safe_title = _ascii_sanitize(title)
        pdf.set_font("Arial", "B", 16)
        pdf.multi_cell(0, 9, _soft_wrap_line(safe_title, 120))
        pdf.ln(3)
        pdf.set_font("Arial", "", 11)
        for line in report_text.splitlines():
            line = _ascii_sanitize(line if isinstance(line, str) else str(line))
            if len(line) > 8000:
                line = line[:8000] + " …(truncated)"
            line = _soft_wrap_line(line, 120)
            pdf.multi_cell(0, 6, line)

    path = _pdf_path(job_id)
    pdf.output(path)
    return path

# -----------------------------
# Pipeline
# -----------------------------
async def _run_pipeline(job_id: str, submission: Submission):
    job = JOBS[job_id]
    job["raw_report"] = ""  # keep raw LLM text for debugging

    try:
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
            # Update UI step and progress (spread across the agent phases)
            job["step"] = section_title
            job["progress"] = min(6 + int(idx * (90 / 11)), 95)
            job["message"] = message

        # Run agents
        report_text = await run_multi_agent(submission.model_dump(), cb)
        job["raw_report"] = report_text or ""

        # Final step: write PDF
        job["step"] = UI_STEPS[-1]
        job["message"] = "Writing PDF…"
        job["progress"] = 97
        await asyncio.sleep(0.2)

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

    except Exception as e:
        # Mark job failed so the UI can display the reason
        job["status"] = "failed"
        job["step"] = "Error"
        job["message"] = f"Pipeline error: {e}"
        job["progress"] = 100
        import traceback; traceback.print_exc()

# -----------------------------
# API endpoints
# -----------------------------
@app.post("/api/submit")
async def submit(submission: Submission):
    try:
        print("SUBMIT received:", {
            "journey_type": submission.journey_type,
            "payload_keys": list(submission.payload.keys())
        })

        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "created_at": time.time(),
            "status": "queued",
            "step": "Queued",
            "progress": 0,
            "message": "Job queued.",
            "pdf_path": None,
            "journey_type": submission.journey_type,
            "raw_report": ""
        }
        asyncio.create_task(_run_pipeline(job_id, submission))
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"/api/submit error: {e}")

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

@app.get("/api/debug/{job_id}/raw")
async def debug_raw(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    raw = job.get("raw_report", "")
    if not raw:
        raw = "(no raw report stored)"
    return PlainTextResponse(raw)

@app.get("/")
async def root():
    return {"ok": True, "service": "Nextify Backend (Gemini)"}
