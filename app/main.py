# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple
import asyncio
import uuid
import time
import os
from pathlib import Path
import unicodedata
import re
import io

from dotenv import load_dotenv
load_dotenv("app/.env")

from .agents import run_multi_agent
from .templates import get_prompt_bundle  # ensure your agents use this

# ---------- ReportLab ----------
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem,
    Table, TableStyle, Image
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------- Matplotlib (for charts) ----------
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # savefig used below

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Nextify Backend (ReportLab PDF)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory job store (MVP)
# -----------------------------
class Submission(BaseModel):
    journey_type: str = Field(..., pattern="^(company|industry|product|idea)$")
    payload: Dict[str, Any] = Field(..., description="Raw form fields from the page")

JOBS: Dict[str, Dict[str, Any]] = {}
PDF_DIR = os.path.join("data", "pdf")
CHART_DIR = os.path.join("data", "charts")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

FONT_DIR = Path(__file__).resolve().parent / "fonts"
DEJAVU_REG = FONT_DIR / "DejaVuSans.ttf"
DEJAVU_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

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
# Helpers
# -----------------------------
def _register_fonts():
    try:
        if DEJAVU_REG.exists():
            pdfmetrics.registerFont(TTFont("DejaVu", str(DEJAVU_REG)))
        if DEJAVU_BOLD.exists():
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(DEJAVU_BOLD)))
    except Exception:
        pass

def _build_styles():
    base = getSampleStyleSheet()
    body_name = "DejaVu" if "DejaVu" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold_name = "DejaVu-Bold" if "DejaVu-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

    title = ParagraphStyle("Title", parent=base["Title"], fontName=bold_name, fontSize=18, leading=22, spaceAfter=8)
    h1 = ParagraphStyle("Heading1", parent=base["Heading1"], fontName=bold_name, fontSize=16, leading=20, spaceBefore=10, spaceAfter=6)
    h2 = ParagraphStyle("Heading2", parent=base["Heading2"], fontName=bold_name, fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("Heading3", parent=base["Heading3"], fontName=bold_name, fontSize=12, leading=16, spaceBefore=8, spaceAfter=6)
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName=body_name, fontSize=10.5, leading=14, spaceAfter=6)

    return {"title": title, "h1": h1, "h2": h2, "h3": h3, "body": body}

def _escape_basic(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _md_inline_to_html(s: str) -> str:
    # very small inline markdown: **bold**, *italic*, __bold__, _italic_
    t = re.sub(r"\*\*(.+?)\*\*", r"«b»\1«/b»", s)
    t = re.sub(r"__(.+?)__",   r"«b»\1«/b»", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"«i»\1«/i»", t)
    t = re.sub(r"_(.+?)_",   r"«i»\1«/i»", t)
    t = _escape_basic(t)
    return t.replace("«b»", "<b>").replace("«/b»", "</b>").replace("«i»", "<i>").replace("«/i»", "</i>")

# ---------- Markdown blocks → Flowables (+ capture RICE tables) ----------
def _parse_md_table(block_lines: List[str]) -> Tuple[List[List[str]], bool]:
    """
    Convert a markdown table block into a 2D list.
    Returns (data, is_rice_table)
    """
    # strip pipes and whitespace
    rows = []
    for ln in block_lines:
        ln = ln.strip()
        if ln.startswith("|"): ln = ln[1:]
        if ln.endswith("|"): ln = ln[:-1]
        parts = [c.strip() for c in ln.split("|")]
        rows.append(parts)

    # detect RICE header
    header = [h.lower() for h in rows[0]]
    is_rice = header[:6] == ["feature", "reach(wk)", "impact(1-5)", "confidence(0-1)", "effort(pw)", "rice"]
    return rows, is_rice

def _extract_rice_scores(rows: List[List[str]]) -> List[Tuple[str, float]]:
    """Return [(feature, rice_score), ...] from a parsed markdown table."""
    out = []
    for r in rows[2:]:  # skip header + separator
        try:
            name = r[0]
            rice = float(str(r[5]).replace(",", "").strip())
            out.append((name, rice))
        except Exception:
            continue
    return out

def _make_rice_chart(job_id: str, scores: List[Tuple[str, float]]) -> str:
    """
    Draw a simple bar chart of RICE scores.
    Saves to data/charts/rice_<job>.png and returns the path.
    """
    if not scores:
        return ""
    labels = [a for a, _ in scores]
    vals   = [b for _, b in scores]

    plt.figure(figsize=(5.4, 3.2))
    plt.bar(labels, vals)  # default style/colors per requirement
    plt.title("RICE Scores")
    plt.xlabel("Feature")
    plt.ylabel("Score")
    plt.tight_layout()
    out_path = os.path.join(CHART_DIR, f"rice_{job_id}.png")
    plt.savefig(out_path)  # Matplotlib savefig doc. :contentReference[oaicite:0]{index=0}
    plt.close()
    return out_path

def _parse_to_flowables(text: str, styles: Dict[str, ParagraphStyle]) -> Tuple[List, List[List[str]]]:
    """
    Markdown-ish formatter:
      - # / ## / ### headings → Paragraph
      - - / * bullets → ListFlowable
      - 1. numbered → ListFlowable
      - markdown tables → Table with styling
      - blank lines → Spacer
    Returns (story_flowables, rice_table_rows_if_any)
    """
    story: List = []
    lines = (text or "").splitlines()
    list_mode, list_buffer = None, []
    rice_rows: List[List[str]] = []

    def flush_list():
        nonlocal list_mode, list_buffer
        if not list_buffer:
            return
        items = [ListItem(Paragraph(_md_inline_to_html(x), styles["body"])) for x in list_buffer]
        if list_mode == "number":
            story.append(ListFlowable(items, bulletType="1", leftIndent=14))
        else:
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14))
        story.append(Spacer(1, 3))
        list_mode, list_buffer = None, []

    i = 0
    N = len(lines)
    while i < N:
        raw = lines[i].rstrip("\n")
        stripped = raw.strip()

        # Markdown table detection: look for header row with '|' then a separator row
        if "|" in stripped and i + 1 < N and re.match(r'^\s*\|?\s*:?-{2,}.*\|.*-+:?\s*\|?\s*$', lines[i + 1]):
            # collect table block
            block = [stripped]
            j = i + 1
            while j < N and "|" in lines[j]:
                block.append(lines[j].strip())
                j += 1
            flush_list()
            data, is_rice = _parse_md_table(block)
            # style a nice table (ReportLab TableStyle docs). :contentReference[oaicite:1]{index=1}
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f5ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2a44")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7cddd")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 6))
            if is_rice and not rice_rows:
                rice_rows = data
            i = j
            continue

        if not stripped:
            flush_list()
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Headings
        if stripped.startswith("### "):
            flush_list()
            story.append(Paragraph(_md_inline_to_html(stripped[4:]), styles["h3"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_list()
            story.append(Paragraph(_md_inline_to_html(stripped[3:]), styles["h2"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flush_list()
            story.append(Paragraph(_md_inline_to_html(stripped[2:]), styles["h1"]))
            i += 1
            continue

        # Bullets
        if stripped.startswith("- ") or stripped.startswith("* "):
            mode = "bullet"
            content = stripped[2:].strip()
            if list_mode not in (None, mode):
                flush_list()
            list_mode = mode
            list_buffer.append(content)
            i += 1
            continue

        # Numbered list
        if re.match(r"^\d+\.\s+", stripped):
            mode = "number"
            content = re.sub(r"^\d+\.\s+", "", stripped).strip()
            if list_mode not in (None, mode):
                flush_list()
            list_mode = mode
            list_buffer.append(content)
            i += 1
            continue

        # Normal paragraph
        flush_list()
        story.append(Paragraph(_md_inline_to_html(stripped), styles["body"]))
        i += 1

    flush_list()
    return story, rice_rows

def _make_report_title(journey_type: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Exact title/filename rules:
    - company : "Next level <company> proposal by Nextify"
    - product : "The next breakthrough <product> proposal by Nextify"
    - idea    : "The idea of <idea> proposal by Nextify"
    - industry: "The next breakthrough in <industry> market proposal by Nextify"
    """
    company  = (payload.get("bench_company") or payload.get("company_name") or "Company").strip()
    product  = (payload.get("product_name") or "Product").strip()
    industry = (payload.get("industry") or "Industry").strip()
    idea     = (payload.get("idea_title") or payload.get("idea_text") or "Idea").strip()

    jt = (journey_type or "").lower().strip()
    if jt == "company":
        title = f"Next level {company} proposal by Nextify"
    elif jt == "product":
        title = f"The next breakthrough {product} proposal by Nextify"
    elif jt == "industry":
        title = f"The next breakthrough in {industry} market proposal by Nextify"
    else:
        title = f"The idea of {idea} proposal by Nextify"

    filename_label = "".join(c for c in title if c not in r'\/:*?"<>|').strip()
    return title, filename_label

def generate_pdf(job_id: str, journey_type: str, payload: Dict[str, Any], report_text: str) -> str:
    if not isinstance(report_text, str) or not report_text.strip():
        report_text = (
            "⚠️ No content was produced by the agent pipeline.\n\n"
            "Please re-run with more details."
        )

    _register_fonts()
    styles = _build_styles()
    title_text, filename_label = _make_report_title(journey_type, payload)
    out_path = os.path.join(PDF_DIR, f"{filename_label}.pdf")

    # Build story from markdown + collect possible RICE table
    story: List = []
    story.append(Paragraph(_md_inline_to_html(title_text), styles["title"]))
    story.append(Spacer(1, 6))

    flowables, rice_rows = _parse_to_flowables(report_text, styles)
    story.extend(flowables)

    # If we saw a RICE table, generate a bar chart and append it
    if rice_rows:
        scores = _extract_rice_scores(rice_rows)
        chart_path = _make_rice_chart(job_id, scores)  # saves PNG. Matplotlib savefig. :contentReference[oaicite:2]{index=2}
        if chart_path and os.path.exists(chart_path):
            story.append(Spacer(1, 6))
            story.append(Paragraph(_md_inline_to_html("RICE Scores (Chart)"), styles["h3"]))
            story.append(Spacer(1, 4))
            story.append(Image(chart_path, width=150*mm, height=90*mm))  # keep aspect tidy
            story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm,
        title=title_text, author="Nextify",
    )
    doc.build(story)
    return out_path

# -----------------------------
# Pipeline
# -----------------------------
async def _run_pipeline(job_id: str, submission: Submission):
    job = JOBS[job_id]
    job["raw_report"] = ""
    try:
        job["status"], job["step"], job["progress"], job["message"] = "running", UI_STEPS[0], 3, "Validating input…"
        await asyncio.sleep(0.2)
        job["step"], job["progress"], job["message"] = UI_STEPS[1], 6, "Spinning up agents…"
        await asyncio.sleep(0.2)

        def cb(idx: int, section_title: str, message: str):
            job["step"], job["progress"], job["message"] = section_title, min(6 + int(idx * (90 / 11)), 95), message

        # NOTE: ensure agents.py uses get_prompt_bundle() internally.
        report_text = await run_multi_agent(submission.model_dump(), cb)
        job["raw_report"] = report_text or ""

        job["step"], job["message"], job["progress"] = UI_STEPS[-1], "Writing PDF…", 97
        await asyncio.sleep(0.2)
        pdf_path = generate_pdf(job_id, submission.journey_type, submission.payload, report_text)
        job.update({"pdf_path": pdf_path, "progress": 100, "step": "Complete", "status": "done", "message": "Report ready."})
    except Exception as e:
        job.update({"status": "failed", "step": "Error", "message": f"Pipeline error: {e}", "progress": 100})
        import traceback; traceback.print_exc()

# -----------------------------
# API endpoints
# -----------------------------
@app.post("/api/submit")
async def submit(submission: Submission):
    try:
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"created_at": time.time(),"status": "queued","step": "Queued","progress": 0,
                        "message": "Job queued.","pdf_path": None,"journey_type": submission.journey_type,"raw_report": ""}
        asyncio.create_task(_run_pipeline(job_id, submission))
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"/api/submit error: {e}")

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id,"status": job["status"],"step": job["step"],"progress": job["progress"],
            "message": job["message"],"ready": job["status"]=="done"}

@app.get("/api/result/{job_id}")
async def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job["pdf_path"]:
        return JSONResponse({"error": "Result not ready"}, status_code=202)
    filename = os.path.basename(job["pdf_path"])
    return FileResponse(job["pdf_path"], media_type="application/pdf", filename=filename)

@app.get("/api/debug/{job_id}/raw")
async def debug_raw(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    raw = job.get("raw_report","")
    return PlainTextResponse(raw or "(no raw report stored)")

@app.get("/")
async def root():
    return {"ok": True, "service": "Nextify Backend (ReportLab PDF)"}
