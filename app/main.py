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

from dotenv import load_dotenv
load_dotenv("app/.env")  # read LLM_PROVIDER / GEMINI_* / OPENAI_* if present

from .agents import run_multi_agent  # orchestrator

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
import matplotlib.pyplot as plt

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Nextify Backend (ReportLab PDF)")

# DEV-friendly CORS (works with rotating ngrok URLs).
# Tighten for production (restrict to your GitHub Pages origin + reserved domain).
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
TXT_DIR = os.path.join("data", "txt")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

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
    """Create paragraph styles using DejaVu if available."""
    base = getSampleStyleSheet()

    # Pick font names depending on registration
    body_name = "DejaVu" if "DejaVu" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold_name = "DejaVu-Bold" if "DejaVu-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

    title = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontName=bold_name,
        fontSize=18,
        leading=22,
        spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "Heading1",
        parent=base["Heading1"],
        fontName=bold_name,
        fontSize=16,
        leading=20,
        spaceBefore=10,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "Heading2",
        parent=base["Heading2"],
        fontName=bold_name,
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "Heading3",
        parent=base["Heading3"],
        fontName=bold_name,
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName=body_name,
        fontSize=10.0,   # slightly smaller so more fits
        leading=13.5,
        spaceAfter=6,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=base["BodyText"],
        fontName=body_name,
        fontSize=9.0,    # smaller for tables to fit nicely
        leading=12,
        spaceAfter=0,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=table_cell,
        fontName=bold_name,
        fontSize=9.2,
    )

    return {"title": title, "h1": h1, "h2": h2, "h3": h3, "body": body,
            "table_cell": table_cell, "table_header": table_header}

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

# ---------- Markdown table parsing & rendering ----------
def _parse_md_table(block_lines: List[str]) -> List[List[str]]:
    """
    Convert a markdown table block into a 2D list (header + rows).
    Removes pure separator rows (---).
    """
    rows: List[List[str]] = []
    for raw in block_lines:
        ln = raw.strip()
        # Allow backslash preceding pipe blocks not necessary; we expect leading '|'
        if ln.startswith("|"): ln = ln[1:]
        if ln.endswith("|"): ln = ln[:-1]
        parts = [c.strip() for c in ln.split("|")]
        # Drop rows that are just separators like :--- or ---:
        if all(re.match(r"^:?-{2,}:?$", c) for c in parts):
            continue
        rows.append(parts)
    return rows

def _is_rice_header(header: List[str]) -> bool:
    canonical = [h.strip().lower().replace(" ", "") for h in header]
    # Accept minor variations (with/without parentheses)
    target = ["feature", "reach(wk)", "impact(1-5)", "confidence(0-1)", "effort(pw)", "rice"]
    canonical_target = [t.replace(" ", "") for t in target]
    # Fallback: allow "reach", "impact", "confidence", "effort", "rice"
    simple_target = ["feature", "reach", "impact", "confidence", "effort", "rice"]
    return canonical[:6] == canonical_target or canonical[:6] == simple_target

def _extract_rice_scores(rows: List[List[str]]) -> List[Tuple[str, float]]:
    """Return [(feature, rice_score), ...] from a parsed markdown table rows (first row is header)."""
    out = []
    if not rows or len(rows[0]) < 6:  # needs RICE columns
        return out
    for r in rows[1:]:
        if len(r) < 6:
            continue
        name = r[0]
        try:
            rice_val = float(str(r[5]).replace(",", "").strip())
        except Exception:
            continue
        out.append((name, rice_val))
    return out

def _make_rice_chart(job_id: str, scores: List[Tuple[str, float]]) -> str:
    """
    Draw a bar chart of RICE scores with 45° label rotation.
    Saves to data/charts/rice_<job>.png and returns the path.
    """
    if not scores:
        return ""
    labels = [a for a, _ in scores]
    vals   = [b for _, b in scores]

    plt.figure(figsize=(5.6, 3.0))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RICE")  # no title as requested
    plt.tight_layout()
    out_path = os.path.join(CHART_DIR, f"rice_{job_id}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path

# ---------- Markdown blocks → Flowables ----------
def _parse_to_flowables(job_id: str, text: str, styles: Dict[str, ParagraphStyle]) -> List:
    """
    Markdown-ish formatter:
    - '#', '##', '###' -> H1/H2/H3
    - '- ' or '* '     -> unordered bullets (grouped)
    - '1. '            -> ordered list (grouped)
    - blocks starting with '|' -> real table with wrapping cells; detect RICE and
      insert bar chart image immediately after that table (no extra title).
    - blank lines     -> spacing
    - otherwise       -> body paragraph
    """
    story: List = []
    lines = (text or "").splitlines()

    list_mode = None  # None | "bullet" | "number"
    list_buffer: List[str] = []
    pipe_buffer: List[str] = []

    def flush_list():
        nonlocal list_mode, list_buffer
        if not list_buffer:
            return
        items = [ListItem(Paragraph(_md_inline_to_html(x), styles["body"])) for x in list_buffer]
        if list_mode == "number":
            story.append(ListFlowable(items, bulletType="1", leftIndent=14))
        else:
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14))
        story.append(Spacer(1, 4))
        list_mode, list_buffer = None, []

    def _render_table(rows: List[List[str]]):
        """Render rows as ReportLab Table with dynamic widths and wrapping cells."""
        if not rows:
            return

        # Prepare header + cells with Paragraph (so wrapping works) and styles
        header = rows[0]
        body_rows = rows[1:]

        # Convert to Paragraph cells
        def cell_para(txt: str, header=False):
            style = styles["table_header"] if header else styles["table_cell"]
            return Paragraph(_md_inline_to_html(txt), style)

        data: List[List[Any]] = [
            [cell_para(h, header=True) for h in header]
        ] + [[cell_para(c) for c in r] for r in body_rows]

        # Compute dynamic column widths: split page width by number of columns with min/max
        # Page width inside margins
        page_width = A4[0] - (14*mm + 14*mm)  # must match margins used in SimpleDocTemplate
        ncols = max(len(r) for r in rows)
        # Heuristic: base each column on equal share, clamp between 50 and 140 mm
        base = page_width / ncols
        col_widths = [max(50*mm, min(140*mm, base)) for _ in range(ncols)]

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), styles["table_header"].fontName, styles["table_header"].fontSize),
            ("FONT", (0, 1), (-1, -1), styles["table_cell"].fontName, styles["table_cell"].fontSize),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBEFORE", (0, 0), (-1, -1), 0.25, colors.HexColor("#1e293b")),
            ("LINEABOVE", (0, 0), (-1, -1), 0.25, colors.HexColor("#1e293b")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#1e293b")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 6))

        # If it's a RICE table, add chart immediately after (no section title)
        if _is_rice_header(header):
            scores = _extract_rice_scores(rows)
            chart_path = _make_rice_chart(job_id, scores)
            if chart_path and os.path.exists(chart_path):
                # reasonable image width (fit within page width), maintain aspect
                img_w = min(150*mm, page_width)
                img_h = img_w * 0.55
                story.append(Image(chart_path, width=img_w, height=img_h))
                story.append(Spacer(1, 6))

    def flush_pipe():
        nonlocal pipe_buffer
        if not pipe_buffer:
            return
        rows = _parse_md_table(pipe_buffer)
        _render_table(rows)
        pipe_buffer = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # blank line
        if not stripped:
            flush_list(); flush_pipe()
            story.append(Spacer(1, 4))
            continue

        # pipe-table line
        if stripped.startswith("|"):
            flush_list()
            pipe_buffer.append(stripped)
            continue

        # headings
        if stripped.startswith("### "):
            flush_list(); flush_pipe()
            story.append(Paragraph(_md_inline_to_html(stripped[4:]), styles["h3"])); continue
        if stripped.startswith("## "):
            flush_list(); flush_pipe()
            story.append(Paragraph(_md_inline_to_html(stripped[3:]), styles["h2"])); continue
        if stripped.startswith("# "):
            flush_list(); flush_pipe()
            story.append(Paragraph(_md_inline_to_html(stripped[2:]), styles["h1"])); continue

        # unordered bullets
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_pipe()
            mode = "bullet"
            content = stripped[2:].strip()
            if list_mode not in (None, mode): flush_list()
            list_mode = mode
            list_buffer.append(content)
            continue

        # numbered list
        if re.match(r"^\d+\.\s+", stripped):
            flush_pipe()
            mode = "number"
            content = re.sub(r"^\d+\.\s+", "", stripped).strip()
            if list_mode not in (None, mode): flush_list()
            list_mode = mode
            list_buffer.append(content)
            continue

        # paragraph
        flush_list(); flush_pipe()
        story.append(Paragraph(_md_inline_to_html(stripped), styles["body"]))

    flush_list(); flush_pipe()
    return story

def _ascii_sanitize(text: str) -> str:
    """Fallback: strip non-ASCII if fonts missing (we aim to keep Unicode with DejaVu)."""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def _make_report_title(journey_type: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Exact title/filename rules (filename = sanitized copy of title):
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
    """Build the PDF with wrapped tables and inline RICE chart."""
    if not isinstance(report_text, str) or not report_text.strip():
        report_text = (
            "⚠️ No content was produced by the agent pipeline.\n\n"
            "Please re-run with more details."
        )

    _register_fonts()
    styles = _build_styles()
    title_text, filename_label = _make_report_title(journey_type, payload)
    out_path = os.path.join(PDF_DIR, f"{filename_label}.pdf")

    story: List = []
    story.append(Paragraph(_md_inline_to_html(title_text), styles["title"]))
    story.append(Spacer(1, 6))

    # parse markdown-ish content into flowables; create chart(s) inline when needed
    story.extend(_parse_to_flowables(job_id, report_text, styles))

    # Create PDF (slightly narrower margins than before)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title_text,
        author="Nextify",
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

        report_text = await run_multi_agent(submission.model_dump(), cb)
        job["raw_report"] = report_text or ""

        # Optional: keep raw text to ./data/txt for debugging
        try:
            with open(os.path.join(TXT_DIR, f"{job_id}.txt"), "w", encoding="utf-8") as f:
                f.write(job["raw_report"])
        except Exception:
            pass

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
    filename = os.path.basename(job["pdf_path"])  # equals the Title (sanitized)
    return FileResponse(job["pdf_path"], media_type="application/pdf", filename=filename)

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
    return {"ok": True, "service": "Nextify Backend (ReportLab PDF)"}
