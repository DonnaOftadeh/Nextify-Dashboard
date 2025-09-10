# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple
import asyncio, uuid, time, os, re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("app/.env")

from .agents import run_multi_agent

# ReportLab
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

# Charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Models / storage
# -----------------------------
class Submission(BaseModel):
    journey_type: str = Field(..., pattern="^(company|industry|product|idea)$")
    payload: Dict[str, Any]

JOBS: Dict[str, Dict[str, Any]] = {}

BASE_DIR  = Path(__file__).resolve().parent.parent
PDF_DIR   = BASE_DIR / "data" / "pdf"
CHART_DIR = BASE_DIR / "data" / "charts"
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

FONT_DIR    = Path(__file__).resolve().parent / "fonts"
DEJAVU_REG  = FONT_DIR / "DejaVuSans.ttf"
DEJAVU_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

# UI steps for status
UI_STEPS = [
    "Parse Submission","Agent Orchestration","Howler Whisperer","The Marauder","The Legilimens",
    "The Seer","Room of Requirement (R1)","The Pensive (v1)","The Headmaster",
    "Room of Requirement (R2)","The Pensive (v2)","The Sorting Hat","The Story Weaver","Write Report (PDF)"
]

# -----------------------------
# Styles & helpers
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
    h1    = ParagraphStyle("Heading1", parent=base["Heading1"], fontName=bold_name, fontSize=16, leading=20, spaceBefore=10, spaceAfter=6)
    h2    = ParagraphStyle("Heading2", parent=base["Heading2"], fontName=bold_name, fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
    h3    = ParagraphStyle("Heading3", parent=base["Heading3"], fontName=bold_name, fontSize=12, leading=16, spaceBefore=8, spaceAfter=6)
    body  = ParagraphStyle("Body", parent=base["BodyText"], fontName=body_name, fontSize=10.0, leading=13.5, spaceAfter=6)
    cell  = ParagraphStyle("Cell", parent=body, fontSize=9, leading=12)  # smaller for tables
    return {"title": title, "h1": h1, "h2": h2, "h3": h3, "body": body, "cell": cell}

def _escape_basic(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _md_inline_to_html(s: str) -> str:
    # very small inline markdown: **bold**, *italic*, __bold__, _italic_
    t = re.sub(r"\*\*(.+?)\*\*", r"«b»\1«/b»", s)
    t = re.sub(r"__(.+?)__",   r"«b»\1«/b»", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"«i»\1«/i»", t)
    t = re.sub(r"_(.+?)_",     r"«i»\1«/i»", t)
    t = _escape_basic(t)
    return t.replace("«b»","<b>").replace("«/b»","</b>").replace("«i»","<i>").replace("«/i»","</i>")

# ---------- Markdown tables ----------
def _is_separator_row(cells: List[str]) -> bool:
    """
    Detect a markdown header separator row like:
    | --- | :---: | ---: |
    """
    if not cells:
        return False
    for c in cells:
        c = c.strip().replace(" ", "")
        if not c:
            return False
        if not re.match(r"^:?-{3,}:?$", c):
            return False
    return True

def _parse_md_table(block_lines: List[str]) -> List[List[str]]:
    # Split into rows/cells
    rows_raw: List[List[str]] = []
    for ln in block_lines:
        s = ln.strip()
        if s.startswith("|"): s = s[1:]
        if s.endswith("|"):   s = s[:-1]
        rows_raw.append([c.strip() for c in s.split("|")])

    # Remove a single header-separator row (the typical second line)
    rows: List[List[str]] = []
    for idx, r in enumerate(rows_raw):
        if idx == 1 and _is_separator_row(r):
            continue
        # also skip any stray separator-only rows later in the table
        if _is_separator_row(r):
            continue
        rows.append(r)
    return rows

def _as_table(flow_rows: List[List[str]], styles: Dict[str, ParagraphStyle], inner_width: float) -> Table:
    # Convert to Paragraph cells for proper wrap
    data = [[Paragraph(_md_inline_to_html(c), styles["cell"]) for c in row] for row in flow_rows]
    ncols = max(len(r) for r in flow_rows)

    # Dynamic equal-width columns within the available inner width
    col_width = inner_width / float(ncols)
    col_widths = [col_width] * ncols

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "DejaVu-Bold" if "DejaVu-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LEADING",  (0,0), (-1,-1), 11),
        ("GRID",     (0,0), (-1,-1), 0.25, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F2F2F2")),
        ("VALIGN",   (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
    ]))
    return tbl

def _parse_to_flowables(
    text: str,
    styles: Dict[str, ParagraphStyle],
    inner_width: float
) -> Tuple[List, List[List[str]], List[int]]:
    """
    Markdown-ish formatter with real ReportLab tables:
    - '#', '##', '###' -> headings
    - '- ' / '* '      -> bullets
    - '1. '            -> ordered bullets
    - pipe tables      -> RL Table with wrapping and dynamic equal widths
    Returns: (story, rice_rows, rice_table_positions)
      - rice_rows: parsed rows of the first RICE table (for chart data)
      - rice_table_positions: indices in 'story' where a RICE chart should be inserted
    """
    story: List = []
    lines = (text or "").splitlines()
    list_mode = None
    list_buffer: List[str] = []
    rice_rows: List[List[str]] = []
    rice_positions: List[int] = []

    def flush_list():
        nonlocal list_mode, list_buffer
        if not list_buffer: return
        items = [ListItem(Paragraph(_md_inline_to_html(x), styles["body"])) for x in list_buffer]
        btype = "1" if list_mode == "number" else "bullet"
        start = None if btype == "1" else "•"
        story.append(ListFlowable(items, bulletType=btype, start=start, leftIndent=14))
        story.append(Spacer(1, 3))
        list_mode, list_buffer = None, []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # table block (header + subsequent rows)
        if stripped.startswith("|"):
            flush_list()
            block = [stripped]; i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip()); i += 1
            rows = _parse_md_table(block)
            if rows and len(rows) >= 1:
                header = [h.lower() for h in rows[0]]
                tbl = _as_table(rows, styles, inner_width)
                story.append(tbl)
                story.append(Spacer(1, 6))
                # if it is a RICE table, remember to place a chart right after this table
                if header[:6] == ["feature","reach","impact","confidence","effort","rice"]:
                    rice_rows = rows
                    rice_positions.append(len(story))  # insert chart at this position (after spacer)
            continue

        if not stripped:
            flush_list(); story.append(Spacer(1, 4)); i += 1; continue

        if stripped.startswith("### "):
            flush_list(); story.append(Paragraph(_md_inline_to_html(stripped[4:]), styles["h3"])); i += 1; continue
        if stripped.startswith("## "):
            flush_list(); story.append(Paragraph(_md_inline_to_html(stripped[3:]), styles["h2"])); i += 1; continue
        if stripped.startswith("# "):
            flush_list(); story.append(Paragraph(_md_inline_to_html(stripped[2:]), styles["h1"])); i += 1; continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            mode = "bullet"
            if list_mode not in (None, mode): flush_list()
            list_mode = mode
            list_buffer.append(stripped[2:].strip()); i += 1; continue

        if re.match(r"^\d+\.\s+", stripped):
            mode = "number"
            if list_mode not in (None, mode): flush_list()
            list_mode = mode
            list_buffer.append(re.sub(r"^\d+\.\s+", "", stripped)); i += 1; continue

        flush_list()
        story.append(Paragraph(_md_inline_to_html(stripped), styles["body"]))
        i += 1

    flush_list()
    return story, rice_rows, rice_positions

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

def _extract_rice_scores(rows: List[List[str]]) -> List[Tuple[str, float]]:
    out = []
    for r in rows[1:]:  # header already stripped; we removed separator lines earlier
        try:
            name = r[0]
            rice = float(str(r[5]).replace(",", "").replace("$","").strip())
            out.append((name, rice))
        except Exception:
            continue
    return out

def _make_rice_chart(job_id: str, scores: List[Tuple[str, float]]) -> str:
    if not scores: return ""
    labels = [a for a,_ in scores]; vals = [b for _,b in scores]
    plt.figure(figsize=(6.0, 3.2))
    plt.bar(labels, vals)
    plt.xlabel("Feature"); plt.ylabel("RICE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = str(CHART_DIR / f"rice_{job_id}.png")
    plt.savefig(out_path); plt.close()
    return out_path

# -----------------------------
# PDF generator
# -----------------------------
def generate_pdf(job_id: str, journey_type: str, payload: Dict[str, Any], report_text: str) -> str:
    if not isinstance(report_text, str) or not report_text.strip():
        report_text = (
            "⚠️ No content was produced by the agent pipeline.\n\n"
            "Please re-run with more details."
        )

    _register_fonts()
    styles = _build_styles()

    # Margins (slightly tighter) and inner width for table sizing
    LEFT = 14 * mm; RIGHT = 14 * mm; TOP = 14 * mm; BOTTOM = 12 * mm
    PAGE_W = A4[0]
    inner_width = PAGE_W - LEFT - RIGHT

    title_text, filename_label = _make_report_title(journey_type, payload)
    out_path = str(PDF_DIR / f"{filename_label}.pdf")

    # Build the story
    story: List = []
    story.append(Paragraph(_md_inline_to_html(title_text), styles["title"]))
    story.append(Spacer(1, 6))

    flowables, rice_rows, rice_positions = _parse_to_flowables(report_text, styles, inner_width)

    # If there is a RICE table, generate chart and insert it right after that table (no heading)
    if rice_rows:
        scores = _extract_rice_scores(rice_rows)
        chart_path = _make_rice_chart(job_id, scores)
        if chart_path and os.path.exists(chart_path):
            # Insert image at each recorded position (usually one)
            for pos in rice_positions:
                flowables.insert(pos, Image(chart_path, width=min(inner_width, 160*mm), height=90*mm))
                flowables.insert(pos, Spacer(1, 4))

    story.extend(flowables)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=TOP, bottomMargin=BOTTOM,
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
    filename = os.path.basename(job["pdf_path"])  # equals title (sanitized)
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
