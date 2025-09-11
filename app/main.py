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

# --------- your orchestrator ---------
from .agents import run_multi_agent

# --------- ReportLab + matplotlib ---------
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Image
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- app ----------
app = FastAPI(title="Nextify Backend (ReportLab PDF)")

# IMPORTANT: allow GitHub Pages + your ngrok URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://donnaoftadeh.github.io",
        "https://2b14bd5555af.ngrok-free.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Submission(BaseModel):
    journey_type: str = Field(..., pattern="^(company|industry|product|idea)$")
    payload: Dict[str, Any]

JOBS: Dict[str, Dict[str, Any]] = {}
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "data" / "pdf"
CHART_DIR = BASE_DIR / "data" / "charts"
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path(__file__).resolve().parent / "fonts"
DEJAVU_REG = FONT_DIR / "DejaVuSans.ttf"
DEJAVU_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

UI_STEPS = [
    "Parse Submission","Agent Orchestration","Howler Whisperer","The Marauder",
    "The Legilimens","The Seer","Room of Requirement (R1)","The Pensive (v1)",
    "The Headmaster","Room of Requirement (R2)","The Pensive (v2)",
    "The Sorting Hat","The Story Weaver","Write Report (PDF)"
]

# ---------- styles / formatting ----------
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
    body  = ParagraphStyle("Body", parent=base["BodyText"], fontName=body_name, fontSize=10.2, leading=13.8, spaceAfter=5)
    mono  = ParagraphStyle("Mono", parent=base["BodyText"], fontName="Courier", fontSize=10, leading=13, spaceAfter=6)
    return {"title": title, "h1": h1, "h2": h2, "h3": h3, "body": body, "mono": mono}

def _escape_basic(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _md_inline_to_html(s: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"«b»\1«/b»", s)
    t = re.sub(r"__(.+?)__",   r"«b»\1«/b»", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"«i»\1«/i»", t)
    t = re.sub(r"_(.+?)_",   r"«i»\1«/i»", t)
    t = _escape_basic(t)
    return t.replace("«b»", "<b>").replace("«/b»", "</b>").replace("«i»", "<i>").replace("«/i»", "</i>")

def _parse_to_flowables(text: str, styles) -> Tuple[List, List[List[str]]]:
    """Markdown-ish → flowables. Also returns any pipe-table rows (for RICE parse)."""
    story: List = []
    lines = (text or "").splitlines()

    list_mode = None
    list_buf: List[str] = []
    pipe_buf: List[str] = []
    rice_rows: List[List[str]] = []

    def flush_list():
        nonlocal list_mode, list_buf
        if list_buf:
            items = [ListItem(Paragraph(_md_inline_to_html(x), styles["body"])) for x in list_buf]
            if list_mode == "number":
                story.append(ListFlowable(items, bulletType="1", leftIndent=14))
            else:
                story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14))
            story.append(Spacer(1, 3))
        list_mode, list_buf = None, []

    def flush_pipe():
        nonlocal pipe_buf, rice_rows
        if not pipe_buf:
            return
        # keep monospaced “visual” table (no separator-only rows)
        cleaned = [ln for ln in pipe_buf if set(ln.replace("|", "").strip()) != {"-"}]
        if cleaned:
            block = "<br/>".join(_escape_basic(row) for row in cleaned)
            story.append(Paragraph(block, styles.get("mono", styles["body"])))
            story.append(Spacer(1, 4))
            # keep original split cells for RICE detection
            rows = []
            for ln in cleaned:
                ln = ln.strip()
                if ln.startswith("|"): ln = ln[1:]
                if ln.endswith("|"): ln = ln[:-1]
                rows.append([c.strip() for c in ln.split("|")])
            rice_rows = rows
        pipe_buf = []

    for raw in lines:
        ln = raw.rstrip()
        st = ln.strip()

        if not st:
            flush_list(); flush_pipe(); story.append(Spacer(1, 4)); continue

        if st.startswith("|"):
            flush_list()
            pipe_buf.append(st)
            continue

        if st.startswith("### "):
            flush_list(); flush_pipe()
            story.append(Paragraph(_md_inline_to_html(st[4:]), styles["h3"])); continue
        if st.startswith("## "):
            flush_list(); flush_pipe()
            story.append(Paragraph(_md_inline_to_html(st[3:]), styles["h2"])); continue
        if st.startswith("# "):
            flush_list(); flush_pipe()
            story.append(Paragraph(_md_inline_to_html(st[2:]), styles["h1"])); continue

        if st.startswith("- ") or st.startswith("* "):
            flush_pipe()
            mode = "bullet"
            if list_mode not in (None, mode): flush_list()
            list_mode = mode
            list_buf.append(st[2:].strip())
            continue

        if re.match(r"^\d+\.\s+", st):
            flush_pipe()
            mode = "number"
            if list_mode not in (None, mode): flush_list()
            list_mode = mode
            list_buf.append(re.sub(r"^\d+\.\s+", "", st).strip())
            continue

        flush_list(); flush_pipe()
        story.append(Paragraph(_md_inline_to_html(st), styles["body"]))

    flush_list(); flush_pipe()
    return story, rice_rows

def _extract_rice_scores(rows: List[List[str]]):
    """Return [(feature, score)] if a valid RICE header is detected."""
    if not rows:
        return []
    header = [c.lower().replace(" ", "") for c in rows[0]]
    # allow variants like "reach(wk)"
    if not (("feature" in header[0]) and ("rice" in "".join(header))):
        return []
    out = []
    for r in rows[2:]:  # skip header + separator
        try:
            name = r[0]
            rice = float(str(r[-1]).replace(",", "").strip())
            out.append((name, rice))
        except Exception:
            continue
    return out

def _make_rice_chart(job_id: str, scores):
    if not scores:
        return ""
    labels = [a for a,_ in scores]
    vals   = [b for _,b in scores]
    plt.figure(figsize=(5.4, 3.2))
    bars = plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RICE")
    plt.tight_layout()
    out = CHART_DIR / f"rice_{job_id}.png"
    plt.savefig(out)
    plt.close()
    return str(out)

def _make_report_title(journey_type: str, payload: Dict[str, Any]) -> Tuple[str, str]:
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

    filename = "".join(c for c in title if c not in r'\/:*?"<>|').strip()
    return title, filename

def generate_pdf(job_id: str, journey_type: str, payload: Dict[str, Any], report_text: str) -> str:
    if not isinstance(report_text, str) or not report_text.strip():
        report_text = "No content produced. Please retry."

    _register_fonts()
    styles = _build_styles()
    title_text, fname = _make_report_title(journey_type, payload)
    out_path = PDF_DIR / f"{fname}.pdf"

    story: List = []
    story.append(Paragraph(_md_inline_to_html(title_text), styles["title"]))
    story.append(Spacer(1, 6))

    flowables, rice_rows = _parse_to_flowables(report_text, styles)
    story.extend(flowables)

    # Optional RICE chart directly after the features table (no title text)
    scores = _extract_rice_scores(rice_rows)
    if scores:
        chart = _make_rice_chart(job_id, scores)
        if chart and os.path.exists(chart):
            story.append(Spacer(1, 4))
            story.append(Image(chart, width=160*mm, height=95*mm))
            story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=14*mm, rightMargin=14*mm,  # slightly narrower
        topMargin=14*mm,  bottomMargin=14*mm,
        title=title_text, author="Nextify"
    )
    doc.build(story)
    return str(out_path)

# ---------- pipeline ----------
async def _run_pipeline(job_id: str, submission: Submission):
    job = JOBS[job_id]
    job["raw_report"] = ""
    try:
        job.update(status="running", step=UI_STEPS[0], progress=3, message="Validating input…")
        await asyncio.sleep(0.15)
        job.update(step=UI_STEPS[1], progress=6, message="Spinning up agents…")
        await asyncio.sleep(0.15)

        def cb(idx: int, section_title: str, message: str):
            job["step"] = section_title
            job["progress"] = min(6 + int(idx * (90 / 11)), 95)
            job["message"] = message

        report_text = await run_multi_agent(submission.model_dump(), cb)
        job["raw_report"] = report_text or ""

        job.update(step=UI_STEPS[-1], message="Writing PDF…", progress=97)
        await asyncio.sleep(0.15)
        pdf_path = generate_pdf(job_id, submission.journey_type, submission.payload, report_text)
        job.update(pdf_path=pdf_path, progress=100, step="Complete", status="done", message="Report ready.")
    except Exception as e:
        job.update(status="failed", step="Error", message=f"Pipeline error: {e}", progress=100)
        import traceback; traceback.print_exc()

# ---------- API ----------
@app.post("/api/submit")
async def submit(submission: Submission):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "created_at": time.time(), "status": "queued", "step": "Queued",
        "progress": 0, "message": "Job queued.", "pdf_path": None,
        "journey_type": submission.journey_type, "raw_report": ""
    }
    asyncio.create_task(_run_pipeline(job_id, submission))
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "step": job["step"],
            "progress": job["progress"], "message": job["message"],
            "ready": job["status"] == "done"}

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
    return PlainTextResponse(job.get("raw_report") or "(no raw report stored)")

@app.get("/")
async def root():
    return {"ok": True, "service": "Nextify Backend (ReportLab PDF)"}
