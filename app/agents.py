# app/agents.py
# Multi-step agent runner using prompt bundles. Prevents context echo.
# No visible provenance announcements. No repetition of previous context.

from __future__ import annotations
import os, re, asyncio
from typing import Dict, Any, Callable, List, Tuple, Optional
from .prompts import build_agent_prompts, SYSTEM_GUARDRAILS

# Provider selection via env
PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower()   # "gemini" or "openai"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_openai_client = None
_gemini_model = None

def _want_gemini_first() -> bool:
    return PROVIDER in ("gemini","google") and bool(GEMINI_API_KEY)

async def _ensure_openai():
    global _openai_client
    if _openai_client: return _openai_client
    from openai import AsyncOpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

async def _ensure_gemini():
    global _gemini_model
    if _gemini_model: return _gemini_model
    import google.generativeai as genai
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model

def _natural_sort_keys(d: Dict[str,str]) -> List[str]:
    def keyer(k: str):
        m = re.match(r"^(\d+)", k); n = int(m.group(1)) if m else 10**9
        return (n, k)
    return sorted(d.keys(), key=keyer)

def _section_title_from_key(k: str) -> str:
    s = re.sub(r"^\d+_?", "", k)
    return s.replace("_"," ").strip().title()

def _summarize_for_context(text: str, max_chars: int = 1600) -> str:
    t = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(t) <= max_chars: return t
    cut = t[:max_chars]
    last_h = cut.rfind("\n## ")
    if last_h > max_chars*0.6: cut = cut[:last_h].strip()
    return cut

def _strip_echoed_context(md: str) -> str:
    """Remove any visible 'Context...' echoes the model might produce."""
    md = re.sub(r"(?ims)^\s*#{1,6}\s*context.*?(?=^\s*#{1,6}\s|\Z)", "", md)
    md = re.sub(r"(?im)^\s*context from previous sections.*$", "", md)
    return md.strip()

def _fix_markdown_tables(md: str) -> str:
    """Normalize single-line pipe rows to help our table parser (main handles visual tables)."""
    lines = md.splitlines(); out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("|") and "|" in ln:
            buf = [ln.rstrip()]; j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.strip().startswith("|") and nxt.count("|") < buf[-1].count("|"):
                    buf[-1] = (buf[-1] + " " + nxt.strip()).replace("  ", " ")
                    j += 1
                else:
                    break
            out.append(re.sub(r"\s*\|\s*", " | ", " ".join(buf)).strip())
            i = j; continue
        out.append(ln); i += 1
    return _strip_echoed_context("\n".join(out))

async def _call_gemini(user_prompt: str, system_message: Optional[str]) -> str:
    model = await _ensure_gemini()
    def _go():
        parts = []
        if system_message:
            parts.append({"role":"user","parts":[system_message]})
        parts.append({"role":"user","parts":[user_prompt]})
        resp = model.generate_content(parts)
        return getattr(resp,"text","") or (resp.candidates[0].content.parts[0].text if getattr(resp,"candidates",None) else "")
    return await asyncio.to_thread(_go)

async def _call_openai(user_prompt: str, system_message: Optional[str]) -> str:
    client = await _ensure_openai()
    messages = []
    if system_message:
        messages.append({"role":"system","content":system_message})
    messages.append({"role":"user","content":user_prompt})
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()

async def _call_with_fallback(user_prompt: str, ctx_summary: str) -> str:
    """Return cleaned text; context is SYSTEM-only with 'do not echo' rules."""
    system = (
        SYSTEM_GUARDRAILS
        + "\n\n[CONTEXT — DO NOT QUOTE OR RESTATE]\n"
        + ctx_summary.strip()
        + "\n[/CONTEXT]\n"
        + "Rules: Use context only to keep names and numbers consistent. "
          "Do not include any 'Context' or 'Previous sections' text in output.\n"
    )
    first = "gemini" if _want_gemini_first() else "openai"
    second = "openai" if first == "gemini" else "gemini"
    try:
        out = await (_call_gemini(user_prompt, system) if first=="gemini" else _call_openai(user_prompt, system))
        if out.strip(): return out.strip()
    except Exception:
        pass
    out2 = await (_call_gemini(user_prompt, system) if second=="gemini" else _call_openai(user_prompt, system))
    return (out2 or "").strip()

async def run_multi_agent(submission: Dict[str, Any], progress_cb: Callable[[int,str,str],None] | None = None) -> str:
    """Generate each section; join without echoing context."""
    journey_type = submission.get("journey_type","")
    payload = submission.get("payload",{}) or {}

    bundle = build_agent_prompts(journey_type, payload)
    if "error" in bundle:
        raise RuntimeError(bundle["error"])

    keys = _natural_sort_keys(bundle)
    if not keys:
        raise RuntimeError("No prompts available for this journey type.")

    all_sections: List[Tuple[str,str]] = []
    ctx = ""

    for idx, k in enumerate(keys, start=1):
        user_prompt = bundle.get(k)
        if not user_prompt:
            continue
        title = _section_title_from_key(k)
        if progress_cb:
            progress_cb(idx, title, f"Generating: {title}")
        raw = await _call_with_fallback(user_prompt, _summarize_for_context(ctx))
        cleaned = _fix_markdown_tables(raw)
        all_sections.append((title, cleaned))
        ctx_piece = f"## {title}\n{_summarize_for_context(cleaned, max_chars=1200)}\n"
        ctx = (ctx + "\n" + ctx_piece).strip()

    out_parts: List[str] = []
    for title, body in all_sections:
        if not re.search(r"^\s*#{1,3}\s", body):
            out_parts.append(f"## {title}")
        out_parts.append(body.strip())

    return "\n\n".join(out_parts).strip() or "No content produced."
