# app/agents.py
import os
import asyncio
from typing import Callable, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv("app/.env")

# Provider selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower().strip()  # 'gemini' | 'openai' | 'auto'

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Lazy imports
_genai = None
_openai_client = None

def _ensure_gemini():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing. Put it in app/.env (not committed).")
        genai.configure(api_key=GEMINI_API_KEY)
        _genai = genai
    return _genai

def _ensure_openai():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing. Put it in app/.env (not committed).")
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

async def _call_gemini(prompt: str) -> str:
    def _call_blocking():
        genai = _ensure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL)
        try:
            resp = model.generate_content(prompt)
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}")
    return await asyncio.to_thread(_call_blocking)

async def _call_openai(prompt: str) -> str:
    def _call_blocking():
        client = _ensure_openai()
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise, structured product strategy assistant. Use clear markdown with '#', '##', '###' headings, bullets, and short paragraphs."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")
    return await asyncio.to_thread(_call_blocking)

async def _ask_llm_single(provider: str, prompt: str) -> str:
    if provider == "gemini":
        return await _call_gemini(prompt)
    elif provider == "openai":
        return await _call_openai(prompt)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

async def _ask_llm(prompt: str) -> str:
    """
    Choose provider based on LLM_PROVIDER:
    - 'gemini' → Gemini only
    - 'openai' → OpenAI only
    - 'auto' (default) → try Gemini, fall back to OpenAI on error or empty
    """
    if LLM_PROVIDER == "gemini":
        return await _ask_llm_single("gemini", prompt)
    if LLM_PROVIDER == "openai":
        return await _ask_llm_single("openai", prompt)

    # AUTO: Gemini first, then OpenAI
    try:
        out = await _ask_llm_single("gemini", prompt)
        if out.strip():
            return out
    except Exception:
        pass
    # fallback
    return await _ask_llm_single("openai", prompt)

from .templates import get_prompt_bundle

# -----------------------------
# Core multi-agent runner
# -----------------------------
async def run_multi_agent(
    submission: Dict,
    progress_cb: Callable[[int, str, str], None] | None = None
) -> str:
    """
    Orchestrate the 'agents' (prompt stages) and produce a single
    consolidated report.
    """
    journey = submission.get("journey_type", "company")
    payload = submission.get("payload", {})

    sections: List[Tuple[str, str]] = get_prompt_bundle(journey, payload)

    final_report_parts: List[str] = []

    subject = (
        payload.get("bench_company")
        or payload.get("company_name")
        or payload.get("industry")
        or payload.get("product_name")
        or payload.get("idea_title")
        or payload.get("idea_text")
        or "Project"
    )

    # Top H1 so the PDF sees a heading (your previous code used plain text)
    final_report_parts.append(f"# Nextify Multi-Agent Report – {subject}")

    for idx, (title, prompt) in enumerate(sections, start=1):
        if progress_cb:
            progress_cb(idx, title, f"{title}…")
        try:
            result = await _ask_llm(prompt)
        except Exception as e:
            result = f"_Agent error while running **{title}**: {e}_"

        clean = (result or "").lstrip()

        # If the agent didn't start with a markdown header, add one to keep structure tidy.
        if not clean.startswith("#"):
            final_report_parts.append(f"## {title}")

        final_report_parts.append(clean.strip())

    return "\n\n".join(final_report_parts)
