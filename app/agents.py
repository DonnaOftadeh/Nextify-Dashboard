# app/agents.py
import os
import json
import asyncio
from typing import Callable, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv("app/.env")

# ---- Choose LLM (Gemini) ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        # Let FastAPI crash early with a clear message
        raise RuntimeError("GEMINI_API_KEY not found. Put it in app/.env (do NOT commit).")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

from .templates import get_prompt_bundle


# -----------------------------
# LLM call helper
# -----------------------------
async def _call_gemini(prompt: str) -> str:
    """
    Async wrapper around Gemini call.
    """
    def _call_blocking():
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        # gemini SDK returns .text
        return resp.text or ""
    return await asyncio.to_thread(_call_blocking)


async def _ask_llm(prompt: str) -> str:
    """
    Dispatch to the active LLM provider (only Gemini in this build).
    """
    if LLM_PROVIDER == "gemini":
        return await _call_gemini(prompt)
    # Fallback (shouldn't happen here)
    return "LLM provider not configured."


# -----------------------------
# Core multi-agent runner
# -----------------------------
async def run_multi_agent(
    submission: Dict,
    progress_cb: Callable[[int, str, str], None] | None = None
) -> str:
    """
    Orchestrate the 'agents' (really: prompt stages) and produce a single
    consolidated report in your v4 format.

    submission: dict produced by pydantic model_dump() in main.py
    progress_cb: callback(index, section_title, message) -> None
    """

    journey = submission.get("journey_type", "company")
    payload = submission.get("payload", {})

    # Build grounding text and prompts for this entry type
    sections: List[Tuple[str, str]] = get_prompt_bundle(journey, payload)

    # Accumulate final report (markdown)
    final_report_parts: List[str] = []

    # Optional header
    subject = (
        payload.get("bench_company")
        or payload.get("company_name")
        or payload.get("industry")
        or payload.get("product_name")
        or payload.get("idea_text")
        or "Project"
    )
    header = f"🎯 Nextify v4 Multi-Agent Output Report – {subject}"
    final_report_parts.append(header)

    # Run each prompt sequentially (you can parallelize later if needed)
    for idx, (title, prompt) in enumerate(sections, start=1):
        if progress_cb:
            progress_cb(idx, title, f"{title}…")

        try:
            result = await _ask_llm(prompt)
        except Exception as e:
            result = f"_Agent error while running **{title}**: {e}_"

        # Ensure section title appears as your notebook format
        final_report_parts.append(f"\n{result}".strip())

    # Join everything
    return "\n\n".join(final_report_parts)
