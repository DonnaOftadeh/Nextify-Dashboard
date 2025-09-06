# app/agents.py
import os
import asyncio
import logging
import traceback
from typing import Callable, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv("app/.env")

# -----------------------------------------------------------------------------
# Logging setup (helpful while you're testing)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nextify.agents")

# -----------------------------------------------------------------------------
# LLM selection (Gemini in this build)
# -----------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not found. Put it in app/.env (do NOT commit).")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

from .templates import get_prompt_bundle


# -----------------------------------------------------------------------------
# LLM call helpers
# -----------------------------------------------------------------------------
async def _call_gemini(prompt: str) -> str:
    """
    Async wrapper for Gemini call.
    Ensures we return a string (possibly empty) rather than throwing.
    """
    def _call_blocking() -> str:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            # simple single-turn generation
            resp = model.generate_content(prompt)
            # Gemini SDK: best-effort text extraction
            text = getattr(resp, "text", None)
            if text:
                return text
            # sometimes candidates exist but .text is empty; try to salvage
            try:
                if resp.candidates:
                    parts = []
                    for c in resp.candidates:
                        if getattr(c, "content", None) and getattr(c.content, "parts", None):
                            for p in c.content.parts:
                                if getattr(p, "text", None):
                                    parts.append(p.text)
                    if parts:
                        return "\n".join(parts)
            except Exception:
                pass
            return ""
        except Exception as e:
            logger.error("Gemini call failed: %s", e)
            logger.debug("Trace:\n%s", traceback.format_exc())
            return ""
    return await asyncio.to_thread(_call_blocking)


async def _ask_llm(prompt: str) -> str:
    """
    Dispatch by provider. Only Gemini implemented here.
    """
    if LLM_PROVIDER == "gemini":
        return await _call_gemini(prompt)
    return "LLM provider not configured."


def _nonempty(text: str, fallback: str) -> str:
    """Ensure we never pass an empty section back to the report."""
    if isinstance(text, str) and text.strip():
        return text
    return fallback


# -----------------------------------------------------------------------------
# Core multi-agent runner
# -----------------------------------------------------------------------------
async def run_multi_agent(
    submission: Dict,
    progress_cb: Callable[[int, str, str], None] | None = None
) -> str:
    """
    Orchestrates the prompt "agents" and produces a single consolidated report
    in your v4 format. Never returns an empty string.
    """
    journey = submission.get("journey_type", "company")
    payload = submission.get("payload", {}) or {}

    # Build prompts for this entry type
    # get_prompt_bundle returns: List[Tuple[title, prompt]]
    sections: List[Tuple[str, str]] = get_prompt_bundle(journey, payload)

    # Subject line for header
    subject = (
        payload.get("bench_company")
        or payload.get("company_name")
        or payload.get("industry")
        or payload.get("product_name")
        or payload.get("idea_text")
        or "Project"
    )

    final_report_parts: List[str] = []
    final_report_parts.append(f"🎯 Nextify v4 Multi-Agent Output Report – {subject}")

    # Run each "agent" (prompt) in sequence
    for idx, (title, prompt) in enumerate(sections, start=1):
        if progress_cb:
            progress_cb(idx, title, f"{title}…")

        try:
            raw = await _ask_llm(prompt)
        except Exception as e:
            logger.error("Agent '%s' raised: %s", title, e)
            logger.debug("Trace:\n%s", traceback.format_exc())
            raw = ""

        # Make sure each section is non-empty and shows a clear fallback if needed
        section_body = _nonempty(
            raw,
            fallback=f"_No output produced for **{title}**. Check LLM/API settings or prompts._"
        )

        # Ensure your notebook format includes the section title
        final_report_parts.append(f"\n{title}\n{section_body.strip()}")

    # Join
    report_text = "\n\n".join(final_report_parts).strip()

    if not report_text:
        # Extremely unlikely with the fallback guards,
        # but just in case, return a helpful placeholder.
        report_text = (
            "⚠️ Agents produced empty output.\n\n"
            "Please verify your API key, model settings, and prompt templates."
        )

    return report_text
