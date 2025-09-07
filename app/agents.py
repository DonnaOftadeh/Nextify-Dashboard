# app/agents.py
import os
import json
import asyncio
from typing import Callable, Dict, List, Tuple
import logging

from dotenv import load_dotenv
load_dotenv("app/.env")

logger = logging.getLogger("nextify.agents")

# ---- Provider Config ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Import SDKs only if keys exist
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

from .templates import get_prompt_bundle


# -----------------------------
# LLM call helpers
# -----------------------------
async def _call_gemini(prompt: str) -> str:
    """Async wrapper for Gemini call."""
    def _blocking():
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        return resp.text or ""
    return await asyncio.to_thread(_blocking)


async def _call_openai(prompt: str) -> str:
    """Async wrapper for OpenAI call."""
    def _blocking():
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    return await asyncio.to_thread(_blocking)


async def _ask_llm(prompt: str) -> str:
    """
    Try Gemini first; fallback to OpenAI if quota exceeded or fails.
    """
    if GEMINI_API_KEY:
        try:
            return await _call_gemini(prompt)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning("Gemini quota exceeded, falling back to OpenAI.")
            else:
                logger.error(f"Gemini error: {e}")
            # fallback to OpenAI if available
            if OPENAI_API_KEY:
                try:
                    return await _call_openai(prompt)
                except Exception as e2:
                    logger.error(f"OpenAI error: {e2}")
                    return "⚠️ Both Gemini and OpenAI failed. Please try again later."
            return "⚠️ Gemini quota exceeded. OpenAI not configured."
    elif OPENAI_API_KEY:
        try:
            return await _call_openai(prompt)
        except Exception as e2:
            logger.error(f"OpenAI error: {e2}")
            return "⚠️ OpenAI request failed. Please try again later."
    else:
        return "⚠️ No LLM provider configured."


# -----------------------------
# Core multi-agent runner
# -----------------------------
async def run_multi_agent(
    submission: Dict,
    progress_cb: Callable[[int, str, str], None] | None = None
) -> str:
    """
    Orchestrate the agents and produce a consolidated report.
    """

    journey = submission.get("journey_type", "company")
    payload = submission.get("payload", {})

    # Build prompts
    sections: List[Tuple[str, str]] = get_prompt_bundle(journey, payload)

    # Report
    final_report_parts: List[str] = []

    subject = (
        payload.get("bench_company")
        or payload.get("company_name")
        or payload.get("industry")
        or payload.get("product_name")
        or payload.get("idea_text")
        or "Project"
    )
    header = f"🎯 Nextify Multi-Agent Report – {subject}"
    final_report_parts.append(header)

    for idx, (title, prompt) in enumerate(sections, start=1):
        if progress_cb:
            progress_cb(idx, title, f"{title}…")

        try:
            result = await _ask_llm(prompt)
        except Exception as e:
            result = f"_Agent error in {title}: {e}_"

        final_report_parts.append(f"\n## {title}\n{result}".strip())

    return "\n\n".join(final_report_parts)
