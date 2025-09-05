# app/agents.py
import os
import asyncio
from typing import Dict, Any, Callable, List

from dotenv import load_dotenv
import google.generativeai as genai

from . import templates as T

load_dotenv()  # load .env when running locally

# --- Gemini Setup ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if LLM_PROVIDER != "gemini":
    raise RuntimeError("This starter only wires Gemini. Set LLM_PROVIDER=gemini in .env")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Put it in app/.env (do NOT commit).")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# helper to call Gemini with system + user text
async def _call_llm(system_prompt: str, user_prompt: str) -> str:
    # google-generativeai python SDK is sync; call in a thread to keep API async-friendly
    def _sync_call():
        # In Gemini, prepend your "system" content into the prompt as guidance text
        full = f"{system_prompt.strip()}\n\nUSER CONTEXT:\n{user_prompt.strip()}"
        resp = model.generate_content(full)
        return resp.text or ""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_call)

def _make_entry_grounding(journey_type: str, payload: Dict[str, Any]) -> str:
    if journey_type == "company":
        return T.ENTRY_GROUNDING["company"].format(
            benchmark_company=payload.get("bench_company") or payload.get("company_name") or "Unknown",
            regions=payload.get("regions") or "Not provided",
            segments=payload.get("segments") or "Not provided",
        )
    if journey_type == "industry":
        return T.ENTRY_GROUNDING["industry"].format(
            industry=payload.get("industry") or "Unknown",
            regions=payload.get("regions") or "Not provided",
            segments=payload.get("segments") or "Not provided",
        )
    if journey_type == "product":
        return T.ENTRY_GROUNDING["product"].format(
            product_name=payload.get("product_name") or "Unknown",
            target_users=payload.get("target_users") or "Not provided",
            value_prop=payload.get("value_prop") or "Not provided",
        )
    if journey_type == "idea":
        return T.ENTRY_GROUNDING["idea"].format(
            idea_text=payload.get("idea_text") or "Unknown",
            goal=payload.get("goal") or "Not provided",
        )
    return "Entry Type: Unknown"

def _user_prompt(base_grounding: str, entry_grounding: str, extra_context: str = "") -> str:
    return f"""{base_grounding}

{entry_grounding}

Additional context:
{extra_context.strip() if extra_context else "N/A"}
"""

# phases the UI will show (aligns with your v4 sections)
AGENT_PHASES = [
    ("Howler Whisperer", T.HOWLER_SYSTEM, "🧠 1. Feedback Summary"),
    ("The Marauder", T.MARAUDER_SYSTEM, "🔍 2. Issue Analysis"),
    ("The Legilimens", T.LEGILIMENS_SYSTEM, "😊 3. Sentiment Analysis"),
    ("The Seer", T.SEER_SYSTEM, "🔭 4. Competitor Insight"),
    ("Room of Requirement (R1)", T.ROOM_REQ_R1_SYSTEM, "💡 5. Feature Ideas – Round 1"),
    ("The Pensive (v1)", T.PENSIVE_V1_SYSTEM, "🧠 6. Strategic Synthesis v1"),
    ("The Headmaster", T.OKR_SYSTEM, "🎯 7. OKR Plan"),
    ("Room of Requirement (R2)", T.ROOM_REQ_R2_SYSTEM, "💡 8. Feature Ideas – Refined Round 2"),
    ("The Pensive (v2)", T.PENSIVE_V2_SYSTEM, "🧠 9. Strategic Synthesis v2"),
    ("The Sorting Hat", T.SORTING_HAT_SYSTEM, "🎯 10. Prioritized Feature List"),
    ("The Story Weaver", T.STORY_WEAVER_SYSTEM, "📣 11. Final Formatted Output"),
]

async def run_multi_agent(
    submission: Dict[str, Any],
    progress_cb: Callable[[int, str, str], None],
) -> str:
    """
    Run all agents sequentially (easy to reason about) but keep the structure so you can
    parallelize later. progress_cb(step_idx, phase_title, message) updates the UI.
    Returns the final combined report text matching your v4 format.
    """
    journey = submission["journey_type"]
    payload = submission["payload"]

    header_subject = payload.get("bench_company") or payload.get("company_name") \
        or payload.get("industry") or payload.get("product_name") \
        or payload.get("idea_text") or "Untitled"

    base = T.BASE_GROUNDING
    entry = _make_entry_grounding(journey, payload)

    sections: List[str] = [T.report_header(header_subject)]

    # optional: include a one-line recap of the entry at the top
    sections.append(f"Entry: {journey.capitalize()} • Context anchored to: {header_subject}\n")

    # iterate agents
    for idx, (agent_name, system_prompt, section_title) in enumerate(AGENT_PHASES, start=1):
        phase_msg = f"{agent_name} is analyzing…"
        progress_cb(idx, section_title, phase_msg)
        user_prompt = _user_prompt(base, entry)
        text = await _call_llm(system_prompt, user_prompt)
        # guardrail: never return empty
        if not text.strip():
            text = "No data produced; please provide clearer inputs or a reference site."
        sections.append(f"{section_title}\n{text}\n")

    # join
    return "\n".join(sections)
