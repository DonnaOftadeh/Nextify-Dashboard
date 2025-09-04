# app/agents.py
#
# Minimal multi-agent runner that can call Gemini or OpenAI.
# It returns (events, final_markdown). The API will translate events
# into the status polling you already built.

import os
from typing import Dict, Any, List, Tuple, Optional

from .prompts import (
    build_system,
    SYSTEM_ORCHESTRATOR, SYSTEM_RESEARCHER, SYSTEM_ANALYST,
    SYSTEM_SYNTHESIZER, SYSTEM_CRITIC, SYSTEM_WRITER,
    user_orchestrator, user_researcher, user_analyst,
    user_synthesizer, user_critic, user_writer
)

# -------- Provider selection (Gemini default) ----------
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
USE_GEMINI = bool(os.getenv("GOOGLE_API_KEY")) or not USE_OPENAI

_client_gemini = None
_client_openai = None

def _ensure_clients():
    global _client_gemini, _client_openai
    if USE_GEMINI and _client_gemini is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _client_gemini = genai.GenerativeModel("gemini-1.5-pro")
    if USE_OPENAI and _client_openai is None:
        from openai import OpenAI
        _client_openai = OpenAI()


def _llm(system: str, user: str) -> str:
    """Call the selected LLM with a system+user pair and return markdown text."""
    _ensure_clients()
    if USE_GEMINI:
        # Gemini supports system via 'system_instruction'
        resp = _client_gemini.generate_content(
            [{"role": "user", "parts": [system + "\n\n" + user]}],
            safety_settings=None,
        )
        return (resp.text or "").strip()

    # OpenAI fallback
    chat = _client_openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return chat.choices[0].message.content.strip()


def run_multi_agent(entry: str, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Execute the orchestrated flow and return (events, final_markdown).
    Each event: {"step": "...", "message": "...", "progress": int}
    """
    events: List[Dict[str, Any]] = []
    progress = 0

    def push(step, msg, p_inc):
        nonlocal progress
        progress = min(100, progress + p_inc)
        events.append({"step": step, "message": msg, "progress": progress})

    system = build_system(entry)

    # 1) Orchestrator
    push("Orchestrate", "Planning tasks and guardrail questions…", 8)
    plan_json = _llm(system + "\n\n" + SYSTEM_ORCHESTRATOR, user_orchestrator(entry, payload))

    # 2) Research
    push("Research", "Gathering facts and comps…", 18)
    research_md = _llm(system + "\n\n" + SYSTEM_RESEARCHER, user_researcher(entry, payload))

    # 3) Analysis
    push("Analysis", "Deriving insights & options…", 22)
    analysis_md = _llm(system + "\n\n" + SYSTEM_ANALYST, user_analyst(entry, payload, research_md))

    # 4) Synthesis
    push("Synthesis", "Drafting OKRs / plan / MVP…", 22)
    synthesis_md = _llm(system + "\n\n" + SYSTEM_SYNTHESIZER, user_synthesizer(entry, payload, analysis_md))

    # 5) Critique
    push("Critique", "Stress testing assumptions…", 12)
    critic_md = _llm(system + "\n\n" + SYSTEM_CRITIC, user_critic(entry, payload, synthesis_md))

    # 6) Final
    push("Compose", "Writing final brief…", 15)
    final_md = _llm(system + "\n\n" + SYSTEM_WRITER,
                    user_writer(entry, payload, research_md, analysis_md, synthesis_md, critic_md))

    push("Complete", "Report ready.", 3)
    return events, final_md
