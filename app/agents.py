# app/agents.py
"""
Agent runner. For now this is a self-contained async runner with a LOCAL stub
that generates coherent text without calling external LLMs, so your backend
works immediately. Later, you can swap `llm_complete` to OpenAI/Gemini.
"""

import asyncio
from typing import Dict, Any, Tuple
import os
import textwrap
from .prompts import build_agent_prompts
from .templates import render_nextify_v4


# ---- LLM adapter (stub) ----
# Swap this later to real OpenAI/Gemini calls.
async def llm_complete(prompt: str, max_tokens: int = 800) -> str:
    """
    Deterministic local stub: produce concise, structured text by trimming the prompt
    and returning a templated paragraph. Replace with real LLM calls later.
    """
    # Make a tiny deterministic "summary" from prompt sections
    head = " ".join(prompt.splitlines()[:10])
    head = textwrap.shorten(head, width=220, placeholder="…")
    return (
        "Summary based on provided context:\n"
        f"- {head}\n"
        "- Key points acknowledged. Where inputs are unknown, asked for clarifications.\n"
        "- Proposed pragmatic steps tailored to entry type and constraints."
    )


# ---- Agent Orchestration ----

SECTION_ORDER = [
    "feedback_summary",
    "issue_analysis",
    "sentiment_analysis",
    "competitor_insight",
    "feature_ideas_round1",
    "strategic_synthesis_v1",
    "okr_plan",
    "feature_ideas_round2",
    "strategic_synthesis_v2",
    "prioritized_features",
    "story_weaver",
]


async def run_agents(journey_type: str, payload: Dict[str, Any], status_cb=None) -> Tuple[Dict[str, str], str]:
    """
    Run all agents (parallel), collect outputs, and assemble the final v4 report.
    status_cb: optional callable(step_name, progress, message)
    Returns: (pieces dict, final_markdown)
    """
    if status_cb:
        status_cb("Prepare Prompts", 5, "Preparing agent prompts…")

    prompts = build_agent_prompts(journey_type, payload)

    async def run_one(key: str):
        if status_cb:
            status_cb(f"Running {key}", None, f"Agent '{key}' is generating…")
        # You can route to different models per key later
        return key, await llm_complete(prompts[key])

    if status_cb:
        status_cb("Parallel Agents", 20, "Launching agents in parallel…")

    # Run agents concurrently
    results = await asyncio.gather(*(run_one(k) for k in SECTION_ORDER))
    pieces = {k: v for k, v in results}

    if status_cb:
        status_cb("Assemble Report", 80, "Assembling final v4 report…")

    final_md = render_nextify_v4(journey_type, payload, pieces)

    if status_cb:
        status_cb("Finalize", 95, "Final report ready.")
    return pieces, final_md
