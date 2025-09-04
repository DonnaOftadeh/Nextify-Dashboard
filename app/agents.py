"""
agents.py
---------
Pluggable agent runner for Nextify.

- If GOOGLE_API_KEY is set -> uses Google Generative AI (Gemini 1.5).
- Else if OPENAI_API_KEY is set -> uses OpenAI (GPT-4o family).
- Else -> uses a stub model (deterministic dummy outputs) so the backend
  and UI flows still work while you wire up real prompts / agents.

Entry point for main.py background jobs:
    await run_journey(journey_type, payload, update)

where:
  - journey_type: "company" | "industry" | "product" | "idea"
  - payload: dict collected from your HTML forms
  - update: callable(status: str, pct: int, meta: dict | None)
            Use it to push progress: update("Collecting context…", 10, {...})
"""

from __future__ import annotations

import os
import asyncio
from typing import Callable, Dict, Any, Optional, List

# ------------------------------------------------------------
# Optional LLM backends
# ------------------------------------------------------------
class LLM:
    """Minimal async LLM wrapper with fallback stub."""

    def __init__(self):
        self.provider = None
        self.model = None

        # Try Gemini first
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=api_key)
                # You can change model here if you like
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.provider = "gemini"
            except Exception as e:
                print("[agents] Gemini import failed, falling back. Error:", e)

        # Otherwise try OpenAI
        if not self.provider and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI  # type: ignore
                self.model = OpenAI()
                self.provider = "openai"
            except Exception as e:
                print("[agents] OpenAI import failed, falling back. Error:", e)

        # Otherwise stub
        if not self.provider:
            self.provider = "stub"

    async def acomplete(self, prompt: str, system: Optional[str] = None) -> str:
        """Return a completion string for a prompt."""
        if self.provider == "gemini":
            # Gemini is synchronous in this SDK; wrap in a thread to avoid blocking.
            import asyncio
            loop = asyncio.get_event_loop()

            def _call():
                parts = []
                if system:
                    parts.append({"text": f"[SYSTEM]\n{system}"})
                parts.append({"text": prompt})
                resp = self.model.generate_content(parts)  # type: ignore
                return resp.text if hasattr(resp, "text") else str(resp)

            return await loop.run_in_executor(None, _call)

        if self.provider == "openai":
            # Using Chat Completions with OpenAI SDK v1
            # Default model: gpt-4o-mini (cheap/fast). Adjust as needed.
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # OpenAI client set in __init__
            resp = self.model.chat.completions.create(  # type: ignore
                model=model_name,
                messages=messages,
                temperature=0.3,
            )
            return resp.choices[0].message.content  # type: ignore

        # Stub: deterministic response to keep the flow working
        return (
            "This is a stubbed agent response. Set GOOGLE_API_KEY or OPENAI_API_KEY "
            "to enable real GenAI outputs."
        )


# ------------------------------------------------------------
# Prompt helpers (you can swap these with your prompts.py later)
# ------------------------------------------------------------

BASE_SYSTEM = (
    "You are Nextify's multi-agent coordinator. Use crisp bullet points. "
    "Be explicit about assumptions and call out missing info. "
    "NEVER invent facts that are not logically implied."
)

# Minimal step prompts per journey. Replace/expand with your final prompts later.
STEP_PROMPTS: Dict[str, Dict[str, str]] = {
    "company": {
        "context": "Benchmark company: {company_name}\nRegion: {region}\n\n"
                   "Summarize the benchmark context and immediate risks or gaps to investigate.",
        "research": "Based on the benchmark company ({company_name}) and region ({region}), "
                    "list 5–8 key market insights and competitor notes.",
        "strategy": "Propose 3 positioning angles versus {company_name}. Include audience, value prop, and GTM notes.",
        "mvp": "Draft an MVP outline with top 5 features, KPIs, and a 4-week plan.",
        "summary": "Summarize findings as an executive brief (<=200 words) and 5 action bullets.",
    },
    "industry": {
        "context": "Industry: {industry}\nRegion: {region}\nSegment: {segment}\n"
                   "Summarize the market context, growth drivers, and regulatory flags.",
        "research": "List top 5 trends, 5 competitors, and 5 whitespace opportunities in {industry}.",
        "strategy": "Propose 3 approaches to enter {industry} targeting {segment} in {region}.",
        "mvp": "Draft 2 pilot MVPs and validation plan (surveys/experiments) for {industry}.",
        "summary": "Executive brief for the above, with next steps and metrics.",
    },
    "product": {
        "context": "Product idea: {product_name}\nUse case: {use_case}\nAudience: {audience}\n"
                   "Summarize the value hypothesis and immediate unknowns.",
        "research": "Identify 5 adjacent products, 5 risks, and 5 differentiators.",
        "strategy": "Define ICP, value prop, pricing hypothesis, and distribution ideas.",
        "mvp": "Outline MVP scope, success metrics, launch channels, and first 2 experiments.",
        "summary": "Condense into a 1-page brief (<=200 words) + 5 action bullets.",
    },
    "idea": {
        "context": "Problem theme: {theme}\nMotivation: {motivation}\n"
                   "Clarify the problem framing and user pain points succinctly.",
        "research": "List 5 user personas, 5 pain points, and 3 opportunity spaces.",
        "strategy": "Suggest 3 solution directions with pros/cons and feasibility.",
        "mvp": "Pick one direction and sketch MVP + test plan (interviews/landing test).",
        "summary": "Executive recap and next-step checklist.",
    },
}

# ------------------------------------------------------------
# Journey runner (sequential now; swap with your graph later)
# ------------------------------------------------------------

ProgressFn = Callable[[str, int, Optional[Dict[str, Any]]], None]

async def _run_step(llm: LLM, system: str, prompt: str) -> str:
    return await llm.acomplete(prompt, system=system)

def _interp(template: str, payload: Dict[str, Any]) -> str:
    try:
        return template.format(**payload)
    except Exception:
        # If some key is missing, just return template (your guardrails in main.py should catch this)
        return template

async def run_journey(journey_type: str, payload: Dict[str, Any], update: ProgressFn) -> Dict[str, Any]:
    """
    Orchestrates the journey steps. Returns a dict with detailed outputs that
    main.py can turn into a PDF + audio later.
    """
    if journey_type not in STEP_PROMPTS:
        raise ValueError(f"Unsupported journey_type: {journey_type}")

    llm = LLM()
    steps = ["context", "research", "strategy", "mvp", "summary"]
    prompts = STEP_PROMPTS[journey_type]
    outputs: Dict[str, str] = {}

    # Small helper to compute percentage nicely
    def pct(i: int) -> int:
        return int((i / len(steps)) * 100)

    update(f"Starting {journey_type} journey…", 1, {"provider": llm.provider})

    # Run steps sequentially (easy to replace with a parallel graph later)
    for i, step in enumerate(steps, start=1):
        label = step.capitalize()
        update(f"{label} analysis…", max(1, pct(i - 1)), {"step": step})

        prompt = _interp(prompts[step], payload)
        text = await _run_step(llm, BASE_SYSTEM, prompt)
        outputs[step] = text

        update(f"{label} complete", pct(i), {"step": step})

        # tiny pause to make progress visible in UI
        await asyncio.sleep(0.15)

    update("Journey complete", 100, {"done": True})

    # Package a canonical result
    return {
        "journey_type": journey_type,
        "payload": payload,
        "provider": llm.provider,
        "outputs": outputs,
        "brief": outputs.get("summary", ""),
    }

# ------------------------------------------------------------
# Example: parallel helper (for when you plug your multi-agent graph)
# ------------------------------------------------------------

async def run_parallel(
    tasks: List[asyncio.Task],
    update: ProgressFn,
    label: str = "Parallel tasks",
):
    """
    Utility to run a set of tasks in parallel and emit a simple status bar.
    Use this when you convert the sequential pipeline to your multi-agent graph.
    """
    update(f"{label} started…", 0, None)
    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    update(f"{label} finished", 100, {"completed": len(done)})
    return [t.result() for t in done]
