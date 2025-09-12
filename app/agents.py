# app/agents.py
import os
import asyncio
from typing import Dict, Any, Callable, List, Tuple

from .prompts import build_agent_prompts, SYSTEM_GUARDRAILS  # use prompts bundle (not templates)

# Optional: simple logger
def _dbg(msg: str):
    print(f"[agents] {msg}", flush=True)

# ---- LLM clients (Gemini + OpenAI) ------------------------------------------
# You can keep your existing client setup if you already had it. This version
# initializes lazily and chooses provider by env var LLM_PROVIDER = gemini|openai.
# Fallback from Gemini → OpenAI happens automatically on failure/limit.

_GEMINI = None
_OPENAI = None

def _get_provider_order() -> List[str]:
    # preferred → fallback
    first = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if first == "openai":
        return ["openai", "gemini"]
    return ["gemini", "openai"]

def _init_gemini():
    global _GEMINI
    if _GEMINI is not None:
        return _GEMINI
    import google.generativeai as genai  # requires google-generativeai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
    _GEMINI = genai.GenerativeModel(model_name)
    return _GEMINI

def _init_openai():
    global _OPENAI
    if _OPENAI is not None:
        return _OPENAI
    from openai import OpenAI  # requires openai>=1.0
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    # store tuple (client, model)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _OPENAI = (client, model)
    return _OPENAI

async def _llm_call(provider: str, system: str, user: str) -> str:
    """Single shot call to the chosen provider, returning plain text."""
    if provider == "gemini":
        gm = _init_gemini()
        # Gemini “system” goes into a preface; keep outputs concise.
        resp = await asyncio.to_thread(
            gm.generate_content,
            [{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
            safety_settings=None,
        )
        return (resp.text or "").strip()

    if provider == "openai":
        client, model = _init_openai()
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    raise RuntimeError(f"Unknown provider: {provider}")

# ---- helpers to keep sections crisp (no repetition) -------------------------
def _section_prompt(base_prompt: str, prior_bullets: List[str]) -> str:
    """
    Append a tiny control tail: use prior sections only for consistency, not for copying
    or summarizing. This prevents repetition while still keeping terms aligned.
    """
    ctx = ""
    if prior_bullets:
        # Keep this short to minimize the chance the model repeats content.
        joined = "; ".join(prior_bullets[-5:])  # last few anchors only
        ctx = (
            "\n\nConstraints for this section ONLY:\n"
            "- Use earlier outputs solely to keep naming and numbers consistent.\n"
            "- Do NOT copy or summarize previous sections.\n"
            f"- Key anchors to stay consistent with: {joined}\n"
        )
    return f"{base_prompt}{ctx}"

def _anchor_lines(section_id: str, text: str) -> List[str]:
    """
    Extract a few short anchors (first heading + first table header, if any)
    to help later sections keep terminology consistent.
    """
    anchors: List[str] = []
    lines = (text or "").splitlines()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("## "):
            anchors.append(s[3:].strip())
            break
    # First pipe header line (if any)
    for ln in lines:
        s = ln.strip()
        if s.startswith("|") and s.endswith("|") and "---" in s:
            anchors.append(s)
            break
    # Keep short
    return [a[:160] for a in anchors][:2]

# ---- main orchestrator ------------------------------------------------------
async def run_multi_agent(submission: Dict[str, Any],
                          progress_cb: Callable[[int, str, str], None]) -> str:
    """
    Orchestrates the 11-step (idea) or 4-step (others) flow using prompts.get_prompt_bundle.
    - Each step is generated independently to avoid repetition.
    - Previous outputs are passed as tiny anchors for consistency only.
    - Provider failover (Gemini → OpenAI) keeps the pipeline running.
    - No model/provider names are printed inside the final report text.
    """
    journey_type: str = (submission.get("journey_type") or "").lower().strip()
    payload: Dict[str, Any] = submission.get("payload") or {}

    bundle: Dict[str, str] = build_agent_prompts(journey_type, payload)
    if "error" in bundle:
        return f"## Error\nUnsupported journey type: {journey_type}"

    # Ordered by numeric prefix in keys: "1_", "2_", ...
    steps: List[Tuple[str, str]] = sorted(bundle.items(), key=lambda kv: kv[0])

    results: List[str] = []
    anchors: List[str] = []

    providers = _get_provider_order()  # e.g., ["gemini", "openai"]
    current_provider = providers[0]

    # Announce via callback (UI only). Nothing goes into the PDF itself.
    progress_cb(0, "Orchestration", f"Provider: {current_provider}")

    for idx, (step_id, base_prompt) in enumerate(steps, start=1):
        # progress: map 1..N to status steps in your UI (callback provided by main.py)
        progress_cb(idx, f"Step {idx}: {step_id}", "Generating…")

        # Keep each step independent; add tiny “consistency only” tail.
        user_prompt = _section_prompt(base_prompt, anchors)

        # Try preferred provider, then fallback once if needed.
        tried = []
        text = ""
        for prov in providers:
            if prov in tried:
                continue
            tried.append(prov)
            try:
                text = await _llm_call(prov, SYSTEM_GUARDRAILS, user_prompt)
                current_provider = prov  # stick to the last successful one
                break
            except Exception as e:
                _dbg(f"{step_id}: provider {prov} failed: {e}")
                # If failure on preferred, we’ll loop and try the other

        if not text:
            raise RuntimeError(f"{step_id}: all providers failed")

        # Light normalization: trim, collapse too many blank lines
        cleaned = _normalize_block(text)

        # Accumulate
        results.append(cleaned)
        anchors.extend(_anchor_lines(step_id, cleaned))

    # Stitch with two newlines between sections.
    final_report = "\n\n".join(results).strip()
    return final_report

# ---- tiny cleaner for model output -----------------------------------------
def _normalize_block(s: str) -> str:
    """
    - Strip surrounding whitespace
    - Collapse >2 blank lines into one
    - Remove leading/trailing stray pipes
    - Ensure table header separators aren't repeated
    """
    s = (s or "").strip()

    # collapse blank runs
    out_lines: List[str] = []
    blank = 0
    for ln in s.splitlines():
        if ln.strip() == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out_lines.append(ln.rstrip())

    # remove pure separator rows that sometimes get duplicated between tables
    cleaned: List[str] = []
    for ln in out_lines:
        st = ln.strip()
        if st.startswith("|") and st.endswith("|"):
            # Keep line; your PDF renderer removes internal --- rows itself,
            # so we leave header separators in place but not duplicate them back to back.
            if cleaned and cleaned[-1].strip() == st and "---" in st:
                continue
        cleaned.append(ln)

    return "\n".join(cleaned).strip()
