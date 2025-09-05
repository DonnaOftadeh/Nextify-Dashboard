# app/templates.py
"""
Prompts designed to reproduce your v4 multi-agent report format exactly,
with the same section titles, emoji, and sub-labels you used in the Spotify example.

We keep the same 11 sections and adapt grounding by journey (company | industry | product | idea).
Each section prompt asks the LLM to return ONLY that section with the exact header.
"""

from typing import Dict, List, Tuple
import json
from datetime import datetime


# -----------------------------
# Grounding blocks
# -----------------------------
def _grounding_for_company(p: Dict) -> str:
    company = p.get("bench_company") or p.get("company_name") or "Unknown Company"
    region = p.get("region") or p.get("target_region") or "Global"
    segment = p.get("segment") or p.get("target_segment") or "General Audience"
    return (
        f"Entry Type: Company Benchmark\n"
        f"Company: {company}\n"
        f"Target Region: {region}\n"
        f"Target Segment: {segment}\n"
        "Goal: Analyze public feedback, issues, sentiment, competition, and propose features, "
        "strategy, OKRs, and a prioritized plan. Maintain the exact v4 structure and labels.\n"
    )


def _grounding_for_industry(p: Dict) -> str:
    industry = p.get("industry") or "Unknown Industry"
    region = p.get("region") or "Global"
    segment = p.get("segment")or p.get("target_segment") or "General Audience"
    return (
        f"Entry Type: Industry Opportunity\n"
        f"Industry: {industry}\n"
        f"Target Region: {region}\n"
        f"Target Segment: {segment}\n"
        "Goal: Surface gaps, user needs, competitive dynamics, and product ideas with strategy, "
        "OKRs, and prioritization. Follow the exact v4 structure and labels.\n"
    )


def _grounding_for_product(p: Dict) -> str:
    name = p.get("product_name") or "Unnamed Product"
    category = p.get("category") or "General"
    region = p.get("region") or "Global"
    return (
        f"Entry Type: Product Concept\n"
        f"Product: {name}\n"
        f"Category: {category}\n"
        f"Target Region: {region}\n"
        "Goal: Validate problems, analyze alternatives, propose features, strategy, OKRs, and "
        "prioritization. Use the exact v4 structure and labels.\n"
    )


def _grounding_for_idea(p: Dict) -> str:
    text = p.get("idea_text") or "No description provided"
    region = p.get("region") or "Global"
    return (
        "Entry Type: Idea (early concept)\n"
        f"Idea Summary: {text}\n"
        f"Target Region: {region}\n"
        "Goal: Clarify the problem, evaluate potential, outline competitive context, propose "
        "features, OKRs, and prioritization. Use the exact v4 structure and labels.\n"
    )


# -----------------------------
# Guardrails + Output contract
# -----------------------------
def _guardrail_block() -> str:
    return (
        "GUARDRAILS:\n"
        "- If inputs are vague or unknown, explicitly say what info is missing and suggest examples.\n"
        "- Use concise bullet lists; keep claims reasonable; mark assumptions.\n"
        "- DO NOT hallucinate metrics; where real data is missing, use placeholders (X, Y).\n"
        "- Keep the EXACT section order, titles, and emojis from v4. Do not add or rename sections.\n"
        "- Use the exact sub-labels seen in the v4 example (e.g., 'Final Issue (summary in one sentence):',\n"
        "  'Optional Root Cause Hypothesis:', 'Suggested Categories', 'Final Picks & Why:', etc.).\n"
        "- Return ONLY the content for the requested section title.\n"
    )


def _output_contract() -> str:
    return (
        "OUTPUT FORMAT CONTRACT (v4):\n"
        "🧠 1. Feedback Summary (Howler Whisperer)\n"
        "🔍 2. Issue Analysis (The Marauder)\n"
        "😊 3. Sentiment Analysis (The Legilimens)\n"
        "🔭 4. Competitor Insight (The Seer)\n"
        "💡 5. Feature Ideas – Round 1 (Room of Requirement)\n"
        "🧠 6. Strategic Synthesis v1 (The Pensive)\n"
        "🎯 7. OKR Plan (The Headmaster)\n"
        "💡 8. Feature Ideas – Refined Round 2 (Room of Requirement)\n"
        "🧠 9. Strategic Synthesis v2 (The Pensive)\n"
        "🎯 10. Prioritized Feature List (The Sorting Hat)\n"
        "📣 11. Final Formatted Output (The Story Weaver)\n"
        "Return exactly that single section, with the same header text and emoji.\n"
    )


# -----------------------------
# Section prompt builders
# -----------------------------
def _section_prompts(grounding: str, payload: Dict) -> List[Tuple[str, str]]:
    """
    Build (title, prompt) pairs. Each prompt reproduces your v4 content/labels.
    """
    pjson = json.dumps(payload, ensure_ascii=False, indent=2)
    guard = _guardrail_block()
    contract = _output_contract()

    sections: List[Tuple[str, str]] = []

    # 1 Feedback Summary
    title = "🧠 1. Feedback Summary (Howler Whisperer)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are the agent **Howler Whisperer**. Produce ONLY the section titled:\n\"{title}\"\n\n"
        "Mirror this structure:\n"
        "- Overall Sentiment (1 short line)\n"
        "- Scores (if known; otherwise note as unknown)\n"
        "- Key Issues & Complaints (bullets grouped like Ads, Free Version Limitations, Customer Service,\n"
        "  Subscription Issues, Artist-Related Issues, App Performance & Bugs, Content & Recommendations, UI/UX, Other)\n"
        "- Positive Feedback (bullets)\n"
        "- Summary Output (a short paragraph)\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 2 Issue Analysis
    title = "🔍 2. Issue Analysis (The Marauder)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Marauder**. Produce ONLY:\n\"{title}\"\n\n"
        "Include exactly these sub-labels and content:\n"
        "- Final Issue (summary in one sentence):\n"
        "- Optional Root Cause Hypothesis:\n"
        "- Suggested Categories (e.g., Bug, UX Flaw, Performance, Feature Gap):\n"
        "- Optional: Brainstorm Potential Solutions (bullet list)\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 3 Sentiment Analysis
    title = "😊 3. Sentiment Analysis (The Legilimens)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Legilimens**. Produce ONLY:\n\"{title}\"\n\n"
        "Mimic the v4 tone and structure:\n"
        "- Overall (1 short paragraph)\n"
        "- Per-source breakdown (e.g., Trustpilot, Google Play, App Store) with bullet highlights\n"
        "- Close with 'In summary:' paragraph\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 4 Competitor Insight
    title = "🔭 4. Competitor Insight (The Seer)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Seer**. Produce ONLY:\n\"{title}\"\n\n"
        "Replicate the v4 content blocks:\n"
        "- Inferred Industry\n"
        "- Key Competitors (bulleted list)\n"
        "- Competitive Positioning Summary (paragraph)\n"
        "- Strengths & Differentiators (bullets)\n"
        "- Weaknesses or Gaps (bullets)\n"
        "- Strategy Type (short line)\n"
        "- Optional: Porter's Five Forces Summary (bulleted per force)\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 5 Feature Ideas – Round 1
    title = "💡 5. Feature Ideas – Round 1 (Room of Requirement)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **Room of Requirement**. Produce ONLY:\n\"{title}\"\n\n"
        "Return exactly 3 options (A,B,C) with this shape:\n"
        "Option A:\n"
        "Feature Title: <title>\n"
        "Description: <short description>\n"
        "Score Each Option:\n"
        "- Originality (1–10): <n>\n"
        "- Feasibility (1–10): <n>\n"
        "- Impact (1–10): <n>\n"
        "Option B: ...\n"
        "Option C: ...\n"
        "Final Picks & Why:\n"
        "- Primary Pick: <Option X> …\n"
        "- Secondary Pick: <Option Y> …\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 6 Strategic Synthesis v1
    title = "🧠 6. Strategic Synthesis v1 (The Pensive)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Pensive**. Produce ONLY:\n\"{title}\"\n\n"
        "Follow the v4 structure:\n"
        "- Opportunity Theme 1: <title + 1-2 sentences>\n"
        "- Opportunity Theme 2: <title + 1-2 sentences>\n"
        "- (Optional) Opportunity Theme 3: …\n"
        "- Risk Theme: <title + 1-2 sentences>\n"
        "- Summary Recommendation: <short paragraph>\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 7 OKR Plan
    title = "🎯 7. OKR Plan (The Headmaster)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Headmaster**. Produce ONLY:\n\"{title}\"\n\n"
        "Create a SMART OKR with placeholders X/Y when unknown. Use the exact v4 style:\n"
        "Objective: <single line>\n"
        "Key Result 1: Increase <metric> from X to Y by the end of [Quarter]. (SMART breakdown)\n"
        "Key Result 2: Decrease <metric> from X% to Y% by the end of [Quarter]. (SMART breakdown)\n"
        "(Optional) Key Result 3: <adoption/engagement metric> to X% within [N] weeks. (SMART breakdown)\n"
        "Explanation and Considerations: <bullets>\n"
        "Why this is a good OKR: <bullets>\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 8 Feature Ideas – Refined Round 2
    title = "💡 8. Feature Ideas – Refined Round 2 (Room of Requirement)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **Room of Requirement** again. Produce ONLY:\n\"{title}\"\n\n"
        "Return 3 options (A,B,C) with the same scoring as Round 1. Close with:\n"
        "Final Picks & Why:\n"
        "- Primary Pick: <Option X> …\n"
        "- Secondary Pick: <Option Y> …\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 9 Strategic Synthesis v2
    title = "🧠 9. Strategic Synthesis v2 (The Pensive)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Pensive**. Produce ONLY:\n\"{title}\"\n\n"
        "Update based on Round 2:\n"
        "- Opportunity Theme 1\n"
        "- Opportunity Theme 2\n"
        "- Risk Theme\n"
        "- Summary Recommendation\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 10 Prioritized Feature List (ICE)
    title = "🎯 10. Prioritized Feature List (The Sorting Hat)"
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Sorting Hat**. Produce ONLY:\n\"{title}\"\n\n"
        "Render an ICE table exactly like v4:\n"
        "Feature\tImpact\tConfidence\tEffort\tICE Score\n"
        "<row 1>\n<row 2>\n<row 3>\n"
        "Decision-Making Rationale: <short paragraph>\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    # 11 Final Formatted Output (Standup)
    title = "📣 11. Final Formatted Output (The Story Weaver)"
    subject = (
        payload.get("bench_company")
        or payload.get("company_name")
        or payload.get("industry")
        or payload.get("product_name")
        or payload.get("idea_text")
        or "Project"
    )
    date_str = datetime.utcnow().strftime("%B %d, %Y")
    prompt = (
        f"{grounding}\n{guard}\n{contract}\n\n"
        f"You are **The Story Weaver**. Produce ONLY:\n\"{title}\"\n\n"
        "Follow the standup format exactly like v4. Use this heading line:\n"
        f"{subject} Product Strategy Standup - {date_str}\n"
        "Then include:\n"
        "- Summary (1 short paragraph)\n"
        "- Key Highlights (bulleted): User Sentiment, Core Issue, Competitive Landscape, Feature Ideas, ICE Prioritization\n"
        "- Recommended Actions (numbered list)\n"
        "- Closing (1 short paragraph)\n\n"
        f"INPUT PAYLOAD:\n{pjson}\n"
    )
    sections.append((title, prompt))

    return sections


# -----------------------------
# Bundle selector
# -----------------------------
def get_prompt_bundle(journey: str, payload: Dict) -> List[Tuple[str, str]]:
    journey = (journey or "company").lower()
    if journey == "company":
        g = _grounding_for_company(payload)
    elif journey == "industry":
        g = _grounding_for_industry(payload)
    elif journey == "product":
        g = _grounding_for_product(payload)
    elif journey == "idea":
        g = _grounding_for_idea(payload)
    else:
        g = _grounding_for_company(payload)

    # Append the contract so every prompt knows the canonical section list
    g = g + "\n" + _output_contract()
    return _section_prompts(g, payload)
