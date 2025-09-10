# app/templates.py
from typing import Dict, List, Tuple

# -------------------------------------------------------------------
# IDEA bundle: short, visual, and structured (1–11 as requested)
# -------------------------------------------------------------------
def idea_bundle(p: Dict) -> List[Tuple[str, str]]:
    title   = (p.get("idea_title") or "New Idea").strip()
    problem = (p.get("problem") or p.get("idea_text") or "Problem not provided").strip()
    users   = (p.get("target_users") or "Define the early adopters").strip()
    stage   = (p.get("current_stage") or "concept").strip()
    cons    = (p.get("constraints") or "none stated").strip()
    region  = (p.get("region") or "Global").strip()

    def h(n: str) -> str:
        return f"## {n}"

    return [
        # 1. Problem & JTBD
        ("1. Problem & JTBD Snapshot", f"""{h("1. Problem & JTBD Snapshot")}
**Idea:** {title} • **Stage:** {stage} • **Region:** {region}

- **Problem (from form):** {problem}
- **Who is it for:** {users}
- **Constraints:** {cons}

**Output (≤120 words):**
- One-sentence **JTBD**: “When ___, I want to ___, so I can ___.”
- 3 bullets: **pain**, **workaround**, **desired outcome**.
"""),

        # 2. Brainstorm & current behaviors
        ("2. Brainstorm & Current Behaviors", f"""{h("2. Brainstorm & Current Behaviors")}
Generate **diverse possibilities** for how target users behave today and could use {title}.
Keep it actionable (≤8 bullets). Group by theme, e.g. *learn*, *make*, *share*, *buy*.
"""),

        # 3. Audience & early adopters (market research)
        ("3. Audience & Early Adopters (Market Research)", f"""{h("3. Audience & Early Adopters (Market Research)")}
Define **2–3 micro-segments** and give **one-line value prop** + **activation channel** for each.
- Segment format: *segment label* — value prop — acquisition channel
"""),

        # 4. Competitors & market size
        ("4. Competitors & Market Size", f"""{h("4. Competitors & Market Size")}
List **4–6 alternatives/competitors** users hire today. For each: **strength** → **gap** (our wedge).
Then give a compact TAM/SAM/SOM snapshot and **reachable market in first quarter**.
"""),

        # 5. Assumptions & risks + tests
        ("5. Assumptions, Risks & How to Test", f"""{h("5. Assumptions, Risks & How to Test")}
List the **top 3 assumptions** and a **1-week test** for each with pass/fail rule.
Provide **example survey/interview prompts** (3–5 sample questions).
"""),

        # 6. Product & feature candidates + RICE
        ("6. Product & Feature Candidates + RICE", f"""{h("6. Product & Feature Candidates + RICE")}
Propose **3 candidate features** (A, B, C) for {title}, tailored to **{users}** solving **{problem}**.
Give a concise benefit statement per feature.

**Then output this exact markdown table (headers and order must match):**

| Feature | Reach(wk) | Impact(1-5) | Confidence(0-1) | Effort(pw) | RICE |
|---|---:|---:|---:|---:|---:|
| A | 300 | 3 | 0.7 | 2 | 315 |
| B | 120 | 4 | 0.6 | 1 | 288 |
| C | 500 | 2 | 0.7 | 4 | 175 |

> The RICE score is `(Reach * Impact * Confidence) / Effort`. Keep numbers realistic.
"""),

        # 7. Lean OKR
        ("7. Lean OKR (Next Quarter)", f"""{h("7. Lean OKR (Next Quarter)")}
Draft **1 Objective** and **2–3 measurable KRs** (baseline → target).
"""),

        # 8. Customer journey storyboard
        ("8. Customer Journey Storyboard (Q1)", f"""{h("8. Customer Journey Storyboard (Q1)")}
Describe **3 storyboard frames** (caption each step in ≤20 words):
1) Discover → 2) Activate → 3) Outcome at end of Q1.
"""),

        # 9. Tools to use
        ("9. Tools to Use", f"""{h("9. Tools to Use")}
List 5–8 tools (e.g., research, prototyping, analytics, user feedback) with one-line reason each.
"""),

        # 10. Synthesis
        ("10. Synthesis Summary", f"""{h("10. Synthesis Summary")}
One short paragraph that synthesizes the decisions above (what to build now and why).
"""),

        # 11. Next 3-month plan (table)
        ("11. Next 3-Month Plan (Table)", f"""{h("11. Next 3-Month Plan (Table)")}
Output a markdown table with **Milestone | Owner | Metric | Target date** (3–6 rows).
"""),
    ]


# -------------------------------------------------------------------
# Other journeys — keep your existing ones if you have them.
# Below are minimal fallbacks so the app continues to run.
# You can replace them later with similarly concise bundles.
# -------------------------------------------------------------------
def company_bundle(p: Dict) -> List[Tuple[str, str]]:
    name   = (p.get("company_name") or p.get("bench_company") or "Company").strip()
    region = (p.get("region") or "Global").strip()
    return [
        ("Company Summary", f"## Company Summary\nProduce a crisp overview for **{name}** in **{region}** (≤150 words)."),
        ("Opportunities", "## Opportunities\nList 5 focused growth/retention opportunities (short bullets)."),
        ("Top 3 Bets + RICE", "## Top 3 Bets + RICE\nProvide a 3-row RICE table (same header format as in idea bundle)."),
        ("Lean OKR", "## Lean OKR\n1 Objective + 2–3 KRs (baseline→target)."),
        ("Plan", "## Plan (Next 8 weeks)\n3 bullets with who/what/metric."),
    ]

def product_bundle(p: Dict) -> List[Tuple[str, str]]:
    prod = (p.get("product_name") or "Product").strip()
    return [
        ("Product Snapshot", f"## Product Snapshot\nWhat job does **{prod}** solve? Keep to 120 words."),
        ("Competitors", "## Competitors\nList 4–6 alternatives and our wedge."),
        ("Feature RICE", "## Feature RICE\n3 features with the standard RICE table."),
        ("Lean OKR", "## Lean OKR\n1 Objective + 2–3 KRs."),
        ("Rollout Plan", "## Rollout Plan (8 weeks)\n3 bullets."),
    ]

def industry_bundle(p: Dict) -> List[Tuple[str, str]]:
    ind = (p.get("industry") or "Industry").strip()
    region = (p.get("region") or "Global").strip()
    return [
        ("Market Snapshot", f"## Market Snapshot\nSummarize **{ind}** in **{region}**."),
        ("Segments", "## Segments\nList buyer roles/segments + pains."),
        ("Landscape", "## Landscape\nKey players + gaps."),
        ("Opportunities", "## Opportunities\n3 themes and why now."),
        ("Lean OKR + Plan", "## Lean OKR + Plan\nOKR + next steps."),
    ]


def get_prompt_bundle(journey: str, payload: Dict) -> List[Tuple[str, str]]:
    j = (journey or "").lower().strip()
    if j == "idea":
        return idea_bundle(payload)
    if j == "company":
        return company_bundle(payload)
    if j == "product":
        return product_bundle(payload)
    if j == "industry":
        return industry_bundle(payload)
    # default
    return idea_bundle(payload)
