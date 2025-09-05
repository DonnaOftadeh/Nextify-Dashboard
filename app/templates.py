# app/templates.py

from typing import Dict, Any

FINAL_NEXTIFY_TEMPLATE = """\
🎯 Nextify v4 Multi-Agent Output Report – {title}

🧠 1. Feedback Summary (Howler Whisperer)
{feedback_summary}

🔍 2. Issue Analysis (The Marauder)
{issue_analysis}

😊 3. Sentiment Analysis (The Legilimens)
{sentiment_analysis}

🔭 4. Competitor Insight (The Seer)
{competitor_insight}

💡 5. Feature Ideas – Round 1 (Room of Requirement)
{feature_ideas_round1}

🧠 6. Strategic Synthesis v1 (The Pensive)
{strategic_synthesis_v1}

🎯 7. OKR Plan (The Headmaster)
{okr_plan}

💡 8. Feature Ideas – Refined Round 2 (Room of Requirement)
{feature_ideas_round2}

🧠 9. Strategic Synthesis v2 (The Pensive)
{strategic_synthesis_v2}

🎯 10. Prioritized Feature List (The Sorting Hat)
{prioritized_features}

📣 11. Final Formatted Output (The Story Weaver)
{story_weaver}
""".strip()


def title_from_payload(journey_type: str, payload: Dict[str, Any]) -> str:
    """
    Create a concise title for the report header based on the entry type.
    """
    if journey_type == "company":
        return payload.get("bench_company") or payload.get("company_name") or "Company"
    if journey_type == "industry":
        base = payload.get("industry") or "Industry"
        region = payload.get("target_region") or payload.get("region")
        return f"{base} – {region}" if region else base
    if journey_type == "product":
        base = payload.get("product_name") or payload.get("product") or "Product"
        segment = payload.get("target_segment") or payload.get("segment")
        return f"{base} – {segment}" if segment else base
    if journey_type == "idea":
        return payload.get("idea_title") or payload.get("idea") or "Idea"
    return payload.get("title") or "User Input"


def render_nextify_v4(journey_type: str, payload: Dict[str, Any], pieces: Dict[str, str]) -> str:
    """
    Render the final markdown report using the v4 persona/section structure.
    Missing sections are filled with an empty string to keep formatting consistent.
    """
    title = title_from_payload(journey_type, payload)
    return FINAL_NEXTIFY_TEMPLATE.format(
        title=title,
        feedback_summary=pieces.get("feedback_summary", ""),
        issue_analysis=pieces.get("issue_analysis", ""),
        sentiment_analysis=pieces.get("sentiment_analysis", ""),
        competitor_insight=pieces.get("competitor_insight", ""),
        feature_ideas_round1=pieces.get("feature_ideas_round1", ""),
        strategic_synthesis_v1=pieces.get("strategic_synthesis_v1", ""),
        okr_plan=pieces.get("okr_plan", ""),
        feature_ideas_round2=pieces.get("feature_ideas_round2", ""),
        strategic_synthesis_v2=pieces.get("strategic_synthesis_v2", ""),
        prioritized_features=pieces.get("prioritized_features", ""),
        story_weaver=pieces.get("story_weaver", ""),
    )
