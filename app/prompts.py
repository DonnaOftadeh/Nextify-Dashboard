# app/prompts.py

prompt_templates = {
    # === FEEDBACK AGENT ===
    "feedback": {
        "description": "Summarizes raw user reviews and NPS comments into concise insights.",
        "wizard": "Howler Whisperer (Feedback Summarizer)",
        "style": "Few-Shot",
        "grounding": [
            "https://www.trustpilot.com/review/app.com",
            "https://play.google.com/store",
            "https://apps.apple.com"
        ],
        "prompt": """
        Here are examples of how to summarize feedback in a concise, structured way:

        [Example 1]
        Input:
        - “Love the content, but the app keeps crashing on older phones.”
        - “Great discovery features! I wish it supported offline.”
        - “UI is clean, but it’s hard to find a way to save playlists quickly.”
        Output:
        - Summary: Users like content discovery and UI, but stability issues on older devices and missing offline mode are pain points. Playlist saving is discoverability issue.
        - Positives: Discovery, UI
        - Negatives: Crashes on old devices, no offline mode
        - Opportunities: Add offline mode, improve playlist saving UX, optimize for legacy devices

        [Example 2]
        Input:
        - “Customer support is slow.”
        - “I had billing issues and couldn’t get help in time.”
        Output:
        - Summary: Support response time and billing help are top user dissatisfaction drivers.
        - Negatives: Slow support, billing assistance
        - Opportunities: Improve SLA, add self-serve billing help

        Task:
        You will receive a context about a product or company along with notes from the user. Summarize the likely customer feedback (or proxy feedback from adjacent products if none exists) in the clear, structured format above.

        Constraints:
        - Keep it short, actionable, and grouped by themes.
        - Include at most 5 bullets per section.

        ### Inputs
        - Company (if any): {company_name}
        - Industry Context: {industry_context}
        - User Notes: {user_notes}

        ### Output (use this exact structure)
        - Summary:
        - Positives:
        - Negatives:
        - Opportunities:
        """,
        "temperature": 0.3
    },

    # === ISSUE FINDER AGENT ===
    "issue": {
        "description": "Identifies root issues and risks based on feedback and domain context.",
        "wizard": "The Marauder (Issue Finder)",
        "prompt": """
        You are the Issue Finder. From the context, produce the most likely root issues, risks, and blockers.
        Use a short, prioritized list with impact and confidence.

        ### Inputs
        - Company (if any): {company_name}
        - Industry Context: {industry_context}
        - User Notes: {user_notes}

        ### Output
        - Top Issues (max 5): [Issue, Why it matters, Impact (High/Med/Low), Confidence (0-100%)]
        - Risks & Unknowns (max 5)
        - Quick Wins (max 5)
        """,
        "temperature": 0.35
    },

    # === SENTIMENT AGENT ===
    "sentiment": {
        "description": "Summarizes sentiment by theme and persona.",
        "wizard": "Legilimens (Sentiment Analyst)",
        "prompt": """
        Derive likely sentiment clusters across typical personas (e.g., casual users, power users, partners).
        If no company is given, infer from industry exemplars.

        ### Inputs
        - Company (if any): {company_name}
        - Industry Context: {industry_context}
        - User Notes: {user_notes}

        ### Output
        - Persona Sentiment Table (max 4 personas): [Persona, Top Likes, Top Dislikes, Overall Sentiment (-,0,+)]
        - Notable Quotes (fabricated but realistic, 3-5 short lines)
        """,
        "temperature": 0.4
    },

    # === COMPETITOR / LANDSCAPE ANALYST ===
    "competitor": {
        "description": "Maps direct and adjacent competitors; highlights differentiators & moats.",
        "wizard": "The Seer (Competitor Analyst)",
        "prompt": """
        Build a concise landscape snapshot. Split into direct competitors (same core job-to-be-done)
        and adjacent alternatives (partial or substitute solutions). Extract differentiators and moats.

        ### Inputs
        - Company (if any): {company_name}
        - Industry Context: {industry_context}
        - User Notes: {user_notes}

        ### Output
        - Direct Competitors (3-5): [Name, What they do best, Weakness, Notable pricing/region]
        - Adjacent Alternatives (3-5)
        - Differentiation Opportunities (3-5)
        - Potential Moats (2-4)
        """,
        "temperature": 0.35
    },

    # === FEATURE IDEATOR ===
    "ideation": {
        "description": "Proposes features / experiments aligned to findings.",
        "wizard": "Room of Requirement (Feature Ideator)",
        "prompt": """
        Generate tightly-scoped features or experiments that directly address the top issues and leverage opportunities.

        ### Inputs
        - Company (if any): {company_name}
        - Industry Context: {industry_context}
        - User Notes: {user_notes}

        ### Output
        - Feature Ideas (5-8): [Name, Problem it solves, Success metric, Complexity (S/M/L)]
        - Experiments (3-5): [Hypothesis, Minimum test, Metric, Decision rule]
        """,
        "temperature": 0.55
    },

    # === STRATEGIC SYNTHESIZER ===
    "synthesis": {
        "description": "Merges all agent outputs into one coherent strategy.",
        "wizard": "The Pensive (Strategic Synthesizer)",
        "prompt": """
        You will receive the outputs from these agents: Feedback, Issue Finder, Sentiment, Competitor, Ideation.
        Synthesize them into a **single concise plan** with clear scope and next steps.

        ### Inputs
        - Company (if any): {company_name}
        - Industry Context: {industry_context}
        - User Notes: {user_notes}
        - Agent Outputs:
          - Feedback: {feedback_output}
          - Issue: {issue_output}
          - Sentiment: {sentiment_output}
          - Competitor: {competitor_output}
          - Ideation: {ideation_output}

        ### Output
        - One-Liner Positioning
        - Target User & Core Job-to-be-Done
        - Top 3 Bets (with why)
        - 30/60/90 (bullet list)
        - Key Risks & How We’ll Learn
        - Metrics to Watch
        """,
        "temperature": 0.35
    },

    # (Optional post-synthesis helpers — we can wire later)
    "prioritization": {
        "description": "Ranks features vs impact/effort.",
        "wizard": "RICE/ICE Ranker",
        "prompt": """
        Prioritize the proposed features using RICE. Keep output short.

        ### Inputs
        - Feature Ideas: {feature_list}

        ### Output
        - RICE Table: [Feature, Reach, Impact(1-3), Confidence(0-100%), Effort(1-3), Score]
        - Top 5 to tackle next
        """,
        "temperature": 0.3
    },

    "okr": {
        "description": "Turns plan into draft OKRs.",
        "wizard": "OKR Shaper",
        "prompt": """
        Convert strategy into OKRs (1–2 Objectives, 3–4 KRs each). Keep crisp, measurable.

        ### Inputs
        - Strategy: {strategy_text}

        ### Output
        - Objective 1:
          - KR1:
          - KR2:
          - KR3:
        - Objective 2:
          - KR1:
          - KR2:
          - KR3:
        """,
        "temperature": 0.25
    },

    "formatter": {
        "description": "Polishes for human-friendly reading.",
        "wizard": "Story Weaver",
        "prompt": """
        Format for exec readability: short sections, strong headings, bullets, and clarity, tone: practical.

        ### Inputs
        - Strategy: {strategy_text}

        ### Final Output
        - Title:
        - Summary:
        - Key Highlights:
        - Recommended Actions:
        - Closing:
        """,
        "temperature": 0.7
    }
}
