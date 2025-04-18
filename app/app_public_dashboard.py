
# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
st.set_page_config(
    page_title="Nextify Dashboard",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# === Load Data ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/all_experiment_view.csv")
        return df
    except:
        return pd.DataFrame()

df = load_data()

# === Sidebar Filters ===
st.sidebar.title("🌟 Nextify Control Panel")
companies = st.sidebar.multiselect("Select Companies", df["Company"].unique() if not df.empty else [])
strategies = st.sidebar.multiselect("Select Strategies", df["Strategy"].unique() if not df.empty else [])
prompt_tags = st.sidebar.multiselect("Select Prompt Tags", df["Prompt Tag"].unique() if not df.empty else [])

# === Apply Filters ===
filtered_df = df.copy()
if companies:
    filtered_df = filtered_df[filtered_df["Company"].isin(companies)]
if strategies:
    filtered_df = filtered_df[filtered_df["Strategy"].isin(strategies)]
if prompt_tags:
    filtered_df = filtered_df[filtered_df["Prompt Tag"].isin(prompt_tags)]


# === Tabs ===
tabs = st.tabs(["Overview", "Scores & Trends", "Prompt Table", "Multi-Agent System", "Embeddings & RAG"])

# === Tab 1: Overview ===
with tabs[0]:
    st.markdown("## 📊 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Run Count", f"{filtered_df['Run'].nunique() if not df.empty else 0}")
    col2.metric("Average LLM Score", f"{filtered_df['LLM Score'].mean():.2f}" if not df.empty else "N/A")
    col3.metric("Average Human Score", f"{filtered_df['Human Score'].mean():.2f}" if not df.empty else "N/A")

    st.markdown("### 🧠 Prompt Evaluation Cards")

    ordered_sections = filtered_df['Section'].dropna().unique().tolist()
    grouped = filtered_df.groupby(["Strategy", "Prompt Tag", "Section"]).agg({
        "LLM Score": "mean",
        "Human Score": "mean",
        "Feedback": lambda x: x.iloc[0],
        "LLM Output Section": lambda x: x.iloc[0]
    }).reset_index()

    for (strategy, tag), group in grouped.groupby(["Strategy", "Prompt Tag"]):
        st.markdown(
            f"""<div style="display: flex; align-items: center; margin-top: 20px; margin-bottom: 10px;">
                <div style="font-size: 20px; margin-right: 10px;">🎯 <span style='color: #000;'>Strategy:</span></div>
                <div style="background-color: #7f9cf5; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-family: Courier New, monospace;">{strategy}</div>
                <div style="font-size: 20px; margin-left: 30px; margin-right: 10px;">🏷️ <span style='color: #000;'>Prompt Tag:</span></div>
                <div style="background-color: #f78fb3; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-family: Courier New, monospace;">{tag}</div>
            </div>""",
            unsafe_allow_html=True
        )

        group['Section'] = pd.Categorical(group['Section'], categories=ordered_sections, ordered=True)
        group = group.sort_values("Section")

        for idx, row in group.iterrows():
            expander_id = f"expand_{strategy}_{tag}_{row['Section']}"
            with st.container():
                expand = st.checkbox(f"⬇️ {row['Section']}", key=expander_id)
                st.markdown(
                    f"""<div style='
                        border-radius: 12px;
                        padding: 16px;
                        background: linear-gradient(to right, #b2f7ef, #7f9cf5, #f78fb3);
                        margin-bottom: 16px;
                        color: #fff;
                        font-family: "Segoe UI", sans-serif;
                    '>
                        <h4 style='margin-bottom: 8px;'>{row['Section']}</h4>
                        <p>🤖 LLM Score: <strong>{row['LLM Score']:.2f}</strong> &nbsp; | &nbsp; 👤 Human Score: <strong>{row['Human Score']:.2f}</strong></p>
                        <p style='font-size: 0.9em;'><strong>Feedback:</strong> {row['Feedback'][:120]}...</p>
                    </div>""",
                    unsafe_allow_html=True
                )

                if expand:
                    st.markdown(f"""
                    #### 📘 Full LLM Output for <span style='color:#7f9cf5; font-weight:bold;'>{row['Section']}</span>
                    """, unsafe_allow_html=True)
                    st.markdown(row["LLM Output Section"], unsafe_allow_html=True)
                    st.markdown(f"""
                    #### 📋 All Entries for <span style='color:#7f9cf5; font-weight:bold;'>{row['Section']}</span>
                    """, unsafe_allow_html=True)
                    subset_df = filtered_df[
                        (filtered_df['Strategy'] == strategy) &
                        (filtered_df['Prompt Tag'] == tag) &
                        (filtered_df['Section'] == row['Section'])
                    ]
                    st.dataframe(subset_df)

    st.markdown("### 🧾 Evaluation Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
# === Tab 2: Scores & Trends ===
with tabs[1]:
    st.markdown("## 📊 <span style='color:#7f9cf5;'>Scores & Trends</span>", unsafe_allow_html=True)

    if not filtered_df.empty:
        selected_strategy = ", ".join(filtered_df['Strategy'].unique())
        selected_tags = ", ".join(filtered_df['Prompt Tag'].unique())

        # 🎯 Strategy & Prompt Tag — split lines, styled
        st.markdown(
            f"""
            <div style='margin-top: 10px; margin-bottom: 10px; font-size: 18px; text-align: left;'>
                <div>
                    🎯 <strong>Strategy:</strong><br>
                    <span style='background-color: #7f9cf5; color: white; padding: 4px 12px; border-radius: 6px;
                                font-family: Courier New, monospace; font-weight: bold;'>{selected_strategy}</span>
                </div>
                <div style="margin-top: 10px;">
                    🏷️ <strong>Prompt Tags:</strong><br>
                    <span style='background-color: #f78fb3; color: white; padding: 4px 12px; border-radius: 6px;
                                font-family: Courier New, monospace; font-weight: bold;'>{selected_tags}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        pivot_metric = filtered_df.pivot_table(
            values=["LLM Score", "Human Score"],
            index="Section",
            columns="Prompt Tag",
            aggfunc="mean"
        )

        # 🔥 LLM Score Heatmap
        st.markdown("---")
        st.markdown(
            "<h4 style='color: #7f9cf5; font-weight: bold; text-align: left;'>🔥 LLM Score Heatmap</h4>",
            unsafe_allow_html=True
        )
        fig_llm = px.imshow(
            pivot_metric["LLM Score"].fillna(0),
            text_auto=True,
            color_continuous_scale=["#b2f7ef", "#7f9cf5", "#f78fb3"]
        )
        fig_llm.update_layout(
            width=900,
            height=500,
            margin=dict(t=20, l=20, r=20, b=20),
            xaxis_title=dict(text="Prompt Tag", font=dict(size=14, color="black", family="Arial", weight="bold")),
            yaxis_title=dict(text="Section", font=dict(size=14, color="black", family="Arial", weight="bold"))
        )
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_llm, use_container_width=False, key="llm_heatmap")
        st.markdown("</div>", unsafe_allow_html=True)

        # 🧠 Human Score Heatmap
        st.markdown("---")
        st.markdown(
            "<h4 style='color: #7f9cf5; font-weight: bold; text-align: left;'>🧠 Human Score Heatmap</h4>",
            unsafe_allow_html=True
        )
        fig_human = px.imshow(
            pivot_metric["Human Score"].fillna(0),
            text_auto=True,
            color_continuous_scale=["#b2f7ef", "#7f9cf5", "#f78fb3"]
        )
        fig_human.update_layout(
            width=900,
            height=500,
            margin=dict(t=20, l=20, r=20, b=20),
            xaxis_title=dict(text="Prompt Tag", font=dict(size=14, color="black", family="Arial", weight="bold")),
            yaxis_title=dict(text="Section", font=dict(size=14, color="black", family="Arial", weight="bold"))
        )
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_human, use_container_width=False, key="human_heatmap")
        st.markdown("</div>", unsafe_allow_html=True)

        # 📊 Combined Score Bar Chart
        st.markdown("---")
        st.markdown(
            "<h4 style='color: #7f9cf5; font-weight: bold; text-align: left;'>📊 Combined Score by Section</h4>",
            unsafe_allow_html=True
        )
        filtered_df["Combined Score"] = filtered_df[["LLM Score", "Human Score"]].mean(axis=1)
        avg_combined = filtered_df.groupby(["Section", "Prompt Tag"])["Combined Score"].mean().reset_index()

        fig_comb = px.bar(
            avg_combined,
            x="Section",
            y="Combined Score",
            color="Prompt Tag",
            color_discrete_sequence=["#b2f7ef", "#7f9cf5", "#f78fb3"]
        )
        fig_comb.update_layout(
            barmode="group",
            width=900,
            margin=dict(t=20, l=20, r=20, b=20),
            xaxis_title=dict(text="Section", font=dict(size=14, family="Arial", color="black", weight="bold")),
            yaxis_title=dict(text="Combined Score", font=dict(size=14, family="Arial", color="black", weight="bold"))
        )
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_comb, use_container_width=False, key="combined_chart")
        st.markdown("</div>", unsafe_allow_html=True)

        # 📋 Table Below
        st.markdown("---")
        st.markdown("### 📋 <span style='color:#7f9cf5;'>Combined Score Table</span>", unsafe_allow_html=True)
        st.dataframe(avg_combined, use_container_width=True, height=350)

# === Tab 3: Prompt Table ===
with tabs[2]:
    st.markdown("## 📋 Full Prompt Evaluation")

    ordered_sections = filtered_df['Section'].dropna().unique().tolist()
    grouped = filtered_df.groupby(["Strategy", "Prompt Tag", "Section"]).agg({
        "LLM Score": "mean",
        "Human Score": "mean",
        "Feedback": lambda x: x.iloc[0],
        "LLM Output Section": lambda x: x.iloc[0]
    }).reset_index()

    for (strategy, tag), group in grouped.groupby(["Strategy", "Prompt Tag"]):
        # Strategy and Prompt Tag Header
        st.markdown(f"""
        <div style="margin-top: 20px; margin-bottom: 10px; text-align: left;">
            <div style='font-size: 20px;'><strong>🎯 Strategy:</strong>
                <span style='background-color: #7f9cf5; color: white; padding: 4px 12px; border-radius: 6px;
                            font-family: Courier New, monospace; font-weight: bold;'>{strategy}</span>
            </div>
            <div style='font-size: 20px; margin-top: 10px;'><strong>🏷️ Prompt Tag:</strong>
                <span style='background-color: #f78fb3; color: white; padding: 4px 12px; border-radius: 6px;
                            font-family: Courier New, monospace; font-weight: bold;'>{tag}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        group['Section'] = pd.Categorical(group['Section'], categories=ordered_sections, ordered=True)
        group = group.sort_values("Section")

        for idx, row in group.iterrows():
            expander_id = f"prompt_tab_expand_{strategy}_{tag}_{row['Section']}"
            with st.container():
                expand = st.checkbox(f"⬇️ {row['Section']}", key=expander_id)
                st.markdown(f'''
                <div style='
                    border-radius: 12px;
                    padding: 16px;
                    background: linear-gradient(to right, #b2f7ef, #7f9cf5, #f78fb3);
                    margin-bottom: 16px;
                    color: #fff;
                    font-family: "Segoe UI", sans-serif;
                '>
                    <h4 style='margin-bottom: 8px;'>{row['Section']}</h4>
                    <p>🤖 LLM Score: <strong>{row['LLM Score']:.2f}</strong> &nbsp; | &nbsp; 👤 Human Score: <strong>{row['Human Score']:.2f}</strong></p>
                    <p style='font-size: 0.9em;'><strong>Feedback:</strong> {row['Feedback'][:120]}...</p>
                </div>
                ''', unsafe_allow_html=True)

                if expand:
                    st.markdown(f"""
                    <h4 style='margin-top: 30px;'>
                    📘 <span style="font-weight: 600;">Full LLM Output for</span>
                    <span style="background-color: #7f9cf5; color: white; padding: 4px 12px; border-radius: 6px;
                    font-family: Courier New, monospace; font-weight: bold;">
                    {row['Section']}
                    </span>
                    </h4>
                    """, unsafe_allow_html=True)

                    st.markdown(row["LLM Output Section"], unsafe_allow_html=True)

                    st.markdown(f"""
                    <h4 style='margin-top: 30px;'>
                    🗂️ <span style="font-weight: 600;">All Entries for</span>
                    <span style="background-color: #7f9cf5; color: white; padding: 4px 12px; border-radius: 6px;
                    font-family: Courier New, monospace; font-weight: bold;">
                    {row['Section']}
                    </span>
                    </h4>
                    """, unsafe_allow_html=True)

                    subset_df = filtered_df[
                        (filtered_df['Strategy'] == strategy) &
                        (filtered_df['Prompt Tag'] == tag) &
                        (filtered_df['Section'] == row['Section'])
                    ]
                    st.dataframe(subset_df)
                    
# === Tab 4: Multi-Agent System ===
with tabs[3]:
    st.markdown("## 🤖 Multi-Agent System")

    st.markdown("This page shows the agents designed to automate different parts of the PM workflow using Generative AI.")
    
    st.image("data/agentic_workflow_diagram.png", caption="🔄 Agentic Workflow: Nextify System Design", use_column_width=True)

    agents = [
        {
            "name": "Competitor & Market Analyst",
            "brief": "Monitors competitor moves and market signals using search + grounding.",
            "full": "This agent scans public sources like Google Search, TrustPilot, and app reviews to track product launches, pricing changes, customer sentiment, and positioning of competitors. It summarizes threats and opportunities."
        },
        {
            "name": "Customer Feedback Synthesizer",
            "brief": "Summarizes NPS, app reviews, and support tickets into clear insights.",
            "full": "This agent classifies user feedback into feature requests, complaints, and sentiments using embeddings and LLM-based summarization. It flags top pain points and common themes."
        },
        {
            "name": "Sentiment & Emotion Detector",
            "brief": "Detects emotional tones in feedback using fine-tuned models.",
            "full": "With a fine-tuned sentiment classifier, this agent can detect not just polarity but also emotions like frustration, delight, confusion, etc., to inform roadmap prioritization and design tweaks."
        },
        {
            "name": "Feature Ideation Engine",
            "brief": "Generates innovative feature ideas based on pain points, market gaps, and vision.",
            "full": "This agent uses LLM prompting with RAG to generate context-aware feature suggestions aligned to product vision and OKRs. It cross-checks with competitor gaps for strategic differentiation."
        },
        {
            "name": "Roadmap & OKR Builder",
            "brief": "Builds a quarterly roadmap and aligns each item to OKRs.",
            "full": "It turns features into roadmap cards with effort/impact tags, estimates priority using ICE scoring or OKR alignment, and formats the output as a structured JSON roadmap."
        },
        {
            "name": "Decision-making Assistant",
            "brief": "Prioritizes roadmap items using ICE, RICE, or custom scoring strategies.",
            "full": "Using OKR weighting and contextual scoring, this agent ranks options and can simulate trade-offs. It also includes a ‘risk check’ to validate if the suggestion violates any product principles or constraints."
        }
    ]

    for i, agent in enumerate(agents):
        expander_id = f"agent_expander_{i}"
        with st.container():
            expand = st.checkbox(f"⬇️ {agent['name']}", key=expander_id)
            st.markdown(f"""
            <div style='
                border-radius: 12px;
                padding: 16px;
                background: linear-gradient(to right, #b2f7ef, #7f9cf5, #f78fb3);
                margin-bottom: 16px;
                color: #fff;
                font-family: "Segoe UI", sans-serif;
            '>
                <h4 style='margin-bottom: 8px;'>{agent['name']}</h4>
                <p style='font-size: 0.9em;'><strong>🧠 What it does:</strong> {agent['brief']}</p>
            </div>
            """, unsafe_allow_html=True)

            if expand:
                st.markdown(f"#### 📘 Full Description of `{agent['name']}`")
                st.markdown(agent['full'])

    # Coming soon section
    st.markdown("""
    <div style='
        margin-top: 30px;
        padding: 12px;
        font-size: 18px;
        font-weight: bold;
        border-left: 5px solid #f78fb3;
        background-color: #fff3f8;
        color: #7f9cf5;
    '>
        🚧 More agents and chaining logic coming soon!
    </div>
    """, unsafe_allow_html=True)

# === Tab 5: Embeddings & RAG (Future) ===
with tabs[4]:
    st.markdown("## 🧠 Embeddings + Retrieval Augmented Generation")
    st.info("Upload files, view document embeddings, and test similarity-based prompting")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}. Embedding + RAG view coming soon.")
