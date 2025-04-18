
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
    st.markdown("## 🤖 Multi-Agent System (Preview)")

    agents = [
        {
            "name": "🧠 Feature Ideator",
            "summary": "Synthesizes market trends, competitor insights, and user feedback to generate actionable feature ideas.",
            "description": """The Feature Ideator collaborates with:
- 📊 Competitor & Market Agent for trends and rival positioning  
- ❤️ Customer Feedback Agent for needs and complaints  
- 😠 Sentiment Agent for emotion-weighted prioritization  
- 🧪 Breakthrough Watcher Agent for industry innovation  
It produces multiple feature concepts with rationales, ready for prioritization."""
        },
        {
            "name": "🎯 Roadmap & OKR Generator",
            "summary": "Turns validated feature proposals into strategic OKRs and a milestone-driven roadmap.",
            "description": """This agent maps each selected feature to:
- 🧭 Company mission/vision  
- 📅 Quarterly delivery goals  
- 📐 Measurable OKRs  
It also requests input from Decision Agent to resolve priority conflicts."""
        },
        {
            "name": "📈 Competitor & Market Agent",
            "summary": "Monitors product updates, pricing, and news using RAG + Search grounding.",
            "description": """This agent uses Google Search and document RAG pipelines to:
- 📰 Scrape competitor blogs, changelogs, and reviews  
- 📉 Compare positioning and growth metrics  
- 🧠 Feed summaries to Feature Ideator and Decision Maker"""
        },
        {
            "name": "💬 Customer Feedback & Sentiment Agent",
            "summary": "Clusters and scores user feedback for topic modeling and emotional tone.",
            "description": """Using embedding-based clustering and Google NLP sentiment scoring:
- 📂 Segments complaints, suggestions, praises  
- 🎭 Detects urgency and emotion  
- 📩 Feeds tagged feedback into Feature Ideator"""
        },
        {
            "name": "📊 Decision Maker",
            "summary": "Orchestrates and ranks agent outputs to recommend what to build next.",
            "description": """Applies weighted scoring to features:
- 📌 Alignment with strategy (via roadmap)  
- 💰 Business impact (from OKRs)  
- ❤️ Customer value (from feedback agent)  
- 🥇 Picks top features for inclusion"""
        },
        {
            "name": "🌍 Breakthrough Watcher",
            "summary": "Scans for game-changing technologies and startup launches in the domain.",
            "description": """Acts as an explorer bot, ingesting:
- 🚀 TechCrunch, HackerNews, arXiv, VentureBeat  
- 📈 Signals of disruption  
Notifies Feature Ideator if relevance detected."""
        }
    ]

    st.markdown("### 🧠 Agent Modules")

    for idx, agent in enumerate(agents):
        expander_id = f"agent_{idx}"
        expand = st.checkbox(f"⬇️ {agent['name']}", key=expander_id)

        st.markdown(f'''
        <div style='
            border-radius: 12px;
            padding: 16px;
            background: linear-gradient(to right, #b2f7ef, #7f9cf5, #f78fb3);
            margin-bottom: 16px;
            color: white;
            font-family: "Segoe UI", sans-serif;
        '>
            <h4 style='margin-bottom: 8px;'>{agent['name']}</h4>
            <p style='font-size: 0.95em;'>{agent['summary']}</p>
        </div>
        ''', unsafe_allow_html=True)

        if expand:
            st.markdown(f"""
            <h4 style='margin-top: 30px;'>
            📘 <span style="font-weight: 600;">Full Description for</span>
            <span style="background-color: #7f9cf5; color: white; padding: 4px 12px; border-radius: 6px;
            font-family: Courier New, monospace; font-weight: bold;">
            {agent['name']}
            </span>
            </h4>
            """, unsafe_allow_html=True)

            st.markdown(agent["description"])

    st.markdown("---")
    st.markdown("### 🧩 Workflow Diagram")
    st.image("https://github.com/DonnaOftadeh/Nextify-Dashboard/raw/main/app/MultiAgent%20Workflow.png", use_container_width=True)

    st.markdown('''
    <div style='
        margin-top: 40px;
        border-radius: 12px;
        padding: 16px;
        background: linear-gradient(to right, #f78fb3, #7f9cf5);
        color: white;
        font-family: "Segoe UI", sans-serif;
        text-align: center;
    '>
        <h4 style='margin: 0;'>🚧 More agents, live chaining & real-time orchestration coming soon!</h4>
    </div>
    ''', unsafe_allow_html=True)


# === Tab 5: Embeddings & RAG (Future) ===
with tabs[4]:
    st.markdown("## 🧠 Embeddings + Retrieval Augmented Generation")
    st.info("Upload files, view document embeddings, and test similarity-based prompting")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}. Embedding + RAG view coming soon.")
