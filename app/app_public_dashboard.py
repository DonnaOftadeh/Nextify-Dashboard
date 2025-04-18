
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
        st.markdown(f'''
            <div style='display: flex; align-items: center; margin-top: 20px; margin-bottom: 10px;'>
                <div style='font-size: 20px; margin-right: 10px;'>🎯 <span style="color: #000;">Strategy:</span></div>
                <div style='
                    background-color: #7f9cf5;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-family: "Courier New", monospace;
                '>{strategy}</div>

                <div style='font-size: 20px; margin-left: 30px; margin-right: 10px;'>🏷️ <span style="color: #000;">Prompt Tag:</span></div>
                <div style='
                    background-color: #f78fb3;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-family: "Courier New", monospace;
                '>{tag}</div>
            </div>
        ''', unsafe_allow_html=True)

        group['Section'] = pd.Categorical(group['Section'], categories=ordered_sections, ordered=True)
        group = group.sort_values("Section")
    
        for idx, row in group.iterrows():
            expander_id = f"expand_{strategy}_{tag}_{row['Section']}"
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
                    st.markdown(f"#### 📘 Full LLM Output for `{row['Section']}`")
                    st.markdown(row["LLM Output Section"], unsafe_allow_html=True)

                    st.markdown(f"#### 📋 All Entries for `{row['Section']}`")
                    subset_df = filtered_df[
                        (filtered_df['Strategy'] == strategy) &
                        (filtered_df['Prompt Tag'] == tag) &
                        (filtered_df['Section'] == row['Section'])
                    ]
                    st.dataframe(subset_df)
st.markdown("### 🧾 Evaluation Table")
st.dataframe(filtered_df, use_container_width=True, height=400)
# === Tab 3: Prompt Table ===
with tabs[2]:
    st.markdown("## 📋 Full Prompt Evaluation")
    selected_prompt = st.selectbox("Choose a Prompt Tag", df["Prompt Tag"].unique() if not df.empty else [])
    if selected_prompt:
        prompt_rows = df[df["Prompt Tag"] == selected_prompt]
        for _, row in prompt_rows.iterrows():
            with st.expander(f"🔹 {row['Section']}" if pd.notna(row['Section']) else "🔹 Section"):
                st.markdown(f"**Metric**: {row['Metric']}")
                st.markdown(f"**LLM Score**: {row['LLM Score']}")
                st.markdown(f"**Human Score**: {row['Human Score']}")
                st.markdown(f"**Feedback**: {row['Feedback']}")
                st.markdown(f"**Lesson**: {row['Lesson']}")
                st.markdown("---")
                st.markdown("**LLM Output Section:**")
                st.markdown(row["LLM Output Section"])

# === Tab 4: Multi-Agent (Preview) ===
with tabs[3]:
    st.markdown("## 🤖 Multi-Agent System (Preview)")
    st.info("This section will contain agent logs, recommendations, and real-time chaining.")
    st.markdown("**Planned agents:**")
    st.markdown("- Feature Ideator")
    st.markdown("- Roadmap Generator")
    st.markdown("- OKR Builder")
    st.markdown("- Competitive Analyst")

    st.markdown("**Coming soon: Upload documents, run retrieval, view multi-agent flow.**")

# === Tab 5: Embeddings & RAG (Future) ===
with tabs[4]:
    st.markdown("## 🧠 Embeddings + Retrieval Augmented Generation")
    st.info("Upload files, view document embeddings, and test similarity-based prompting")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}. Embedding + RAG view coming soon.")
