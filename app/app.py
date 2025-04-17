import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
    df = pd.read_csv("data/all_experiment_view.csv")
    return df

df = load_data()

# === Color Palette ===
blue_yellow = LinearSegmentedColormap.from_list(
    "blue_yellow", ["#f7e733", "#3e5efb"]
)
norm = Normalize(vmin=1, vmax=5)

# === Sidebar Filters ===
st.sidebar.title("🌟 Nextify Control Panel")
companies = st.sidebar.multiselect("Select Companies", df["Company"].unique())
strategies = st.sidebar.multiselect("Select Strategies", df["Strategy"].unique())
prompt_tags = st.sidebar.multiselect("Select Prompt Tags", df["Prompt Tag"].unique())

# === Filter Data ===
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
    col1.metric("Run Count", f"{filtered_df['Run'].nunique()}")
    col2.metric("Avg. LLM Score", f"{filtered_df['LLM Score'].mean():.2f}")
    col3.metric("Avg. Human Score", f"{filtered_df['Human Score'].mean():.2f}")

    st.markdown("### 📈 Human vs LLM Scores (All Models)")
    pivot = filtered_df.pivot_table(index="Section", columns="Prompt Tag", values=["LLM Score", "Human Score"], aggfunc="mean")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(pivot["LLM Score"], annot=True, fmt=".1f", cmap=blue_yellow, ax=ax[0], cbar_kws={'label': 'LLM Score'})
    ax[0].set_title("LLM Scores by Section")

    sns.heatmap(pivot["Human Score"], annot=True, fmt=".1f", cmap=blue_yellow, ax=ax[1], cbar_kws={'label': 'Human Score'})
    ax[1].set_title("Human Scores by Section")
    st.pyplot(fig)

    st.markdown("### 🧠 Evaluation Summary Table")
    summary = filtered_df.groupby(["Section", "Prompt Tag"])[["LLM Score", "Human Score"]].mean().round(2).reset_index()
    st.dataframe(summary)

# === Tab 2: Scores & Trends ===
with tabs[1]:
    st.markdown("## 🔍 Score Comparison: Before vs Improved")

    before = filtered_df[filtered_df["Prompt Tag"].str.contains("baseline", case=False)]
    improved = filtered_df[filtered_df["Prompt Tag"].str.contains("improved", case=False)]

    if not before.empty and not improved.empty:
        comparison = before.merge(improved, on=["Company", "Strategy", "Section"], suffixes=("_Before", "_Improved"))
        comparison["Combined Score_Before"] = comparison[["LLM Score_Before", "Human Score_Before"]].mean(axis=1)
        comparison["Combined Score_Improved"] = comparison[["LLM Score_Improved", "Human Score_Improved"]].mean(axis=1)

        # Bar charts for Human and LLM scores
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(data=comparison, x="Section", y="Human Score_Improved", color="#3e5efb", label="Improved", ax=axs[0])
        sns.barplot(data=comparison, x="Section", y="Human Score_Before", color="#f7e733", label="Before", ax=axs[0])
        axs[0].set_title("👤 Human Scores")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
        axs[0].legend()

        sns.barplot(data=comparison, x="Section", y="LLM Score_Improved", color="#3e5efb", label="Improved", ax=axs[1])
        sns.barplot(data=comparison, x="Section", y="LLM Score_Before", color="#f7e733", label="Before", ax=axs[1])
        axs[1].set_title("🤖 LLM Scores")
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)
        axs[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### 🧠 Combined Score Heatmap")
        heat_data = pd.DataFrame({
            "LLM Δ": comparison["LLM Score_Improved"] - comparison["LLM Score_Before"],
            "Human Δ": comparison["Human Score_Improved"] - comparison["Human Score_Before"],
            "Combined Δ": comparison["Combined Score_Improved"] - comparison["Combined Score_Before"]
        }, index=comparison["Section"])

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.heatmap(heat_data, annot=True, cmap=blue_yellow, fmt=".2f", center=0)
        plt.title("📈 Score Improvements by Section")
        st.pyplot(fig2)

# === Tab 3: Prompt Table ===
with tabs[2]:
    st.markdown("## 📋 Full Prompt Evaluation")
    selected_prompt = st.selectbox("Choose a Prompt Tag", df["Prompt Tag"].unique())
    if selected_prompt:
        prompt_rows = df[df["Prompt Tag"] == selected_prompt]
        for _, row in prompt_rows.iterrows():
            with st.expander(f"🔹 {row['Section']}"):
                st.markdown(f"**Metric**: {row['Metric']}")
                st.markdown(f"**LLM Score**: {row['LLM Score']}")
                st.markdown(f"**Human Score**: {row['Human Score']}")
                st.markdown(f"**Feedback**: {row['Feedback']}")
                st.markdown(f"**Lesson**: {row['Lesson']}")
                st.markdown("---")
                st.markdown("**LLM Output Section:**")
                st.markdown(row["LLM Output Section"])

# === Tab 4: Multi-Agent ===
with tabs[3]:
    st.markdown("## 🤖 Multi-Agent System (Preview)")
    st.info("This section will contain agent logs, recommendations, and real-time chaining.")
    st.markdown("**Planned agents:**\n- Feature Ideator\n- Roadmap Generator\n- OKR Builder\n- Competitive Analyst")
    st.markdown("**Coming soon: Upload documents, run retrieval, view multi-agent flow.**")

# === Tab 5: Embeddings ===
with tabs[4]:
    st.markdown("## 🧠 Embeddings + Retrieval Augmented Generation")
    st.info("Upload files, view document embeddings, and test similarity-based prompting")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}. Embedding + RAG view coming soon.")
