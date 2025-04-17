
import streamlit as st
import pandas as pd
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
    return pd.read_csv("data/all_experiment_view.csv")

df = load_data()

# === Color Palette ===
purple_blue_pink = LinearSegmentedColormap.from_list("purple_blue_pink", ["#b2f7ef", "#7f9cf5", "#f78fb3"])
norm = Normalize(vmin=1, vmax=5)

# === Sidebar Filters ===
st.sidebar.title("🌟 Nextify Control Panel")
companies = st.sidebar.multiselect("Select Companies", df["Company"].unique())
strategies = st.sidebar.multiselect("Select Strategies", df["Strategy"].unique())
prompt_tags = st.sidebar.multiselect("Select Prompt Tags", df["Prompt Tag"].unique())

filtered_df = df.copy()
if companies:
    filtered_df = filtered_df[filtered_df["Company"].isin(companies)]
if strategies:
    filtered_df = filtered_df[filtered_df["Strategy"].isin(strategies)]
if prompt_tags:
    filtered_df = filtered_df[filtered_df["Prompt Tag"].isin(prompt_tags)]

tabs = st.tabs(["Overview", "Scores & Trends", "Prompt Table", "Multi-Agent System", "Embeddings & RAG"])

# === Tab 1: Overview ===
with tabs[0]:
    st.markdown("## 📊 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧪 Total Runs", f"{filtered_df['Run'].nunique()}")
    col2.metric("🤖 Avg. LLM Score", f"{filtered_df['LLM Score'].mean():.2f}")
    col3.metric("👤 Avg. Human Score", f"{filtered_df['Human Score'].mean():.2f}")

    st.markdown("### 📈 Score Heatmaps (LLM vs Human)")
    pivot = filtered_df.pivot_table(index="Section", columns="Prompt Tag", values=["LLM Score", "Human Score"], aggfunc="mean")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(pivot["LLM Score"], annot=True, fmt=".1f", cmap=purple_blue_pink, ax=ax[0], cbar_kws={'label': 'LLM Score'})
    ax[0].set_title("🤖 LLM Scores by Section")
    ax[0].tick_params(axis='x', labelrotation=45)

    sns.heatmap(pivot["Human Score"], annot=True, fmt=".1f", cmap=purple_blue_pink, ax=ax[1], cbar_kws={'label': 'Human Score'})
    ax[1].set_title("👤 Human Scores by Section")
    ax[1].tick_params(axis='x', labelrotation=45)
    st.pyplot(fig)

    st.markdown("### 🧠 Prompt Evaluation Cards")
    section_cards = filtered_df.groupby("Section").agg({
        "LLM Score": "mean",
        "Human Score": "mean",
        "Feedback": lambda x: x.iloc[0] if not x.empty else ""
    }).reset_index()

    for _, row in section_cards.iterrows():
        with st.container():
            if st.button(f"View details for: {row['Section']}"):
                st.markdown(f"""
                <div style='background-color: white; padding: 20px; border-radius: 10px;'>
                    <h4>{row['Section']}</h4>
                    <p><strong>LLM Score:</strong> {row['LLM Score']:.2f} &nbsp; | &nbsp; <strong>Human Score:</strong> {row['Human Score']:.2f}</p>
                    <p><strong>Feedback:</strong> {row['Feedback']}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("### 📋 Full Table View")
    st.dataframe(filtered_df)

# === Tab 2: Scores & Trends ===
with tabs[1]:
    st.markdown("## 🔍 Score Comparison: Before vs Improved")
    before = filtered_df[filtered_df["Prompt Tag"].str.contains("baseline", case=False, na=False)]
    improved = filtered_df[filtered_df["Prompt Tag"].str.contains("improved", case=False, na=False)]

    if not before.empty and not improved.empty:
        comparison = before.merge(improved, on=["Company", "Strategy", "Section"], suffixes=("_Before", "_Improved"))
        comparison["Combined Score_Before"] = comparison[["LLM Score_Before", "Human Score_Before"]].mean(axis=1)
        comparison["Combined Score_Improved"] = comparison[["LLM Score_Improved", "Human Score_Improved"]].mean(axis=1)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(data=comparison, x="Section", y="Human Score_Improved", color="#7f9cf5", label="Improved", ax=axs[0])
        sns.barplot(data=comparison, x="Section", y="Human Score_Before", color="#f78fb3", label="Before", ax=axs[0])
        axs[0].set_title("👤 Human Scores")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
        axs[0].legend()

        sns.barplot(data=comparison, x="Section", y="LLM Score_Improved", color="#7f9cf5", label="Improved", ax=axs[1])
        sns.barplot(data=comparison, x="Section", y="LLM Score_Before", color="#f78fb3", label="Before", ax=axs[1])
        axs[1].set_title("🤖 LLM Scores")
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)
        axs[1].legend()
        st.pyplot(fig)

        st.markdown("### 🔥 Score Improvement Heatmap")
        score_improvement = pd.DataFrame({
            "LLM Δ": comparison["LLM Score_Improved"] - comparison["LLM Score_Before"],
            "Human Δ": comparison["Human Score_Improved"] - comparison["Human Score_Before"],
            "Combined Δ": comparison["Combined Score_Improved"] - comparison["Combined Score_Before"],
        }, index=comparison["Section"])
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.heatmap(score_improvement, annot=True, cmap=purple_blue_pink, center=0, fmt=".2f")
        ax2.set_title("Score Improvements by Section")
        st.pyplot(fig2)

        comparison.to_csv("data/model_comparison.csv", index=False)
