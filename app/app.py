
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    return pd.read_csv("all_experiment_view.csv")

df = load_data()

# === Color Palette ===
pinkish = LinearSegmentedColormap.from_list("green_blue_pink", ["#b2f7ef", "#7f9cf5", "#f78fb3"])

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

if filtered_df.empty:
    st.warning("⚠️ No matching data found. Try resetting filters.")
    st.stop()

if not companies and not strategies and not prompt_tags:
    filtered_df = df.copy()

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 + len(pivot.columns), 6))
    sns.heatmap(pivot["LLM Score"], annot=True, fmt=".1f", cmap=pinkish, ax=ax1)
    ax1.set_title("🤖 LLM Scores by Section")
    ax1.tick_params(axis='x', labelrotation=45)

    sns.heatmap(pivot["Human Score"], annot=True, fmt=".1f", cmap=pinkish, ax=ax2)
    ax2.set_title("👤 Human Scores by Section")
    ax2.tick_params(axis='x', labelrotation=45)

    fig.subplots_adjust(wspace=0.4)
    st.pyplot(fig)

    st.markdown("### 📋 Data Table")
    st.dataframe(filtered_df)
