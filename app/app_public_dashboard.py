# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# === Config ===
st.set_page_config(
    page_title="Nextify Dashboard",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# === Clean Light Theme Styling ===
st.markdown("""
    <style>
        body {
            background-color: #FAFAFA;
            color: #1C1C1E;
        }
        .stApp {
            background-color: #FAFAFA;
        }
        .css-18e3th9, .css-1d391kg {
            background-color: transparent;
            color: #1C1C1E;
            border: none;
            box-shadow: none;
        }
        .stButton>button {
            background-color: #4CC9F0;
            color: white;
            border: none;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #3ABCD4;
            color: white;
        }
        .stTabs [data-baseweb="tab"] {
            color: #6E6E6E;
        }
        .stTabs [aria-selected="true"] {
            color: #4CC9F0;
            border-bottom: 2px solid #4CC9F0;
        }
        .stSelectbox, .stMultiSelect, .stDataFrame, .stExpander {
            background-color: transparent;
            color: #1C1C1E;
            border: none;
        }
        .metric-label {
            color: #6E6E6E;
        }
    </style>
""", unsafe_allow_html=True)

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

    st.markdown("### 📈 Human vs LLM Scores per Run")
    if not filtered_df.empty:
        fig = px.bar(filtered_df, x="Run", y=["LLM Score", "Human Score"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧾 Evaluation Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)

# === Tab 2: Scores & Trends ===
with tabs[1]:
    st.markdown("## 📊 Scores & Trends")
    if not filtered_df.empty:
        avg_by_strategy = filtered_df.groupby("Strategy")[["LLM Score", "Human Score"]].mean().reset_index()
        fig2 = px.bar(avg_by_strategy, x="Strategy", y=["LLM Score", "Human Score"], barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

        avg_by_metric = filtered_df.groupby("Metric")[["LLM Score", "Human Score"]].mean().reset_index()
        fig3 = px.line(avg_by_metric, x="Metric", y="LLM Score", markers=True, title="LLM Score by Metric")
        st.plotly_chart(fig3, use_container_width=True)

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
                st.markdown("**LLM Output Section**:")
                st.markdown(row["LLM Output Section"])

# === Tab 4: Multi-Agent (Coming Soon) ===
with tabs[3]:
    st.markdown("## 🤖 Multi-Agent System (Preview)")
    st.info("This section will contain agent logs, recommendations, and real-time chaining.")
    st.markdown("**Planned agents:**")
    st.markdown("- Feature Ideator\n- Roadmap Generator\n- OKR Builder\n- Competitive Analyst")
    st.markdown("**Coming soon: Upload documents, run retrieval, view multi-agent flow.**")

# === Tab 5: Embeddings & RAG (Coming Soon) ===
with tabs[4]:
    st.markdown("## 🧠 Embeddings + Retrieval Augmented Generation")
    st.info("Upload files, view document embeddings, and test similarity-based prompting")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}. Embedding + RAG view coming soon.")
