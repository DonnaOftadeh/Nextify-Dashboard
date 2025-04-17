# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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

# === Tab 1: Overview (Refined) ===
with tabs[0]:
    st.markdown("## 📊 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧪 Total Runs", f"{filtered_df['Run'].nunique()}")
    col2.metric("🤖 Avg. LLM Score", f"{filtered_df['LLM Score'].mean():.2f}")
    col3.metric("👤 Avg. Human Score", f"{filtered_df['Human Score'].mean():.2f}")

    st.markdown("### 📈 Score Heatmaps (LLM vs Human)")
    pivot = filtered_df.pivot_table(index="Section", columns="Prompt Tag", values=["LLM Score", "Human Score"], aggfunc="mean")
    num_tags = len(pivot["LLM Score"].columns)
    

    st.markdown("### 🧠 Prompt Evaluation Cards")
    section_cards = filtered_df.groupby("Section").agg({
        "LLM Score": "mean",
        "Human Score": "mean",
        "Feedback": lambda x: x.iloc[0] if not x.empty else ""
    }).reset_index()

    clicked_section = st.session_state.get("clicked_section", None)

    for _, row in section_cards.iterrows():
        section_name = row["Section"]
        key = f"card_{section_name}"
        card_clicked = st.button(f"{section_name}", key=key)
        st.markdown(f'''
        <div style='
            border-radius: 12px;
            padding: 16px;
            background: linear-gradient(to right, #b2f7ef, #7f9cf5, #f78fb3);
            margin-bottom: 8px;
            color: #fff;
            font-family: "Segoe UI", sans-serif;
        '>
            <h4 style='margin-bottom: 8px;'>{section_name}</h4>
            <p>🤖 LLM Score: <strong>{row['LLM Score']:.2f}</strong> &nbsp; | &nbsp; 👤 Human Score: <strong>{row['Human Score']:.2f}</strong></p>
            <p style='font-size: 0.9em;'><strong>Feedback:</strong> {row['Feedback'][:120]}...</p>
        </div>
        ''', unsafe_allow_html=True)

        if card_clicked:
            st.session_state["clicked_section"] = section_name

        if clicked_section == section_name:
            matching_rows = filtered_df[filtered_df["Section"] == section_name]
            for _, subrow in matching_rows.iterrows():
                st.markdown(f'''
                <div style='background-color:#ffffff; padding:16px; border-radius:10px; margin-bottom:10px;'>
                    <strong>Prompt Tag:</strong> {subrow["Prompt Tag"]}  
                    <br><strong>LLM Score:</strong> {subrow["LLM Score"]} | <strong>Human Score:</strong> {subrow["Human Score"]}
                    <br><strong>Metric:</strong> {subrow["Metric"]}
                    <br><strong>Feedback:</strong> {subrow["Feedback"]}
                    <br><strong>Lesson:</strong> {subrow["Lesson"]}
                    <br><strong>LLM Output Section:</strong><br>
                    <pre style="white-space:pre-wrap;">{subrow["LLM Output Section"]}</pre>
                </div>
                ''', unsafe_allow_html=True)

    st.markdown("### 📋 Full Table View")
    st.dataframe(filtered_df)
    st.markdown("## 📊 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Run Count", f"{filtered_df['Run'].nunique() if not df.empty else 0}")
    col2.metric("Average LLM Score", f"{filtered_df['LLM Score'].mean():.2f}" if not df.empty else "N/A")
    col3.metric("Average Human Score", f"{filtered_df['Human Score'].mean():.2f}" if not df.empty else "N/A")

    st.markdown("### 🧾 Evaluation Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)


# === Tab 2: Scores & Trends (Improved) ===
with tabs[1]:
    st.markdown("## 🔍 Score Comparison: Before vs Improved")
    before = filtered_df[filtered_df["Prompt Tag"].str.contains("baseline", case=False, na=False)]
    improved = filtered_df[filtered_df["Prompt Tag"].str.contains("improved", case=False, na=False)]

    if not before.empty and not improved.empty:
        comparison = before.merge(improved, on=["Company", "Strategy", "Section"], suffixes=("_Before", "_Improved"))
        comparison["Combined Score_Before"] = comparison[["LLM Score_Before", "Human Score_Before"]].mean(axis=1)
        comparison["Combined Score_Improved"] = comparison[["LLM Score_Improved", "Human Score_Improved"]].mean(axis=1)

        st.markdown("### 📊 Bar Charts – LLM & Human Scores")
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(data=comparison, x="Section", y="Human Score_Before", color="#f7e733", label="Before", ax=axs[0])
        sns.barplot(data=comparison, x="Section", y="Human Score_Improved", color="#7f9cf5", label="Improved", ax=axs[0])
        axs[0].set_title("👤 Human Scores")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
        axs[0].legend()

        sns.barplot(data=comparison, x="Section", y="LLM Score_Before", color="#f7e733", label="Before", ax=axs[1])
        sns.barplot(data=comparison, x="Section", y="LLM Score_Improved", color="#7f9cf5", label="Improved", ax=axs[1])
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
        sns.heatmap(score_improvement, annot=True, cmap=pinkish, center=0, fmt=".2f")
        ax2.set_title("Score Improvements by Section")
        st.pyplot(fig2)

        st.markdown("### 📘 Detailed Comparison Table")
        styled_df = comparison[[
            "Company", "Strategy", "Prompt Tag_Before", "Prompt Tag_Improved", "Section",
            "LLM Score_Before", "LLM Score_Improved",
            "Human Score_Before", "Human Score_Improved",
            "Combined Score_Before", "Combined Score_Improved"
        ]].rename(columns={
            "Prompt Tag_Before": "Prompt Tag (Before)",
            "Prompt Tag_Improved": "Prompt Tag (Improved)"
        })

        def highlight_scores(val):
            try:
                val = float(val)
                if val >= 4.5:
                    return "background-color: #b2f7ef"
                elif val >= 4.0:
                    return "background-color: #7f9cf5"
                else:
                    return "background-color: #f78fb3"
            except:
                return ""

        st.dataframe(styled_df.style.applymap(highlight_scores, subset=[
            "LLM Score_Before", "LLM Score_Improved",
            "Human Score_Before", "Human Score_Improved",
            "Combined Score_Before", "Combined Score_Improved"
        ]))

        comparison.to_csv("data/model_comparison.csv", index=False)
    else:
        st.warning("Baseline and improved prompt tags not found in the selected filters.")

with tabs[1]:
    st.markdown("## 📊 Scores & Trends")

    if not filtered_df.empty:
        st.markdown(f"**Selected Prompt Tags:** {', '.join(filtered_df['Prompt Tag'].unique())}")

        # 🟦 Heatmap by Metric and Prompt
        pivot_metric = filtered_df.pivot_table(
            values=["LLM Score", "Human Score"],
            index="Section",
            columns="Prompt Tag",
            aggfunc="mean"
        )

        st.markdown("### 🔥 LLM Score Heatmap")
        fig_llm = px.imshow(pivot_metric["LLM Score"].fillna(0), text_auto=True, color_continuous_scale="blues")
        st.plotly_chart(fig_llm, use_container_width=True)

        st.markdown("### 🧠 Human Score Heatmap")
        fig_human = px.imshow(pivot_metric["Human Score"].fillna(0), text_auto=True, color_continuous_scale="greens")
        st.plotly_chart(fig_human, use_container_width=True)

        # 🎯 Combined Score per Section
        st.markdown("### 📊 Combined Score by Section (LLM + Human Average)")
        filtered_df["Combined Score"] = filtered_df[["LLM Score", "Human Score"]].mean(axis=1)
        avg_combined = filtered_df.groupby(["Section", "Prompt Tag"])["Combined Score"].mean().reset_index()
        fig_comb = px.bar(avg_combined, x="Section", y="Combined Score", color="Prompt Tag", barmode="group")
        st.plotly_chart(fig_comb, use_container_width=True)


# === Tab 3: Prompt Table (Card-Based) ===
with tabs[2]:
    st.markdown("## 📋 Full Prompt Evaluation")
    selected_prompt = st.selectbox("Choose a Prompt Tag", df["Prompt Tag"].unique())
    if selected_prompt:
        prompt_rows = df[df["Prompt Tag"] == selected_prompt]

        clicked_section = st.session_state.get("tab3_clicked_section", None)

        for _, row in prompt_rows.iterrows():
            section = row["Section"]
            strategy = row["Strategy"]
            key = f"prompt_card_{section}"

            if st.button(f"{section} ({strategy})", key=key):
                st.session_state["tab3_clicked_section"] = section

            st.markdown(f"""
            <div style='
                border-radius: 12px;
                padding: 16px;
                background: linear-gradient(to right, #b2f7ef, #7f9cf5, #f78fb3);
                margin-bottom: 8px;
                color: #fff;
                font-family: "Segoe UI", sans-serif;
            '>
                <h4 style='margin-bottom: 8px;'>{section} <span style='font-size:0.8em;'>({strategy})</span></h4>
                <p>🤖 LLM Score: <strong>{row['LLM Score']:.2f}</strong> &nbsp; | &nbsp; 👤 Human Score: <strong>{row['Human Score']:.2f}</strong></p>
                <p style='font-size: 0.9em;'><strong>Feedback:</strong> {row['Feedback'][:120]}...</p>
            </div>
            """, unsafe_allow_html=True)

            if clicked_section == section:
                st.markdown(f"""
                <div style='background-color:#ffffff; padding:16px; border-radius:10px; margin-bottom:10px;'>
                    <strong>Prompt Tag:</strong> {row["Prompt Tag"]}  
                    <br><strong>LLM Score:</strong> {row["LLM Score"]} | <strong>Human Score:</strong> {row["Human Score"]}
                    <br><strong>Metric:</strong> {row["Metric"]}
                    <br><strong>Feedback:</strong> {row["Feedback"]}
                    <br><strong>Lesson:</strong> {row["Lesson"]}
                    <br><strong>LLM Output Section:</strong><br>
                    <pre style="white-space:pre-wrap;">{row["LLM Output Section"]}</pre>
                </div>
                """, unsafe_allow_html=True)

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

# === Tab 4: Multi-Agent (Preview) ===
with tabs[3]:
    st.markdown("## 🤖 Multi-Agent System (Preview)")
    st.info("This section will contain agent logs, recommendations, and real-time chaining.")
    st.markdown("**Planned agents:**")
    st.markdown("- Feature Ideator\n- Roadmap Generator\n- OKR Builder\n- Competitive Analyst")
    st.markdown("**Coming soon: Upload documents, run retrieval, view multi-agent flow.**")

# === Tab 5: Embeddings & RAG (Future) ===
with tabs[4]:
    st.markdown("## 🧠 Embeddings + Retrieval Augmented Generation")
    st.info("Upload files, view document embeddings, and test similarity-based prompting")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}. Embedding + RAG view coming soon.")
