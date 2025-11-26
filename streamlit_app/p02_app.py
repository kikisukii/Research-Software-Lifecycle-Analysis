import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os  # <--- 新增
from p01_inference import run_git_analysis

# Page Config
st.set_page_config(page_title="Research Software Lifecycle", layout="wide")

# Colors for plotting (V2 Scheme)
STAGE_COLORS = {
    "Baseline": "#f8b862",
    "Internal Development": "#38b48b",
    "Release Phase": "#9d5b8b",
    "Peak Activity": "#e9546b",
    "Maintenance": "#89c3eb",
    "Dormant": "#9ea1a3",
    "Dead": "#383c3c",
}


def main():
    st.title("🧬 Research Software Lifecycle Detector")
    st.markdown("""
    **Enter a GitHub repository URL** to automatically detect its lifecycle phases over time 
    (e.g., *Peak Activity*, *Maintenance*, *Dormant*).
    """)

    # Input Area
    col1, col2 = st.columns([3, 1])
    with col1:
        repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Analyze", type="primary")

    if run_btn and repo_url:
        with st.spinner("Cloning repo and crunching numbers... (This may take 10-20s)"):
            try:
                # --- 核心修改开始 ---
                # 获取当前脚本所在的目录，确保无论在哪里运行都能找到模型
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "model_bundle_v2.pkl")
                # --- 核心修改结束 ---

                df = run_git_analysis(repo_url, model_path)

                # Plotting with Plotly
                st.success(f"Analysis complete! Found {len(df)} weeks of history.")

                # Create interactive plot
                fig = go.Figure()

                # Add Commits Line (Smoothing for visual clarity)
                df["commits_smooth"] = df["commits"].rolling(window=4, center=True).mean()

                fig.add_trace(go.Scatter(
                    x=df['week_date'],
                    y=df['commits_smooth'],
                    mode='lines',
                    line=dict(color='#333333', width=2),
                    name='Commits (Avg)',
                    customdata=df['stage_name'],
                    hovertemplate="<b>Date:</b> %{x}<br><b>Commits:</b> %{y:.1f}<br><b>Phase:</b> %{customdata}<extra></extra>"
                ))

                fig.update_layout(
                    title=f"Lifecycle Timeline: {repo_url}",
                    xaxis_title="Time",
                    yaxis_title="Weekly Activity",
                    hovermode="x unified",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show Data Table (Optional)
                with st.expander("View Raw Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Error analyzing repository: {str(e)}")
                st.info("Check if the repository is public and the URL is correct.")


if __name__ == "__main__":
    main()