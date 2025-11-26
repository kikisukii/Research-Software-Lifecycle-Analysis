import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from p01_inference import run_git_analysis

# Page Configuration
st.set_page_config(page_title="Research Software Lifecycle Detector", layout="wide")

# Stage Color Mapping (Aligned with V2 Visualization)
STAGE_COLORS = {
    "Baseline": "#f8b862",
    "Internal Development": "#38b48b",
    "Release Phase": "#9d5b8b",
    "Peak Activity": "#e9546b",
    "Maintenance": "#89c3eb",
    "Dormant": "#9ea1a3",
    "Dead": "#383c3c",
}


def build_segments(df):
    """
    Merge consecutive weeks with the same stage into continuous segments.
    Used for drawing background color bands on the timeline.
    """
    if df.empty: return []
    segments = []
    df = df.sort_values("week_date").reset_index(drop=True)
    start_date = df.iloc[0]["week_date"]
    current_stage = df.iloc[0]["stage_name"]

    for i in range(1, len(df)):
        stage = df.iloc[i]["stage_name"]
        if stage != current_stage:
            segments.append((start_date, df.iloc[i]["week_date"], current_stage))
            current_stage = stage
            start_date = df.iloc[i]["week_date"]
    segments.append((start_date, df.iloc[-1]["week_date"], current_stage))
    return segments


def main():
    st.title("🧬 Research Software Lifecycle Detector (Full v2)")

    # --- [Version Tag] ---
    # Update this timestamp manually before pushing to confirm deployment on the web app.
    st.caption("🚀 Version updated: 0.0.3")
    # ---------------------

    # Validate Secrets for API Access
    if "GITHUB_TOKEN" not in st.secrets:
        st.error("⚠️ GitHub Token missing! Please add 'GITHUB_TOKEN' in Streamlit Secrets.")
        st.stop()

    token = st.secrets["GITHUB_TOKEN"]

    # UI Layout: Input + Button
    col1, col2 = st.columns([3, 1])
    with col1:
        repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Analyze", type="primary")

    if run_btn and repo_url:
        with st.spinner("Cloning repo & Fetching API (Issues/Releases)... This takes time..."):
            try:
                # 1. Resolve Model Path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "model_bundle_v2.pkl")

                # 2. Run Inference Pipeline (with Token)
                df = run_git_analysis(repo_url, model_path, token)

                st.success(f"Analysis complete! Weeks processed: {len(df)}")

                # --- Visualization (Replicating V2 Style) ---
                fig = go.Figure()

                # A. Draw Background Stage Colors
                segments = build_segments(df)
                shapes = []
                for start, end, stage in segments:
                    shapes.append(dict(
                        type="rect", xref="x", yref="paper", x0=start, x1=end, y0=0, y1=1,
                        fillcolor=STAGE_COLORS.get(stage, "#eee"), opacity=0.4, layer="below", line_width=0
                    ))
                fig.update_layout(shapes=shapes)

                # B. Draw Metrics (All 4 Signals - 8w Rolling)

                # Signal 1: Commits (Main)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['commits_8w_sum'],
                    mode='lines', line=dict(color='#333333', width=2), name='Commits (8w)',
                    customdata=df['stage_name'],
                    hovertemplate="<b>Week:</b> %{x}<br><b>Commits:</b> %{y}<br><b>Phase:</b> %{customdata}<extra></extra>"
                ))

                # Signal 2: Contributors (Interactive toggle)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['contributors_8w_unique'],
                    mode='lines', line=dict(color='#1f77b4', width=1.5), name='Contributors (8w)',
                    visible='legendonly'  # User clicks legend to show
                ))

                # Signal 3: Issues Closed (Interactive toggle)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['issues_closed_8w_count'],
                    mode='lines', line=dict(color='#ff7f0e', width=1.5), name='Issues Closed (8w)',
                    visible='legendonly'
                ))

                # Signal 4: Releases (Dotted line, shown if present)
                if df['releases_8w_count'].sum() > 0:
                    fig.add_trace(go.Scatter(
                        x=df['week_date'], y=df['releases_8w_count'],
                        mode='lines', line=dict(color='#882255', width=1.5, dash='dot'), name='Releases (8w)'
                    ))

                # C. Layout Configurations
                fig.update_layout(
                    title=f"Lifecycle Timeline (v2): {repo_url}",
                    xaxis_title="Time", yaxis_title="Activity (8-week rolling)",
                    hovermode="x unified", height=600, template="plotly_white",
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
                )

                st.plotly_chart(fig, use_container_width=True)

                # Data Table Expander
                with st.expander("View Underlying Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()