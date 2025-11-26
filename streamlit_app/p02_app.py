import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from p01_inference import run_git_analysis

# Page Config
st.set_page_config(page_title="Research Software Lifecycle Detector", layout="wide")

# Colors (V2 Standard Palette)
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
    """Merge consecutive weeks with the same stage."""
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

    # --- Version Tag ---
    st.caption("🚀 Version updated: 0.0.4")
    # -------------------

    if "GITHUB_TOKEN" not in st.secrets:
        st.error("⚠️ GitHub Token missing! Please add 'GITHUB_TOKEN' in Streamlit Secrets.")
        st.stop()

    token = st.secrets["GITHUB_TOKEN"]

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
                # 1. Setup Path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "model_bundle_v2.pkl")

                # 2. Run Inference
                df = run_git_analysis(repo_url, model_path, token)

                st.success(f"Analysis complete! Weeks: {len(df)}")

                # --- Visualization: 4 Subplots (V2 Paper Style) ---

                # Create 4 rows, 1 column, shared X axis
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        "Commits (8w rolling)",
                        "Active Contributors (8w rolling)",
                        "Issues Closed (8w rolling)",
                        "Releases (8w rolling)"
                    )
                )

                # --- Metric 1: Commits (Row 1) ---
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['commits_8w_sum'],
                    mode='lines', line=dict(color='#333333', width=1.8),
                    name='Commits',
                    customdata=df['stage_name'],
                    hovertemplate="<b>Week:</b> %{x}<br><b>Commits:</b> %{y}<br><b>Phase:</b> %{customdata}<extra></extra>"
                ), row=1, col=1)

                # --- Metric 2: Contributors (Row 2) ---
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['contributors_8w_unique'],
                    mode='lines', line=dict(color='#1f77b4', width=1.5),  # Blue
                    name='Contributors',
                    hovertemplate="<b>Contribs:</b> %{y}<extra></extra>"
                ), row=2, col=1)

                # --- Metric 3: Issues (Row 3) ---
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['issues_closed_8w_count'],
                    mode='lines', line=dict(color='#ff7f0e', width=1.5),  # Orange
                    name='Issues Closed',
                    hovertemplate="<b>Issues:</b> %{y}<extra></extra>"
                ), row=3, col=1)

                # --- Metric 4: Releases (Row 4) ---
                # Plot releases as a filled area or line to make small values visible
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['releases_8w_count'],
                    mode='lines', line=dict(color='#9467bd', width=1.5),  # Purple
                    fill='tozeroy',  # Fill to make it pop even if low
                    name='Releases',
                    hovertemplate="<b>Releases:</b> %{y}<extra></extra>"
                ), row=4, col=1)

                # --- Background Colors (Apply to ALL Rows) ---
                # We use shapes with xref='x', yref='paper' to span the full height
                segments = build_segments(df)
                shapes = []
                for start, end, stage in segments:
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="paper",  # This makes the color band cover all 4 subplots vertically
                        x0=start, x1=end,
                        y0=0, y1=1,
                        fillcolor=STAGE_COLORS.get(stage, "#eee"),
                        opacity=0.35,  # Slightly transparent
                        layer="below",
                        line_width=0
                    ))

                fig.update_layout(
                    title=f"Lifecycle Timeline (v2): {repo_url}",
                    height=900,  # Taller figure for 4 rows
                    shapes=shapes,
                    hovermode="x unified",
                    # --- FORCE WHITE THEME (Fixes the "Ugly" issue) ---
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    margin=dict(l=60, r=40, t=80, b=60),
                    showlegend=False  # Hide legend to save space, titles are enough
                )

                # Add a manual legend for Stages (Optional, using HTML or just reliance on Hover)
                # For now, Hover info is sufficient for stage ID.

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()