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


def smooth_series(series, window=3):
    """Apply visual smoothing matching V2 local script."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def main():
    st.title("🧬 Research Software Lifecycle Detector (Full v2)")

    # --- Version Tag ---
    st.caption("🚀 Version updated: 0.0.5")
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

                # --- Visualization: Exact V2 Replica ---

                # Create 4 rows with generous spacing
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,  # Increase gap between charts
                    subplot_titles=(
                        "Commits (8w rolling)",
                        "Active Contributors (8w rolling)",
                        "Issues Closed (8w rolling)",
                        "Releases (8w rolling)"
                    )
                )

                # Apply Smoothing (Window=3) to match V2 local plots
                c8_smooth = smooth_series(df['commits_8w_sum'])
                u8_smooth = smooth_series(df['contributors_8w_unique'])
                i8_smooth = smooth_series(df['issues_closed_8w_count'])
                # Releases usually don't need much smoothing, but consistent style helps
                r8_smooth = df['releases_8w_count']

                # --- Metric 1: Commits (Row 1) ---
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=c8_smooth,
                    mode='lines', line=dict(color='#333333', width=1.8),
                    name='Commits',
                    customdata=df['stage_name'],
                    hovertemplate="<b>Week:</b> %{x}<br><b>Commits:</b> %{y:.1f}<br><b>Phase:</b> %{customdata}<extra></extra>"
                ), row=1, col=1)

                # --- Metric 2: Contributors (Row 2) ---
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=u8_smooth,
                    mode='lines', line=dict(color='#1f77b4', width=1.5),  # Blue
                    name='Contributors',
                    hovertemplate="<b>Contribs:</b> %{y:.1f}<extra></extra>"
                ), row=2, col=1)

                # --- Metric 3: Issues (Row 3) ---
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=i8_smooth,
                    mode='lines', line=dict(color='#ff7f0e', width=1.5),  # Orange
                    name='Issues Closed',
                    hovertemplate="<b>Issues:</b> %{y:.1f}<extra></extra>"
                ), row=3, col=1)

                # --- Metric 4: Releases (Row 4) ---
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=r8_smooth,
                    mode='lines', line=dict(color='#9467bd', width=1.5),  # Purple
                    fill='tozeroy',  # Fill slightly to mimic density
                    name='Releases',
                    hovertemplate="<b>Releases:</b> %{y}<extra></extra>"
                ), row=4, col=1)

                # --- Background Colors (Clean Shapes) ---
                segments = build_segments(df)
                shapes = []
                for start, end, stage in segments:
                    shapes.append(dict(
                        type="rect",
                        xref="x", yref="paper",  # Span full height
                        x0=start, x1=end,
                        y0=0, y1=1,
                        fillcolor=STAGE_COLORS.get(stage, "#eee"),
                        opacity=0.35,
                        layer="below",
                        line_width=0  # CRITICAL: Removes the black vertical lines
                    ))

                fig.update_layout(
                    title=f"Lifecycle Timeline (v2): {repo_url}",
                    height=1000,  # Taller: 1000px to avoid squashing
                    shapes=shapes,
                    hovermode="x unified",
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    margin=dict(l=60, r=40, t=80, b=60),
                    showlegend=False
                )

                # --- Clean Axes (Remove black lines, keep grid) ---
                common_axis_config = dict(
                    showgrid=True,
                    gridcolor='#eeeeee',  # Light grey grid
                    zeroline=False,  # Remove the thick zero line
                    showline=True,  # Keep outer frame
                    linecolor='#cccccc',  # Light frame
                    mirror=True
                )

                fig.update_xaxes(**common_axis_config)
                fig.update_yaxes(**common_axis_config)

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()