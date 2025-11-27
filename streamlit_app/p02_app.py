import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from p01_inference import run_git_analysis

# Page Config (Layout wide)
st.set_page_config(page_title="Research Software Lifecycle Detector", layout="wide")

# --- 1. Colors (V2 Palette - Pastel/Clean) ---
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
    """Apply slight visual smoothing (V2 standard)."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def main():
    st.title("🧬 Research Software Lifecycle Detector (Full v2)")

    # --- Version Tag (For your verification) ---
    st.caption("🚀 Version updated: 0.0.6")
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
        with st.spinner("Fetching Data & Analyzing..."):
            try:
                # 1. Path Setup
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "model_bundle_v2.pkl")

                # 2. Run Inference
                df = run_git_analysis(repo_url, model_path, token)

                st.success(f"Analysis complete! Weeks: {len(df)}")

                # --- Visualization: High Contrast 4-Row Plot ---

                # Create 4 rows
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.06,  # Nice gap between plots
                    subplot_titles=(
                        "Commits (8w rolling)",
                        "Contributors (8w rolling)",
                        "Issues Closed (8w rolling)",
                        "Releases (8w rolling)"
                    )
                )

                # Smoothing
                c8 = smooth_series(df['commits_8w_sum'])
                u8 = smooth_series(df['contributors_8w_unique'])
                i8 = smooth_series(df['issues_closed_8w_count'])
                r8 = df['releases_8w_count']  # No smoothing for releases

                # --- Traces (Lines) ---
                # Row 1: Commits (Dark Grey)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=c8,
                    mode='lines', line=dict(color='#333333', width=2),
                    name='Commits', customdata=df['stage_name'],
                    hovertemplate="Week: %{x}<br>Commits: %{y:.1f}<br>Phase: %{customdata}<extra></extra>"
                ), row=1, col=1)

                # Row 2: Contributors (Blue)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=u8,
                    mode='lines', line=dict(color='#1f77b4', width=2),
                    name='Contributors',
                    hovertemplate="Contribs: %{y:.1f}<extra></extra>"
                ), row=2, col=1)

                # Row 3: Issues (Orange)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=i8,
                    mode='lines', line=dict(color='#ff7f0e', width=2),
                    name='Issues',
                    hovertemplate="Issues: %{y:.1f}<extra></extra>"
                ), row=3, col=1)

                # Row 4: Releases (Purple, Dashed, NO FILL)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=r8,
                    mode='lines', line=dict(color='#9467bd', width=2, dash='dot'),  # Dashed line only
                    name='Releases',
                    hovertemplate="Releases: %{y}<extra></extra>"
                ), row=4, col=1)

                # --- Background Colors (Per Subplot) ---
                segments = build_segments(df)

                # Loop through each subplot row (1 to 4) to add vrects
                # This ensures colors stay INSIDE the box, not in the gaps
                for row_idx in range(1, 5):
                    for start, end, stage in segments:
                        fig.add_vrect(
                            x0=start, x1=end,
                            fillcolor=STAGE_COLORS.get(stage, "#eee"),
                            opacity=0.3,  # Lighter opacity for better readability
                            layer="below",
                            line_width=0,
                            row=row_idx, col=1
                        )

                # --- Layout: Force "Academic White" ---
                fig.update_layout(
                    title=dict(text=f"Lifecycle Timeline (v2): {repo_url}", font=dict(color="black", size=20)),
                    height=1000,
                    hovermode="x unified",
                    template="plotly_white",  # Base template
                    paper_bgcolor="white",  # Force white background outside
                    plot_bgcolor="white",  # Force white background inside
                    margin=dict(l=60, r=40, t=80, b=60),
                    showlegend=False,
                    font=dict(color="black")  # Force ALL text to be black
                )

                # --- Axis Styling (Sharp Black Lines) ---
                common_axis = dict(
                    showgrid=True, gridcolor='#f0f0f0',  # Very light grid
                    zeroline=False,
                    showline=True, linecolor='black', linewidth=1,  # Black border
                    mirror=True,
                    tickfont=dict(color='black', size=12),  # Sharp black ticks
                    title_font=dict(color='black')
                )

                fig.update_xaxes(**common_axis)
                fig.update_yaxes(**common_axis)

                # Specific Y-axis titles
                fig.update_yaxes(title_text="Count", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()