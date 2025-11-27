import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from p01_inference import run_git_analysis

# Page Config
st.set_page_config(page_title="Research Software Lifecycle Detector", layout="wide")

# --- Colors (V2 Standard) ---
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
    """V2 Smoothing logic (Causes decimals, which is expected)."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def main():
    st.title("🧬 Research Software Lifecycle Detector (Full v2)")
    st.caption("🚀 Version updated: 0.0.8")

    if "GITHUB_TOKEN" not in st.secrets:
        st.error("⚠️ GitHub Token missing in Secrets.")
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
        with st.spinner("Fetching & Analyzing (Git Clone + API)..."):
            try:
                # 1. Inference
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "model_bundle_v2.pkl")
                df = run_git_analysis(repo_url, model_path, token)

                st.success(f"Analysis complete! Weeks: {len(df)}")

                # --- Data Prep ---
                # Smooth curves (Note: this creates decimals)
                c8_s = smooth_series(df['commits_8w_sum'])
                u8_s = smooth_series(df['contributors_8w_unique'])
                i8_s = smooth_series(df['issues_closed_8w_count'])
                r8_s = df['releases_8w_count']  # No smoothing for releases

                # Prepare Custom Data for Hover (All info in every hover)
                custom_data = np.stack((
                    df['stage_name'],
                    c8_s.round(1),
                    u8_s.round(1),
                    i8_s.round(1),
                    r8_s
                ), axis=-1)

                hover_template = (
                        "<b>Week:</b> %{x|%Y-%m-%d}<br>" +
                        "<b>Stage:</b> %{customdata[0]}<br>" +
                        "<br>" +
                        "Commits (Avg): %{customdata[1]}<br>" +
                        "Contributors (Avg): %{customdata[2]}<br>" +
                        "Issues (Avg): %{customdata[3]}<br>" +
                        "Releases: %{customdata[4]}" +
                        "<extra></extra>"
                )

                # --- Visualization ---
                # REMOVED subplot_titles to clean up the top
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03
                )

                # Row 1: Commits
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=c8_s,
                    mode='lines', line=dict(color='#333333', width=2),
                    name='Commits', customdata=custom_data, hovertemplate=hover_template
                ), row=1, col=1)

                # Row 2: Contributors
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=u8_s,
                    mode='lines', line=dict(color='#1f77b4', width=2),
                    name='Contributors', customdata=custom_data, hovertemplate=hover_template
                ), row=2, col=1)

                # Row 3: Issues
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=i8_s,
                    mode='lines', line=dict(color='#ff7f0e', width=2),
                    name='Issues', customdata=custom_data, hovertemplate=hover_template
                ), row=3, col=1)

                # Row 4: Releases (Solid Line)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=r8_s,
                    mode='lines', line=dict(color='#9467bd', width=2),
                    name='Releases', customdata=custom_data, hovertemplate=hover_template
                ), row=4, col=1)

                # --- Inner Labels (Annotations instead of Titles) ---
                # This puts the label INSIDE the box, top-right, colored
                labels = [
                    (1, "Commits (8w)", "#333333"),
                    (2, "Contributors (8w)", "#1f77b4"),
                    (3, "Issues Closed (8w)", "#ff7f0e"),
                    (4, "Releases (8w)", "#9467bd")
                ]
                for row, text, color in labels:
                    fig.add_annotation(
                        xref=f"x{row} domain" if row == 1 else "x domain",  # simple domain ref
                        yref=f"y{row} domain" if row == 1 else f"y{row} domain",
                        # For subplots, referencing domains can be tricky in loop, using paper ref relative to subplot
                        # Easier method: use 'xref'='paper' is hard.
                        # Let's use standard add_annotation to the specific subplot row/col
                        text=f"<b>{text}</b>",
                        showarrow=False,
                        x=0.99, y=0.95,  # Top Right corner
                        xref=f"x{row} domain" if row > 1 else "x domain",
                        yref=f"y{row} domain" if row > 1 else "y domain",
                        font=dict(color=color, size=14)
                    )

                # --- Background Colors ---
                segments = build_segments(df)
                for row_idx in range(1, 5):
                    for start, end, stage in segments:
                        fig.add_vrect(
                            x0=start, x1=end,
                            fillcolor=STAGE_COLORS.get(stage, "#eee"),
                            opacity=0.4,
                            layer="below",
                            line_width=0,  # No border for color blocks
                            row=row_idx, col=1
                        )

                # --- Layout: Tight & Clean ---
                # Lock X range to exact data limits (Removes side gaps)
                min_date = df['week_date'].min()
                max_date = df['week_date'].max()

                fig.update_layout(
                    title=dict(text=f"Lifecycle Timeline (v2): {repo_url}", font=dict(color="black", size=18)),
                    height=1000,
                    hovermode="x",
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    margin=dict(l=60, r=40, t=60, b=60),  # Reduced top margin since titles are gone
                    showlegend=False
                )

                # Axis Styling: NO GRID (Clean Look)
                common_axis = dict(
                    showgrid=False,  # CRITICAL: Removes all internal grid lines
                    zeroline=False,
                    showline=True, linecolor='black', linewidth=1,  # Outer Frame
                    mirror=True,  # Frame on all sides
                    tickfont=dict(color='black', size=11),
                    range=[min_date, max_date]  # CRITICAL: Forces "Tight Fit"
                )

                fig.update_xaxes(**common_axis)
                fig.update_yaxes(**common_axis)

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()