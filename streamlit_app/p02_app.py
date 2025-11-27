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
            # Use current week as end to ensure continuity (no gaps)
            segments.append((start_date, df.iloc[i]["week_date"], current_stage))
            current_stage = stage
            start_date = df.iloc[i]["week_date"]
    segments.append((start_date, df.iloc[-1]["week_date"], current_stage))
    return segments


def smooth_series(series, window=3):
    """V2 Smoothing logic."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def main():
    st.title("🧬 Research Software Lifecycle Detector (Full v2)")
    st.caption("🚀 Version updated: 0.0.7")

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

                # --- Data Prep for Visualization ---
                # Smooth curves
                df['c8_s'] = smooth_series(df['commits_8w_sum'])
                df['u8_s'] = smooth_series(df['contributors_8w_unique'])
                df['i8_s'] = smooth_series(df['issues_closed_8w_count'])
                df['r8_s'] = df['releases_8w_count']  # No smooth for releases

                # Prepare Custom Data for Hover
                # We stack all info into a single array so EVERY trace knows everything
                # Structure: [Stage, Commits, Contribs, Issues, Releases]
                custom_data = np.stack((
                    df['stage_name'],
                    df['c8_s'].round(1),
                    df['u8_s'].round(1),
                    df['i8_s'].round(1),
                    df['r8_s']
                ), axis=-1)

                # Define the Universal Hover Template
                # This ensures that no matter which line you hover, you see the full context
                hover_template = (
                        "<b>Week:</b> %{x|%Y-%m-%d}<br>" +
                        "<b>Stage:</b> %{customdata[0]}<br>" +
                        "<br>" +
                        "Commits: %{customdata[1]}<br>" +
                        "Contributors: %{customdata[2]}<br>" +
                        "Issues Closed: %{customdata[3]}<br>" +
                        "Releases: %{customdata[4]}" +
                        "<extra></extra>"  # Hides the trace name box on the side
                )

                # --- Visualization ---
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,  # Tighter gap
                    subplot_titles=("Commits", "Contributors", "Issues", "Releases")
                )

                # Row 1: Commits
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['c8_s'],
                    mode='lines', line=dict(color='#333333', width=2),
                    name='Commits', customdata=custom_data, hovertemplate=hover_template
                ), row=1, col=1)

                # Row 2: Contributors
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['u8_s'],
                    mode='lines', line=dict(color='#1f77b4', width=2),
                    name='Contributors', customdata=custom_data, hovertemplate=hover_template
                ), row=2, col=1)

                # Row 3: Issues
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['i8_s'],
                    mode='lines', line=dict(color='#ff7f0e', width=2),
                    name='Issues', customdata=custom_data, hovertemplate=hover_template
                ), row=3, col=1)

                # Row 4: Releases (Solid Line now, no dash)
                fig.add_trace(go.Scatter(
                    x=df['week_date'], y=df['r8_s'],
                    mode='lines', line=dict(color='#9467bd', width=2),  # Solid purple
                    name='Releases', customdata=custom_data, hovertemplate=hover_template
                ), row=4, col=1)

                # --- Background Colors ---
                segments = build_segments(df)
                # Loop through 4 subplots
                for row_idx in range(1, 5):
                    for start, end, stage in segments:
                        fig.add_vrect(
                            x0=start, x1=end,
                            fillcolor=STAGE_COLORS.get(stage, "#eee"),
                            opacity=0.4,  # Slightly more opaque
                            layer="below",
                            line_width=0,  # Removes vertical white lines
                            row=row_idx, col=1
                        )

                # --- Layout: Perfect Border & Layout ---
                # Lock X-axis range to remove white padding on sides
                min_date = df['week_date'].min()
                max_date = df['week_date'].max()

                fig.update_layout(
                    title=dict(text=f"Lifecycle Timeline (v2): {repo_url}", font=dict(color="black", size=18)),
                    height=1000,
                    hovermode="x",  # Simple x-line hover
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    margin=dict(l=60, r=40, t=80, b=60),
                    showlegend=False
                )

                # Axis Styling: Remove inner Grid, Keep outer Box
                common_axis = dict(
                    showgrid=False,  # REMOVES THE WHITE GRID LINES inside color blocks
                    zeroline=False,
                    showline=True, linecolor='black', linewidth=1,  # Sharp black border
                    mirror=True,  # Border on all 4 sides
                    tickfont=dict(color='black', size=11),
                    range=[min_date, max_date]  # LOCKS the range (No white side gaps)
                )

                fig.update_xaxes(**common_axis)
                fig.update_yaxes(**common_axis)

                # Keep horizontal grid only for readability? Optional.
                # User asked for "no strange white lines", usually vertical grid causes this.
                # Let's enable Horizontal Grid ONLY, disable Vertical.
                fig.update_yaxes(showgrid=True, gridcolor='#e5e5e5', gridwidth=1)

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()