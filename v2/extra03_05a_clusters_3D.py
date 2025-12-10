import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob


def find_latest_data_file():
    """
    Automatically finds the latest '05a_cluster_assignments_v2_*.csv' file.
    Searches in standard relative paths and recursively in v2_data.
    Please check the 'cluster_map' part to make sure the cluster id maps the phases names!!! It's not an automatic map
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define search patterns relative to the script location
    search_patterns = [
        # 1. Same directory (if moved manually)
        os.path.join(script_dir, "05a_cluster_assignments_v2_*.csv"),

        # 2. Standard Thesis Structure: v2_data/05a_cluster_and_profiles_v2/
        os.path.join(script_dir, "..", "v2_data", "05a_cluster_and_profiles_v2", "05a_cluster_assignments_v2_*.csv"),

        # 3. User Custom Path: v2_data/05b_name_and_label_v2/ (Your specific case)
        os.path.join(script_dir, "..", "v2_data", "05b_name_and_label_v2", "05a_cluster_assignments_v2_*.csv"),

        # 4. Fallback: Recursive search in v2_data (finds it anywhere inside v2_data)
        os.path.join(script_dir, "..", "v2_data", "**", "05a_cluster_assignments_v2_*.csv")
    ]

    print(f"[*] Searching for data file '05a_cluster_assignments_v2_*.csv'...")

    candidates = []
    for pattern in search_patterns:
        found = glob.glob(pattern, recursive=True)
        if found:
            candidates.extend(found)

    # Remove duplicates and sort
    candidates = sorted(list(set(candidates)))

    if not candidates:
        print("[!] Debug - Checked locations relative to script:")
        print(f"    Script dir: {script_dir}")
        print("    Target: ../v2_data/**/05a_cluster_assignments_v2_*.csv")
        raise FileNotFoundError(
            "Could not find any '05a_cluster_assignments_v2_*.csv' file. Please check directory structure."
        )

    # Get the latest file by timestamp
    latest_file = candidates[-1]
    print(f"[*] Auto-detected latest data file:\n    -> {os.path.abspath(latest_file)}")
    return latest_file


def main():
    # --- 1. Load Data ---
    try:
        csv_path = find_latest_data_file()
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Critical Error: {e}")
        return

    # --- 2. Map Stage Names ---
    #Edit phases names here!!!!
    cluster_map = {
        0: "Dormant",
        1: "Internal Development",
        2: "Release Phase",
        3: "Peak Activity",
        4: "Baseline",
        5: "Maintenance"
    }

    if 'cluster' not in df.columns:
        print("[!] Error: Column 'cluster' not found in CSV.")
        return

    df['Stage'] = df['cluster'].map(cluster_map)

    # --- 3. Define Color Palette ---
    color_map = {
        "Baseline": "#f8b862",
        "Internal Development": "#19448e",
        "Release Phase": "#9d5b8b",
        "Peak Activity": "#38b48b",
        "Maintenance": "#89c3eb",
        "Dormant": "#9ea1a3"
    }

    # --- 4. Filter Outliers (Top 0.5%) ---
    print("[*] Filtering outliers for clearer visualization...")
    p_low, p_high = 0.5, 99.5
    mask = (
            (df['PC1'] >= np.percentile(df['PC1'], p_low)) & (df['PC1'] <= np.percentile(df['PC1'], p_high)) &
            (df['PC2'] >= np.percentile(df['PC2'], p_low)) & (df['PC2'] <= np.percentile(df['PC2'], p_high)) &
            (df['PC3'] >= np.percentile(df['PC3'], p_low)) & (df['PC3'] <= np.percentile(df['PC3'], p_high))
    )
    df_clean = df[mask].copy()

    # --- 5. Generate 3D Plot ---
    print(f"[*] Generating 3D plot with {len(df_clean)} points...")

    # === 修改 1：加上约等号 (≈) ===
    # 使用 HTML 实体 &approx; 或者直接符号 ≈
    title_pc1 = (
        'PC1 (Activity)<br>'
        '≈ <sup>0.58*Com + 0.57*Ctrb + 0.47*Iss + 0.33*Rel</sup>'
    )
    title_pc2 = (
        'PC2 (Delivery)<br>'
        '≈ <sup>0.78*Rel + 0.34*Iss - 0.34*Com - 0.39*Ctrb</sup>'
    )
    title_pc3 = (
        'PC3 (Maintenance)<br>'
        '≈ <sup>0.81*Iss - 0.52*Rel - 0.21*Com - 0.16*Ctrb</sup>'
    )

    fig = px.scatter_3d(
        df_clean, x='PC1', y='PC2', z='PC3',
        color='Stage', color_discrete_map=color_map,
        hover_name='repo',
        hover_data={'week_dt': True, 'commits_8w_sum': True, 'cluster': False, 'Stage': False},
        opacity=0.6,
        title="Interactive 3D Lifecycle Clusters (v2)"
    )

    # === 修改 2：粒子大小改为 1 (极致细腻) ===
    fig.update_traces(marker=dict(size=1))

    # Clean axis style
    axis_style = dict(
        showbackground=False, showgrid=False,
        zeroline=True, zerolinewidth=5, zerolinecolor='black',
        showline=True, linecolor='black', linewidth=3,
        title_font=dict(size=12, family='Arial'),
        tickfont=dict(size=10)
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(**axis_style, title=title_pc1),
            yaxis=dict(**axis_style, title=title_pc2),
            zaxis=dict(**axis_style, title=title_pc3),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.6))
        ),
        legend=dict(
            title_text='Lifecycle Phase',
            itemsizing='constant',
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='Black',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # --- 6. Save Output ---
    data_dir = os.path.dirname(csv_path)
    output_filename = os.path.join(data_dir, "v2_3d_clusters_interactive.html")

    fig.write_html(output_filename)

    print("-" * 60)
    print(f"[*] Success! Interactive plot saved to:\n    {os.path.abspath(output_filename)}")
    print("-" * 60)


if __name__ == "__main__":
    main()