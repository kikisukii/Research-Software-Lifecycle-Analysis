# -*- coding: utf-8 -*-
"""
v2/05d_compare_run.py
功能：从数据中随机抽取 10 个仓库，然后同时驱动“8w版”和“Weekly版”画图脚本进行绘制。
确保两个文件夹生成的是同一批仓库的图。
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# --- 配置：请确认你的文件名 ---

SCRIPT_8W = "05c_plot.py"
SCRIPT_WEEKLY = "05c_test_weekly.py"

# --- 路径 ---
THIS = Path(__file__).resolve()
DIR_V2 = THIS.parent
DATA_DIR = DIR_V2.parent / "v2_data" / "05_b"


def get_latest_data():
    # 找到最新的 05b 文件
    cands = sorted(DATA_DIR.glob("05b_assignments_with_stage_AND_DEAD_v2_*.csv"))
    if not cands:
        print(f"[错误] 在 {DATA_DIR} 找不到 05b 数据文件，请先运行 03 和 05b。")
        sys.exit(1)
    return cands[-1]


def main():
    # 1. 读取仓库列表
    csv_file = get_latest_data()
    print(f"[1/4] 读取数据: {csv_file.name}")
    df = pd.read_csv(csv_file)
    all_repos = sorted(df["repo"].unique())

    # 2. 随机抽 10 个
    rng = np.random.default_rng(42)  # 种子固定，每次运行抽的都一样
    targets = rng.choice(all_repos, size=10, replace=False)
    print(f"[2/4] 选中目标仓库 (n=10):")
    for r in targets:
        print(f"  - {r}")

    # 3. 构造命令参数
    # 形式: python script.py --repo r1 --repo r2 ...
    cmd_args = []
    for r in targets:
        cmd_args.append("--repo")
        cmd_args.append(r)

    # 4. 运行 8w 版 (05c_plot_repo_v2.py)
    print(f"\n[3/4] 正在运行 8w 版画图 ({SCRIPT_8W})...")
    try:
        subprocess.run([sys.executable, str(DIR_V2 / SCRIPT_8W)] + cmd_args, check=True)
    except Exception as e:
        print(f"[错误] 运行 8w 版失败: {e}")

    # 5. 运行 Weekly 版 (05c_plot_repo_v2_weekly.py)
    print(f"\n[4/4] 正在运行 Weekly 版画图 ({SCRIPT_WEEKLY})...")
    try:
        subprocess.run([sys.executable, str(DIR_V2 / SCRIPT_WEEKLY)] + cmd_args, check=True)
    except Exception as e:
        print(f"[错误] 运行 Weekly 版失败: {e}")

    print("\n" + "=" * 40)
    print("对比完成！请查看以下两个文件夹:")
    print(f"1. 8w 版图表:     v2_data/05_c_viz/")
    print(f"2. Weekly 版图表: v2_data/05_c_weekly_test/")
    print("=" * 40)


if __name__ == "__main__":
    main()