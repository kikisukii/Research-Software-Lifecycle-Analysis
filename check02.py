import pandas as pd, numpy as np, os
base = r"v2_data/02_dat"  # 如需改目录自行改

stamp = "20251114_004622"
fn = lambda n: os.path.join(base, f"{n}_{stamp}.csv")

meta   = pd.read_csv(fn("02_gh_repo_meta"))
comm   = pd.read_csv(fn("02_gh_commit_weekly"))
contrw = pd.read_csv(fn("02_contributors_weekly"))
issues = pd.read_csv(fn("02_gh_issues"))
rels   = pd.read_csv(fn("02_gh_releases"))
stat   = pd.read_csv(fn("02_api_status"))

# A1: (repo, week_unix) 唯一（commits / contributors）
assert not comm.duplicated(["repo","week_unix"]).any(), "commits 存在重复 (repo,week)"
assert not contrw.duplicated(["repo","week_unix"]).any(), "contributors_weekly 存在重复 (repo,week)"

# A2: 周起点=周日 00:00:00 UTC
d = pd.to_datetime(comm["week_unix"], unit="s", utc=True)
assert (d.dt.dayofweek == 6).all() and (d.dt.hour==0).all() and (d.dt.minute==0).all(), "周界未对齐周日 00:00 UTC"

# A3: commits_source 分布（git 为主；少量 api_52w/none 合理）
print("commits_source 分布：\n", stat["commits_source"].value_counts(dropna=False))

# A4: issues 已剔除 PR（无 pull_request 字段；状态集合理）
assert "pull_request" not in issues.columns, "issues 仍含 PR 字段（应在 02 已剔除）"
assert set(issues["state"].dropna().unique()) <= {"open","closed"}, "issues.state 出现异常取值"

# A5: releases 字段完备；布尔列类型正确性（宽松检查）
for c in ["is_prerelease","draft"]:
    assert c in rels.columns, f"releases 缺少列 {c}"

# A6: contributors_weekly 覆盖到有提交的周（允许存在提交但无作者邮箱的极端情况）
m = comm[["repo","week_unix"]].merge(contrw[["repo","week_unix"]], how="left", on=["repo","week_unix"], indicator=True)
missing_auth_weeks = m[m["_merge"]=="left_only"]
print("存在提交但无 weekly-contributors 的周数：", len(missing_auth_weeks))
# 若有少量是可以接受的（git log 中个别提交邮箱缺失/匿名），后续在 03 用 complete-case 过滤

# A7: meta 默认分支合理性（实际会含 main/master/开发分支名）
print("默认分支样例：", meta["default_branch"].value_counts().head(10))

# A8: releases/ issues 拉取状态
print("issues_api_status 分布：\n", stat["issues_api_status"].value_counts(dropna=False))
print("releases_api_status 分布：\n", stat["releases_api_status"].value_counts(dropna=False))

# A9: 覆盖仓数量（与 meta 行数一致；极少数 meta not available 会在 status 里记录）
repos_meta = set(meta["repo"])
repos_comm = set(comm["repo"])
assert repos_comm.issubset(repos_meta), "commits 中存在 meta 未覆盖的 repo（检查 api_status 的 meta 失败情况）"

# A10: 样例抽查 —— 将一个 repo 的 week_unix 转为日期看前后 3 行
sample = comm["repo"].iloc[0]
print("Sample repo:", sample)
print(comm[comm["repo"]==sample].sort_values("week_unix").head(3).assign(week_dt=lambda x: pd.to_datetime(x["week_unix"], unit="s", utc=True)))
print(comm[comm["repo"]==sample].sort_values("week_unix").tail(3).assign(week_dt=lambda x: pd.to_datetime(x["week_unix"], unit="s", utc=True)))
