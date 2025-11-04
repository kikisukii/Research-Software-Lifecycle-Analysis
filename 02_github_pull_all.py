# 02_github_pull_all.py
import os, csv, time, sys, requests, subprocess, shutil
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# ========= Config =========
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# !!! 改成你的 01 文件名 !!!
INPUT_FILENAME = "01_rsd_software_all_20251028_235202.csv"
INPUT = os.path.join(DATA_DIR, INPUT_FILENAME)

# Commits 获取策略
USE_GIT_FOR_COMMITS = True            # 优先 git 全历史；失败回退 API(52周)
GIT_DEFAULT_BRANCH_ONLY = True        # True=只统计默认分支；False=统计所有分支
CLEAN_GIT_CACHE_AFTER = True          # True=跑完删除克隆缓存，省磁盘

# ========= Outputs =========
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUT_META          = os.path.join(DATA_DIR, f"02_gh_repo_meta_{timestamp}.csv")
OUT_COMMITS       = os.path.join(DATA_DIR, f"02_gh_commit_weekly_{timestamp}.csv")
OUT_RELEASES      = os.path.join(DATA_DIR, f"02_gh_releases_{timestamp}.csv")
OUT_ISSUES        = os.path.join(DATA_DIR, f"02_gh_issues_{timestamp}.csv")
OUT_COMMIT_STATUS = os.path.join(DATA_DIR, f"02_gh_commit_status_{timestamp}.csv")

# ========= Auth / Session =========
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    print("Error: GITHUB_TOKEN not found in .env file.")
    print('Please add GITHUB_TOKEN="your_token" to .env')
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ========= Utils =========
def week_start_sunday_unix(dt_utc: datetime) -> int:
    """对齐 GitHub API：周起点为【周日 00:00 UTC】"""
    # weekday(): Monday=0 ... Sunday=6 ；周日偏移 = (weekday+1)%7 == 0 表示周日
    offset = (dt_utc.weekday() + 1) % 7
    wk = dt_utc - timedelta(days=offset)
    wk = wk.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(wk.timestamp())

# ========= API: meta / commits(52w) / releases / issues =========
def get_repo_meta(owner_repo, writer_meta):
    try:
        r = SESSION.get(f"https://api.github.com/repos/{owner_repo}")
        if r.status_code == 404:
            print(f"... {owner_repo} meta 404 (not found/private). Skipping this repo.")
            return False, None, None
        r.raise_for_status()
        j = r.json()
        writer_meta.writerow({
            "repo": owner_repo,
            "id": j.get("id"),
            "is_fork": j.get("fork"),
            "created_at": j.get("created_at"),
            "updated_at": j.get("updated_at"),
            "pushed_at": j.get("pushed_at"),
            "size": j.get("size"),
            "stars": j.get("stargazers_count"),
            "watchers": j.get("watchers_count"),
            "forks": j.get("forks_count"),
            "open_issues": j.get("open_issues_count"),
            "license": j.get("license", {}).get("name") if j.get("license") else None,
            "default_branch": j.get("default_branch"),
            "topics": ";".join(j.get("topics", [])),
        })
        print(f"... repo {owner_repo} meta done")
        return True, r.headers.get("X-RateLimit-Remaining"), j.get("default_branch")
    except Exception as e:
        print(f"!!! Error getting meta for {owner_repo}: {e}")
        return False, None, None

def get_commit_activity_api(owner_repo, writer_commits):
    """API：只拿最近52周；周起点已经是周日00:00UTC；直接写入 repo/week_unix/commits"""
    url = f"https://api.github.com/repos/{owner_repo}/stats/commit_activity"
    for attempt in range(5):
        try:
            r = SESSION.get(url)
            if r.status_code == 200:
                data = r.json() or []
                for week in data:
                    writer_commits.writerow({
                        "repo": owner_repo,
                        "week_unix": week.get("week"),
                        "commits": week.get("total")
                    })
                print(f"... {owner_repo} commit_activity (API) done")
                return True
            if r.status_code == 204:
                print(f"... {owner_repo} has no commit history (204).")
                return True
            if r.status_code == 404:
                print(f"... {owner_repo} commit_activity 404 (gone/no access).")
                return False
            print(f"... {owner_repo} commit stats {r.status_code} (retrying {attempt+1}/5)...")
            time.sleep(3)
        except Exception as e:
            print(f"!!! Error getting commit_activity(API) for {owner_repo} (attempt {attempt+1}/5): {e}")
            time.sleep(2)
    print(f"!!! commit_activity(API) pending for {owner_repo} (no data after retries).")
    return False

# ========= GIT: full history weekly =========
def get_commit_activity_git(owner_repo, writer_commits, default_branch=None):
    """
    用 git 拉全历史提交时间，按【周日 00:00 UTC】聚合为周粒度。
    - 裸库 + 过滤 blobs：减少带宽/磁盘
    - 默认只统计 default_branch（更接近 API）；如需全分支，把 GIT_DEFAULT_BRANCH_ONLY 设为 False
    """
    url = f"https://github.com/{owner_repo}.git"
    cache_root = os.path.join(DATA_DIR, "_gitcache")
    os.makedirs(cache_root, exist_ok=True)
    repo_dir = os.path.join(cache_root, owner_repo.replace("/", "__"))

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)

    try:
        subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none", "--no-checkout", "--quiet", url, repo_dir],
            check=True
        )
        log_cmd = ["git", "-C", repo_dir, "log", "--pretty=%ct"]
        if GIT_DEFAULT_BRANCH_ONLY and default_branch:
            log_cmd.append(default_branch)
        else:
            log_cmd.append("--all")

        res = subprocess.run(log_cmd, check=True, capture_output=True, text=True)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        if not lines:
            print(f"... {owner_repo} has no commits (git).")
            return True  # 空历史也算成功

        buckets = {}
        for ts_str in lines:
            ts = int(ts_str)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            week_unix = week_start_sunday_unix(dt)
            buckets[week_unix] = buckets.get(week_unix, 0) + 1

        for wk, cnt in sorted(buckets.items()):
            writer_commits.writerow({"repo": owner_repo, "week_unix": wk, "commits": cnt})

        print(f"... {owner_repo} commit_activity via git done ({len(buckets)} weeks)")
        return True

    except Exception as e:
        print(f"!!! git clone/log failed for {owner_repo}: {e}")
        return False
    finally:
        if CLEAN_GIT_CACHE_AFTER:
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception:
                pass

# ========= API: releases / issues (full history via pagination) =========
def get_all_releases(owner_repo, writer_releases):
    url = f"https://api.github.com/repos/{owner_repo}/releases?per_page=100"
    any_success = False
    while url:
        try:
            r = SESSION.get(url)
            if r.status_code == 404:
                print(f"... {owner_repo} releases 404 (no access/gone). Skipping releases.")
                return
            r.raise_for_status()
            rels = r.json() or []
            for rel in rels:
                writer_releases.writerow({
                    "repo": owner_repo,
                    "tag": rel.get("tag_name"),
                    "name": rel.get("name"),
                    "created_at": rel.get("created_at"),
                    "published_at": rel.get("published_at"),
                    "is_prerelease": rel.get("prerelease")
                })
                any_success = True
            url = r.links.get("next", {}).get("url")
        except Exception as e:
            print(f"!!! Error getting releases for {owner_repo}: {e}")
            break
    if any_success:
        print(f"... {owner_repo} releases done")
    else:
        print(f"... {owner_repo} releases skipped / none")

def get_all_issues(owner_repo, writer_issues):
    url = f"https://api.github.com/repos/{owner_repo}/issues?state=all&per_page=100&direction=asc"
    page = 1
    any_success = False
    while url:
        try:
            r = SESSION.get(url)
            if r.status_code == 404:
                print(f"... {owner_repo} issues 404 (no access/gone). Skipping issues.")
                return
            r.raise_for_status()
            data = r.json() or []
            if not data:
                break
            for item in data:
                writer_issues.writerow({
                    "repo": owner_repo,
                    "number": item.get("number"),
                    "is_pr": "pull_request" in item,
                    "state": item.get("state"),
                    "created_at": item.get("created_at"),
                    "closed_at": item.get("closed_at"),
                    "user_login": item.get("user", {}).get("login") if item.get("user") else None,
                    "comments": item.get("comments")
                })
                any_success = True
            print(f"... {owner_repo} issues page {page} done")
            page += 1
            url = r.links.get("next", {}).get("url")
        except Exception as e:
            print(f"!!! Error getting issues for {owner_repo} on page {page}: {e}")
            break
    if any_success:
        print(f"... {owner_repo} issues done")
    else:
        print(f"... {owner_repo} issues skipped / none")

# ========= Main =========
def main():
    if not os.path.exists(INPUT):
        print(f"Error: Input file not found: {INPUT}")
        print("Please check INPUT_FILENAME.")
        return

    # read repos
    repos = []
    with open(INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("github_owner_repo"):
                repos.extend(row.get("github_owner_repo").split(';'))
    repos = sorted(list(set(r for r in repos if r)))

    print(f"Found {len(repos)} unique repositories to process.")
    print(f"Output prefix timestamp: {timestamp}")
    print(f"Meta:           {OUT_META}")
    print(f"Commits:        {OUT_COMMITS}")
    print(f"Releases:       {OUT_RELEASES}")
    print(f"Issues:         {OUT_ISSUES}")
    print(f"Commit status:  {OUT_COMMIT_STATUS}")

    with open(OUT_META, 'w', newline='', encoding='utf-8') as f_meta, \
         open(OUT_COMMITS, 'w', newline='', encoding='utf-8') as f_commits, \
         open(OUT_RELEASES, 'w', newline='', encoding='utf-8') as f_releases, \
         open(OUT_ISSUES, 'w', newline='', encoding='utf-8') as f_issues, \
         open(OUT_COMMIT_STATUS, 'w', newline='', encoding='utf-8') as f_cstatus:

        w_meta = csv.DictWriter(f_meta, fieldnames=[
            "repo","id","is_fork","created_at","updated_at","pushed_at","size",
            "stars","watchers","forks","open_issues","license","default_branch","topics"
        ])
        w_commits = csv.DictWriter(f_commits, fieldnames=["repo","week_unix","commits"])
        w_releases = csv.DictWriter(f_releases, fieldnames=["repo","tag","name","created_at","published_at","is_prerelease"])
        w_issues = csv.DictWriter(f_issues, fieldnames=["repo","number","is_pr","state","created_at","closed_at","user_login","comments"])
        w_cstatus = csv.DictWriter(f_cstatus, fieldnames=["repo","has_commit_data"])

        w_meta.writeheader(); w_commits.writeheader(); w_releases.writeheader(); w_issues.writeheader(); w_cstatus.writeheader()

        print(f"Starting processing for {len(repos)} repositories...")

        for i, owner_repo in enumerate(repos):
            print(f"\n--- Processing {i+1}/{len(repos)}: {owner_repo} ---")

            # 1) META
            ok_meta, rem_s, default_branch = get_repo_meta(owner_repo, w_meta)
            if not ok_meta:
                print(f"--- Skipping {owner_repo} because meta failed ---")
                w_cstatus.writerow({"repo": owner_repo, "has_commit_data": 0})
                continue

            # 简单限流
            if rem_s:
                try:
                    rem = int(rem_s)
                    if rem < 50:
                        print(f"Rate limit remaining {rem}, sleeping 60s...")
                        time.sleep(60)
                except ValueError:
                    pass

            # 2) COMMITS — prefer git; fallback API
            ok_commits = False
            if USE_GIT_FOR_COMMITS:
                ok_commits = get_commit_activity_git(owner_repo, w_commits, default_branch=default_branch)
            if not ok_commits:
                ok_commits = get_commit_activity_api(owner_repo, w_commits)
            w_cstatus.writerow({"repo": owner_repo, "has_commit_data": 1 if ok_commits else 0})

            # 3) Releases
            get_all_releases(owner_repo, w_releases)

            # 4) Issues
            get_all_issues(owner_repo, w_issues)

            print(f"--- Finished processing: {owner_repo} ---")

    print("\nAll processing complete!")

if __name__ == "__main__":
    main()
