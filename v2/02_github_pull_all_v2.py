# v2/02_github_pull_all_v2.py
# Usage: Run directly in PyCharm (no arguments required).
# Requirements: requests, python-dotenv, Git CLI installed; project root contains .env with GITHUB_TOKEN.
# Outputs: ../v2_data/02_dat/ with current UTC <STAMP_PULL>.
# Notes:
#   - Weekly boundary aligned to GitHub: Sunday 00:00:00 (UTC).
#   - Commits/contributors are weekly; issues/releases are event-level (full history); PRs are excluded from issues.
#   - Git is primary (full history). If Git fails, fallback to /stats/commit_activity (52w) and /commits list for unique authors.

import os
import sys
import csv
import re
import time
import json
import shutil
import argparse
import subprocess
import glob
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import requests
from dotenv import load_dotenv

# -------------------------------
# Paths & constants
# -------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
OUT_DIR = os.path.join(ROOT_DIR, "v2_data", "02_dat")
CACHE_DIR = os.path.join(ROOT_DIR, "v2_data", "_gitcache")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

UTC = timezone.utc
STAMP_PULL = datetime.utcnow().strftime("%Y%m%d_%H%M%S")  # batch timestamp in UTC

# -------------------------------
# Environment (.env in project root)
# -------------------------------
load_dotenv(os.path.join(ROOT_DIR, ".env"))
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    print("Error: GITHUB_TOKEN not found in project-root .env")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# -------------------------------
# Output files
# -------------------------------
FN_META       = os.path.join(OUT_DIR, f"02_gh_repo_meta_{STAMP_PULL}.csv")
FN_COMMITS    = os.path.join(OUT_DIR, f"02_gh_commit_weekly_{STAMP_PULL}.csv")
FN_CONTR_W    = os.path.join(OUT_DIR, f"02_contributors_weekly_{STAMP_PULL}.csv")
FN_CONTR_RAW  = os.path.join(OUT_DIR, f"02_contributors_raw_{STAMP_PULL}.csv")   # raw audit (local only)
FN_ISSUES     = os.path.join(OUT_DIR, f"02_gh_issues_{STAMP_PULL}.csv")          # PRs removed here
FN_RELEASES   = os.path.join(OUT_DIR, f"02_gh_releases_{STAMP_PULL}.csv")
FN_API_STATUS = os.path.join(OUT_DIR, f"02_api_status_{STAMP_PULL}.csv")
FN_PULL_META  = os.path.join(OUT_DIR, f"02_pull_meta_{STAMP_PULL}.json")

# -------------------------------
# Helpers
# -------------------------------
def week_start_sunday_unix(dt_utc: datetime) -> int:
    """Return epoch for the week start aligned to GitHub: Sunday 00:00:00 (UTC)."""
    # Monday=0..Sunday=6; distance to Sunday:
    delta_days = (dt_utc.weekday() + 1) % 7
    start = (dt_utc - timedelta(days=delta_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp())

def parse_stamp_rsd_from_path(path: str) -> str:
    """Extract <STAMP> like 20251028_235202 from filename."""
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    return m.group(1) if m else ""

def find_latest_rsd_csv():
    """Pick the latest 01_rsd_software_all_<STAMP>.csv in project root (by timestamp in filename)."""
    candidates = []
    for p in glob.glob(os.path.join(ROOT_DIR, "01_rsd_software_all_*.csv")):
        m = re.search(r"01_rsd_software_all_(\d{8}_\d{6})\.csv$", os.path.basename(p))
        if m:
            candidates.append((m.group(1), p))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    ts, path = candidates[-1]
    return path, ts

def rate_limit_guard(resp):
    """Simple rate-limit safety: if remaining is low, sleep a short period."""
    if resp is None:
        return
    try:
        rem = int(resp.headers.get("X-RateLimit-Remaining", "1000"))
        if rem < 30:
            reset = resp.headers.get("X-RateLimit-Reset")
            wait = 60
            if reset:
                try:
                    reset_ts = int(reset)
                    now_ts = int(time.time())
                    wait = max(5, min(180, reset_ts - now_ts))
                except:
                    pass
            print(f"... rate limit low ({rem}), sleep {wait}s")
            time.sleep(wait)
    except:
        pass

# -------------------------------
# API: repo metadata
# -------------------------------
def api_repo_meta(owner_repo):
    r = SESSION.get(f"https://api.github.com/repos/{owner_repo}")
    if r.status_code in (404, 410, 403):
        return None, {404: "not_found_404", 410: "gone_410", 403: "forbidden_403"}[r.status_code]
    try:
        r.raise_for_status()
        rate_limit_guard(r)
        return r.json(), "ok"
    except Exception as e:
        print(f"!!! meta error {owner_repo}: {e}")
        return None, "error"

# -------------------------------
# API: 52w commit activity (fallback)
# -------------------------------
def api_commit_activity_52w(owner_repo):
    """Fallback: GitHub /stats/commit_activity (52 weeks). Handles 202-pending with retries."""
    url = f"https://api.github.com/repos/{owner_repo}/stats/commit_activity"
    for attempt in range(7):
        try:
            r = SESSION.get(url)
            if r.status_code in (202, 204):
                print(f"... {owner_repo} commit_activity pending ({r.status_code}), retry {attempt+1}/7")
                time.sleep(3)
                continue
            if r.status_code in (404, 410, 403):
                return None, {404: "not_found_404", 410: "gone_410", 403: "forbidden_403"}[r.status_code]
            r.raise_for_status()
            rate_limit_guard(r)
            return r.json() or [], "ok"
        except Exception as e:
            print(f"!!! commit_activity error {owner_repo}: {e}")
            time.sleep(2)
    return None, "pending_timeout"

# -------------------------------
# API: releases (full history)
# -------------------------------
def api_all_releases(owner_repo, writer):
    status = "not_attempted"
    total = 0
    url = f"https://api.github.com/repos/{owner_repo}/releases?per_page=100"
    while url:
        try:
            r = SESSION.get(url)
            if r.status_code in (404, 410, 403):
                status = {404: "not_found_404", 410: "gone_410", 403: "forbidden_403"}[r.status_code]
                break
            r.raise_for_status()
            rows = r.json() or []
            for rel in rows:
                writer.writerow({
                    "repo": owner_repo,
                    "id": rel.get("id"),
                    "tag_name": rel.get("tag_name"),
                    "published_at": rel.get("published_at"),
                    "is_prerelease": rel.get("prerelease"),
                    "draft": rel.get("draft"),
                    "author_login": rel.get("author", {}).get("login") if rel.get("author") else None
                })
                total += 1
            rate_limit_guard(r)
            url = r.links.get("next", {}).get("url")
            status = "ok"
        except Exception as e:
            print(f"!!! releases error {owner_repo}: {e}")
            status = "error"
            break
    return status, total

# -------------------------------
# API: issues (full history) — PRs excluded here
# -------------------------------
def api_all_issues_no_pr(owner_repo, writer):
    status = "not_attempted"
    total = 0
    url = f"https://api.github.com/repos/{owner_repo}/issues?state=all&per_page=100&direction=asc&sort=created"
    page = 1
    while url:
        try:
            r = SESSION.get(url)
            if r.status_code in (404, 410, 403):
                status = {404: "not_found_404", 410: "gone_410", 403: "forbidden_403"}[r.status_code]
                break
            r.raise_for_status()
            arr = r.json() or []
            if not arr:
                break
            for it in arr:
                # Exclude PRs at the 02 stage
                if "pull_request" in it:
                    continue
                writer.writerow({
                    "repo": owner_repo,
                    "number": it.get("number"),
                    "state": it.get("state"),
                    "created_at": it.get("created_at"),
                    "closed_at": it.get("closed_at"),
                    "user_login": it.get("user", {}).get("login") if it.get("user") else None,
                    "comments": it.get("comments")
                })
                total += 1
            rate_limit_guard(r)
            url = r.links.get("next", {}).get("url")
            page += 1
            status = "ok"
        except Exception as e:
            print(f"!!! issues error {owner_repo} page{page}: {e}")
            status = "error"
            break
    return status, total

# -------------------------------
# GIT: weekly commits + weekly unique contributors (default branch; include merges)
# -------------------------------
def git_weekly_commits_and_contributors(owner_repo, default_branch, commits_writer, contr_weekly_writer, contr_raw_writer):
    """
    Clone repo as --bare, fetch a real local branch ref, and run 'git log <branch>'.
    We do NOT use 'origin/<branch>' because bare repo may not create remote-tracking refs automatically.
    We include merge commits to align with GitHub /stats/commit_activity semantics.
    Returns: (ok: bool, earliest_week_unix: int|None, coverage_weeks: 'all'|int, contributors_rows: int)
    """
    url = f"https://github.com/{owner_repo}.git"
    repo_dir = os.path.join(CACHE_DIR, owner_repo.replace("/", "__"))
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)
    try:
        subprocess.run(["git", "clone", "--bare", "--filter=blob:none", "--quiet", url, repo_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! git clone failed {owner_repo}: {e}")
        return False, None, 0, 0

    # Try default branch first; then fallback to main/master; finally probe any head via ls-remote.
    candidates = [default_branch, "main", "master"]
    picked = None

    def try_fetch_branch(branch_name: str) -> bool:
        try:
            # Fetch remote head into a local head ref explicitly.
            subprocess.run([
                "git", "--git-dir", repo_dir, "fetch", "origin",
                f"+refs/heads/{branch_name}:refs/heads/{branch_name}", "--quiet"
            ], check=True)
            # Verify the local ref exists.
            subprocess.check_output(
                ["git", "--git-dir", repo_dir, "rev-parse", "--verify", branch_name],
                text=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    for br in candidates:
        if br and try_fetch_branch(br):
            picked = br
            break

    if not picked:
        # Probe remote heads and pick the first one that fetches successfully.
        try:
            out = subprocess.check_output(["git", "ls-remote", "--heads", url], text=True, encoding="utf-8", errors="ignore")
            for line in out.splitlines():
                if "\trefs/heads/" in line:
                    br = line.split("\trefs/heads/")[-1].strip()
                    if try_fetch_branch(br):
                        picked = br
                        break
        except subprocess.CalledProcessError:
            pass

    if not picked:
        print(f"... no branch fetched for {owner_repo} (empty repo or unusual refs)")
        return False, None, 0, 0

    print(f"... fetched branch: {picked}")

    # Log over the local branch ref (NOT origin/<branch>)
    try:
        log = subprocess.check_output(
            ["git", "--git-dir", repo_dir, "log", picked,
             "--pretty=%H%x09%ct%x09%ae%x09%an", "--date-order"],
            text=True, encoding="utf-8", errors="ignore"
        )
    except subprocess.CalledProcessError as e:
        print(f"!!! git log failed {owner_repo} {picked}: {e}")
        return False, None, 0, 0

    weekly_commits = defaultdict(int)
    weekly_contributors = defaultdict(set)
    earliest_week = None

    for line in log.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        sha, ts, email, author = parts[0], parts[1], (parts[2] or "").lower(), parts[3]
        try:
            dt = datetime.fromtimestamp(int(ts), tz=UTC)
        except:
            continue
        w = week_start_sunday_unix(dt)
        weekly_commits[w] += 1
        if email:
            weekly_contributors[w].add(email)
        contr_raw_writer.writerow({
            "repo": owner_repo,
            "commit_sha": sha,
            "commit_date_utc": dt.isoformat().replace("+00:00", "Z"),
            "author_email": email,
            "author_name": author,
            "default_branch": picked
        })
        if earliest_week is None or w < earliest_week:
            earliest_week = w

    for w in sorted(weekly_commits.keys()):
        commits_writer.writerow({
            "repo": owner_repo,
            "week_unix": w,
            "commits": weekly_commits[w],
            "commit_source": "git",
            "default_branch": picked
        })
    for w in sorted(weekly_contributors.keys()):
        contr_weekly_writer.writerow({
            "repo": owner_repo,
            "week_unix": w,
            "contributors": len(weekly_contributors[w]),
            "contributors_source": "git"
        })

    contributors_rows = sum(len(s) for s in weekly_contributors.values())
    return True, earliest_week, "all", contributors_rows

# -------------------------------
# API fallback: /commits list for 52w (unique authors + optional commit counts)
# -------------------------------
def api_weekly_from_commits(owner_repo, since_epoch_unix, commits_writer, contr_weekly_writer):
    """
    Use /commits?since=... (52 weeks window) to compute weekly commits and unique authors.
    If commits_writer is None, we only write contributors (authors) weekly counts.
    Returns: (status: str, total_processed_rows: int)
    """
    status = "not_attempted"
    total = 0
    since_iso = datetime.fromtimestamp(since_epoch_unix, tz=UTC).isoformat().replace("+00:00", "Z")
    url = f"https://api.github.com/repos/{owner_repo}/commits?per_page=100&since={since_iso}"
    weekly_commits = defaultdict(int)
    weekly_contributors = defaultdict(set)

    while url:
        try:
            r = SESSION.get(url)
            if r.status_code in (404, 410, 403):
                status = {404: "not_found_404", 410: "gone_410", 403: "forbidden_403"}[r.status_code]
                break
            r.raise_for_status()
            arr = r.json() or []
            if not arr:
                break
            for c in arr:
                dt_str = c.get("commit", {}).get("author", {}).get("date") or c.get("commit", {}).get("committer", {}).get("date")
                if not dt_str:
                    continue
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(UTC)
                except:
                    continue
                w = week_start_sunday_unix(dt)
                weekly_commits[w] += 1
                login = (c.get("author", {}) or {}).get("login") or (c.get("committer", {}) or {}).get("login") or ""
                email = (c.get("commit", {}).get("author", {}) or {}).get("email") or (c.get("commit", {}).get("committer", {}) or {}).get("email") or ""
                key = (login or email or "").lower()
                if key:
                    weekly_contributors[w].add(key)
                total += 1
            rate_limit_guard(r)
            url = r.links.get("next", {}).get("url")
            status = "ok"
        except Exception as e:
            print(f"!!! commits api list error {owner_repo}: {e}")
            status = "error"
            break

    for w in sorted(weekly_commits.keys()):
        if commits_writer is not None:  # allow "authors only"
            commits_writer.writerow({
                "repo": owner_repo,
                "week_unix": w,
                "commits": weekly_commits[w],
                "commit_source": "api_commits_list",
                "default_branch": ""  # unknown
            })
    for w in sorted(weekly_contributors.keys()):
        contr_weekly_writer.writerow({
            "repo": owner_repo,
            "week_unix": w,
            "contributors": len(weekly_contributors[w]),
            "contributors_source": "api_commits_list"
        })
    return status, total

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsd_csv", required=False, help="Optional; if omitted, auto-pick the latest 01_rsd_software_all_*.csv in project root.")
    ap.add_argument("--clean_cache", action="store_true", help="Clean git bare cache after finishing.")
    args = ap.parse_args()

    rsd_csv = args.rsd_csv
    STAMP_RSD = None
    if not rsd_csv:
        rsd_csv, STAMP_RSD = find_latest_rsd_csv()
        if not rsd_csv:
            print("Not found: 01_rsd_software_all_*.csv in project root. Place the file or use --rsd_csv.")
            sys.exit(1)
        print(f"[auto] Using latest RSD list: {os.path.basename(rsd_csv)}")
    else:
        if not os.path.exists(rsd_csv):
            print(f"Input not found: {rsd_csv}")
            sys.exit(1)
        STAMP_RSD = parse_stamp_rsd_from_path(rsd_csv)

    # Read repo list (column: github_owner_repo or repo; multiple repos separated by ';')
    repos = []
    with open(rsd_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("github_owner_repo") or row.get("repo") or ""
            if not v:
                continue
            for r in str(v).split(";"):
                r = r.strip()
                if r:
                    repos.append(r)
    repos = sorted(list(set(repos)))
    print(f"Found {len(repos)} repos.")

    # Writers
    w_meta = csv.DictWriter(open(FN_META, "w", encoding="utf-8", newline=""),
                            fieldnames=["repo", "id", "fork", "created_at", "updated_at", "pushed_at", "size",
                                        "stars", "watchers", "forks", "open_issues", "license", "default_branch", "topics"])
    w_meta.writeheader()

    w_commits = csv.DictWriter(open(FN_COMMITS, "w", encoding="utf-8", newline=""),
                               fieldnames=["repo", "week_unix", "commits", "commit_source", "default_branch"])
    w_commits.writeheader()

    w_contr_w = csv.DictWriter(open(FN_CONTR_W, "w", encoding="utf-8", newline=""),
                               fieldnames=["repo", "week_unix", "contributors", "contributors_source"])
    w_contr_w.writeheader()

    w_contr_raw = csv.DictWriter(open(FN_CONTR_RAW, "w", encoding="utf-8", newline=""),
                                 fieldnames=["repo", "commit_sha", "commit_date_utc", "author_email", "author_name", "default_branch"])
    w_contr_raw.writeheader()

    w_issues = csv.DictWriter(open(FN_ISSUES, "w", encoding="utf-8", newline=""),
                              fieldnames=["repo", "number", "state", "created_at", "closed_at", "user_login", "comments"])
    w_issues.writeheader()

    w_releases = csv.DictWriter(open(FN_RELEASES, "w", encoding="utf-8", newline=""),
                                fieldnames=["repo", "id", "tag_name", "published_at", "is_prerelease", "draft", "author_login"])
    w_releases.writeheader()

    w_status = csv.DictWriter(open(FN_API_STATUS, "w", encoding="utf-8", newline=""),
                              fieldnames=["repo", "commits_source", "commits_earliest_week_unix", "commits_coverage_weeks",
                                          "contributors_status", "contributors_total_rows",
                                          "issues_api_status", "issues_total_rows",
                                          "releases_api_status", "releases_total_rows"])
    w_status.writeheader()

    # Iterate repos
    for idx, owner_repo in enumerate(repos, 1):
        print(f"\n--- [{idx}/{len(repos)}] {owner_repo} ---")

        # Repo meta
        meta, meta_status = api_repo_meta(owner_repo)
        if not meta:
            print(f"... skip repo (meta not available): {owner_repo}")
            w_status.writerow({
                "repo": owner_repo,
                "commits_source": "none", "commits_earliest_week_unix": "", "commits_coverage_weeks": 0,
                "contributors_status": "not_available", "contributors_total_rows": 0,
                "issues_api_status": "not_available", "issues_total_rows": 0,
                "releases_api_status": "not_available", "releases_total_rows": 0
            })
            continue

        default_branch = meta.get("default_branch") or "main"
        w_meta.writerow({
            "repo": owner_repo,
            "id": meta.get("id"),
            "fork": meta.get("fork"),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "pushed_at": meta.get("pushed_at"),
            "size": meta.get("size"),
            "stars": meta.get("stargazers_count"),
            "watchers": meta.get("watchers_count"),
            "forks": meta.get("forks_count"),
            "open_issues": meta.get("open_issues_count"),
            "license": (meta.get("license") or {}).get("name") if meta.get("license") else None,
            "default_branch": default_branch,
            "topics": ";".join(meta.get("topics") or [])
        })

        # Commits + Contributors via Git (full history). If Git fails, fallback to API.
        commits_source = "none"
        commits_earliest = ""
        commits_coverage = 0
        contributors_status = "not_attempted"
        contributors_total = 0

        ok_git = False
        try:
            ok_git, earliest_w, coverage, contr_rows = git_weekly_commits_and_contributors(
                owner_repo, default_branch, w_commits, w_contr_w, w_contr_raw
            )
            if ok_git:
                commits_source = "git"
                commits_earliest = earliest_w
                commits_coverage = coverage
                contributors_status = "ok_git"
                contributors_total = contr_rows
        except Exception as e:
            print(f"!!! git pipeline error {owner_repo}: {e}")

        if not ok_git:
            # Fallback path: 52w stats for commits, and /commits list for authors
            stats, st_status = api_commit_activity_52w(owner_repo)
            if st_status == "ok" and stats:
                earliest = None
                for w in stats:
                    week_unix = int(w.get("week") or 0)
                    total = int(w.get("total") or 0)
                    if week_unix:
                        w_commits.writerow({
                            "repo": owner_repo,
                            "week_unix": week_unix,
                            "commits": total,
                            "commit_source": "api_52w",
                            "default_branch": default_branch or ""
                        })
                        if earliest is None or week_unix < earliest:
                            earliest = week_unix
                commits_source = "api_52w"
                commits_earliest = earliest
                commits_coverage = 52

                st2, tot = api_weekly_from_commits(
                    owner_repo,
                    earliest or int((datetime.utcnow() - timedelta(weeks=52)).timestamp()),
                    commits_writer=None,                 # authors only here
                    contr_weekly_writer=w_contr_w
                )
                contributors_status = st2
                contributors_total = tot
            else:
                commits_source = st_status
                contributors_status = "not_available"

        # Releases & Issues (PRs already excluded)
        rel_status, rel_total = api_all_releases(owner_repo, w_releases)
        iss_status, iss_total = api_all_issues_no_pr(owner_repo, w_issues)

        # Status row
        w_status.writerow({
            "repo": owner_repo,
            "commits_source": commits_source,
            "commits_earliest_week_unix": commits_earliest,
            "commits_coverage_weeks": commits_coverage,
            "contributors_status": contributors_status,
            "contributors_total_rows": contributors_total,
            "issues_api_status": iss_status,
            "issues_total_rows": iss_total,
            "releases_api_status": rel_status,
            "releases_total_rows": rel_total
        })

    # Batch meta
    with open(FN_PULL_META, "w", encoding="utf-8") as f:
        json.dump({
            "stamp_pull": STAMP_PULL,
            "stamp_rsd": parse_stamp_rsd_from_path(rsd_csv),
            "out_dir": os.path.abspath(OUT_DIR),
            "git_cache": os.path.abspath(CACHE_DIR),
            "note": "v2 02_dat outputs; UTC times; issues exclude PRs; week_unix = Sunday 00:00:00 UTC"
        }, f, ensure_ascii=False, indent=2)

    print("\nAll done.")
    print(f"Outputs @ {OUT_DIR}")
    print(f"- meta        : {os.path.basename(FN_META)}")
    print(f"- commits     : {os.path.basename(FN_COMMITS)}")
    print(f"- contributors: {os.path.basename(FN_CONTR_W)} (raw: {os.path.basename(FN_CONTR_RAW)})")
    print(f"- issues      : {os.path.basename(FN_ISSUES)}")
    print(f"- releases    : {os.path.basename(FN_RELEASES)}")
    print(f"- api_status  : {os.path.basename(FN_API_STATUS)}")
    print(f"- pull_meta   : {os.path.basename(FN_PULL_META)}")

    if args.clean_cache:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
