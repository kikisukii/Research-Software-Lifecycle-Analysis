import os, csv, time, requests, sys
from dotenv import load_dotenv

# --- Define Data Directory ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- !!! IMPORTANT: MANUAL STEP !!! ---
# You must update this filename to match the actual output file from script 01.
# Go to your 'data' folder and copy the full filename of the RSD list.
# Example: "01_rsd_software_all_20251028_235500.csv"
# ---------------------------------------------------------------------
INPUT_FILENAME = "01_rsd_software_all_20251028_235202.csv"  # <-- UPDATE THIS
# ---------------------------------------------------------------------

INPUT = os.path.join(DATA_DIR, INPUT_FILENAME)

# --- Output Files ---
# A single timestamp will be used for all output files from this run
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUT_META = os.path.join(DATA_DIR, f"02_gh_repo_meta_{timestamp}.csv")
OUT_COMMITS = os.path.join(DATA_DIR, f"02_gh_commit_weekly_{timestamp}.csv")
OUT_RELEASES = os.path.join(DATA_DIR, f"02_gh_releases_{timestamp}.csv")
OUT_ISSUES = os.path.join(DATA_DIR, f"02_gh_issues_{timestamp}.csv")

# Load .env file from the project root (since this script is in the root)
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    print("Error: GITHUB_TOKEN not found in .env file.")
    print("Please add GITHUB_TOKEN=\"your_token\" to your .env file.");
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28"
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# --- API Helper Functions ---

def get_repo_meta(owner_repo, writer):
    # 1. Metadata
    try:
        r = SESSION.get(f"https://api.github.com/repos/{owner_repo}")
        r.raise_for_status()
        j = r.json()
        writer.writerow({
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
            "topics": ";".join(j.get("topics", []))
        })
        print(f"... repo {owner_repo} meta done")
        return r.headers.get("X-RateLimit-Remaining")
    except Exception as e:
        print(f"!!! Error getting meta for {owner_repo}: {e}")
        return None


def get_commit_activity(owner_repo, writer):
    # 2. Commit Activity (Weekly)
    # This endpoint can return 202 if data is being computed
    for _ in range(5):  # Try 5 times
        try:
            r = SESSION.get(f"https://api.github.com/repos/{owner_repo}/stats/commit_activity")
            if r.status_code == 202:  # 202: Still computing...
                print(f"... repo {owner_repo} commit stats 202 (retrying)...")
                time.sleep(3)
                continue
            r.raise_for_status()
            for week_data in r.json():
                writer.writerow({
                    "repo": owner_repo,
                    "week_unix": week_data.get("week"),
                    "commits": week_data.get("total")
                })
            print(f"... repo {owner_repo} commit_activity done")
            return
        except Exception as e:
            print(f"!!! Error getting commit_activity for {owner_repo}: {e}")
            time.sleep(1)  # Wait before retrying on error
    print(f"!!! Failed to get commit_activity for {owner_repo} after retries.")


def get_all_releases(owner_repo, writer):
    # 3. Releases
    url = f"https://api.github.com/repos/{owner_repo}/releases?per_page=100"
    while url:
        try:
            r = SESSION.get(url)
            r.raise_for_status()
            for rel in r.json():
                writer.writerow({
                    "repo": owner_repo,
                    "tag": rel.get("tag_name"),
                    "name": rel.get("name"),
                    "created_at": rel.get("created_at"),
                    "published_at": rel.get("published_at"),
                    "is_prerelease": rel.get("prerelease")
                })
            # (Recursively fetch all pages)
            url = r.links.get("next", {}).get("url")
        except Exception as e:
            print(f"!!! Error getting releases for {owner_repo}: {e}")
            url = None  # Stop on error
    print(f"... repo {owner_repo} releases done")


def get_all_issues(owner_repo, writer):
    # 4. Issues & PRs (combined)
    url = f"https://api.github.com/repos/{owner_repo}/issues?state=all&per_page=100&direction=asc"
    page = 1
    while url:
        try:
            r = SESSION.get(url)
            r.raise_for_status()
            data = r.json()
            if not data:
                break  # No more data

            for item in data:
                writer.writerow({
                    "repo": owner_repo,
                    "number": item.get("number"),
                    "is_pr": "pull_request" in item,
                    "state": item.get("state"),
                    "created_at": item.get("created_at"),
                    "closed_at": item.get("closed_at"),
                    "user_login": item.get("user", {}).get("login") if item.get("user") else None,
                    "comments": item.get("comments")
                })

            print(f"... repo {owner_repo} issues page {page} done")
            page += 1
            url = r.links.get("next", {}).get("url")
        except Exception as e:
            print(f"!!! Error getting issues for {owner_repo} on page {page}: {e}")
            url = None  # Stop on error
    print(f"... repo {owner_repo} issues done")


# --- Main Execution ---

def main():
    # Read list of repos to process
    repos = []
    # Check if input file exists
    if not os.path.exists(INPUT):
        print(f"Error: Input file not found: {INPUT}")
        print(f"Please check the INPUT_FILENAME variable at the top of this script.")
        return

    with open(INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("github_owner_repo"):
                repos.extend(row.get("github_owner_repo").split(';'))

    # Ensure unique list
    repos = sorted(list(set(r for r in repos if r)))

    print(f"Found {len(repos)} unique repositories to process.")
    print(f"Output files will be prefixed with timestamp: {timestamp}")
    print(f"Meta:    {OUT_META}")
    print(f"Commits: {OUT_COMMITS}")
    print(f"Releases: {OUT_RELEASES}")
    print(f"Issues:  {OUT_ISSUES}")

    # Write to CSV files
    with open(OUT_META, 'w', newline='', encoding='utf-8') as f_meta, \
            open(OUT_COMMITS, 'w', newline='', encoding='utf-8') as f_commits, \
            open(OUT_RELEASES, 'w', newline='', encoding='utf-8') as f_releases, \
            open(OUT_ISSUES, 'w', newline='', encoding='utf-8') as f_issues:

        # Setup writers
        w_meta = csv.DictWriter(f_meta,
                                fieldnames=["repo", "id", "is_fork", "created_at", "updated_at", "pushed_at", "size",
                                            "stars", "watchers", "forks", "open_issues", "license", "default_branch",
                                            "topics"])
        w_commits = csv.DictWriter(f_commits, fieldnames=["repo", "week_unix", "commits"])
        w_releases = csv.DictWriter(f_releases,
                                    fieldnames=["repo", "tag", "name", "created_at", "published_at", "is_prerelease"])
        w_issues = csv.DictWriter(f_issues, fieldnames=["repo", "number", "is_pr", "state", "created_at", "closed_at",
                                                        "user_login", "comments"])

        w_meta.writeheader()
        w_commits.writeheader()
        w_releases.writeheader()
        w_issues.writeheader()

        print(f"Starting processing for {len(repos)} repositories...")

        for i, owner_repo in enumerate(repos):
            print(f"\n--- Processing {i + 1}/{len(repos)}: {owner_repo} ---")

            # 1. Get Metadata (also checks rate limit)
            rem_s = get_repo_meta(owner_repo, w_meta)

            # (Rate limit check)
            if rem_s:
                try:
                    rem = int(rem_s)
                    if rem < 50:  # Be safe
                        print(f"Rate limit remaining {rem}, sleeping for 60s...")
                        time.sleep(60)
                except ValueError:
                    pass  # Couldn't parse header, just continue

            # 2. Get Commit Activity
            get_commit_activity(owner_repo, w_commits)

            # 3. Get Releases
            get_all_releases(owner_repo, w_releases)

            # 4. Get Issues & PRs
            get_all_issues(owner_repo, w_issues)

            print(f"--- Finished processing: {owner_repo} ---")

    print("\nAll processing complete!")


if __name__ == "__main__":
    main()