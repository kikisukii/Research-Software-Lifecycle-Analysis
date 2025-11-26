import os, csv, re, sys, time
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

# --- Load .env from the project root directory ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# 改成直接在当前目录找 .env，去掉那个 '..'
dotenv_path = os.path.join(script_dir, '.env')

load_dotenv(dotenv_path=dotenv_path)
TOKEN = os.getenv("RSD_TOKEN")

if not TOKEN:
    print(f"Error: RSD_TOKEN not found in {dotenv_path}")
    print("Please create .env in the project root and add: RSD_TOKEN=\"your_token\"");
    sys.exit(1)
# --- End of .env loading ---

BASE = "https://research-software-directory.org/api/v2"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "Prefer": "count=exact"  # Ask the server to return the total count in Content-Range
}


# --- Helper functions ---
def extract_repo_urls(embedded):
    """Extract all URLs from the embedded repository_url(url) structure"""
    urls = []
    block = embedded.get("repository_url")
    if isinstance(block, dict):
        block = [block]
    if isinstance(block, list):
        for r in block:
            if isinstance(r, dict):
                u = r.get("url")
                if isinstance(u, str):
                    urls.append(u)
    # Deduplicate
    seen = set();
    out = []
    for u in urls:
        if u not in seen:
            out.append(u);
            seen.add(u)
    return out


def to_owner_repo(url: str):
    """Parse https://github.com/owner/repo[.git] into owner/repo"""
    try:
        u = urlparse(url)
        if u.netloc != "github.com":
            return None
        m = re.match(r"^/([^/]+)/([^/]+?)(?:\.git)?/?$", u.path)
        return f"{m.group(1)}/{m.group(2)}" if m else None
    except Exception:
        return None


def fetch_page(after_id=None, limit=500):
    """
    Use "keyset pagination": fetch one page ordered by id ascending,
    the next page uses id > last_id to continue, avoiding slowness when offset is large.
    Fetch repository_url(url) as an embedded field in one go to reduce round trips.
    """
    params = {
        "select": "id,slug,brand_name,description,repository_url(url)",
        "order": "id.asc",
        "limit": str(limit),
    }
    if after_id is not None:
        params["id"] = f"gt.{after_id}"
    r = requests.get(f"{BASE}/software", headers=HEADERS, params=params, timeout=60)
    r.raise_for_status()
    total = None
    cr = r.headers.get("Content-Range")
    # Can get total count if format is like "0-499/12345"
    if cr and "/" in cr and not cr.endswith("/*"):
        try:
            total = int(cr.split("/")[-1])
        except Exception:
            total = None
    return r.json(), total


def main():
    # --- Define output directory and file ---
    # 1. Define the data directory (relative to the project root, one level up)
    DATA_DIR = script_dir

    # 2. Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # 3. Create the filename with prefix and timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"01_rsd_software_all_{timestamp}.csv"

    # 4. Combine the directory and filename
    out_path = os.path.join(DATA_DIR, filename)
    # ---

    fields = ["id", "slug", "brand_name", "description", "repo_urls", "github_owner_repo"]

    count = 0
    total = None
    last_id = None
    page_size = 500

    print(f"Starting fetch. Data will be saved to: {out_path}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        while True:
            rows, t = fetch_page(after_id=last_id, limit=page_size)
            if total is None and t is not None:
                total = t
            if not rows:
                break

            for row in rows:
                sid = row.get("id")
                slug = row.get("slug")
                name = row.get("brand_name") or slug or ""
                desc = (row.get("description") or "").replace("\n", " ").strip()

                urls = extract_repo_urls(row)
                owners = [x for x in (to_owner_repo(u) for u in urls) if x]

                w.writerow({
                    "id": sid,
                    "slug": slug,
                    "brand_name": name,
                    "description": desc,
                    "repo_urls": ";".join(urls),
                    "github_owner_repo": ";".join(owners)
                })

                last_id = sid
                count += 1

            # Progress indicator
            if total:
                print(f"Fetched {count}/{total} ...")
            else:
                print(f"Fetched {count} ...")

            # Be polite, avoid hitting the server too hard continuously
            time.sleep(0.2)

    print(f"Done! Data saved to -> {out_path}")


if __name__ == "__main__":
    main()