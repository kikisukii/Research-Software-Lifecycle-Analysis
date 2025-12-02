# Thesis Research Software Lifecycle Identification Pipeline (v1/v2)

This repository contains the data processing pipeline for identifying research software life cycle phases using GitHub signals. The pipeline creates weekly aggregated features and applies unsupervised clustering (K-Means).

## 📋 Prerequisites

Before running the scripts, ensure you have the following installed:

### 1. Python Environment
* **Python 3.10+** is required.
* Install dependencies via `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Git (Crucial)
* **Git** must be installed and accessible via the system command line (PATH).
* The script uses `git clone` to retrieve full commit histories which are more accurate than the API's limited window.
* To verify, run: `git --version` in your terminal.

### 3. GitHub Token
* A valid GitHub Personal Access Token (Classic) is required to fetch metadata, issues, and releases.
* Create a file named `.env` in the **project root directory**.
* Add your token inside:
    ```env
    GITHUB_TOKEN="ghp_YourActualTokenHere"
    ```

---

## 📂 Directory Structure

Ensure your files are organized as follows before running:

```text
project_root/
├── .env                              # [Create This] Your API Token
├── requirements.txt                  # [Create This] Python dependencies
├── 01_rsd_software_all_....csv       # [Input] Source file from RSD
├── v1/                               # [Code] Scripts folder
│   ├── 02_github_pull_all_v1.py
│   ├── 03_build_features_v1.py
│   └── ...
└── v1_data/                          # [Output] Auto-generated data folder
    ├── 02_dat/                       # Output CSVs will appear here
    └── _gitcache/                    # Temporary git clones

