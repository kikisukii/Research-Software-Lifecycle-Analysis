# Learning to Identify Life Cycle Phases in Research Software Projects


---

##  Abstract

This repository contains the source code, data processing pipelines, and analysis artifacts for the Master Thesis: *"Learning to Identify Life Cycle Phases in Research Software Projects"*.

Research software often lacks formal development oversight. This study proposes an automated, unsupervised framework to identify life cycle phases (e.g., *Peak Activity, Internal Development, Maintenance, Dormant*) using public GitHub activity data.

The project implements a progressive experimental design:
* **Phase I (v1 - Baseline):** A single-signal approach using **Commit Frequency** and **Momentum** ($M_{8/24}$).
* **Phase II (v2 - Advanced):** A multi-signal approach (Commits, Contributors, Issues, Releases) utilizing **PCA** and **K-Means Clustering** ($K=6$) to capture complex evolution patterns.

##  Repository Structure

The project follows a modular structure separating the baseline (v1), the advanced model (v2), and the web interface.

```text
project_root/
├── .env                        # [Required] API Tokens (Git/RSD)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── v1/                         # [Phase I] Baseline Model Scripts
│   ├── 01_rsd_pull_all.py          # [Common] Fetches software list from RSD
│   ├── 02_github_pull_all_v1.py    # Fetches weekly Commit data
│   ├── 03_build_features_v1.py     # Constructs Momentum features
│   ├── 04_k_sweep_v1.py            # Grid search for K and Alpha
│   ├── 05_a_fit_kmeans_v1.py       # Fits model and saves assignments
│   ├── 05_b_label_only_v1.py       # Applies semantic labels to phases
│   └── 05_c_plot_v1_8w.py          # Generates visualization (v1 style)
│
├── v2/                         # [Phase II] Advanced Model Scripts (Main Contribution)
│   ├── 02_github_pull_all_v2.py    # Fetches Multi-signal data (Git + API)
│   ├── 03_build_features_v2.py     # Constructs 8-week rolling features (log1p)
│   ├── 04_select_k_v2.py           # PCA Analysis & K-Selection
│   ├── 05a_cluster_and_profiles... # Fits model (K=6), extracts profiles
│   ├── 05b_name_and_label_v2.py    # Maps clusters to Semantic Names
│   ├── 05c_plot_v2_8w.py           # Generates visualization (v2 style)
│   ├── 07_export_bundle_v2.py      # Exports model pickle for Web App
│   ├── 08_survey_candidates.py     # Stratified sampling for evaluation
│   └── 09_plot_from_selection.py   # Survey figure generation
│
├── streamlit_app/              # Interactive Web Tool
│   ├── p01_inference.py        # Inference engine (Git clone -> Prediction)
│   └── p02_app.py              # Streamlit frontend
│
├── v1_data/                    # Data artifacts for Phase I
└── v2_data/                    # Data artifacts for Phase II
    ├── 02_dat/                 # Raw weekly data
    ├── 03_features/            # Engineered features
    ├── 05_a_kmeans/            # Clustering results
    └── 07_bundle/              # Contains 'model_bundle_v2.pkl'
```

##  Installation & Prerequisites

The project requires **Python 3.10+** and **Git** installed on the system path.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/Research-Software-Lifecycle-Analysis.git
    cd Research-Software-Lifecycle-Analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration (.env):**
    Create a `.env` file in the project root to handle API limits.
    ```env
    # Required for Step 01
    RSD_TOKEN="your_rsd_token"
    # Required for Step 02 & Web App (to avoid rate limits)
    GITHUB_TOKEN="your_github_token"
    ```

---

##  Reproduction Pipeline

To reproduce the results presented in the thesis, follow the steps below sequentially.

### Step 0: Data Collection (Common)
Retrieve the list of research software from the Research Software Directory (RSD).
```bash
python v1/01_rsd_pull_all.py
```
*Output:* `01_rsd_software_all_<timestamp>.csv`

### Step 1: Phase I (Baseline Model)
The baseline model relies solely on commit activity and momentum.

```bash
# 1. Fetch Data (Commits only)
python v1/02_github_pull_all_v1.py

# 2. Feature Engineering (Momentum M8_24)
python v1/03_build_features_v1.py

# 3. Model Selection (Grid Search K & Alpha)
python v1/04_k_sweep_v1.py

# 4. Training & Inference (K=4, Alpha=1.9 as per thesis)
python v1/05_a_fit_kmeans_v1.py --k 4 --alpha 1.9
python v1/05_b_label_only_v1.py

# 5. Visualization
python v1/05_c_plot_v1_8w.py
```

### Step 2: Phase II (Advanced Model)
The advanced model uses 4 signals (Commits, Contributors, Issues, Releases) and PCA.

```bash
# 1. Fetch Data (Multi-signal: Git Clone + API Fallback)
python v2/02_github_pull_all_v2.py

# 2. Feature Engineering (8-week rolling, log1p, complete-case)
python v2/03_build_features_v2.py

# 3. Model Selection (PCA Analysis & K-Selection)
python v2/04_select_k_v2.py

# 4. Clustering & Mapping (Fit K=6)
python v2/05a_cluster_and_profiles_v2.py
python v2/05b_name_and_label_v2.py

# 5. Visualization (Mid-week alignment)
python v2/05c_plot_v2_8w.py
```

### Step 3: Evaluation & Web Tool
Tools used for the human evaluation survey and the interactive demo.

**Evaluation Sampling:**
```bash
python v2/08_survey_candidates.py  # Stratified random sampling
python v2/09_plot_from_selection.py # Generate survey figures
```

**Run Web Tool Locally:**
Ensure the model bundle is exported, then launch Streamlit.
```bash
# Export trained model to .pkl
python v2/07_export_bundle_v2.py

# Launch App
streamlit run streamlit_app/p02_app.py
```

##  Methodology Highlights

### 1. Temporal Alignment
All data is aligned to a weekly grid starting on **Sunday 00:00 UTC**. An **8-week rolling window** is applied to all signals to smooth intermittency while retaining state inertia

### 2. The "Dead" Rule
A heuristic rule is applied to identify project end-of-life:
> A period is marked as **Dead** if both commits and releases are zero for **24 consecutive weeks**.

### 3. Phase Definitions (v2)
Unsupervised clustering ($K=6$) identified the following semantic phases:

| Phase | Description | Key Signals |
| :--- | :--- | :--- |
| **Peak Activity** | High intensity coding & delivery | High Commits, Issues, Releases |
| **Internal Dev.** | Intense coding, zero delivery | High Commits, **Zero Releases** |
| **Release Phase** | Coding slows; frequent releases | Low Commits, **High Releases** |
| **Maintenance** | Bug-fixing; responding to users | **High Issues**, Low Commits |
| **Baseline** | Low but steady activity | Low (non-zero) on all signals |
| **Dormant** | Barely moving | Near zero activity |

##  License

This project is open-source. Please refer to the `LICENSE` file for details.



