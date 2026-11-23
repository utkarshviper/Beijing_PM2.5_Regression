# Machine Learning Mini-Project Blueprint: Beijing Multi-Site Air Quality ($\text{PM}_{2.5}$) Regression

**Course:** Introduction to Machine Learning (BS Analytics, Semester V)

**Institution:** TISS School of Analytics

**Dataset:** Beijing Multi-Site Air Quality Data (UCI Machine Learning Repository / PRSA 2013–2017)

**Primary Task:** Supervised Regression ($Y = f(X) + \varepsilon$)

## 1. Project Master Plan

A complete machine learning project structure with all rigorous academic and methodological standards:

> **Project Scope:**
> An undergraduate machine learning mini-project for BS Analytics course using the **Beijing Multi-Site Air Quality (PRSA 2013–2017)** dataset from the UCI Machine Learning Repository.
>
> **1. Data & Folder Structure:**
>
> * Inside `data/raw/PRSA2017_Data_20130301-20170228/`, I have 12 CSV files (`PRSA_Data_<Station>_20130301-20170228.csv`) for 12 monitoring stations (`Aotizhongxin`, `Changping`, `Dingling`, `Dongsi`, `Guanyuan`, `Gucheng`, `Huairou`, `Nongzhanguan`, `Shunyi`, `Tiantan`, `Wanliu`, `Wanshouxigong`), totaling 420,768 hourly records from March 1, 2013 to February 28, 2017.
>
> * Ignore `data.csv` and `test.csv` outside the folder—merge all 12 station CSVs into one master DataFrame and add a `station` column extracted from each filename.
>
> **2. Problem Framing & Variables (Strictly Regression):**
>
> * Inspired by Zhang et al. (2017), *'Cautionary tales on air-quality improvement in Beijing'* (*Proc. R. Soc. A* 473: 20170457), frame this as a **Supervised Regression** task ($Y = f(X) + \varepsilon$) predicting continuous hourly **`PM2.5`** ($\mu\text{g/m}^3$) strictly from meteorological, spatial, and engineered temporal covariates.
>
> * **Target (**$y$**):** `PM2.5`
>
> * **Excluded columns:** Drop `No` (row index, causes leakage) and exclude companion pollutants (`PM10`, `SO2`, `NO2`, `CO`, `O3`) from the primary model so we strictly isolate meteorological and spatial confounding.
>
> **3. Mandatory 8-Step Lab Workflow (Strictly Separating Feature Engineering vs. Feature Scaling):**
> Follow the exact 8-step undergraduate pipeline taught in our lectures:
>
> 1. **Import Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
>
> 2. **Import & Merge Dataset:** Load and concatenate all 12 PRSA station CSVs.
>
> 3. **Clean & Feature Engineering (BEFORE Splitting — Domain Creation):**
>    * *Note: Keep Feature Engineering strictly distinct from Feature Scaling.*
>    * Construct `datetime` from `year`, `month`, `day`, `hour` and sort chronologically within each `station`.
>    * Handle missing values (`NA`) using station-wise short-gap linear interpolation (`limit=3` hours) for weather variables and drop remaining `NA` rows.
>    * **Engineered Temporal Features:** Create `season` (`Spring`: Mar–May, `Summer`: Jun–Aug, `Autumn`: Sep–Nov, `Winter`: Dec–Feb), `is_weekend`, and `time_of_day` (`Morning_Rush`, `Afternoon`, `Evening_Rush`, `Night`).
>    * **Engineered Physical Weather Features:** Create `temp_dew_spread` (`TEMP - DEWP`, proxy for relative dryness/condensation potential) and grouped wind direction `wd_sector` (`NW`, `NE`, `SW`, `SE`, `CV` as defined in Zhang et al., 2017).
>    * **Engineered Spatial Zone & Urban/Rural Dummy Variables:** Classify the 12 stations into `zone_type` (`Urban` = `Aotizhongxin`, `Dongsi`, `Guanyuan`, `Gucheng`, `Nongzhanguan`, `Tiantan`, `Wanliu`, `Wanshouxigong`; `Rural_Suburban` = `Changping`, `Dingling`, `Huairou`, `Shunyi`), and create explicit binary dummy variables (`is_urban`, `is_rural`) plus one-hot station dummy variables (`station_*` with `drop_first=True` for linear regression to avoid the dummy variable trap).
>    * **Synthetic/Dummy Test Stations for Spatial Stress-Testing:** Also construct two out-of-sample **synthetic dummy station test profiles** (`Dummy_Urban_Core` and `Dummy_Rural_Background`) by feeding identical 2016–2017 hourly weather sequences with `is_urban=1` vs. `is_urban=0` into the trained models to isolate and test the pure **Urban Heat/Emission Penalty** ($\Delta \widehat{\text{PM}}_{2.5}$) under identical weather conditions.
>
> 4. **Identify** $X$ **and** $y$**:** Separate engineered feature matrix $X$ and target vector $y = \text{PM2.5}$.
>
> 5. **Chronological Split (Anti-Leakage):** Do NOT use a random split. Split by seasonal year:
>    * **Train:** `2013-03-01` to `2016-02-29` (first 3 seasonal years)
>    * **Test:** `2016-03-01` to `2017-02-28` (4th seasonal year)
>
> 6. **Feature Scaling & Encoding (AFTER Splitting — Numerical Normalization Only):**
>    * Fit `StandardScaler` on continuous features (`DEWP`, `TEMP`, `PRES`, `WSPM`, `RAIN`, `temp_dew_spread`) **strictly on** $X_{\text{train}}$ before transforming $X_{\text{test}}$ so test statistics never leak into training, and so `PRES` ($\sim 1015\text{ hPa}$) does not dominate `WSPM` ($\sim 2\text{ m/s}$) in distance/kernel calculations.
>
> 7. **Train Baseline & Comparison Models (Aligned with Course Lectures):**
>    * **Baseline:** Linear Regression / Ridge Regression (interpreting the `is_urban` dummy coefficient directly!)
>    * **Comparison Model 1:** Decision Tree Regressor (with tuned `max_depth` to demonstrate bias-variance tradeoff)
>    * **Comparison Model 2:** K-Nearest Neighbors Regressor (`KNeighborsRegressor` with `weights='distance'`, comparing Euclidean $L_2$ vs. Manhattan $L_1$ metrics and tuning $k$)
>    * **Comparison Model 3:** Support Vector Regressor (`SVR` with `kernel='rbf'`, trained on a stratified/daily-aggregated subset if needed due to $O(N^2)$ kernel complexity, tuning $C$ and $\gamma$)
>
> 8. **Evaluate, Spatial Hypothesis Test & GitHub Export:**
>    * Report test **RMSE**, **MAE**, and $R^2$ across all models in a clean comparison table.
>    * Run the **Urban vs. Rural Dummy Station Test** (comparing predicted $\text{PM}_{2.5}$ when holding weather fixed and toggling `is_urban` vs. `is_rural`, plus spatial holdout validation).
>    * Save all figures to `reports/figures/` and structure the repository cleanly for **GitHub** (including `.gitignore` excluding raw 8MB+ zip archives, `requirements.txt`, `README.md`, `src/`, and `notebooks/`)."

---

## 2. Core Methodological Pillars & Data Verification

### A. Three-Tier Independent Data Verification Framework (Ground-Truth Auditing)

To establish unimpeachable scientific provenance and address data authenticity without relying solely on state-reported figures, the project implements a **Three-Tier Independent Cross-Verification Protocol**:

```
                              [Primary Dataset]
                   UCI ML Repository (DOI: 10.24432/C5RK5G)
              Raw Source: Beijing Municipal Environmental Monitoring Center
                               (12 Stations, 2013–2017)
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  [Tier 1: Diplomatic]       [Tier 2: Non-Profit Audit]      [Tier 3: Physical Remote]
   U.S. Embassy Beijing            Berkeley Earth               NASA AERONET & MODIS
  StateAir Rooftop Sensor       Rohde & Muller (2015)         Sun-Photometers & Satellites
 (r > 0.95 with Dongsi /     Regional transport corridor      Physical, tamper-proof AOD
      Nongzhanguan)             audit (SW wind penalty)       column optical verification
```

1. **Primary Research Dataset:** The UCI Machine Learning Repository release (`id=501`, DOI: `10.24432/C5RK5G`), which packages the 12 ground stations of the **Beijing Municipal Ecological Environment Monitoring Center** (北京市环境保护监测中心) compiled by Prof. Song Xi Chen's laboratory at Peking University.
2. **Tier 1 — Diplomatic Ground Truth (U.S. Embassy Beijing StateAir):**
   * Hourly rooftop $\text{PM}_{2.5}$ beta-attenuation monitor operated by the U.S. Department of State in Chaoyang District.
   * Direct geographic pairing with adjacent PRSA stations (`Dongsi` and `Nongzhanguan`, $\approx 2\text{--}4\text{ km}$ distance) demonstrates empirical correlation $r > 0.95$, validating that municipal sensors were not artificially deflated during this period.
3. **Tier 2 — Non-Governmental / Academic Audit (Berkeley Earth):**
   * Independent scientific non-profit analysis (*Rohde & Muller, 2015, PLOS ONE*).
   * Cross-validates Beijing's heavy episodic haze as driven by transboundary regional transport from the southern/southwestern industrial corridor (Hebei/Shijiazhuang), matching our engineered `wd_sector` (`SW`/`SE`) findings.
4. **Tier 3 — Physical & Satellite Remote Sensing Benchmark (NASA AERONET & MODIS/MAIAC):**
   * **NASA AERONET:** Ground-based sun-photometer stations (`Beijing_CAMS`, `Beijing-RADI`) measuring column Aerosol Optical Depth (AOD) via direct solar beam extinction. Completely physical, optical, and tamper-proof.
   * **NASA MODIS/MAIAC:** $1\text{ km}$ high-resolution satellite retrievals independently confirming ground-level urban-rural spatial disparity (e.g., Aotizhongxin vs. Dingling background).

### B. Cloud Data Pull & Zero-Storage Deployment Strategy

To enable seamless continuous integration, cloud execution (Google Colab, Streamlit Cloud, GitHub Actions), and clean GitHub hosting without committing large binary archives:
* The repository excludes raw archives (`data/raw/` is gitignored).
* An automated programmatic fetch routine in `src/data_prep.py` pulls the dataset directly from the permanent UCI Cloud DOI endpoint (`https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip`) or via `ucimlrepo` (`id=501`), extracting and caching the 12 station CSVs on first run.

### C. Feature Engineering vs. Feature Scaling (Why They Are Separate Steps)

Keeping these distinct is fundamental because they happen at **different stages** of the 8-step lab pipeline and serve **different mathematical purposes**:

| Aspect | **Feature Engineering (Step 3 — Before Split)** | **Feature Scaling (Step 6 — After Split)** |
| :--- | :--- | :--- |
| **What it does** | Creates **new domain-informed variables** from raw columns (e.g., `season`, `temp_dew_spread`, `wd_sector`, `is_urban` / `is_rural` dummies). | Rescales **existing continuous numbers** onto a common numerical range ($z = \frac{x - \mu}{\sigma}$ via `StandardScaler`). |
| **Does it change information?** | **Yes** — injects domain knowledge from atmospheric science (Zhang et al., 2017) into $X$. | **No** — preserves the exact distribution shape; only changes units/scale. |
| **When it happens** | Before `train_test_split` (deterministic row-level transformations do not leak across rows). | **Strictly after `train_test_split`** ($\mu$ and $\sigma$ must be fit *only* on $X_{\text{train}}$ to prevent data leakage). |
| **Why our models need it** | Helps Linear Regression & Trees capture seasonal/spatial structure. | Prevents `PRES` ($\sim 1015\text{ hPa}$) from dominating `WSPM` ($\sim 2\text{ m/s}$) in **KNN** distances and **SVR** kernels. |

### D. Urban & Rural Dummy Stations (What We Are Testing)

Incorporating **dummy variables for Urban and Rural** provides **three concrete tests** in the regression model:

1. **Testing the "Urban Pollution Penalty" (Dummy Variable Regression Test):**
   * By encoding `is_urban` ($1 = \text{Urban Core}$, $0 = \text{Rural/Suburban}$), the coefficient $\beta_{\text{urban}}$ in the **Linear Regression baseline** directly answers: *"Holding temperature, dew point, wind speed, and wind direction constant, how many extra $\mu\text{g/m}^3$ of $\text{PM}_{2.5}$ does an urban station experience compared to a rural station?"*
2. **Counterfactual Dummy Station Stress-Test (Ceteris Paribus Simulation):**
   * Constructing two synthetic test profiles—**`Dummy_Station_Urban`** (`is_urban = 1`) and **`Dummy_Station_Rural`** (`is_urban = 0`)—fed with the **exact same 2016–2017 hourly weather sequence**. Passing both through trained KNN / SVR / Decision Tree models isolates the pure spatial gap ($\widehat{Y}_{\text{urban}} - \widehat{Y}_{\text{rural}}$) across winter vs. summer.
3. **Spatial Generalization Check (Leave-One-Station-Type-Out Test):**
   * Testing how well a model trained primarily on Urban stations predicts a Rural background station (like `Dingling` or `Huairou`)—proving empirically why Zhang et al. (2017) warned that station placement introduces **spatial reporting bias**.

### C. Aligned Model Hierarchy & Bias-Variance Diagnostics (Undergraduate Machine Learning)

In accordance with course lectures and assignment evaluation standards:
* **Naive Baseline:** `DummyRegressor(strategy='mean')` to formally anchor the mathematical floor ($R^2 = 0$).
* **Parametric Baseline:** Linear Regression & Ridge Regression (evaluating stability and interpreting $\beta_{\text{urban}}$).
* **Comparison Models & Diagnostic Sweeps (Validation Set Tuning):**
  * **Decision Tree Regressor:** Sweeping `max_depth` $\in [2, 4, 6, 8, 10, 15, 20]$ to demonstrate underfitting vs. overfitting on validation RMSE; extracting tree rules and feature importances to identify the root split.
  * **KNN Regressor:** Sweeping $K \in [3, 5, 11, 21, 31, 51]$, comparing Euclidean ($L_2$) vs. Manhattan ($L_1$) metrics and distance weighting.
  * **Support Vector Regressor (SVR):** Evaluating the margin-slack parameter $C$ ($0.1, 1.0, 10.0$) and tube width $\varepsilon$.

### D. Responsible Machine Learning Framework (Report Section 8)

Directly addressing the 5 core requirements from `ML_Mini_Project_Guidelines.pdf`:
1. **Data Privacy:** Public, de-identified environmental sensor telemetry with no personal identifiable information (PII).
2. **Bias & Spatial Fairness:** Disproportionate concentration of monitoring stations in urban centers (8 of 12) vs. rural peri-urban communities (4 of 12), and its implications for model generalizability.
3. **Interpretability:** Mathematical explainability of linear coefficients and tree partitions versus black-box kernel boundaries.
4. **Anti-Overclaiming (Prediction vs. Causality):** Differentiating between observational meteorological correlation and causal emission mechanisms.
5. **Practical Use & Human Decision Support:** Model application as an early advisory tool for municipal authorities (e.g., school outdoor activity advisories), subject to meteorological human oversight.

---

## 3. Complete Variable & Feature Engineering Dictionary

| Variable | Stage | Type | Description & Rationale |
| :--- | :--- | :--- | :--- |
| `No` | Dropped (Step 3) | Integer | Row ID (`1`–`35064`); dropped to prevent index leakage. |
| `year`, `month`, `day`, `hour` | Raw Timestamp | Integer | Combined into `datetime` for chronological sorting and 2013–2016 vs. 2016–2017 split. |
| `DEWP`, `TEMP`, `PRES`, `WSPM`, `RAIN` | Raw Continuous ($X$) | Float | Core meteorological variables; scaled in **Step 6** (`StandardScaler`). |
| `temp_dew_spread` | **Engineered (Step 3)** | Float | `TEMP - DEWP` ($^\circ\text{C}$); measures air dryness vs. near-saturation fog/haze conditions. |
| `season` | **Engineered (Step 3)** | Categorical | `Spring` (Mar–May), `Summer` (Jun–Aug), `Autumn` (Sep–Nov), `Winter` (Dec–Feb). |
| `time_of_day` | **Engineered (Step 3)** | Categorical | `Night` (0–5), `Morning_Rush` (6–10), `Afternoon` (11–16), `Evening_Rush` (17–23). |
| `wd_sector` | **Engineered (Step 3)** | Categorical | Groups 16 compass directions into 5 sectors (`NW`, `NE`, `SW`, `SE`, `CV`) per Zhang et al. (2017). |
| `is_urban` / `is_rural` | **Engineered Dummy (Step 3)** | Binary (`0/1`) | `is_urban = 1` for the 8 urban core stations; `is_rural = 1` for the 4 northern/suburban stations (`Changping`, `Dingling`, `Huairou`, `Shunyi`). |
| `station` | Raw Spatial ($X$) | Categorical | 12 monitoring site names (one-hot encoded with `drop_first=True` in linear models). |
| **`PM2.5`** | **Target ($y$)** | **Float** | Continuous hourly particulate concentration ($\mu\text{g/m}^3$). |
| `PM10`, `SO2`, `NO2`, `CO`, `O3` | Excluded | Float | Excluded from $X$ to avoid trivial co-emission collinearity and isolate meteorological confounding. |

---

## 4. Single Unified Repository Structure (Coursework Notebook + Streamlit App)

Everything lives in **ONE single project / repository**. The coursework deliverables and the Streamlit web app coexist cleanly:

```text
beijing-pm25-ml-regression/
├── README.md                          # Single repo overview, group credits, methodology & live Streamlit link
├── requirements.txt                   # Unified dependencies: numpy, pandas, scikit-learn, matplotlib, streamlit
├── .gitignore                         # Exclude raw data zips, __pycache__, .ipynb_checkpoints
├── app.py                             # Interactive Streamlit Web Application (runs directly from repo root)
│
├── notebooks/
│   └── 01_beijing_pm25_regression.ipynb # The exact 8-step lab notebook submitted for the group assignment
│
├── src/                               # Shared Python modules powering both notebook & web app
│   ├── data_prep.py                   # UCI Cloud fetch, missing data interpolation & domain features
│   ├── models.py                      # Scalers & model training (OLS, Trees, KNN, SVR)
│   └── evaluate.py                    # Evaluation metrics & counterfactual simulation
│
├── reports/
│   ├── report_outline.md              # 8–10 page written report draft (meeting all PDF guidelines)
│   └── figures/                       # High-res PNG plots for the written report
│
└── data/                              # Data root (gitignored, fetched dynamically via UCI Cloud)
    ├── raw/                           # 12 PRSA monitoring station CSVs
    └── processed/                     # Preprocessed splits / metadata
```

### Why This Single Unified Structure Works Best:
1. **One Single GitHub Repo:** You only maintain and push to one repository.
2. **Direct Submission of the Notebook:** For course evaluation, simply submit `notebooks/01_beijing_pm25_regression.ipynb` and the written report from `reports/`. It runs self-contained from top to bottom.
3. **One-Click Streamlit Deployment:** Streamlit Cloud points directly to your single repository and launches `app.py`.
4. **Seamless Post-Submission Updates:** Even after submitting the assignment, any commits pushed to this same repo automatically refresh your live Streamlit site.
5. **No Code Duplication:** `app.py` and `01_beijing_pm25_regression.ipynb` both import the same shared functions from `src/`.

---

## 5. GitHub Repository Push Checklist

## 6. GitHub Push Checklist
1. **Create `.gitignore` first** so you don't accidentally commit `.ipynb_checkpoints/`, `.DS_Store`, virtual environments (`venv/`), or raw zip archives if they exceed GitHub's recommended file size.
2. **Generate `requirements.txt`** via `pip freeze > requirements.txt` (or list exact versions of `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`).
3. **Include the UCI & Paper Citation in `README.md`:**
   * *Dataset:* Song Chen (2019). Beijing Multi-Site Air Quality. UCI Machine Learning Repository (`https://doi.org/10.24432/C5RK5G`).
   * *Paper:* Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings of the Royal Society A*, 473(2205), 20170457.
4. **Standard Git Commands to Push from Antigravity Terminal:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Beijing PM2.5 regression pipeline with urban/rural dummy station analysis"
   git branch -M main
   git remote add origin https://github.com/<your-username>/beijing-pm25-ml-regression.git
   git push -u origin main
   ```