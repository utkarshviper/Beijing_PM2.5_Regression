# Beijing Multi-Site Air Quality (PM2.5) Supervised Regression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/UCI-Dataset%20501-orange.svg)](https://doi.org/10.24432/C5RK5G)

An undergraduate machine learning mini-project for **Introduction to Machine Learning (BS Analytics, Semester V)** at TISS School of Analytics.

---

## 📌 Project Overview
This repository builds a continuous **Supervised Regression** ($Y = f(X) + \varepsilon$) pipeline predicting hourly continuous particulate matter (**$\text{PM}_{2.5}$**, $\mu\text{g/m}^3$) across 12 Beijing monitoring stations (420,768 records from March 1, 2013 to February 28, 2017).

### 🎯 Key Scientific & Course Objectives:
1. **The 2015 Peak AQI Policy Milestone:** Analyzes the historic 2014–2015 particulate peak and the subsequent policy-driven decline following municipal clean-air directives.
2. **Three-Tier Independent Data Verification:** Validates the primary Beijing Municipal Environmental Monitoring Center dataset against:
   * **Tier 1 (Diplomatic Ground Truth):** U.S. Embassy Beijing StateAir rooftop monitor ($r > 0.95$).
   * **Tier 2 (Non-Profit Audit):** Berkeley Earth regional transboundary transport findings (*Rohde & Muller, 2015*).
   * **Tier 3 (Physical Remote Sensing):** NASA AERONET sun-photometers and MODIS/MAIAC satellite AOD.
3. **Canonical 8-Step Lab Pipeline:**
   * Strict separation between **Feature Engineering (Step 3 - Domain Creation before split)** and **Feature Scaling (Step 6 - Numerical normalization strictly after split)**.
   * Anti-leakage chronological split (Train: 2013–2016, Held-out Test: 2016–2017).
   * Excludes companion co-pollutants (`PM10`, `SO2`, `NO2`, `CO`, `O3`) to avoid trivial collinearity and strictly isolate meteorological and spatial confounding.
4. **Spatial Hypothesis & Counterfactual Dummy Station Test:** Quantifies the **Urban Pollution Penalty** ($\Delta \widehat{\text{PM}}_{2.5}$) by feeding identical test weather sequences into trained models with `is_urban=1` vs. `is_urban=0`.
5. **Responsible Machine Learning:** Comprehensive evaluation of data privacy, spatial bias, model interpretability, and anti-overclaiming.

---

## 📂 Repository Structure

```text
beijing-pm25-ml-regression/
├── README.md                          # Repository documentation, methodology & team credits
├── requirements.txt                   # Project dependencies (pandas, scikit-learn, matplotlib, streamlit)
├── .gitignore                         # Excludes raw data zips, __pycache__, .ipynb_checkpoints
├── app.py                             # Interactive Streamlit Web Application (portfolio demo)
│
├── notebooks/
│   └── 01_beijing_pm25_regression.ipynb # Self-contained 8-step lab notebook submitted for group grading
│
├── src/                               # Shared Python engine powering both notebook and web app
│   ├── data_prep.py                   # Automated UCI cloud fetch, interpolation & feature engineering
│   ├── models.py                      # Scalers & model architectures (OLS, Ridge, Trees, KNN, SVR)
│   ├── evaluate.py                    # Metrics calculation & counterfactual simulation
│   ├── build_notebook.py              # Notebook generation script
│   └── generate_figures.py            # High-resolution report figure generation
│
├── reports/
│   ├── report_outline.md              # 8–10 page written report draft matching rubric
│   └── figures/                       # High-res publication plots (PNG)
│
└── data/                              # Data root (gitignored, fetched dynamically via UCI Cloud)
    ├── raw/                           # 12 PRSA monitoring station CSVs
    └── processed/                     # Preprocessed splits / metadata preview
```

---

## 📊 Model Evaluation Results

All six syllabus-aligned models were trained under the strict anti-leakage chronological split (Train: 2013–2016, Held-out Test: 2016–2017) and evaluated on a standardized uniform test sample ($N = 15,000$ held-out records):

| Model Family | Algorithm | Train RMSE ($\mu\text{g/m}^3$) | Test RMSE ($\mu\text{g/m}^3$) | Test MAE ($\mu\text{g/m}^3$) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline** | Mean Dummy Regressor | 80.48 | 81.28 | 59.55 | -0.0012 |
| **Linear Parametric** | Ordinary Least Squares (OLS) | 68.00 | 67.50 | 48.41 | 0.3096 |
| **Regularized Linear**| Ridge Regression ($L_2$, $\alpha=1.0$) | 68.00 | 67.50 | 48.41 | 0.3096 |
| **Tree-Based** | Decision Tree (`max_depth=8`) | 61.03 | **63.74** | **44.26** | **0.3844** |
| **Distance-Based** | K-Nearest Neighbors ($K=15$) | 0.00 | **63.62** | **44.52** | **0.3865** |
| **Kernel Method** | Support Vector Regressor (RBF) | 63.18 | **62.07** | **40.03** | **0.4162** |

### 🏙️ Decoupled Spatial Counterfactual & Permutations (P&C) Stress-Test
To quantify Dr. Bhandare's **Synthetic Urban Station (`is_urban=1`) vs. Synthetic Rural Station (`is_urban=0`)** without multicollinearity, we fit a decoupled spatial model yielding an identifiable linear marginal urban penalty of **$\beta_{\text{urban}} = +8.75\,\mu\text{g/m}^3$**. 

* **Winter Penalty:** Localized heating emissions and thermal inversions cause an urban seasonal swing of **$+30.47\,\mu\text{g/m}^3$**.
* **Summer Convective Dissipation:** Strong vertical mixing and monsoon precipitation equalize spatial disparities across the basin ($\Delta \approx 0.01\,\mu\text{g/m}^3$).

---

## 🚀 Quickstart & Reproducibility

### 1. Setup & Ingestion
The pipeline automatically pulls the dataset directly from the UCI Machine Learning Repository (DOI: `10.24432/C5RK5G`) if not already cached:

```bash
# Clone the repository
git clone <your-github-repo-url>
cd ML_Mini_Project

# Install dependencies
pip install -r requirements.txt

# Launch the interactive lab notebook
jupyter notebook notebooks/01_beijing_pm25_regression.ipynb

# Or launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## 📄 Deliverables Summary
* **Jupyter Notebook:** [`notebooks/01_beijing_pm25_regression.ipynb`](notebooks/01_beijing_pm25_regression.ipynb) (28 cells fully executed with all embedded visual outputs & HTML tables).
* **Final Report (Word DOCX):** [`reports/final_project_report.docx`](reports/final_project_report.docx) (Formally styled, ~10 double-spaced pages with embedded figures).
* **Final Report (Markdown):** [`reports/final_project_report.md`](reports/final_project_report.md).
* **Interactive App:** [`app.py`](app.py) (Streamlit dashboard with live prediction, counterfactual simulation & 3-tier audit).

Evaluated on the held-out 2016–2017 seasonal test set (104,081 hourly observations):

| Model Family | Model Description | Test RMSE ($\mu\text{g/m}^3$) | Test MAE ($\mu\text{g/m}^3$) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Naive Baseline** | Mean Predictor (`DummyRegressor`) | 82.61 | 60.02 | -0.0009 |
| **Parametric Baseline** | Ordinary Least Squares (`LinearRegression`) | 68.49 | 48.51 | 0.3119 |
| **Parametric Baseline** | Ridge Regression ($L_2$ Regularization) | 68.49 | 48.51 | 0.3119 |
| **Tree-Based** | Decision Tree Regressor (`max_depth=8`) | **64.43** | **44.67** | **0.3912** |
| **Distance-Based** | K-Nearest Neighbors ($K=15$, Distance-Weighted) | 63.85 | 42.10 | 0.4020 |
| **Kernel Method** | Support Vector Regressor (RBF Kernel, $C=10.0$) | 62.15 | 40.80 | 0.4340 |

* **Empirical Urban Pollution Penalty ($\beta_{\text{urban}}$):** Linear regression directly isolates a marginal penalty of **$+4.55\,\mu\text{g/m}^3$** in urban centers holding all meteorological variables constant.

---

## 🚀 Quickstart & Execution

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/<your-username>/beijing-pm25-ml-regression.git
cd beijing-pm25-ml-regression
pip install -r requirements.txt
```

### 2. Run the Submission Notebook
Launch Jupyter to explore the step-by-step pipeline:
```bash
jupyter notebook notebooks/01_beijing_pm25_regression.ipynb
```

### 3. Launch the Streamlit Web Application
Run the interactive dashboard locally:
```bash
streamlit run app.py
```

---

## 👥 Contributors (Group Assignment)
* BS Analytics, Batch 2024–2028, TISS 

---

## 📚 References & Citations
1. **Dataset:** Chen, S. (2019). *Beijing Multi-Site Air Quality Data*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G).
2. **Atmospheric Study:** Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). *Cautionary tales on air-quality improvement in Beijing*. Proceedings of the Royal Society A, 473(2205), 20170457.
3. **Independent Verification:** Rohde, R. A., & Muller, R. A. (2015). *Air pollution in China: Mapping of concentrations and sources*. PLOS ONE, 10(8), e0135749.
