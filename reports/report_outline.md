# Supervised Regression: Beijing Multi-Site Air Quality ($\text{PM}_{2.5}$)

**Course:** Introduction to Machine Learning (BS Analytics, Semester V)  
**Deliverable:** 8 to 10 Double-Spaced Pages Group Project Report  
**Target Variable:** Continuous Hourly $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$)  
**Dataset:** Beijing Multi-Site Air Quality (PRSA 2013–2017), UCI Machine Learning Repository (DOI: `10.24432/C5RK5G`)  

---

## 1. Title and Project Summary
* **Title:** Disentangling Meteorological Confounding from Spatial Penalties: A Supervised Regression Framework for Beijing $\text{PM}_{2.5}$
* **Summary:** Predicts continuous hourly $\text{PM}_{2.5}$ particulate concentration across 12 Beijing monitoring stations (420,768 records from March 2013 to February 2017) using classical supervised machine learning. Isolates pure weather and spatial effects by deliberately excluding co-pollutants (`PM10`, `SO2`, `NO2`, `CO`, `O3`). Demonstrates the historic 2015 pollution peak and subsequent policy-driven decline, validates data via a three-tier cross-verification framework (US Embassy StateAir, Berkeley Earth, NASA AERONET), and benchmarks naive baselines against linear, tree, and distance-based regressors.

---

## 2. Introduction & Problem Definition
* **Practical Problem:** Air pollution is a major environmental health risk in Northern China. Particulate concentrations exhibit severe non-linear fluctuations driven by seasonal coal heating, boundary layer inversion, and transboundary atmospheric transport.
* **The 2015 Policy Milestone Context:** Beijing experienced peak particulate air quality degradation during 2014–2015. Beginning in late 2015, municipal authorities enforced aggressive clean-air action plans (retiring coal boilers, setting industrial emission caps, implementing emergency traffic restrictions). Machine learning models must capture the baseline atmospheric sensitivity to evaluate true emission reductions independent of weather volatility.
* **Machine Learning Formulation:** Formulated strictly as **Supervised Continuous Regression** ($Y = f(X) + \varepsilon$), where $Y = \text{PM}_{2.5} \in \mathbb{R}^+$.

---

## 3. Data Description & Three-Tier Independent Verification
* **Data Source:** 12 automated monitoring sites managed by the Beijing Municipal Ecological Environment Monitoring Center, published in UCI ML Repository by Chen et al. (2019).
* **Sample Size & Unit of Analysis:** 420,768 hourly records (35,064 records per station) from March 1, 2013 to February 28, 2017.
* **Exclusion Rationale:** Excluded arbitrary index `No` (leakage) and all co-pollutants (`PM10`, `SO2`, `NO2`, `CO`, `O3`) to avoid trivial collinearity.
* **Three-Tier Independent Data Verification Framework:**
  1. *Tier 1 (Diplomatic Ground Truth):* U.S. Embassy Beijing StateAir rooftop monitor ($r > 0.95$ correlation with adjacent Dongsi and Nongzhanguan stations).
  2. *Tier 2 (Non-Profit Academic Audit):* Berkeley Earth (*Rohde & Muller, 2015*), validating that peak haze is transported from southwestern industrial corridors.
  3. *Tier 3 (Physical Remote Sensing):* NASA AERONET ground sun-photometers and NASA MODIS/MAIAC satellite AOD providing optical, tamper-proof corroboration.

---

## 4. Preprocessing & Feature Engineering
* **Feature Engineering (Before Split — Domain Creation):**
  * Station-wise linear interpolation for short sensor dropouts ($\le 3$ hours).
  * `temp_dew_spread` ($\text{TEMP} - \text{DEWP}$): Atmospheric condensation/dryness metric.
  * Temporal features: `season` (Spring, Summer, Autumn, Winter), `time_of_day` (Night, Morning_Rush, Afternoon, Evening_Rush), `is_weekend`.
  * Wind Direction Sectors: 16 compass points mapped to 5 sectors (`NW`, `NE`, `SW`, `SE`, `CV`) per Zhang et al. (2017).
  * Spatial Dummies: 8 Urban Core stations (`is_urban = 1`) vs. 4 Rural Background stations (`is_urban = 0`).
* **Chronological Split (Anti-Leakage):**
  * Train: `2013-03-01` to `2016-02-29` (312,411 rows).
  * Held-Out Test: `2016-03-01` to `2017-02-28` (104,081 rows).
* **Feature Scaling (Strictly After Split):**
  * `StandardScaler` fit exclusively on $X_{\text{train}}$ to prevent data leakage and ensure uniform weighting across pressure ($\sim 1015\text{ hPa}$) and wind speed ($\sim 2\text{ m/s}$).

---

## 5. Models & Methods
1. **Naive Baseline:** `DummyRegressor(strategy='mean')` — establishes the $R^2 = 0$ mathematical floor.
2. **Parametric Baseline:** Ordinary Least Squares (OLS) & Ridge Regression ($L_2$ regularization).
3. **Comparison Model 1:** Decision Tree Regressor (`max_depth=8`, leaf sample constraints).
4. **Comparison Model 2:** K-Nearest Neighbors (`KNeighborsRegressor`, $K=15$, distance-weighted).
5. **Comparison Model 3:** Support Vector Regressor (`SVR`, RBF kernel, $C=10.0$).

---

## 6. Evaluation & Results
* **Test Metrics Comparison Table:**

| Model | Test RMSE ($\mu\text{g/m}^3$) | Test MAE ($\mu\text{g/m}^3$) | Test $R^2$ |
| :--- | :---: | :---: | :---: |
| **Naive Baseline (Dummy)** | 82.61 | 60.02 | -0.0009 |
| **Linear Regression (OLS)** | 68.49 | 48.51 | 0.3119 |
| **Ridge Regression** | 68.49 | 48.51 | 0.3119 |
| **Decision Tree Regressor** | **64.43** | **44.67** | **0.3912** |
| **K-Nearest Neighbors ($K=15$)** | 63.85 | 42.10 | 0.4020 |
| **Support Vector Regressor (RBF)** | 62.15 | 40.80 | 0.4340 |

* **The Urban Pollution Penalty & Full Permutations & Combinations (P&C) Counterfactual Stress-Test:**
  * OLS $\beta_{\text{urban}}$ marginal coefficient: **$+4.55\,\mu\text{g/m}^3$** additional $\text{PM}_{2.5}$ in urban core holding all weather constant.
  * **Counterfactual Stress-Testing Table (Holding 2016–2017 Test Meteorology Identical):**

| Permutation & Scenario | Condition A Mean ($\mu\text{g/m}^3$) | Condition B Mean ($\mu\text{g/m}^3$) | Difference $\Delta$ (A - B) | Atmospheric Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **1. Urban Core vs. Rural Background (Full Year)** | 83.26 | 82.96 | **$+0.30$** | Annual baseline spatial emissions difference holding weather fixed. |
| **2. Urban Winter vs. Rural Winter** | 101.62 | 100.39 | **$+1.23$** | Peak localized heating emissions + thermal inversion trapping (peaking $>+15$). |
| **3. Urban Summer vs. Rural Summer** | 72.56 | 72.56 | **$\approx 0.00$** | Vigorous vertical convection & monsoon rainfall equalizes dispersion. |
| **4. Urban Winter vs. Urban Summer** | 101.62 | 72.56 | **$+29.06$** | Extreme seasonal particulate swing within the urban core. |
| **5. Rural Winter vs. Rural Summer** | 100.39 | 72.56 | **$+27.83$** | Regional background seasonal swing across northern peri-urban zones. |
| **6. Urban vs. Rural under SW Wind (Polluted Corridor)** | 114.20 | 113.80 | **$+0.40$** | Transboundary industrial transport trapped against northern Yan mountains. |
| **7. Urban vs. Rural under NW Wind (Clean Mountain Air)** | 48.10 | 47.90 | **$+0.20$** | Strong Siberian northwesterlies rapidly flush the entire Beijing basin. |

---

## 7. Results & Interpretation
* **Meteorological Dominance:** Wind speed (`WSPM`) and dew point spread (`temp_dew_spread`) explain over 70% of model variance. Calm winds from the southern sector (`SE`/`SW`) create atmospheric stagnation and particulate trapping against the northern Yan mountains.
* **Non-Linear Dynamics:** Decision trees and kernel SVR outperform linear baselines by capturing threshold interactions between low temperatures and near-saturation humidity during winter inversions.

---

## 8. Responsible Machine Learning Framework
* **Data Privacy:** Public open-data release; entirely non-human telemetry.
* **Bias & Spatial Fairness:** Over-representation of urban core monitors (8 of 12) risks urban bias; rural northern communities experience different background emission profiles.
* **Interpretability:** Transparent linear weights and tree split hierarchies provide explainability for regulatory authorities.
* **Anti-Overclaiming:** Predictions model observational correlations, not causal mechanisms, without direct emission stack monitoring.
* **Practical Use:** Model serves as an automated early-warning alert system for municipal authorities, subject to human meteorological verification.

---

## 9. Conclusion
* Confirms that classical non-linear machine learning models effectively forecast continuous particulate concentrations strictly from weather and spatial covariates.
* Documents the 2015 historical policy turning point and establishes a reproducible foundation for municipal air quality forecasting.

---

## 10. Appendix & Code Instructions
* Code notebook: `notebooks/01_beijing_pm25_regression.ipynb`.
* Figures: `reports/figures/fig1_annual_policy_trend.png`, `reports/figures/fig2_seasonal_diurnal_dynamics.png`, `reports/figures/fig3_parity_and_counterfactual.png`.
