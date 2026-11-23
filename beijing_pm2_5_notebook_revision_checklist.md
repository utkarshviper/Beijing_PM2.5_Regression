# Revision Checklist: `01_beijing_pm25_regression.ipynb`
**Aligned with Dr. Purushottam Bhandare’s Syllabus, 8-Step Lab Pipeline & Mini-Project Rubric**

This updated checklist resolves every technical bug found during the strict audit while preserving all syllabus-aligned models (`Dummy`, `OLS`, `Ridge`, `Decision Tree`, `KNN`, `SVR`), your 8-step lab separation between **Feature Engineering (Step 3)** and **Feature Scaling (Step 6)**, and Dr. Bhandare’s **Urban vs. Rural Dummy Station** experiment.

---

## 1. Keep Step 3 (Feature Engineering) vs. Step 6 (Feature Scaling) Distinct, But Fix Boundary Leakage (Steps 3 & 5)
* **Why this matters for your viva/report:** Per Dr. Bhandare’s instruction, deterministic **Feature Engineering** (`temp_dew_spread`, `wind_sector`, `is_urban`, `season`, `rush_hour`) belongs in **Step 3** (before splitting), whereas **Feature Scaling** (`StandardScaler`) belongs in **Step 6** (after splitting) to prevent data leakage.
* **What must be fixed:** Station-wise linear interpolation (`limit=3`) currently runs in Step 3 *before* the chronological split in Step 5, allowing up to 3 hours of test data to bleed across `2016-03-01`. Move interpolation so it runs *after* splitting in Step 5.

```python
# Keep deterministic feature engineering in Step 3, but move interpolation to Step 5 (after splitting):
numeric_weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

train_df = train_df.sort_values(['station', 'datetime']).copy()
test_df = test_df.sort_values(['station', 'datetime']).copy()

train_df[numeric_weather_cols] = (
    train_df.groupby('station')[numeric_weather_cols]
    .transform(lambda s: s.interpolate(method='linear', limit=3))
)
test_df[numeric_weather_cols] = (
    test_df.groupby('station')[numeric_weather_cols]
    .transform(lambda s: s.interpolate(method='linear', limit=3))
)

# Drop remaining missing values separately in train and test
train_df = train_df.dropna(subset=['PM2.5'] + numeric_weather_cols)
test_df = test_df.dropna(subset=['PM2.5'] + numeric_weather_cols)
```

---

## 2. Fix Exact Multicollinearity & Dr. Bhandare’s Urban vs. Rural Dummy Station Test (Steps 4, 6 & 8)
* **Issue:**
  1. `temp_dew_spread` (`TEMP - DEWP`) is currently passed into unregularized `LinearRegression` alongside both `TEMP` and `DEWP`.
  2. `is_urban` is an exact linear combination of the 12 `station_*` one-hot columns.
  3. In Step 8, simulating `Dummy_Station_Urban` (`is_urban = 1`) vs. `Dummy_Station_Rural` (`is_urban = 0`) on `dt_model` leaves the `station_*` one-hot columns unchanged, feeding impossible combinations (e.g., a rural station with `is_urban = 1`) into the tree.
* **Required Fix:**
  * Drop `DEWP` (keeping `TEMP` and `temp_dew_spread`) before fitting unregularized `LinearRegression` so coefficients are mathematically stable.
  * For Dr. Bhandare’s **Urban vs. Rural Dummy Station test**, fit a dedicated spatial-zone model **without** the 12 individual `station_*` columns (using `is_urban` as the spatial indicator instead). This cleanly tests how identical 2016–2017 weather conditions translate into predicted $\text{PM}_{2.5}$ at a synthetic Urban station versus a synthetic Rural station without contradicting station dummies.
  * Frame the result in plain language as a **conditional predictive difference** rather than a causal claim.

```python
# Dedicated feature set for Dr. Bhandare's Urban vs. Rural Dummy Station Test
# (Excludes individual station_* dummies and redundant DEWP so is_urban is cleanly isolated)
zone_num_cols = ['TEMP', 'PRES', 'temp_dew_spread', 'RAIN', 'WSPM', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
zone_cat_cols = ['wd_sector', 'time_bin']  # Exclude 'station'

# Fit OLS / Ridge and DecisionTreeRegressor on [zone_num_cols + zone_cat_cols + ['is_urban']]
# Then simulate Dummy_Station_Urban (is_urban=1) vs. Dummy_Station_Rural (is_urban=0) on identical 2016–2017 test weather
```

---

## 3. Keep All 6 Syllabus Models (`Dummy`, `OLS`, `Ridge`, `Decision Tree`, `KNN`, `SVR`), But Standardize the Test Set (Step 8)
* **Syllabus Alignment:** Keep `DummyRegressor`, `LinearRegression`, `Ridge`, `DecisionTreeRegressor`, `KNeighborsRegressor`, and `SVR`—do **not** replace them with out-of-syllabus ensemble methods (such as Random Forest or XGBoost).
* **Issue:** Currently, `Dummy`, `OLS`, `Ridge`, and `Decision Tree` are evaluated on the full test set (`X_test_scaled`), while `KNN` and `SVR` are evaluated only on `X_test_scaled.iloc[:10000]`. Because the dataframe is sorted by `['station', 'datetime']`, slicing the first 10,000 rows evaluates KNN and SVR on only 1–2 stations, invalidating the comparison table and bar charts.
* **Required Fix:** Evaluate **all 6 models on the exact same test set**. If full-test inference for `SVR` / `KNN` is too slow on your machine, create one fixed, stratified/random test sample (`n=15000`, `random_state=42`) and evaluate **every single model** (including Dummy, OLS, Ridge, and Decision Tree) on that exact same test subset.

```python
# Fair evaluation on an identical test subset across ALL 6 syllabus models:
eval_idx = X_test_scaled.sample(n=15000, random_state=42).index
X_test_eval = X_test_scaled.loc[eval_idx]
y_test_eval = y_test.loc[eval_idx]

# Pass X_test_eval and y_test_eval into EVERY model's RMSE, MAE, and R^2 calculation
```

---

## 4. Add Syllabus-Aligned Diagnostics & Missing Rubric Sections
To meet all grading requirements while staying strictly within Dr. Bhandare’s lecture scope (avoiding out-of-syllabus tools like SHAP or Permutation Importance):

### A. Bias–Variance Tradeoff Diagnostic (Train vs. Test Metrics)
* **Rubric & Lecture Alignment:** Directly demonstrates the **Bias–Variance Tradeoff** (Topics 1, 6, 7) and checks for overfitting/generalizability.
* **Action:** Report **Train RMSE / MAE / $R^2$** alongside **Test RMSE / MAE / $R^2$** for all 6 models in your comparison table.

### B. Syllabus-Aligned Feature Importance & Coefficient Plots
* **Rubric & Lecture Alignment:** The guidelines require explaining *"which features appear important,"* and your Responsible ML section already references tree splits on `temp_dew_spread`.
* **Action:** Plot horizontal bar charts of:
  1. `dt_model.feature_importances_` (Decision Tree Gini/MSE split importance)
  2. `ridge_model.coef_` (Standardized linear weights)
  Verify whether `temp_dew_spread` / `DEWP` and `wd_sector` emerge as primary drivers, connecting your findings directly to Zhang et al. (2017).

### C. Plain-Language Interpretation & Conclusion
* **Rubric Alignment:** The rubric requires explaining model performance *"in plain language, not only by reporting metrics"* and ending with a **Conclusion**.
* **Action:** Add two concise markdown cells after Step 8:
  1. **Plain-Language Results Summary:** Translate the best model's RMSE and MAE into $\mu\text{g/m}^3$ units relative to the WHO/China 24-hour guideline ($35\ \mu\text{g/m}^3$), and explain why meteorology alone captures baseline dispersion trends while missing sudden local emission spikes.
  2. **Conclusion & Future Work:** Summarize which syllabus models generalized best, note the trade-off between linear interpretability (Ridge) and non-linear threshold capture (Decision Tree / KNN / SVR), and outline concrete improvements (e.g., adding lagged $\text{PM}_{2.5}$ features or hyperparameter grid search).

---

## 5. Execute All Cells & Prepare Clean GitHub Repository Structure
* **Execution Requirement:** Restart the kernel (`Kernel -> Restart & Run All`) and save `01_beijing_pm25_regression.ipynb` so all `execution_count` numbers, tables, and charts are embedded in the file.
* **GitHub Push Checklist:**
  * Ensure your `.gitignore` excludes raw/extracted multi-megabyte CSVs (`data/raw/*.csv`, `data/raw/*.zip`, `.ipynb_checkpoints/`) since your notebook automatically fetches the UCI dataset into `data/raw/`.
  * Include a concise `README.md` with your group details, problem statement, 8-step workflow summary, and citation for Zhang et al. (2017) (*Proceedings of the Royal Society A*, 473: 20170457).