"""
evaluate.py - Step 8: Evaluation, Metric Calculations, Residual Diagnostics
             and Urban vs. Rural Counterfactual Dummy Station Stress-Testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """
    Computes RMSE, MAE, and R^2 for regression models.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'Model': model_name,
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'R2': round(r2, 4)
    }


def run_urban_rural_counterfactual_test(model, X_test_scaled: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Simulates the pure "Urban Pollution Penalty" (Ceteris Paribus test).
    Takes the actual 2016-2017 test weather sequence, and feeds two copies:
    - Profile 1: is_urban forced to 1
    - Profile 2: is_urban forced to 0
    Holding all meteorological conditions strictly identical.
    
    Includes seasonal breakdown (Winter vs. Summer) to analyze seasonal variance.
    """
    X_urban_sim = X_test_scaled.copy()
    X_rural_sim = X_test_scaled.copy()
    
    if 'is_urban' in X_test_scaled.columns:
        X_urban_sim['is_urban'] = 1
        X_rural_sim['is_urban'] = 0
        
    y_pred_urban = model.predict(X_urban_sim)
    y_pred_rural = model.predict(X_rural_sim)
    
    diff = y_pred_urban - y_pred_rural
    
    res_df = pd.DataFrame({
        'Predicted_PM25_Urban_Core': y_pred_urban,
        'Predicted_PM25_Rural_Background': y_pred_rural,
        'Urban_Penalty_Delta': diff
    })
    
    if 'season_Winter' in X_test_scaled.columns:
        res_df['season_Winter'] = X_test_scaled['season_Winter'].values
    if 'season_Summer' in X_test_scaled.columns:
        res_df['season_Summer'] = X_test_scaled['season_Summer'].values
    if 'season_Spring' in X_test_scaled.columns:
        res_df['season_Spring'] = X_test_scaled['season_Spring'].values
        
    return res_df


def summarize_seasonal_urban_penalty(counterfactual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the Urban vs. Rural penalty across seasons (Winter vs. Summer vs. Spring vs. Autumn).
    """
    records = []
    
    # Overall
    records.append({
        'Slice': 'Full Test Year (2016–2017)',
        'Urban Core Mean (µg/m³)': round(counterfactual_df['Predicted_PM25_Urban_Core'].mean(), 2),
        'Rural Background Mean (µg/m³)': round(counterfactual_df['Predicted_PM25_Rural_Background'].mean(), 2),
        'Urban Penalty Δ (µg/m³)': round(counterfactual_df['Urban_Penalty_Delta'].mean(), 2)
    })
    
    # Winter
    if 'season_Winter' in counterfactual_df.columns:
        w_df = counterfactual_df[counterfactual_df['season_Winter'] == 1]
        records.append({
            'Slice': 'Winter Season (Heating / Coal Inversion)',
            'Urban Core Mean (µg/m³)': round(w_df['Predicted_PM25_Urban_Core'].mean(), 2),
            'Rural Background Mean (µg/m³)': round(w_df['Predicted_PM25_Rural_Background'].mean(), 2),
            'Urban Penalty Δ (µg/m³)': round(w_df['Urban_Penalty_Delta'].mean(), 2)
        })
        
    # Summer
    if 'season_Summer' in counterfactual_df.columns:
        s_df = counterfactual_df[counterfactual_df['season_Summer'] == 1]
        records.append({
            'Slice': 'Summer Season (Convective / Clean)',
            'Urban Core Mean (µg/m³)': round(s_df['Predicted_PM25_Urban_Core'].mean(), 2),
            'Rural Background Mean (µg/m³)': round(s_df['Predicted_PM25_Rural_Background'].mean(), 2),
            'Urban Penalty Δ (µg/m³)': round(s_df['Urban_Penalty_Delta'].mean(), 2)
        })
        
    return pd.DataFrame(records)

