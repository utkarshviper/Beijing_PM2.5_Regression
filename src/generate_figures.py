"""
generate_figures.py - Generates publication-ready figures for the 8-10 page report
and exports them to reports/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import ensure_dataset_ready, load_and_preprocess_data
from src.models import prepare_feature_matrices, train_baseline_models, train_decision_tree
from src.evaluate import calculate_metrics, run_urban_rural_counterfactual_test

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_annual_policy_trend(df: pd.DataFrame):
    """
    Figure 1: Historical AQI / PM2.5 Trajectory (2013-2017) showing the 2014-2015 peak
    and subsequent decline following aggressive municipal clean air action plans.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    annual_stats = df.groupby(df['datetime'].dt.year)['PM2.5'].agg(['mean', 'median'])
    years = annual_stats.index.values
    
    ax.plot(years, annual_stats['mean'], marker='o', linewidth=2.5, color='#b22222', label='Annual Mean PM2.5 (µg/m³)')
    ax.plot(years, annual_stats['median'], marker='s', linewidth=2.0, linestyle='--', color='#2e8b57', label='Annual Median PM2.5 (µg/m³)')
    
    ax.axvline(x=2015, color='#4682b4', linestyle=':', linewidth=2, label='2015 Clean Air Policy Turning Point')
    ax.annotate('2015 Peak & Aggressive Policy Implementation\n(Coal boiler phase-out, vehicle restrictions)',
                xy=(2015, 80), xytext=(2013.2, 95),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fdf5e6", ec="#b22222", lw=1))
    
    ax.set_title("Beijing Atmospheric Trajectory: Annual PM2.5 Trends & 2015 Policy Turning Point", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Calendar Year", fontsize=10, fontweight='bold')
    ax.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=10, fontweight='bold')
    ax.set_xticks(years)
    ax.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig1_annual_policy_trend.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")


def plot_seasonal_diurnal_patterns(df: pd.DataFrame):
    """
    Figure 2: Seasonal & Diurnal particulate dynamics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    season_means = df.groupby('season')['PM2.5'].mean().reindex(season_order)
    
    axes[0].bar(season_means.index, season_means.values, color=['#7bc043', '#fdf498', '#f37736', '#d11141'], edgecolor='black', alpha=0.85)
    axes[0].set_title("Seasonal Mean PM2.5 Concentrations (Winter Coal Peak)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10, fontweight='bold')
    for i, v in enumerate(season_means.values):
        axes[0].text(i, v + 2, f"{v:.1f}", ha='center', fontweight='bold', fontsize=9)
        
    diurnal = df.groupby(['hour', 'zone_type'])['PM2.5'].mean().unstack()
    axes[1].plot(diurnal.index, diurnal['Urban'], color='#b22222', marker='o', linewidth=2, label='Urban Core')
    axes[1].plot(diurnal.index, diurnal['Rural_Suburban'], color='#2e8b57', marker='s', linewidth=2, linestyle='--', label='Rural Background')
    axes[1].set_title("Diurnal Profile: Urban Penalty Across 24-Hour Cycle", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Hour of Day (0–23)", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=10, fontweight='bold')
    axes[1].legend(frameon=True)
    axes[1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig2_seasonal_diurnal_dynamics.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")


def plot_residuals_and_counterfactual(dt_model, data_dict: dict):
    """
    Figure 3: Model Parity & Counterfactual Urban vs. Rural Seasonal Penalty Simulation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Subsample 2000 points from test set for crisp scatter
    np.random.seed(42)
    sample_idx = np.random.choice(len(data_dict['y_test']), size=2000, replace=False)
    y_test_sub = data_dict['y_test'][sample_idx]
    y_pred_sub = dt_model.predict(data_dict['X_test_scaled'].iloc[sample_idx])
    
    # Left: Parity plot
    axes[0].scatter(y_test_sub, y_pred_sub, alpha=0.3, color='#4682b4', edgecolors='none', s=25)
    axes[0].plot([0, 500], [0, 500], 'r--', linewidth=2, label='1:1 Parity Line')
    axes[0].set_xlim(0, 450)
    axes[0].set_ylim(0, 450)
    axes[0].set_title("Decision Tree: Observed vs. Predicted PM2.5 (Test Set)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Observed PM2.5 (µg/m³)", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=10, fontweight='bold')
    axes[0].legend(frameon=True)
    
    # Right: Counterfactual Urban Penalty: Winter vs. Summer
    sim_df = run_urban_rural_counterfactual_test(dt_model, data_dict['X_test_scaled'], data_dict['feature_names'])
    
    winter_mask = sim_df['season_Winter'] == 1
    summer_mask = sim_df['season_Summer'] == 1
    
    sns.kdeplot(sim_df[winter_mask]['Urban_Penalty_Delta'], ax=axes[1], color='#b22222', linewidth=2.5,
                label=f"Winter Penalty (Mean: +{sim_df[winter_mask]['Urban_Penalty_Delta'].mean():.2f} µg/m³)")
    sns.kdeplot(sim_df[summer_mask]['Urban_Penalty_Delta'], ax=axes[1], color='#2e8b57', linewidth=2.5, linestyle='--',
                label=f"Summer Penalty (Mean: {sim_df[summer_mask]['Urban_Penalty_Delta'].mean():.2f} µg/m³)")
    
    axes[1].axvline(0, color='gray', linestyle=':', linewidth=1.5)
    axes[1].set_title("Counterfactual Urban Penalty: Winter (Heating) vs. Summer (Clean)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Predicted Urban Penalty Δ (Urban - Rural, µg/m³)", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Density", fontsize=10, fontweight='bold')
    axes[1].legend(frameon=True)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig3_parity_and_counterfactual.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    st_dir = "data/raw/PRSA_Data_20130301-20170228"
    print("Loading data for figure generation...")
    df = load_and_preprocess_data(st_dir)
    plot_annual_policy_trend(df)
    plot_seasonal_diurnal_patterns(df)
    
    print("Training Decision Tree for diagnostic plots...")
    data = prepare_feature_matrices(df)
    dt = train_decision_tree(data, max_depth=8)
    plot_residuals_and_counterfactual(dt, data)
    print("All report figures successfully generated.")
