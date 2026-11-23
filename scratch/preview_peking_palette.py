"""
preview_peking_palette.py - Generates standalone preview charts demonstrating
the Imperial Beijing / Peking aesthetic palette on actual project metrics and data.
Saved to ML_Mini_Project/scratch/palette_preview/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom Imperial Beijing / Peking Aesthetic Styling
PEKING_PALETTE = {
    'bg': '#FAF8F2',               # Traditional Xuan Rice Paper antique warm white
    'grid': '#E5DFD3',             # Soft parchment grid
    'text': '#2B231D',             # Deep Chinese Lacquer Black/Charcoal
    'forbidden_red': '#9B1D20',    # Forbidden City Palace Wall Vermilion / Cinnabar
    'imperial_gold': '#DDA12A',    # Imperial Glazed Roof Tile Gold
    'temple_blue': '#1F4E79',      # Temple of Heaven Cloisonne Blue
    'mountain_jade': '#2D6A4F',    # Summer Palace / Yan Mountain Pine Jade
    'hutong_grey': '#6C757D',      # Classical Beijing Hutong Grey Brick
}

plt.rcParams.update({
    'figure.facecolor': PEKING_PALETTE['bg'],
    'axes.facecolor': PEKING_PALETTE['bg'],
    'axes.edgecolor': PEKING_PALETTE['text'],
    'axes.labelcolor': PEKING_PALETTE['text'],
    'xtick.color': PEKING_PALETTE['text'],
    'ytick.color': PEKING_PALETTE['text'],
    'grid.color': PEKING_PALETTE['grid'],
    'font.family': 'sans-serif',
})

OUT_DIR = "ML_Mini_Project/scratch/palette_preview"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_palette_sample_card():
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=300)
    colors = [
        ('Forbidden City Vermilion\n(#9B1D20)', PEKING_PALETTE['forbidden_red']),
        ('Imperial Roof Gold\n(#DDA12A)', PEKING_PALETTE['imperial_gold']),
        ('Temple of Heaven Blue\n(#1F4E79)', PEKING_PALETTE['temple_blue']),
        ('Summer Palace Jade\n(#2D6A4F)', PEKING_PALETTE['mountain_jade']),
        ('Hutong Grey Brick\n(#6C757D)', PEKING_PALETTE['hutong_grey']),
    ]
    
    for i, (name, hex_code) in enumerate(colors):
        ax.bar(i, 1, color=hex_code, edgecolor='#2B231D', width=0.85, linewidth=1.2)
        ax.text(i, 0.5, name, ha='center', va='center', color='white' if i != 1 else '#2B231D',
                fontweight='bold', fontsize=8.5)
        
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    ax.set_title("Imperial Beijing Architectural & Cultural Palette", fontsize=11, fontweight='bold', pad=10, color=PEKING_PALETTE['text'])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "preview_01_color_swatch.png"))
    plt.close()


def generate_preview_model_benchmark():
    models = ['Naive Baseline', 'Linear OLS', 'Ridge Reg', 'Decision Tree', 'KNN (K=15)', 'SVR (RBF)']
    rmse_vals = [82.61, 68.49, 68.49, 64.43, 63.85, 62.15]
    r2_vals = [-0.001, 0.312, 0.312, 0.391, 0.402, 0.434]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    
    # Left: RMSE using Peking Colors
    bar_colors = [
        PEKING_PALETTE['hutong_grey'],
        PEKING_PALETTE['imperial_gold'],
        PEKING_PALETTE['imperial_gold'],
        PEKING_PALETTE['temple_blue'],
        PEKING_PALETTE['mountain_jade'],
        PEKING_PALETTE['forbidden_red'],
    ]
    
    axes[0].bar(models, rmse_vals, color=bar_colors, edgecolor=PEKING_PALETTE['text'], width=0.6, linewidth=1.1)
    axes[0].set_title("Test RMSE Across Models (Lower is Better)", fontsize=11, fontweight='bold', color=PEKING_PALETTE['text'])
    axes[0].set_ylabel("Test RMSE (µg/m³)", fontsize=10, fontweight='bold')
    axes[0].set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(rmse_vals):
        axes[0].text(i, v + 1.2, f"{v:.1f}", ha='center', fontsize=8.5, fontweight='bold', color=PEKING_PALETTE['text'])
        
    # Right: R2 Scores
    axes[1].bar(models, r2_vals, color=bar_colors, edgecolor=PEKING_PALETTE['text'], width=0.6, linewidth=1.1)
    axes[1].axhline(0, color=PEKING_PALETTE['hutong_grey'], linestyle='--', linewidth=1.2)
    axes[1].set_title("Variance Explained ($R^2$ Score) (Higher is Better)", fontsize=11, fontweight='bold', color=PEKING_PALETTE['text'])
    axes[1].set_ylabel("Test $R^2$", fontsize=10, fontweight='bold')
    axes[1].set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(r2_vals):
        pos_y = v + 0.012 if v >= 0 else 0.012
        axes[1].text(i, pos_y, f"{v:.3f}", ha='center', fontsize=8.5, fontweight='bold', color=PEKING_PALETTE['text'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "preview_02_model_benchmark.png"))
    plt.close()


def generate_preview_counterfactual():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    
    # Left: 4-Quadrant Scenarios
    scenarios = ['Urban Winter\n(Inversion & Coal)', 'Rural Winter\n(Background Cold)', 'Urban Summer\n(Convection)', 'Rural Summer\n(Clean Mountain)']
    means = [101.62, 100.39, 72.56, 72.56]
    scenario_colors = [
        PEKING_PALETTE['forbidden_red'],
        '#C25953',                      # Muted terra cotta
        PEKING_PALETTE['temple_blue'],
        PEKING_PALETTE['mountain_jade']
    ]
    
    axes[0].bar(scenarios, means, color=scenario_colors, edgecolor=PEKING_PALETTE['text'], width=0.55, linewidth=1.1)
    axes[0].set_title("Counterfactual Mean PM2.5: Urban vs. Rural in Winter vs. Summer", fontsize=10.5, fontweight='bold', color=PEKING_PALETTE['text'])
    axes[0].set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=10, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, m in enumerate(means):
        axes[0].text(i, m + 1.5, f"{m:.1f}", ha='center', fontweight='bold', fontsize=9, color=PEKING_PALETTE['text'])
        
    # Right: KDE Distributions
    np.random.seed(42)
    winter_sim = np.random.normal(1.23, 2.8, 3000)
    summer_sim = np.random.normal(0.00, 0.4, 3000)
    
    from scipy.stats import gaussian_kde
    x_grid = np.linspace(-6, 12, 300)
    kde_w = gaussian_kde(winter_sim)(x_grid)
    kde_s = gaussian_kde(summer_sim)(x_grid)
    
    axes[1].plot(x_grid, kde_w, color=PEKING_PALETTE['forbidden_red'], linewidth=2.4, label='Winter Season Penalty (Peak Heating Inversion)')
    axes[1].fill_between(x_grid, kde_w, color=PEKING_PALETTE['forbidden_red'], alpha=0.25)
    
    axes[1].plot(x_grid, kde_s, color=PEKING_PALETTE['mountain_jade'], linewidth=2.4, linestyle='--', label='Summer Season Penalty (Convective Dispersion)')
    axes[1].fill_between(x_grid, kde_s, color=PEKING_PALETTE['mountain_jade'], alpha=0.25)
    
    axes[1].axvline(0, color=PEKING_PALETTE['hutong_grey'], linestyle=':', linewidth=1.5)
    axes[1].set_title("Counterfactual Urban Penalty: Winter (Heating) vs. Summer (Clean)", fontsize=10.5, fontweight='bold', color=PEKING_PALETTE['text'])
    axes[1].set_xlabel("Predicted Urban Penalty Δ (Urban Core - Rural, µg/m³)", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Density", fontsize=10, fontweight='bold')
    axes[1].legend(frameon=True, facecolor=PEKING_PALETTE['bg'], edgecolor=PEKING_PALETTE['grid'])
    axes[1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "preview_03_counterfactual_matrix.png"))
    plt.close()


if __name__ == '__main__':
    generate_palette_sample_card()
    generate_preview_model_benchmark()
    generate_preview_counterfactual()
    print("All preview figures generated successfully in ML_Mini_Project/scratch/palette_preview/")
