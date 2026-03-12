"""
=============================================================================
📊 STATISTICS: THALAMOCORTICAL DYSRHYTHMIA (TCD) MARKERS
=============================================================================
Objective:
    Verify the presence of TCD biomarkers in the TDBrain dataset.
    According to TCD theory, chronic pain is associated with:
    1. Increased Theta Power (low-frequency slowing).
    2. Decreased Alpha Power (peak frequency shift).

    Focus: TDBrain dataset only (Healthy vs. Pain) to ensure results 
    are not confounded by site effects from the external dataset.

Input:
    - results/final_dataset.csv

Output:
    - Statistical Report (Console): Mean values and P-values.
    - Figure: results/figures/tcd_boxplot.png

Execution:
    python ./FM_thesis_ML/src/Visualizations_ML/stats_tcd.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import os
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, FIGURES_DIR, CHANNELS

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR

# Ensure output directory exists
IMG_DIR.mkdir(parents=True, exist_ok=True)


def check_tcd():
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    print("📊 Calculating TCD Biomarkers (TDBrain: Healthy vs Pain)...")
    df = pd.read_csv(DATA_FILE)

    # 1. Filter: TDBrain Only & EC Condition
    # We restrict this to TDBrain to avoid scanner/site confounds
    df = df[
        (df['Condition'] == 'EC') & 
        (df['Group_Detailed'].isin(['TDBrain_Healthy', 'TDBrain_ChronicPain']))
    ].copy()

    if df.empty:
        print("⚠️ Warning: No valid TDBrain data found (Healthy/Pain in EC).")
        return

    # 2. Calculate Global Features (Mean across 20 channels)
    # Theta
    theta_cols = [f"{ch}_Theta" for ch in CHANNELS if f"{ch}_Theta" in df.columns]
    if theta_cols:
        df['Global_Theta'] = df[theta_cols].mean(axis=1)
    
    # Alpha
    alpha_cols = [f"{ch}_Alpha" for ch in CHANNELS if f"{ch}_Alpha" in df.columns]
    if alpha_cols:
        df['Global_Alpha'] = df[alpha_cols].mean(axis=1)

    # 3. Aggregate per Subject
    # Average all epochs for a subject to get one value per person
    model_df = df.groupby(['Subject', 'Group_Detailed'])[['Global_Theta', 'Global_Alpha']].mean().reset_index()

    # 4. Statistical Analysis
    h = model_df[model_df['Group_Detailed'] == 'TDBrain_Healthy']
    p = model_df[model_df['Group_Detailed'] == 'TDBrain_ChronicPain']

    print(f"\n--- STATISTICAL RESULTS (N_Healthy={len(h)}, N_Pain={len(p)}) ---")
    print("-" * 90)
    
    metrics = ['Global_Theta', 'Global_Alpha']
    
    for metric in metrics:
        val_h = h[metric].values
        val_p = p[metric].values
        
        # Normality check
        _, nh = shapiro(val_h)
        _, np_val = shapiro(val_p)
        
        # Select Test: T-Test if normal, Mann-Whitney if non-normal
        if nh > 0.05 and np_val > 0.05:
            stat, pval = ttest_ind(val_h, val_p, equal_var=False)
            test_type = "T-Test"
        else:
            stat, pval = mannwhitneyu(val_h, val_p)
            test_type = "Mann-Whitney"
        
        # Determine Direction
        direction = "HIGHER" if val_p.mean() > val_h.mean() else "LOWER"
        
        print(f"{metric:<15} | H: {val_h.mean():.3f} vs P: {val_p.mean():.3f} | Pain is {direction} | p={pval:.4f} ({test_type})")

    # 5. Visualization (Boxplots)
    # Melting for Seaborn
    long_df = model_df.melt(id_vars=['Subject', 'Group_Detailed'], value_vars=metrics, var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    
    # Custom palette matching previous plots
    sns.boxplot(data=long_df, x='Metric', y='Value', hue='Group_Detailed', 
                palette={'TDBrain_Healthy': 'forestgreen', 'TDBrain_ChronicPain': 'firebrick'}, 
                showfliers=False)
    
    plt.title("Thalamocortical Dysrhythmia Biomarkers (TDBrain)")
    plt.ylabel("Relative Power")
    plt.tight_layout()
    
    save_path = os.path.join(IMG_DIR, "tcd_boxplot.png")
    plt.savefig(save_path)
    print(f"\n✅ Plot saved to: {save_path}")

if __name__ == "__main__":
    check_tcd()
