"""
=============================================================================
📊 STATISTICAL ANALYSIS: GLOBAL POWER (NO TBR)
=============================================================================
Objective:
    Calculate statistical differences in Global Relative Power (averaged 
    across 20 channels) between Healthy Controls and Chronic Pain patients.

Focus:
    1. Relative Power (Delta, Theta, Alpha, Beta, Gamma).
    2. False Discovery Rate (FDR) correction using Benjamini-Hochberg.

Input:
    - results/final_dataset.csv

Output:
    - Statistical Table (Console): P-values (Uncorrected & FDR Corrected).
    - Trend Report: Text summary for thesis results section.

Execution:
    python ./FM_thesis_ML/src/Visualizations_ML/stats_global_differences.py
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from statsmodels.stats.multitest import multipletests
import os
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, CHANNELS, BANDS

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"

# Channels and Bands from Config
# CHANNELS = [...] 
# BANDS = dict(...) -> keys list for iteration
BAND_NAMES = list(BANDS.keys())

def get_col_name(df, ch, band):
    """Retrieves column name, handling alias variations (e.g., T7 vs T3)."""
    col = f"{ch}_{band}"
    if col in df.columns: return col
    mapping = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    if ch in mapping:
        alt = f"{mapping[ch]}_{band}"
        if alt in df.columns: return alt
    return None

def run_global_stats():
    if not DATA_FILE.exists():
        print("❌ Error: Data file not found.")
        return

    print("📊 Calculating Global Statistics (Healthy vs. Pain)...")
    df = pd.read_csv(DATA_FILE)

    # 1. Filter: TDBrain EC Only
    # We focus on the internal dataset to avoid site effects in global power
    df = df[
        (df['Condition'] == 'EC') & 
        (df['Group_Detailed'].isin(['TDBrain_Healthy', 'TDBrain_ChronicPain']))
    ].copy()

    if df.empty:
        print("⚠️ Warning: No valid TDBrain data found for stats.")
        return

    # 2. Calculate Global Features (Mean across 20 channels)
    stats_metrics = []
    
    
    
    for band in BAND_NAMES:
        cols = []
        for ch in CHANNELS:
            c = get_col_name(df, ch, band)
            if c: cols.append(c)
        
        if cols:
            df[f'Global_{band}'] = df[cols].mean(axis=1)
            stats_metrics.append(f'Global_{band}')

    # Aggregate per Subject (Average epochs)
    df_sub = df.groupby(['Subject', 'Group_Detailed'])[stats_metrics].mean().reset_index()

    # Split Groups
    group_h = df_sub[df_sub['Group_Detailed'] == 'TDBrain_Healthy']
    group_p = df_sub[df_sub['Group_Detailed'] == 'TDBrain_ChronicPain']

    print(f"\n   N(Healthy) = {len(group_h)}")
    print(f"   N(Pain)    = {len(group_p)}")
    print("-" * 90)
    print(f"{'BAND':<15} | {'MEAN (H)':<10} | {'MEAN (P)':<10} | {'TEST':<12} | {'P-VAL':<10} | {'P-FDR':<10}")
    print("-" * 90)

    # 3. Statistical Loop
    p_values = []
    
    for metric in stats_metrics:
        x = group_h[metric].values
        y = group_p[metric].values

        # Normality Check (Shapiro-Wilk)
        # Use try/except for robustness against small samples/constant data
        try:
            _, p_norm_x = shapiro(x)
            _, p_norm_y = shapiro(y)
            is_normal = (p_norm_x > 0.05 and p_norm_y > 0.05)
        except:
            is_normal = False
        
        # Select Test
        if is_normal:
            stat, p = ttest_ind(x, y, equal_var=False)
        else:
            stat, p = mannwhitneyu(x, y)
        
        p_values.append(p)
    
    # FDR Correction (Benjamini-Hochberg)
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # Print Results
    results_text = ""
    for i, metric in enumerate(stats_metrics):
        x = group_h[metric].values
        y = group_p[metric].values
        p_uncorr = p_values[i]
        p_fdr = p_corrected[i]
        
        # Re-check normality just for the print label logic (simplified)
        try:
            _, px = shapiro(x)
            _, py = shapiro(y)
            test = "T-Test" if (px > 0.05 and py > 0.05) else "Mann-W"
        except:
            test = "Mann-W"

        sig = "*" if rejected[i] else ""
        print(f"{metric:<15} | {x.mean():.3f}      | {y.mean():.3f}      | {test:<12} | {p_uncorr:.3f}      | {p_fdr:.3f} {sig}")

        # Generate Thesis Text
        band_name = metric.replace("Global_", "")
        if p_uncorr < 0.15: # Reporting trends (p < 0.15)
            direction = "higher" if y.mean() > x.mean() else "lower"
            sig_level = "significantly" if rejected[i] else "trend-level"
            results_text += f"- Global {band_name}: Pain group showed a {sig_level} {direction} power (M={y.mean():.3f} vs {x.mean():.3f}, p={p_uncorr:.3f}).\n"

    print("-" * 90)
    print("\n📝 THESIS TREND REPORT:")
    if results_text:
        print(results_text)
    else:
        print("   No strong global effects observed (suggests effects are localized).")

if __name__ == "__main__":
    run_global_stats()