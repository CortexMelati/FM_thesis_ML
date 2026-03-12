"""
=============================================================================
🔥 VISUALIZATION: COMPREHENSIVE HEATMAP ANALYSIS (ROBUSTNESS CHECK)
=============================================================================
Objective:
    Generate four distinct T-value heatmaps to validate the Occipital Theta 
    biomarker and demonstrate the efficacy of the harmonization method.

    1. TDBrain Standard: The baseline result (Healthy vs. Pain).
    2. TDBrain Harmonized: A robustness check. Applies the 'No-Delta' normalization
       to the internal dataset to prove the biomarker persists.
    3. Merged Raw: Visual proof of Site Effects (Delta bias).
    4. Merged Harmonized: Visual proof of the correction strategy.

Input:
    - results/final_dataset.csv

Output: 
    - results/figures/heatmap_1_tdbrain_standard.png
    - results/figures/heatmap_2_tdbrain_harmonized.png
    - results/figures/heatmap_3_merged_raw.png
    - results/figures/heatmap_4_merged_harmonized.png
    - Terminal output of exact T-values for thesis text.

Execution:
    python ./FM_thesis_ML/src/Visualizations_Prep/visualize_heatmap.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, FIGURES_DIR, CHANNELS

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR
IMG_DIR.mkdir(parents=True, exist_ok=True)

def get_col_name(df, ch, band):
    """Retrieves column name, handling potential alias variations."""
    col = f"{ch}_{band}"
    if col in df.columns: return col
    mapping = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    if ch in mapping:
        alt_col = f"{mapping[ch]}_{band}"
        if alt_col in df.columns: return alt_col
    return None

def map_merged_groups(group):
    """Maps specific dataset subgroups to broader 'Healthy' or 'Pain' categories."""
    if group in ['TDBrain_Healthy', 'TDBrain_Unknown_NoIndication']: return 'Healthy_Combined'
    elif group in ['TDBrain_ChronicPain', 'External_CP']: return 'Pain_Combined'
    return 'Other'

# =============================================================================
# HELPER: GENERIC PLOTTER
# =============================================================================
def plot_heatmap_generic(t_matrix, bands, title, filename):
    plt.figure(figsize=(8, 10))
    sns.heatmap(t_matrix, annot=True, fmt=".1f", cmap="coolwarm", center=0,
                xticklabels=bands, yticklabels=CHANNELS, cbar_kws={'label': 'T-Value'})
    plt.title(title)
    plt.tight_layout()
    path = IMG_DIR / filename
    plt.savefig(path)
    print(f"   ✅ Saved: {path.name}")

# =============================================================================
# 1. TDBRAIN STANDARD (MAIN RESULT)
# =============================================================================
def run_tdbrain_standard(df_full):
    print("\n[1/4] Generating TDBrain Standard Heatmap...")
    df = df_full[(df_full['Condition'] == 'EC') & (df_full['Group_Detailed'].isin(['TDBrain_Healthy', 'TDBrain_ChronicPain']))].copy()
    BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    t_matrix = np.zeros((len(CHANNELS), len(BANDS)))

    for i, ch in enumerate(CHANNELS):
        for j, band in enumerate(BANDS):
            col = get_col_name(df, ch, band)
            if col:
                sub = df.groupby(['Subject', 'Group_Detailed'])[col].mean().reset_index()
                h = sub[sub['Group_Detailed'] == 'TDBrain_Healthy'][col]
                p = sub[sub['Group_Detailed'] == 'TDBrain_ChronicPain'][col]
                t_stat, _ = ttest_ind(p, h, nan_policy='omit')
                t_matrix[i, j] = t_stat
                
    plot_heatmap_generic(t_matrix, BANDS, "Fig 1: TDBrain Standard (Main Result)", "heatmap_1_tdbrain_standard.png")

# =============================================================================
# 2. TDBRAIN HARMONIZED (ROBUSTNESS CHECK)
# =============================================================================
def run_tdbrain_harmonized(df_full):
    print("\n[2/4] Generating TDBrain Harmonized Heatmap (Robustness Check)...")
    df = df_full[(df_full['Condition'] == 'EC') & (df_full['Group_Detailed'].isin(['TDBrain_Healthy', 'TDBrain_ChronicPain']))].copy()
    BANDS = ['Theta', 'Alpha', 'Beta', 'Gamma']
    
    for ch in CHANNELS:
        delta_col = get_col_name(df, ch, 'Delta')
        if delta_col:
            denom = (1.0 - df[delta_col]).clip(lower=0.01)
            for band in BANDS:
                col = get_col_name(df, ch, band)
                if col: df[f"{ch}_{band}_Norm"] = df[col] / denom

    t_matrix = np.zeros((len(CHANNELS), len(BANDS)))
    for i, ch in enumerate(CHANNELS):
        for j, band in enumerate(BANDS):
            col = f"{ch}_{band}_Norm"
            if col in df.columns:
                sub = df.groupby(['Subject', 'Group_Detailed'])[col].mean().reset_index()
                h = sub[sub['Group_Detailed'] == 'TDBrain_Healthy'][col]
                p = sub[sub['Group_Detailed'] == 'TDBrain_ChronicPain'][col]
                t_stat, _ = ttest_ind(p, h, nan_policy='omit')
                t_matrix[i, j] = t_stat
                
    plot_heatmap_generic(t_matrix, BANDS, "Fig 2: TDBrain Harmonized (Robustness Check)", "heatmap_2_tdbrain_harmonized.png")

# =============================================================================
# 3 & 4. MERGED ANALYSES (RAW & HARMONIZED)
# =============================================================================
def run_merged_analyses(df_full):
    print("\n[3/4 & 4/4] Generating Merged Heatmaps (Site Effect Proof)...")
    df = df_full[df_full['Condition'] == 'EC'].copy()
    
    df['New_Label'] = df['Group_Detailed'].apply(map_merged_groups)
    df = df[df['New_Label'] != 'Other'].copy()

    # --- RAW MERGED (With Delta) ---
    BANDS_RAW = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    t_mat_raw = np.zeros((len(CHANNELS), len(BANDS_RAW)))
    
    for i, ch in enumerate(CHANNELS):
        for j, band in enumerate(BANDS_RAW):
            col = get_col_name(df, ch, band)
            if col:
                sub = df.groupby(['Subject', 'New_Label'])[col].mean().reset_index()
                h = sub[sub['New_Label'] == 'Healthy_Combined'][col]
                p = sub[sub['New_Label'] == 'Pain_Combined'][col]
                t, _ = ttest_ind(p, h, equal_var=False, nan_policy='omit')
                t_mat_raw[i, j] = t
                
    plot_heatmap_generic(t_mat_raw, BANDS_RAW, "Fig 3: Merged RAW (Site Effect Evidence)", "heatmap_3_merged_raw.png")

    # --- HARMONIZED MERGED (Correction) ---
    BANDS_HARM = ['Theta', 'Alpha', 'Beta', 'Gamma']
    
    for ch in CHANNELS:
        delta_col = get_col_name(df, ch, 'Delta')
        if delta_col:
            denom = (1.0 - df[delta_col]).clip(lower=0.01)
            for band in BANDS_HARM:
                col = get_col_name(df, ch, band)
                if col: df[f"{ch}_{band}_Norm"] = df[col] / denom

    t_mat_harm = np.zeros((len(CHANNELS), len(BANDS_HARM)))
    for i, ch in enumerate(CHANNELS):
        for j, band in enumerate(BANDS_HARM):
            col = f"{ch}_{band}_Norm"
            if col in df.columns:
                sub = df.groupby(['Subject', 'New_Label'])[col].mean().reset_index()
                h = sub[sub['New_Label'] == 'Healthy_Combined'][col]
                p = sub[sub['New_Label'] == 'Pain_Combined'][col]
                t, _ = ttest_ind(p, h, equal_var=False, nan_policy='omit')
                t_mat_harm[i, j] = t
                
    plot_heatmap_generic(t_mat_harm, BANDS_HARM, "Fig 4: Merged Harmonized (Site Effect Correction)", "heatmap_4_merged_harmonized.png")

    # =========================================================================
    # EXTRACT VALUES FOR LATEX THESIS TEXT
    # =========================================================================
    print("\n" + "="*60)
    print("📝 EXACT T-VALUES FOR LATEX TEXT (Merged Harmonized)")
    print("="*60)
    for j, band in enumerate(BANDS_HARM):
        band_vals = []
        for i, ch in enumerate(CHANNELS):
            band_vals.append((ch, t_mat_harm[i, j]))
        
        # Sort by T-value descending
        band_vals.sort(key=lambda x: x[1], reverse=True)
        
        print(f"[{band.upper()}]")
        print("  Highest in Pain (Positive T): " + ", ".join([f"{c} (T={v:.2f})" for c, v in band_vals[:3]]))
        print("  Lowest in Pain  (Negative T): " + ", ".join([f"{c} (T={v:.2f})" for c, v in band_vals[-3:]]))
    print("="*60 + "\n")

if __name__ == "__main__":
    if DATA_FILE.exists():
        df_full = pd.read_csv(DATA_FILE)
        run_tdbrain_standard(df_full)
        run_tdbrain_harmonized(df_full)
        run_merged_analyses(df_full)
        print(f"🚀 Complete! Check the '{IMG_DIR.name}' folder for outputs.")
    else:
        print(f"❌ Data file not found: {DATA_FILE}")