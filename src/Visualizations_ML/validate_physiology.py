"""
=============================================================================
✅ DATA QUALITY VALIDATION: PHYSIOLOGICAL CHECK
=============================================================================
Objective:
    Generate statistics and figures for the "Data Quality Validation" section (4.X)
    of the thesis. This ensures the EEG data contains genuine physiological signals
    before proceeding to Machine Learning.

Validations:
    1. Alpha Blocking (Berger Effect):
       - Compares Global Alpha Power between Eyes Open (EO) and Eyes Closed (EC).
       - Expectation: Significant increase in Alpha power during EC.
       - Output: Figure 1 (2x2 Grid) + Statistical Report.

    2. Age Correlation (Healthy Aging):
       - Analyzes the correlation between Age and Global Alpha Power (in EC condition).
       - Expectation: Slight negative correlation (Alpha decreases with age).
       - Output: Figure 2 (2x2 Grid) + Statistical Report.

Input:
    - results/final_dataset.csv

Output:
    - results/figures/validation_grid_reactivity.png
    - results/figures/validation_grid_age.png
    - results/figures/validation_boxplot_groups.png
    - results/validation_stats_report.txt (T-test and Pearson correlation results)

Execution:
    python ./FM_thesis_ML/src/Visualizations_ML/validate_physiology.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, pearsonr
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
REPORT_FILE = RESULTS_DIR / "validation_stats_report.txt"

# Ensure output directory exists
IMG_DIR.mkdir(parents=True, exist_ok=True)

# The 4 Groups to validate
GROUPS_TO_CHECK = [
    'TDBrain_Healthy', 
    'TDBrain_ChronicPain', 
    'TDBrain_Unknown_NoIndication', 
    'External_CP'
]

# Use CHANNELS from config (Standard 10-20)
# CHANNELS = [...] <-- Removed hardcoded list

def log(f, text):
    """Helper to print text to console and write to file simultaneously."""
    print(text)
    if f: f.write(text + "\n")

def get_col_name(df, ch, band):
    """Retrieves column name, handling potential alias variations (e.g., T7 vs T3)."""
    col = f"{ch}_{band}"
    if col in df.columns: return col
    mapping = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    if ch in mapping:
        alt_col = f"{mapping[ch]}_{band}"
        if alt_col in df.columns: return alt_col
    return None

def calculate_global_alpha(df):
    """Calculates the mean Alpha power across all 20 channels."""
    alpha_cols = []
    for ch in CHANNELS:
        col = get_col_name(df, ch, 'Alpha')
        if col: alpha_cols.append(col)
    
    if not alpha_cols:
        return None
    return df[alpha_cols].mean(axis=1)

# ==============================================================================
# FIGURE 1: ALPHA REACTIVITY GRID (2x2)
# ==============================================================================
def create_reactivity_grid(df, f):
    
    print("📊 Generating Figure 1: Alpha Reactivity Grid (Berger Effect)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Calculate Global Alpha if missing
    if 'Global_Alpha' not in df.columns:
        df['Global_Alpha'] = calculate_global_alpha(df)

    for i, group in enumerate(GROUPS_TO_CHECK):
        ax = axes[i]
        
        # Filter Data for current group
        sub = df[df['Group_Detailed'] == group].copy()
        
        # Pivot to create separate columns for EC and EO
        # We need unique values per Subject and Condition
        df_react = sub.pivot_table(index='Subject', columns='Condition', values='Global_Alpha')

        if 'EC' not in df_react.columns or 'EO' not in df_react.columns:
            ax.text(0.5, 0.5, "Missing EC/EO Data", ha='center')
            continue

        # Keep only subjects with BOTH conditions
        df_paired = df_react.dropna(subset=['EC', 'EO'])
        
        if len(df_paired) > 1:
            # Paired T-Test
            stat, p_val = ttest_rel(df_paired['EC'], df_paired['EO'])
            mean_ec = df_paired['EC'].mean()
            mean_eo = df_paired['EO'].mean()
            increase = ((mean_ec - mean_eo) / mean_eo) * 100
            
            # Log results (Crucial for Thesis text)
            p_text = "< .001" if p_val < 0.001 else f"= {p_val:.3f}"
            log(f, f"[{group:<30}] Reactivity: N={len(df_paired)}, Increase=+{increase:.1f}%, p {p_text}")

            # Scatterplot
            ax.scatter(df_paired['EO'], df_paired['EC'], alpha=0.6, color='#1f77b4', edgecolor='k')
            
            # Identity line (x=y) -> Points above this line indicate EC > EO
            limit = max(df_paired['EO'].max(), df_paired['EC'].max()) * 1.1
            ax.plot([0, limit], [0, limit], 'r--', label='No Effect', alpha=0.7)
            
            # Styling
            short_name = group.replace("TDBrain_", "").replace("External_", "Ext. ")
            ax.set_title(f"{short_name}\n(Mean Increase: +{increase:.1f}%)", fontsize=10, fontweight='bold')
            
            # Only set labels on outer edges of the grid
            if i in [2, 3]: ax.set_xlabel("Eyes Open Alpha")
            if i in [0, 2]: ax.set_ylabel("Eyes Closed Alpha")
            
            ax.grid(True, alpha=0.2)
            
            # Display P-value on plot
            ax.text(0.05, 0.95, f"p {p_text}", transform=ax.transAxes, 
                    verticalalignment='top', fontsize=9, 
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(IMG_DIR / "validation_grid_reactivity.png", dpi=300)
    plt.close()

# ==============================================================================
# FIGURE 2: AGE CORRELATION GRID (2x2)
# ==============================================================================
def create_age_grid(df, f):
    print("📊 Generating Figure 2: Age Correlation Grid...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Only verify Age correlations in EC condition
    df_ec = df[df['Condition'] == 'EC'].copy()
    if 'Global_Alpha' not in df_ec.columns:
         df_ec['Global_Alpha'] = calculate_global_alpha(df_ec)

    colors = ['green', 'red', 'gray', 'orange']

    for i, group in enumerate(GROUPS_TO_CHECK):
        ax = axes[i]
        sub = df_ec[df_ec['Group_Detailed'] == group].copy()
        
        # Aggregate per Subject (take mean of epochs, first value of Age)
        if 'Age' in sub.columns:
            df_age = sub.groupby('Subject').agg({'Global_Alpha': 'mean', 'Age': 'first'}).dropna()

            if len(df_age) > 3:
                r, p = pearsonr(df_age['Age'], df_age['Global_Alpha'])
                
                # Log results
                p_text = "< .001" if p < 0.001 else f"= {p:.3f}"
                log(f, f"[{group:<30}] Age Corr:  N={len(df_age)}, r={r:.2f}, p {p_text}")

                # Regression Plot
                sns.regplot(data=df_age, x='Age', y='Global_Alpha', ax=ax, 
                            scatter_kws={'alpha':0.5, 'edgecolor':'k'}, 
                            line_kws={'color': colors[i]}, color=colors[i])
                
                short_name = group.replace("TDBrain_", "").replace("External_", "Ext. ")
                ax.set_title(f"{short_name}", fontsize=10, fontweight='bold')
                
                if i in [2, 3]: ax.set_xlabel("Age (Years)")
                if i in [0, 2]: ax.set_ylabel("Global Alpha (EC)")
                ax.grid(True, alpha=0.2)
                
                ax.text(0.95, 0.95, f"r={r:.2f}\np {p_text}", 
                        transform=ax.transAxes, ha='right', va='top', 
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "Not enough data", ha='center')
        else:
             ax.text(0.5, 0.5, "No Age Data", ha='center')

    plt.tight_layout()
    plt.savefig(IMG_DIR / "validation_grid_age.png", dpi=300)
    plt.close()

# ==============================================================================
# FIGURE 3: BOXPLOT COMPARISON
# ==============================================================================
def create_comparison_boxplot(df):
    print("📊 Generating Figure 3: Group Power Comparison...")
    
    # Filter EC condition
    subset = df[(df['Condition'] == 'EC') & (df['Group_Detailed'].isin(GROUPS_TO_CHECK))].copy()
    if 'Global_Alpha' not in subset.columns:
        subset['Global_Alpha'] = calculate_global_alpha(subset)

    # Aggregate per subject
    subset_agg = subset.groupby(['Subject', 'Group_Detailed'])['Global_Alpha'].mean().reset_index()
    
    label_map = {
        'TDBrain_Healthy': 'Healthy\n(TD)',
        'TDBrain_ChronicPain': 'Pain\n(TD)',
        'TDBrain_Unknown_NoIndication': 'Unknown\n(TD)',
        'External_CP': 'Pain\n(Ext)'
    }
    subset_agg['Label'] = subset_agg['Group_Detailed'].map(label_map)
    
    order_labels = [label_map[g] for g in GROUPS_TO_CHECK]

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=subset_agg, x='Label', y='Global_Alpha', hue='Label', legend=False, 
                order=order_labels, palette="Set2", showfliers=False)
    sns.stripplot(data=subset_agg, x='Label', y='Global_Alpha', order=order_labels, 
                  color='black', alpha=0.3, jitter=True)
    
    plt.title("Resting State Alpha Power Comparison (EC)")
    plt.ylabel("Global Relative Alpha Power")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "validation_boxplot_groups.png", dpi=300)
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
    else:
        df = pd.read_csv(DATA_FILE)
        
        # Open report file for writing
        with open(REPORT_FILE, 'w') as f:
            log(f, "=== VALIDATION REPORT ===")
            create_reactivity_grid(df, f)
            log(f, "\n=== AGE REPORT ===")
            create_age_grid(df, f)
        
        create_comparison_boxplot(df)
        print(f"\n✅ Validation Complete! Stats saved to: {REPORT_FILE.name}")