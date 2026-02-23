"""
=============================================================================
📊 VISUALIZATION: SITE EFFECTS & HARMONIZATION
=============================================================================
Objective:
    1. RAW PROFILE: Visualize the substantial Delta offset in the External dataset (Site B),
       demonstrating the "Site Effect" caused by different scanner hardware/settings.
    2. HARMONIZED PROFILE: Demonstrate that after 'No-Delta Normalization', the spectral
       curves of the two datasets align significantly better.

Input:
    - results/final_dataset.csv

Output: 
    - results/figures/site_effect_profile_raw.png
    - results/figures/site_effect_profile_harmonized.png

Execution:
    python ./FM_thesis_ML/src/Visualizations_ML/visualize_site_effect.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, FIGURES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR

# Ensure output directory exists (handled by config, but good practice)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Frequency Bands
BANDS_RAW = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
BANDS_HARM = ['Theta', 'Alpha', 'Beta', 'Gamma']  # Delta excluded for harmonization

# Helper function to retrieve band columns
def get_band_cols(df, band):
    """Retrieve column names for a specific frequency band, excluding global/norm columns."""
    return [c for c in df.columns if c.endswith(f"_{band}") and 'Global' not in c and 'Norm' not in c]

def plot_site_effects():
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    print("📊 Starting Site Effect Visualization...")
    df = pd.read_csv(DATA_FILE)

    # Filter Data (EC condition + 3 Groups of interest)
    # Note: Ensure these group names match what is produced by 'final_prep.py'
    groups = ['TDBrain_Healthy', 'TDBrain_ChronicPain', 'External_CP']
    df = df[(df['Condition'] == 'EC') & (df['Group_Detailed'].isin(groups))].copy()

    if df.empty:
        print("⚠️ Warning: No data found for the specified groups/condition.")
        return

    # Rename groups for cleaner legend labels
    # We count N per group for the label
    counts = df.groupby('Group_Detailed')['Subject'].nunique()
    
    group_map = {
        'TDBrain_Healthy': f'TDBrain: Healthy (N={counts.get("TDBrain_Healthy", 0)})',
        'TDBrain_ChronicPain': f'TDBrain: Pain (N={counts.get("TDBrain_ChronicPain", 0)})',
        'External_CP': f'External: Pain (N={counts.get("External_CP", 0)})'
    }
    
    # Create a new column for plotting
    df['Dataset Group'] = df['Group_Detailed'].map(group_map)
    
    # Custom Color Palette
    # Note: Keys must match values in group_map
    custom_pal = {
        group_map['TDBrain_Healthy']: 'forestgreen', 
        group_map['TDBrain_ChronicPain']: 'firebrick', 
        group_map['External_CP']: 'darkorange'
    }

    # =========================================================================
    # PLOT 1: RAW DATA (INCLUDING DELTA)
    # =========================================================================
    print("   1. Generating Raw Spectral Profile...")
    plot_data_raw = []
    
    for band in BANDS_RAW:
        # Get all channel columns for this band
        cols = get_band_cols(df, band)
        
        # Calculate mean across channels (Global Average for this band)
        # axis=1 computes mean across columns for each row
        global_val = df[cols].mean(axis=1)
        
        temp = df[['Subject', 'Dataset Group']].copy()
        temp['Relative_Power'] = global_val
        temp['Frequency Band'] = band
        plot_data_raw.append(temp)
    
    # Combine all bands into one DataFrame
    melted_raw = pd.concat(plot_data_raw)
    melted_raw['Frequency Band'] = pd.Categorical(melted_raw['Frequency Band'], categories=BANDS_RAW, ordered=True)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot Lineplot with Error Bars (standard error of mean by default)
    try:
        sns.lineplot(data=melted_raw, x='Frequency Band', y='Relative_Power', hue='Dataset Group', 
                     style='Dataset Group', palette=custom_pal, markers=True, linewidth=2.5, err_style='bars')
        
        plt.title("Evidence of Site Effects: Raw Spectral Profile", fontsize=14, fontweight='bold')
        plt.ylabel("Global Relative Power", fontsize=12)
        
        # Annotation for Delta Offset (approximate position)
        plt.annotate('Site Effect (Delta Offset)', xy=(0, 0.45), xytext=(0.5, 0.55), # Adjust coordinates as needed based on actual plot
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='darkred')
        
        plt.tight_layout()
        save_path_raw = IMG_DIR / "site_effect_profile_raw.png"
        plt.savefig(save_path_raw)
        print(f"     -> Saved: {save_path_raw.name}")
    except Exception as e:
        print(f"❌ Error plotting Raw Profile: {e}")

    # =========================================================================
    # PLOT 2: HARMONIZED DATA (WITHOUT DELTA)
    # =========================================================================
    print("   2. Generating Harmonized Profile...")
    
    # Calculate Global Delta for normalization reference
    delta_cols = get_band_cols(df, 'Delta')
    # Pre-calculate global delta per row
    global_delta = df[delta_cols].mean(axis=1)
    
    plot_data_harm = []
    
    for band in BANDS_HARM:
        cols = get_band_cols(df, band)
        global_val = df[cols].mean(axis=1)
        
        # Harmonization Formula: New = Old / (1 - Delta)
        # This re-distributes the remaining power (1-Delta) to sum to 1
        denominator = (1.0 - global_delta).clip(lower=0.01) # avoid div by zero
        norm_val = global_val / denominator
        
        temp = df[['Subject', 'Dataset Group']].copy()
        temp['Relative_Power'] = norm_val
        temp['Frequency Band'] = band
        plot_data_harm.append(temp)
        
    melted_harm = pd.concat(plot_data_harm)
    melted_harm['Frequency Band'] = pd.Categorical(melted_harm['Frequency Band'], categories=BANDS_HARM, ordered=True)

    plt.figure(figsize=(10, 6))
    try:
        sns.lineplot(data=melted_harm, x='Frequency Band', y='Relative_Power', hue='Dataset Group', 
                     style='Dataset Group', palette=custom_pal, markers=True, linewidth=2.5, err_style='bars')
        
        plt.title("Harmonized Spectral Profile (Delta Excluded)", fontsize=14, fontweight='bold')
        plt.ylabel("Normalized Relative Power", fontsize=12)
        
        plt.tight_layout()
        save_path_harm = IMG_DIR / "site_effect_profile_harmonized.png"
        plt.savefig(save_path_harm)
        print(f"     -> Saved: {save_path_harm.name}")
        
    except Exception as e:
        print(f"❌ Error plotting Harmonized Profile: {e}")
    
    print("✅ Complete! Two plots saved in figures/.")

if __name__ == "__main__":
    plot_site_effects()