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

    2. Age Correlation (Healthy Aging - Global Power):
       - Analyzes the correlation between Age and Global Alpha Power (in EC condition).
       - Expectation: Slight negative correlation (Alpha decreases with age).
       - Output: Figure 2 (2x2 Grid) + Statistical Report.

    3. Group Comparison Boxplot:
       - Visualizes the distribution of resting-state Alpha across groups.
       - Output: Figure 3 (Boxplot).

    4. Age Correlation (Alpha Peak Frequency - APF):
       - Calculates the individual Alpha Peak Frequency directly from .npy files.
       - Correlates APF with Age to satisfy clinical aging markers.
       - Output: Figure 4 (Scatterplot) + Statistical Report.

Input:
    - results/final_dataset.csv
    - results/**/*_EC_cleaned.npy (For APF calculation)

Output:
    - results/figures/validation_grid_reactivity.png
    - results/figures/validation_grid_age.png
    - results/figures/validation_boxplot_groups.png
    - results/figures/validation_apf_age.png
    - results/validation_stats_report.txt (T-test and Pearson correlation results)

Execution:
    python ./FM_thesis_ML/src/Visualizations_Prep/validate_physiology.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, pearsonr
import os
import sys
import glob
import mne
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, FIGURES_DIR, CHANNELS, SFREQ

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR
REPORT_FILE = RESULTS_DIR / "validation_stats_report.txt"

# Search paths for APF calculation
NPY_SEARCH_PATH = os.path.join(str(RESULTS_DIR), "**", "*_EC_cleaned.npy")
POSTERIOR_CHANNELS = ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4']

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
            log(f, f"[{group:<30}] Reactivity: N={len(df_paired):<3}, Increase=+{increase:.1f}%, p {p_text}")

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
    print("\n📊 Generating Figure 2: Age Correlation Grid (Global Alpha)...")
    
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
                log(f, f"[{group:<30}] Age Corr:  N={len(df_age):<3}, r={r:>5.2f}, p {p_text}")

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
    print("\n📊 Generating Figure 3: Group Power Comparison...")
    
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
# FIGURE 4: ALPHA PEAK FREQUENCY (APF) AGE CORRELATION
# ==============================================================================
def create_apf_analysis(df, f):
    print("\n🧠 Generating Figure 4: Alpha Peak Frequency (APF) Extraction...")
    
    # Extract unique subjects and their metadata
    meta_df = df.drop_duplicates(subset=['Subject'])[['Subject', 'Age', 'Group_Detailed']].set_index('Subject')
    npy_files = glob.glob(NPY_SEARCH_PATH, recursive=True)
    
    # Filter to only subjects present in our final dataset
    valid_subjects = meta_df.index.astype(str).tolist()
    files_to_process = [file for file in npy_files if os.path.basename(file).split('_')[0] in valid_subjects]
    
    results = []
    
    # MNE Info object
    info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types='eeg')
    
    for file_path in tqdm(files_to_process, desc="Extracting APF"):
        sub_id = os.path.basename(file_path).split('_')[0]
        try:
            # Load 3D numpy array: (epochs, channels, times)
            data = np.load(file_path)
            epochs = mne.EpochsArray(data, info, verbose=False)
            
            # Compute PSD for Alpha range (7-13 Hz) on posterior channels
            psd = epochs.compute_psd(method='welch', fmin=7.0, fmax=13.0, picks=POSTERIOR_CHANNELS, verbose=False)
            psd_data, freqs = psd.get_data(return_freqs=True)
            
            # Average power across epochs and channels, find peak
            mean_psd = psd_data.mean(axis=(0, 1))
            peak_idx = np.argmax(mean_psd)
            
            results.append({
                'Subject': sub_id,
                'Age': meta_df.loc[sub_id, 'Age'],
                'Group_Detailed': meta_df.loc[sub_id, 'Group_Detailed'],
                'Alpha_Peak_Freq': freqs[peak_idx]
            })
        except Exception:
            continue

    df_apf = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    colors = ['green', 'red', 'gray', 'orange']
    
    for i, group in enumerate(GROUPS_TO_CHECK):
        group_data = df_apf[df_apf['Group_Detailed'] == group]
        if len(group_data) > 2:
            r, p = pearsonr(group_data['Age'], group_data['Alpha_Peak_Freq'])
            
            # Log results
            p_text = "< .001" if p < 0.001 else f"= {p:.3f}"
            log(f, f"[{group:<30}] APF Corr:  N={len(group_data):<3}, r={r:>5.2f}, p {p_text}")
            
            # Regression Plot
            sns.regplot(data=group_data, x='Age', y='Alpha_Peak_Freq', 
                        label=f"{group.replace('TDBrain_', '').replace('External_', 'Ext. ')} (r={r:.2f})", 
                        scatter_kws={'alpha':0.6}, line_kws={'color': colors[i]}, color=colors[i])

    plt.title("Age-Related Slowing of Alpha Peak Frequency (APF)")
    plt.xlabel("Age (Years)")
    plt.ylabel("Alpha Peak Frequency (Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "validation_apf_age.png", dpi=300)
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
            
            log(f, "\n=== AGE REPORT (GLOBAL ALPHA POWER) ===")
            create_age_grid(df, f)
            
            log(f, "\n=== AGE REPORT (ALPHA PEAK FREQUENCY) ===")
            create_apf_analysis(df, f)
        
        create_comparison_boxplot(df)
        print(f"\n✅ Validation Complete! Stats saved to: {REPORT_FILE.name}")