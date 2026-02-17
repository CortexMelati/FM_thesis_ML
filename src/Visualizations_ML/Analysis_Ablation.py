"""
=============================================================================
🔍 ANALYSIS: FEATURE ABLATION (BAND IMPORTANCE)
=============================================================================
Objective:
    Determine which frequency band is crucial for classification performance.
    Method: 'Leave-One-Band-Out' (Ablation Study).
    
    1. Train with ALL bands (Baseline).
    2. Train with all bands EXCEPT Delta.
    3. Train with all bands EXCEPT Theta.
    ... and so on.

    Focus: Logistic Regression on Merged Dataset (EC condition).
    
    Output:
    - Barplot showing absolute accuracy per configuration.
    - Barplot showing the relative impact (gain/loss) of removing a band.

Execution:
    python ./FM_thesis_ML/src/Visualizations_ML/Analysis_Ablation.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, FIGURES_DIR, CHANNELS, BANDS

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR / "ablation_analysis"

IMG_DIR.mkdir(parents=True, exist_ok=True)

ALL_BANDS = list(BANDS.keys())

def get_features_excluding(df, exclude_band=None):
    cols = []
    if exclude_band is None:
        target_bands = ALL_BANDS
    else:
        target_bands = [b for b in ALL_BANDS if b != exclude_band]
    
    for ch in CHANNELS:
        for band in target_bands:
            candidates = [f"{ch}_{band}", f"{ch}_{band}_Rel", f"{ch}_{band}_Norm"] 
            for c in candidates:
                if c in df.columns:
                    cols.append(c)
                    break
    return cols

def run_ablation():
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    print("🚀 START FEATURE ABLATION ANALYSIS...")
    
    df = pd.read_csv(DATA_FILE)
    df = df[df['Condition'] == 'EC'].copy()
    
    valid = ['TDBrain_Healthy', 'TDBrain_Unknown_NoIndication', 'TDBrain_ChronicPain', 'External_CP']
    df = df[df['Group_Detailed'].isin(valid)].copy()
    
    if df.empty:
        print("⚠️ Warning: No valid data found for ablation study.")
        return

    df['Label'] = df['Group_Detailed'].apply(lambda g: 1 if 'Pain' in g or 'CP' in g else 0)
    y = df['Label']
    groups = df['Subject']
    
    # Using Logistic Regression config from ML_Main (Scenario 3)
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear', C=1, penalty='l1'))
    ])
    
    # --- EXACT SAME LOGIC AS ML_MAIN.PY ---
    n_minority = df.groupby('Label')['Subject'].nunique().min()
    actual_splits = min(8, n_minority)
    if actual_splits < 2: actual_splits = 2
    
    print(f"   Using StratifiedGroupKFold with {actual_splits} splits.")
    cv = StratifiedGroupKFold(n_splits=actual_splits)
    
    results = []
    ablation_scenarios = [None] + ALL_BANDS 
    
    for exclude in ablation_scenarios:
        scen_name = "Baseline (All)" if exclude is None else f"No {exclude}"
        print(f"   ⚙️ Testing: {scen_name}...", end=" ")
        
        feats = get_features_excluding(df, exclude)
        
        if not feats:
            print("-> Skipped (No features found)")
            continue

        X = df[feats]
        
        # Cross Validation
        scores = cross_val_score(pipeline, X, y, groups=groups, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        mean_score = np.mean(scores)
        
        print(f"-> BalAcc: {mean_score:.3f}")
        
        results.append({
            'Configuration': scen_name,
            'Removed Band': exclude if exclude else 'None',
            'Balanced Accuracy': mean_score
        })

    if not results:
        print("❌ No results generated.")
        return

    df_res = pd.DataFrame(results)
    
    baseline_rows = df_res[df_res['Removed Band'] == 'None']
    if baseline_rows.empty:
         print("❌ Baseline run failed, cannot calculate impact.")
         return

    baseline_score = baseline_rows['Balanced Accuracy'].values[0]
    df_res['Impact'] = df_res['Balanced Accuracy'] - baseline_score
    
    print("\n📊 ABLATION RESULTS:")
    print(df_res[['Configuration', 'Balanced Accuracy', 'Impact']])
    
    plt.figure(figsize=(10, 6))
    colors = ['grey' if x == 'None' else 'dodgerblue' for x in df_res['Removed Band']]
    sns.barplot(data=df_res, x='Configuration', y='Balanced Accuracy', palette=colors)
    plt.axhline(baseline_score, color='red', linestyle='--', label='Baseline Accuracy')
    plt.ylim(0.5, 0.85)
    plt.title("Impact of Removing Frequency Bands")
    plt.xticks(rotation=45)
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "ablation_absolute_scores.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df_impact = df_res[df_res['Removed Band'] != 'None'].copy()
    if not df_impact.empty:
        clrs = ['red' if x < 0 else 'green' for x in df_impact['Impact']]
        sns.barplot(data=df_impact, x='Configuration', y='Impact', palette=clrs)
        plt.axhline(0, color='black')
        plt.title("Performance Change when Removing a Band")
        plt.ylabel("Change in Accuracy (vs Baseline)")
        plt.text(0, 0.01, "Positive = Removed Noise", color='green', fontweight='bold')
        plt.text(0, -0.01, "Negative = Removed Signal", color='red', fontweight='bold')
        plt.tight_layout()
        plt.savefig(IMG_DIR / "ablation_impact.png")
        plt.close()

    df_res.to_csv(IMG_DIR / "ablation_results.csv", index=False)
    print(f"\n✅ Analysis Complete. Images saved to {IMG_DIR.name}")

if __name__ == "__main__":
    run_ablation()