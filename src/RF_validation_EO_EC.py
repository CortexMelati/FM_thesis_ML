"""
=============================================================================
FEATURE VALIDATION: EYES OPEN vs. EYES CLOSED CLASSIFICATION
=============================================================================
Objective:
    Validate signal quality by classifying Eyes Open (EO) vs. Eyes Closed (EC)
    conditions across different dataset subsets.
    
    A high classification accuracy indicates that the EEG signal contains 
    genuine physiological information (specifically the Berger effect/Alpha blocking).

Datasets Evaluated:
    1. TDBrain Healthy (N=37)
    2. Chronic Pain Dataset (External) (N=62)
    3. TDBrain No Indication (Unknown) (N=155)
    4. Combined Dataset (All of the above)

    - Approximately 12 epochs per participant (N=254 total subjects).

Methodology:
    - Features: 20 Regional Features (4 Regions x 5 Frequency Bands).
    - Model: Random Forest (Optimized via GridSearch).
    - Validation: GroupKFold (5-splits), ensuring subjects are not mixed between train/test.

Outputs:
    - Epoch-level Accuracy per dataset.
    - Classification Reports (Precision, Recall, F1).
    - Feature Importance ranking (Occipital Alpha should be top).

Input: results/final_dataset.csv
Execution: python ./FM_thesis_ML/src/Visualizations_ML/RF_validation_EO_EC.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from config import RESULTS_DIR, FIGURES_DIR, BANDS

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR

# Ensure output directory exists
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Extract band names from config dictionary keys ['Delta', 'Theta', ...]
BAND_NAMES = list(BANDS.keys())

# Region Mapping: Average specific channels to create regional features
REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Central': ['C3', 'Cz', 'C4', 'T7', 'T8'], 
    'Parietal': ['P3', 'Pz', 'P4', 'P7', 'P8'], 
    'Occipital': ['O1', 'O2']
}

# 🎛️ HYPERPARAMETER GRID
PARAM_GRID = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_leaf': [1, 5]
}

def get_col_name(df, ch, band):
    """Retrieves column name, handling channel aliases (e.g., T7 vs T3)."""
    col = f"{ch}_{band}"
    if col in df.columns: return col
    mapping = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    if ch in mapping:
        alt = f"{mapping[ch]}_{band}"
        if alt in df.columns: return alt
    return None

def run_tuned_validation():
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    print("📜 STARTING EO vs. EC VALIDATION (FULL REPORT)...")
    df_all = pd.read_csv(DATA_FILE)

    # 1. Filter Data: Only EO/EC conditions and relevant groups
    relevant_groups = ['TDBrain_Healthy', 'External_CP', 'TDBrain_Unknown_NoIndication']
    df_all = df_all[
        (df_all['Condition'].isin(['EC', 'EO'])) & 
        (df_all['Group_Detailed'].isin(relevant_groups))
    ].copy()

    if df_all.empty:
        print("⚠️ Warning: No valid data found after filtering.")
        return

    # 2. Compute Regional Features
    

[Image of brain lobes and regions]

    feature_cols = []
    for reg_name, channels in REGIONS.items():
        for band in BAND_NAMES:
            cols = []
            for ch in channels:
                c = get_col_name(df_all, ch, band)
                if c: cols.append(c)
            
            if cols:
                # Calculate mean power for the region
                feat_name = f"{reg_name}_{band}"
                df_all[feat_name] = df_all[cols].mean(axis=1)
                feature_cols.append(feat_name)
    
    print(f"   Total Generated Features: {len(feature_cols)}")

    # 3. Define Datasets for Analysis
    datasets = {
        '1. TDBrain Healthy':      df_all[df_all['Group_Detailed'] == 'TDBrain_Healthy'],
        '2. External CP':          df_all[df_all['Group_Detailed'] == 'External_CP'],
        '3. TDBrain No Indication':df_all[df_all['Group_Detailed'] == 'TDBrain_Unknown_NoIndication'],
        '4. Combined Dataset':     df_all
    }

    # =========================================================================
    # VALIDATION LOOP
    # =========================================================================
    for ds_name, df in datasets.items():
        if df.empty: continue

        print("\n" + "="*60)
        print(f"📊 DATASET: {ds_name} (N_Sub={df['Subject'].nunique()})")
        print("="*60)

        X = df[feature_cols]
        y = df['Condition'].map({'EC': 0, 'EO': 1})
        groups = df['Subject']

        # Construct Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        # A. GRID SEARCH (Optimization)
        print("   🔍 Tuning Hyperparameters...")
        # Use 3-fold GroupKFold for speed during tuning
        gkf_tune = GroupKFold(n_splits=3) 
        
        grid = GridSearchCV(pipeline, PARAM_GRID, cv=gkf_tune, scoring='accuracy', n_jobs=-1)
        grid.fit(X, y, groups=groups)
        
        best_model = grid.best_estimator_
        print(f"      ✅ Best Parameters: {grid.best_params_}")

        # B. FINAL VALIDATION (Evaluation)
        # Use 5-fold GroupKFold for robust performance estimation
        gkf_final = GroupKFold(n_splits=5)
        y_pred = cross_val_predict(best_model, X, y, cv=gkf_final, groups=groups)

        # C. RESULTS REPORT
        print("\n   🎯 CLASSIFICATION REPORT:")
        print(classification_report(y, y_pred, target_names=['Eyes Closed (0)', 'Eyes Open (1)']))
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        print("   🧩 CONFUSION MATRIX:")
        print(f"      [TN, FP] = {cm[0]}")
        print(f"      [FN, TP] = {cm[1]}")
        
        # D. FEATURE IMPORTANCE
        best_model.fit(X, y) # Refit on full data to get importances
        imps = best_model.named_steps['rf'].feature_importances_
        indices = np.argsort(imps)[::-1]
        
        print("\n   🔝 TOP 3 FEATURES (Expectation: Occipital Alpha):")
        for i in range(min(3, len(indices))):
            print(f"      {feature_cols[indices[i]]:<20}: {imps[indices[i]]:.4f}")

    print("\n✅ Validation Complete. Ensure 'Occipital_Alpha' is among the top features.")

if __name__ == "__main__":
    run_tuned_validation()