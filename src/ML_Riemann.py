"""
=============================================================================
🧠 ML RIEMANN: DELTA vs. NO-DELTA COMPARISON
=============================================================================
Objective:
Compare the performance of Riemannian Geometry classification using comprehensive metrics.
This script tests whether high classification accuracy is driven by 'Site Effects' 
(specifically the Delta band) or genuine physiological signals.

Configurations:
1. With Delta (Broadband 1-100 Hz) -> May contain scanner-specific noise/artifacts.
2. No Delta (High-pass > 4 Hz)    -> Physiologically focused (Theta/Alpha/Beta/Gamma).

Metrics Evaluated:
- Sensitivity (Recall Pain): The ability to correctly identify chronic pain patients.
- Specificity (Recall Healthy): The ability to correctly identify healthy controls.
- Balanced Accuracy, F1-Score, ROC AUC.
- Confusion Matrices & Classification Reports.

Execution:
    python ./FM_thesis_ML/src/ML_Riemann.py
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import mne 
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from config import RESULTS_DIR, FIGURES_DIR, TDBRAIN_DIR, SFREQ

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                             classification_report, confusion_matrix, roc_curve, auc)

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR / "riemann_comparison_full"

# Ensure output directory exists
IMG_DIR.mkdir(parents=True, exist_ok=True)

# File containing IDs of subjects with 'Unknown' status but 'NaN' indication
UNKNOWN_FILTER_FILE = TDBRAIN_DIR / "TDBRAIN_participants_UNKNOWN_NaNs.xlsx"

# Search paths for the cleaned .npy files (Where preprocess_pipeline.py saved them)
# Converting to strings for glob compatibility
SEARCH_PATHS = [
    str(RESULTS_DIR / "TDBrain" / "healthy"),
    str(RESULTS_DIR / "TDBrain" / "chronicpain"),
    str(RESULTS_DIR / "TDBrain" / "unknown"),      
    str(RESULTS_DIR / "TDBrain" / "unknown_nans"), # Added based on pipeline logic
    str(RESULTS_DIR / "chronicpain"),
]

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================
def get_no_indication_subjects():
    """
    Retrieves a list of subject IDs classified as 'Unknown' with no specific indication.
    Used for filtering subjects during data loading.
    """
    if not UNKNOWN_FILTER_FILE.exists(): return []
    try:
        df = pd.read_excel(UNKNOWN_FILTER_FILE)
        subject_ids = df.iloc[:, 0].astype(str).tolist()
        return [s if s.startswith('sub-') else f"sub-{s}" for s in subject_ids]
    except: return []

def find_npy_file(subject_id, condition):
    """
    Locates the .npy file for a specific subject and condition within the search paths.
    """
    filename = f"{subject_id}_{condition}_cleaned.npy"
    for root_path in SEARCH_PATHS:
        # Use recursive glob to find file
        matches = glob.glob(os.path.join(root_path, "**", filename), recursive=True)
        if matches: return matches[0]
    return None

def apply_filter(epochs_data, l_freq):
    """
    Applies a high-pass filter to the epoch data using MNE.
    
    Args:
        epochs_data (np.ndarray): The raw epoch data.
        l_freq (float): Low cut-off frequency (e.g., 4.0 Hz).
        
    Returns:
        np.ndarray: Filtered data.
    """
    print(f"      🧹 Filtering data (High-pass > {l_freq} Hz)...")
    # verbose=False to keep the output clean
    filtered_data = mne.filter.filter_data(epochs_data, SFREQ, l_freq=l_freq, h_freq=None, verbose=False)
    return filtered_data

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generates and saves a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', cbar=False,
                xticklabels=['Pred: HC', 'Pred: Pain'],
                yticklabels=['True: HC', 'True: Pain'])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename)
    plt.close()

def plot_roc_curve_combined(roc_data, title, filename):
    """Plots multiple ROC curves on a single figure for comparison."""
    plt.figure(figsize=(8, 6))
    for item in roc_data:
        plt.plot(item['fpr'], item['tpr'], lw=2, label=f"{item['label']} (AUC = {item['auc']:.2f})")
    
    # Diagonal line (random guess)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename)
    plt.close()

# =============================================================================
# 2. DATA LOADING (RAW EPOCHS)
# =============================================================================
def load_raw_epochs(condition='EC'):
    """
    Loads raw .npy epoch files for all valid subjects based on the master CSV.
    """
    print(f"🔍 Loading raw .npy files for condition: {condition}...")
    
    if not CSV_FILE.exists():
        print(f"❌ Error: CSV file not found at {CSV_FILE}")
        return None, None, None

    df = pd.read_csv(CSV_FILE)
    no_indication_ids = get_no_indication_subjects()
    
    # Filter Logic: Select valid subjects
    def is_valid_subject(row):
        group = row['Group_Detailed']
        sub_id = row['Subject']
        if group == 'TDBrain_Healthy': return True
        if group in ['TDBrain_ChronicPain', 'External_CP']: return True
        if group == 'TDBrain_Unknown_NoIndication' or 'Unknown' in group:
            if sub_id in no_indication_ids: return True
        return False

    df = df[df.apply(is_valid_subject, axis=1)].copy()
    subjects = df['Subject'].unique()

    epoch_list = []
    labels = []
    groups = [] 
    
    for sub in subjects:
        # Get group for first occurrence of subject
        group_det = df[df['Subject'] == sub]['Group_Detailed'].iloc[0]
        # Label: 1 = Pain, 0 = Healthy
        label = 1 if group_det in ['TDBrain_ChronicPain', 'External_CP'] else 0
        
        file_path = find_npy_file(sub, condition)
        if file_path and os.path.exists(file_path):
            try:
                # Load epochs: (n_epochs, n_channels, n_times)
                data = np.load(file_path)
                epoch_list.append(data)
                labels.extend([label] * data.shape[0])
                groups.extend([sub] * data.shape[0])
            except: pass

    if not epoch_list: 
        print("❌ No epochs loaded.")
        return None, None, None

    X_raw = np.concatenate(epoch_list, axis=0)
    y = np.array(labels)
    groups = np.array(groups)
    
    print(f"✅ Loaded {X_raw.shape[0]} epochs from {len(np.unique(groups))} subjects.")
    return X_raw, y, groups

# =============================================================================
# 3. MAIN LOOP
# =============================================================================
def run_comparison_full():
    
    print("🚀 START RIEMANN FULL COMPARISON (SENSITIVITY/SPECIFICITY)...")
    
    scenarios = ['EC', 'EO']
    
    # Compare two settings: Broadband vs. High-Pass (No Delta)
    filter_settings = [
        {'name': 'With Delta (Broadband)', 'filter_hz': None}, 
        {'name': 'No Delta (> 4Hz)',       'filter_hz': 4.0}
    ]
    
    results = []
    
    for scen in scenarios:
        print(f"\n{'='*60}")
        print(f"🌍 SCENARIO: {scen}")
        print(f"{'='*60}")
        
        # 1. Load Raw Data
        X_raw, y, groups = load_raw_epochs(scen)
        if X_raw is None: continue
        
        roc_data_list = []

        for setting in filter_settings:
            name = setting['name']
            f_hz = setting['filter_hz']
            
            print(f"\n   ⚙️ Processing: {name}")
            
            # 2. Apply Filter (if required)
            if f_hz is not None:
                X_curr = apply_filter(X_raw.copy(), l_freq=f_hz)
            else:
                X_curr = X_raw 
            
            # 3. Compute Covariance Matrices (OAS Estimator)
            print("      Calculating Covariances...")
            covs = Covariances(estimator='oas').transform(X_curr)
            
            # 4. Construct Pipeline
            # Tangent Space -> Scaling -> Logistic Regression
            pipe = Pipeline([
                ('ts', TangentSpace()), 
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='newton-cholesky')) 
                # Note: 'newton-cholesky' solver handles Riemannian features robustly
            ])
            
            # 5. Cross Validation & Metrics Collection
            outer_cv = StratifiedGroupKFold(n_splits=8)
            
            y_true_all = []
            y_pred_all = []
            y_proba_all = []
            
            print("      Running Cross-Validation...")
            for train_idx, test_idx in outer_cv.split(covs, y, groups):
                X_train, X_test = covs[train_idx], covs[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Fit & Predict
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_prob = pipe.predict_proba(X_test)[:, 1]
                
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
                y_proba_all.extend(y_prob)
            
            # --- CALCULATE METRICS ---
            # Sensitivity = Recall for Class 1 (Pain)
            sens = recall_score(y_true_all, y_pred_all, pos_label=1)
            
            # Specificity = Recall for Class 0 (Healthy)
            spec = recall_score(y_true_all, y_pred_all, pos_label=0)
            
            bal_acc = (sens + spec) / 2
            f1 = f1_score(y_true_all, y_pred_all, pos_label=1)
            
            fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
            roc_auc = auc(fpr, tpr)
            
            # --- OUTPUT TO CONSOLE ---
            print(f"\n      📊 RESULTS: {name}")
            print(f"      Sensitivity (Pain):  {sens:.3f}")
            print(f"      Specificity (HC):    {spec:.3f}")
            print(f"      Balanced Accuracy:   {bal_acc:.3f}")
            print(f"      F1-Score:            {f1:.3f}")
            print(f"      ROC AUC:             {roc_auc:.3f}")
            
            print("\n      --- Classification Report ---")
            print(classification_report(y_true_all, y_pred_all, target_names=['Healthy', 'Pain']))
            
            # --- PLOTS ---
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(">", "gt")
            
            # Confusion Matrix
            cm_fname = f"cm_{scen}_{safe_name}.png"
            plot_confusion_matrix(y_true_all, y_pred_all, f"CM: {scen} - {name}", cm_fname)
            
            # Collect ROC Data
            roc_data_list.append({'label': name, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

            # Store in Results List
            results.append({
                'Scenario': scen,
                'Filter': name,
                'Sensitivity': sens,
                'Specificity': spec,
                'Balanced Accuracy': bal_acc,
                'F1-Score': f1,
                'ROC AUC': roc_auc
            })

        # Plot ROC Combined (Delta vs No Delta)
        plot_roc_curve_combined(roc_data_list, f"Impact of Delta Band - {scen}", f"roc_compare_delta_{scen}.png")

    # Final Table
    if results:
        df_res = pd.DataFrame(results)
        print("\n📊 FINAL COMPARISON TABLE:")
        print(df_res)
        
        df_res.to_csv(IMG_DIR / "riemann_comparison_full_results.csv", index=False)
        print(f"\n✅ Done. Results saved to {IMG_DIR}")

if __name__ == "__main__":
    run_comparison_full()