"""
=============================================================================
ML PIPELINE: EPOCH-BASED CLASSIFICATION BENCHMARK
=============================================================================

Overview:
    This script executes a comprehensive machine learning benchmark for 
    classifying Chronic Pain using Resting-State EEG spectral features.
    It evaluates multiple algorithms across different experimental scenarios
    using a Nested Cross-Validation scheme.

Experimental Conditions:
    1. EC (Eyes Closed)
    2. EO (Eyes Open)
    3. COMBINED (Stacked conditions)

Scenarios:
    1. TDBrain Pure:     Healthy Controls vs. Chronic Pain (Internal Dataset).
    2. TDBrain Extended: Includes subjects with informal indications.
    3. Merged Dataset:   TDBrain + External Chronic Pain Dataset.
                         (Note: Delta band is excluded in Scen 3 to mitigate site effects).

Models Evaluated:
    - Logistic Regression (LR): Baseline for linear separability.
    - XGBoost: Gradient boosting for structured data.
    - Random Forest (RF): Robust ensemble method.
    - LDA: Standard baseline in BCI/EEG literature.
    - SVM: Kernel-based classification for non-linear boundaries.
    - MLP: Multi-Layer Perceptron (Neural Network baseline).
    - Dummy: Random baseline for performance comparison.

Outputs:
    - final_benchmark_mega.csv:      Quantitative performance metrics (Accuracy, Sensitivity, etc.).
    - figures/detailed_metrics/:     ROC Curves, Confusion Matrices, and Barplots.
    - hyperparameter_report.txt:     Log of optimal hyperparameters found via GridSearch.

Execution:
    python ./FM_thesis_ML/src/ML_Main.py
=============================================================================
"""
from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from tqdm import tqdm  # Progress bar for execution tracking
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from config import RESULTS_DIR, FIGURES_DIR, CHANNELS, BANDS

# Pipeline & Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

# Metrics
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc, f1_score

# --- IMPORT MODELS ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR / "detailed_metrics"
REPORT_FILE = RESULTS_DIR / "hyperparameter_report.txt"
CSV_OUTPUT = RESULTS_DIR / "final_benchmark_mega.csv"

# Ensure output directory exists
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Feature Definitions
# Dynamically load from Config
ALL_BANDS = list(BANDS.keys())
NO_DELTA_BANDS = [b for b in ALL_BANDS if b != 'Delta'] # Excluded in Scenario 3
# CHANNELS imported from config

# =============================================================================
# 1. MODEL CONFIGURATION (ALGORITHMS)
# =============================================================================
MODELS = {
    'Dummy': {
        # Baseline: Stratified random guessing based on class distribution
        'model': DummyClassifier(strategy='stratified', random_state=42),
        'params': {} 
    },
    'LR': {
        # Logistic Regression
        # solver='liblinear' supports L1 (Lasso) and L2 (Ridge) regularization.
        'model': LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear'),
        'params': {
            'clf__C': [0.1, 1, 5, 10],   # Low C = Strong regularization (prevents overfitting), High C = Weak regularization
            'clf__penalty': ['l1', 'l2'] # L1 allows for feature selection
        }
    },
    'LDA': {
        # Linear Discriminant Analysis
        # Split into 2 configs because 'svd' solver does not support shrinkage.
        'model': LinearDiscriminantAnalysis(),
        'params': [
            {'clf__solver': ['svd'], 'clf__shrinkage': [None]},
            {'clf__solver': ['lsqr'], 'clf__shrinkage': [None, 'auto', 0.1, 0.5]} # Shrinkage improves performance in high-dimensional spaces
        ]
    },
    'SVM': {
        # Support Vector Machine (RBF Kernel)
        'model': SVC(kernel='rbf', probability=True, class_weight='balanced', cache_size=1000),
        'params': {
            'clf__C': [0.1, 1, 5, 10, 50, 75, 100], 
            'clf__gamma': ['scale', 0.01, 1, 10]
        }
    },
    'RF': {
        # Random Forest
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'clf__n_estimators': [100, 200, 400], 
            'clf__max_depth': [None, 10, 20, 40], # Shallow trees (e.g., 10) often generalize better on noisy EEG data
            'clf__min_samples_leaf': [2, 5, 10],
            'clf__criterion': ['gini', 'entropy', 'log_loss']
        }
    },
    'XGBoost': {
        # Gradient Boosting
        # eval_metric='logloss' prevents warnings.
        'model': XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=1),
        'params': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5],            
            'clf__learning_rate': [0.01, 0.1],    
            'clf__scale_pos_weight': [1, 3] # 1 = No correction, 3 = Prioritize minority class (Pain)
        }
    },
    'MLP': {
        # Multi-Layer Perceptron (Neural Net)
        'model': MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
        'params': {
            'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'clf__alpha': [0.0001, 0.01],         
            'clf__learning_rate_init': [0.001, 0.01]
        }
    }
}

# =============================================================================
# 2. DATA PREPARATION & PLOTTING FUNCTIONS
# =============================================================================
def get_feature_cols(df, bands):
    """
    Retrieves feature columns for the specified frequency bands.
    Searches for Raw, Normalized, and Relative power columns.
    """
    cols = []
    for ch in CHANNELS:
        for band in bands:
            # Candidates: standard, normalized, or relative power naming conventions
            candidates = [f"{ch}_{band}", f"{ch}_{band}_Norm", f"{ch}_{band}_Rel"]
            for c in candidates:
                if c in df.columns:
                    cols.append(c)
                    break
    return cols

def prepare_scenario_data(df_full, scenario_id):
    """
    Filters the dataset based on the selected Scenario configuration.
    
    Args:
        df_full (pd.DataFrame): The complete dataset.
        scenario_id (int): 1, 2, or 3.
        
    Returns:
        pd.DataFrame, list: Filtered dataframe and list of feature columns.
    """
    df = df_full.copy()
    
    if scenario_id == 1: # TDBrain Pure
        valid = ['TDBrain_Healthy', 'TDBrain_ChronicPain']
        df = df[df['Group_Detailed'].isin(valid)].copy()
        # Explicit mapping: Healthy=0, Pain=1
        df['Label'] = df['Group_Detailed'].map({'TDBrain_Healthy': 0, 'TDBrain_ChronicPain': 1})
        feats = get_feature_cols(df, ALL_BANDS)

    elif scenario_id == 2: # TDBrain Extended
        valid = ['TDBrain_Healthy', 'TDBrain_Unknown_NoIndication', 'TDBrain_ChronicPain']
        df = df[df['Group_Detailed'].isin(valid)].copy()
        df['Label'] = df['Group_Detailed'].apply(lambda g: 1 if g == 'TDBrain_ChronicPain' else 0)
        feats = get_feature_cols(df, ALL_BANDS)

    elif scenario_id == 3: # Merged (TDBrain + External)
        valid = ['TDBrain_Healthy', 'TDBrain_Unknown_NoIndication', 'TDBrain_ChronicPain', 'External_CP']
        df = df[df['Group_Detailed'].isin(valid)].copy()
        # Label all Pain groups as 1
        df['Label'] = df['Group_Detailed'].apply(lambda g: 1 if 'Pain' in g or 'CP' in g else 0)
        # Exclude Delta band to prevent site effects
        feats = get_feature_cols(df, NO_DELTA_BANDS)

    return df, feats

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
    # Use Path object
    plt.savefig(IMG_DIR / filename)
    plt.close()

def plot_roc_curve_combined(roc_data, title, filename):
    """Plots combined ROC curves for model comparison."""
    plt.figure(figsize=(8, 6))
    for item in roc_data:
        plt.plot(item['fpr'], item['tpr'], lw=2, label=f"{item['model']} (AUC = {item['auc']:.2f})")
    
    # 0.5 Baseline (Random Guess)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # Use Path object
    plt.savefig(IMG_DIR / filename)
    plt.close()

# =============================================================================
# 3. MAIN EXECUTION LOOP
# =============================================================================
def run_benchmark():
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    print("🚀 STARTING COMPREHENSIVE BENCHMARK (EC / EO / COMBINED)...")
    
    # Initialize Report File
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("🏆 HYPERPARAMETER REPORT - WINNING SETTINGS PER MODEL\n")
        f.write("====================================================\n\n")

    df_raw = pd.read_csv(DATA_FILE)
    
    conditions = ['EC', 'EO', 'COMBINED']
    scenarios = [1, 2, 3]
    
    results = []

    for cond in conditions:
        print(f"\n{'#'*60}")
        print(f"👀 CONDITION: {cond}")
        print(f"{'#'*60}")
        
        if cond == 'COMBINED':
            df_full = df_raw.copy()
        else:
            df_full = df_raw[df_raw['Condition'] == cond].copy()
            
        if df_full.empty: continue

        for scen_id in scenarios:
            print(f"\n 🌍 SCENARIO {scen_id} ({cond})")
            print(f" {'-'*30}")
            
            with open(REPORT_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n--- {cond} | SCENARIO {scen_id} ---\n")

            df_model, features = prepare_scenario_data(df_full, scen_id)
            X = df_model[features]
            y = df_model['Label']
            groups = df_model['Subject']
            
            print(f"      Epochs: {len(y)} | Class Balance: {y.value_counts().to_dict()}")

            roc_data_list = []
            
            # --- OUTER LOOP (Validation) ---
            # StratifiedGroupKFold ensures subjects are not split between train/test
            outer_cv = StratifiedGroupKFold(n_splits=8)
            
            for model_name, config in MODELS.items():
                # RobustScaler is used in Scenario 3 to handle outliers from merged datasets
                scaler = RobustScaler() if scen_id == 3 else StandardScaler()
                pipeline = Pipeline([('scaler', scaler), ('clf', config['model'])])

                # --- INNER LOOP (Hyperparameter Tuning) ---
                # n_splits=3 to prevent crashes on smaller subsets
                inner_cv = StratifiedGroupKFold(n_splits=3) 
                grid = GridSearchCV(pipeline, config['params'], cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1)

                y_true_all = []
                y_pred_all = []
                y_proba_all = []
                fold_bal_accs = []
                fold_best_params = [] 
                
                # Progress Bar setup
                desc_text = f"      Training {model_name}"
                pbar = tqdm(outer_cv.split(X, y, groups), total=8, desc=desc_text, leave=False)
                
                try:
                    for train_idx, test_idx in pbar:
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        groups_train = groups.iloc[train_idx]
                        
                        # Safety check: skip fold if class distribution is insufficient
                        if len(np.unique(groups_train)) < 3: continue

                        # 1. Train & Tune (Inner Loop)
                        grid.fit(X_train, y_train, groups=groups_train)
                        
                        # 2. Record best parameters
                        best_params = grid.best_params_
                        fold_best_params.append(str(best_params))

                        # 3. Validate on Test Set (Outer Loop)
                        best_model = grid.best_estimator_
                        y_pred = best_model.predict(X_test)
                        
                        # Get probabilities for ROC curve
                        if hasattr(best_model, "predict_proba"):
                            y_prob = best_model.predict_proba(X_test)[:, 1]
                        elif hasattr(best_model, "decision_function"):
                            y_prob = best_model.decision_function(X_test)
                        else:
                            y_prob = np.zeros_like(y_pred, dtype=float)

                        y_true_all.extend(y_test)
                        y_pred_all.extend(y_pred)
                        y_proba_all.extend(y_prob)
                        
                        # Calculate Balanced Accuracy for this fold
                        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
                        spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
                        fold_bal_accs.append((recall + spec) / 2)
                    
                    if not fold_bal_accs: continue

                    # Log the most frequent best hyperparameters across folds
                    if fold_best_params:
                        most_common = Counter(fold_best_params).most_common(1)
                        winner_params = most_common[0][0]
                        winner_count = most_common[0][1]
                        
                        with open(REPORT_FILE, "a", encoding="utf-8") as f:
                            f.write(f"[{model_name}] Winner ({winner_count}/8 folds): {winner_params}\n")

                    # Calculate aggregated metrics
                    mean_bal_acc = np.mean(fold_bal_accs)
                    f1 = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
                    sens = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
                    spec = recall_score(y_true_all, y_pred_all, pos_label=0, zero_division=0)
                    
                    tqdm.write(f"      ✅ {model_name}: BalAcc: {mean_bal_acc:.3f} | Sens: {sens:.3f} | Spec: {spec:.3f}")
                    
                    # Generate Confusion Matrix
                    cm_fname = f"cm_{cond}_scen{scen_id}_{model_name}.png"
                    plot_confusion_matrix(y_true_all, y_pred_all, f"CM: {model_name} ({cond}-Scen {scen_id})", cm_fname)
                    
                    # Compute ROC AUC
                    try:
                        fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
                        roc_auc = auc(fpr, tpr)
                    except: fpr, tpr, roc_auc = [0], [0], 0.5
                    
                    roc_data_list.append({'model': model_name, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

                    results.append({
                        'Condition': cond,
                        'Scenario': f"Scenario {scen_id}",
                        'Model': model_name,
                        'Balanced Accuracy': mean_bal_acc,
                        'Sensitivity': sens,
                        'Specificity': spec,
                        'F1-Score': f1,
                        'ROC AUC': roc_auc
                    })

                except Exception as e:
                    tqdm.write(f"      ❌ FAILED {model_name}: {e}")
                    continue

            # Plot Combined ROC curves for this scenario
            plot_roc_curve_combined(roc_data_list, f"ROC: {cond} - Scenario {scen_id}", f"roc_{cond}_scen{scen_id}_comparison.png")

    # Save final results and create summary plots
    if results:
        res_df = pd.DataFrame(results)
        print("\n📊 FINAL MEGA BENCHMARK RESULTS (Preview):")
        print(res_df.pivot_table(index=['Condition', 'Scenario'], columns='Model', values='Balanced Accuracy'))
        
        res_df.to_csv(CSV_OUTPUT, index=False)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=res_df, x='Model', y='Balanced Accuracy', hue='Scenario', palette='Set2')
        plt.axhline(0.5, color='red', linestyle='--', label='Random Chance (0.5)')
        plt.ylim(0.4, 0.9)
        plt.title("Epoch-Level Classification Performance (Across all conditions)")
        plt.legend()
        plt.savefig(IMG_DIR / "benchmark_barplot_mega.png")
        
        print(f"\n✅ All results saved in: {IMG_DIR}")
        print(f"✅ Hyperparameter report: {REPORT_FILE}")

if __name__ == "__main__":
    run_benchmark()