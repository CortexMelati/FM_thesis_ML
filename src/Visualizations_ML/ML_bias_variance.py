"""
=============================================================================
🔍 ML INSPECTION: BIAS/VARIANCE & FEATURE DIRECTIONALITY
=============================================================================
Objective:
1. Bias vs. Variance Assessment: 
   Compare model performance on Training sets vs. Test sets.
   - Large Gap = Overfitting (High Variance).
   - Both Low  = Underfitting (High Bias).

2. Feature Directionality (Coefficients):
   - Identify which spectral features drive the prediction of 'Chronic Pain'.
   - Red Bars  = High feature value correlates with PAIN.
   - Blue Bars = High feature value correlates with HEALTHY.

Focus: Scenario 3 (Merged Dataset), No-Delta Features, Logistic Regression.

Execution:
    python ./FM_thesis_ML/src/Visualizations_ML/ML_bias_variance.py
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

from config import RESULTS_DIR, FIGURES_DIR, CHANNELS

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = RESULTS_DIR / "final_dataset.csv"
IMG_DIR = FIGURES_DIR

IMG_DIR.mkdir(parents=True, exist_ok=True)

NO_DELTA_BANDS = ['Theta', 'Alpha', 'Beta', 'Gamma'] 

def get_feature_cols(df):
    cols = []
    for ch in CHANNELS:
        for band in NO_DELTA_BANDS:
            candidates = [f"{ch}_{band}", f"{ch}_{band}_Norm", f"{ch}_{band}_Rel"]
            for c in candidates:
                if c in df.columns:
                    cols.append(c)
                    break 
    return cols

def prepare_data(df_full):
    print("   🛠️  Preparing Scenario 3: Merged (No Delta)...")
    df = df_full[df_full['Condition'] == 'EC'].copy()
    valid = ['TDBrain_Healthy', 'TDBrain_Unknown_NoIndication', 'TDBrain_ChronicPain', 'External_CP']
    df = df[df['Group_Detailed'].isin(valid)].copy()
    
    # Labeling (1=Pain, 0=Healthy)
    df['Label'] = df['Group_Detailed'].apply(lambda g: 1 if 'Pain' in g or 'CP' in g else 0)
    
    feats = get_feature_cols(df)
    return df, feats

def run_inspection():
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return
        
    print("🚀 START MODEL INSPECTION (BIAS/VARIANCE & COEFFICIENTS)...")
    
    df_full = pd.read_csv(DATA_FILE)
    df, features = prepare_data(df_full)
    
    X = df[features]
    y = df['Label']
    groups = df['Subject']
    
    # ---------------------------------------------------------
    # 1. BIAS / VARIANCE CHECK (Train vs Test Score)
    # ---------------------------------------------------------
    print("\n📊 1. Assessing Bias & Variance (Train vs. Test Performance)...")
    
    # Using Logistic Regression config from ML_Main (Scenario 3)
    model = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced', solver='liblinear'))
    ])
    
    # --- EXACT SAME LOGIC AS ML_MAIN.PY ---
    n_minority = df.groupby('Label')['Subject'].nunique().min()
    actual_splits = min(8, n_minority)
    if actual_splits < 2: actual_splits = 2
    
    print(f"      Using StratifiedGroupKFold with {actual_splits} splits.")
    cv = StratifiedGroupKFold(n_splits=actual_splits)
    
    scores = cross_validate(model, X, y, groups=groups, cv=cv, 
                            scoring=['balanced_accuracy', 'recall'], 
                            return_train_score=True)
    
    train_acc = np.mean(scores['train_balanced_accuracy'])
    test_acc = np.mean(scores['test_balanced_accuracy'])
    gap = train_acc - test_acc
    
    print(f"      Train Accuracy: {train_acc:.3f}")
    print(f"      Test Accuracy:  {test_acc:.3f}")
    print(f"      Gap (Variance): {gap:.3f}")
    
    if gap > 0.10:
        print("      ⚠️  High Variance (Overfitting). The model is memorizing the training data.")
    elif train_acc < 0.60:
        print("      ⚠️  High Bias (Underfitting). The model fails to capture the underlying patterns.")
    else:
        print("      ✅ Good Fit! The model generalizes well to unseen data.")

    plt.figure(figsize=(6, 5))
    plt.bar(['Train Score', 'Test Score'], [train_acc, test_acc], color=['#d3d3d3', '#4e79a7'])
    plt.ylim(0.5, 1.0)
    plt.ylabel("Balanced Accuracy")
    plt.title("Bias-Variance Check\n(Small gap indicates good generalization)")
    
    save_path_bias = IMG_DIR / "bias_variance_check.png"
    plt.savefig(save_path_bias)
    plt.close()

    # ---------------------------------------------------------
    # 2. FEATURE COEFFICIENTS
    # ---------------------------------------------------------
    print("\n🔍 2. Extracting Coefficients (Feature Directionality)...")
    model.fit(X, y)
    coefs = model.named_steps['clf'].coef_[0]
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coef', ascending=False).head(20) 
    coef_df['Color'] = coef_df['Coefficient'].apply(lambda x: 'red' if x > 0 else 'blue')
    
    plt.figure(figsize=(10, 8))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=coef_df['Color'])
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel("Coefficient Value (Impact on Log-Odds)")
    plt.title("Top 20 Features Directionality")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    save_path_coef = IMG_DIR / "feature_coefficients_direction.png"
    plt.savefig(save_path_coef)
    
    print(f"✅ Inspection Complete. Plots saved in: {IMG_DIR.name}")

if __name__ == "__main__":
    run_inspection()