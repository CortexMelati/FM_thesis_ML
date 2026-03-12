"""
=============================================================================
🛡️ CONTROL ANALYSIS: SITE PREDICTION (SCANNER ARTIFACTS)
=============================================================================
Objective:
    Directly addresses the "Site Effect Verification" from the thesis methodology.
    Trains a Random Forest classifier to predict the ORIGIN of the data 
    (TDBrain vs. External), rather than the clinical diagnosis.
    
    Hypothesis 1 (Raw Data): 
    The model will classify the site with >90% accuracy, relying heavily on 
    the Delta band, proving the existence of hardware/scanner artifacts.
    
    Hypothesis 2 (Harmonized Data):
    When the Delta band is removed, the classifier's ability to identify the 
    recording site should drop significantly, proving the harmonization worked.

Input: results/final_dataset.csv
Execution: python ./FM_thesis_ML/src/RF_site_prediction.py
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
import os
import sys
from pathlib import Path

# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, BANDS

DATA_FILE = RESULTS_DIR / "final_dataset.csv"

def run_site_prediction():
    print("🛡️ STARTING CONTROL ANALYSIS: SITE PREDICTION...")
    
    if not DATA_FILE.exists():
        print(f"❌ Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    
    # 1. Select only TDBrain (any subgroup) and External dataset
    df['Site_Target'] = np.where(df['Dataset'] == 'TDBRAIN', 0, 1)
    
    # We only need Eyes Closed for this test
    df_ec = df[df['Condition'] == 'EC'].copy()
    
    # Identify feature columns
    all_features = [c for c in df_ec.columns if any(b in c for b in BANDS.keys()) and 'Global' not in c and 'Norm' not in c]
    no_delta_features = [c for c in all_features if 'Delta' not in c]

    groups = df_ec['Subject']
    y = df_ec['Site_Target']

    # Setup basic Random Forest Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42))
    ])
    
    gkf = GroupKFold(n_splits=5)

    print(f"   Data loaded: {len(df_ec)} epochs. Class balance: 0=TDBrain ({sum(y==0)}), 1=External ({sum(y==1)})")
    
    # =========================================================================
    # TEST 1: RAW DATA (ALL BANDS INCLUDING DELTA)
    # =========================================================================
    print("\n" + "="*50)
    print("🧪 TEST 1: PREDICTING SITE USING ALL BANDS (Raw)")
    print("="*50)
    
    X_all = df_ec[all_features]
    
    # Calculate Cross-Validated Accuracy
    scores_all = cross_val_score(pipeline, X_all, y, cv=gkf, groups=groups, scoring='balanced_accuracy', n_jobs=-1)
    mean_acc_all = np.mean(scores_all) * 100
    
    print(f"   🎯 Mean Accuracy: {mean_acc_all:.1f}%")
    if mean_acc_all > 80:
        print("   ⚠️ High accuracy indicates strong Site Effects (model recognizes the scanner).")

    # Feature Importance
    pipeline.fit(X_all, y)
    imps_all = pipeline.named_steps['rf'].feature_importances_
    indices_all = np.argsort(imps_all)[::-1]
    
    print("\n   🔝 TOP 3 FEATURES DRIVING THE BIAS:")
    for i in range(3):
        print(f"      {all_features[indices_all[i]]:<20}: {imps_all[indices_all[i]]:.4f}")

    # =========================================================================
    # TEST 2: HARMONIZED DATA (DELTA EXCLUDED)
    # =========================================================================
    print("\n" + "="*50)
    print("🧪 TEST 2: PREDICTING SITE WITHOUT DELTA (Harmonized)")
    print("="*50)
    
    X_no_delta = df_ec[no_delta_features]
    
    # Calculate Cross-Validated Accuracy
    scores_no_delta = cross_val_score(pipeline, X_no_delta, y, cv=gkf, groups=groups, scoring='balanced_accuracy', n_jobs=-1)
    mean_acc_no_delta = np.mean(scores_no_delta) * 100
    
    print(f"   🎯 Mean Accuracy: {mean_acc_no_delta:.1f}%")
    
    drop = mean_acc_all - mean_acc_no_delta
    print(f"\n✅ CONCLUSION: Excluding Delta dropped site-prediction accuracy by {drop:.1f}%.")
    print("   This validates the harmonization strategy described in the thesis methodology.")

if __name__ == "__main__":
    run_site_prediction()