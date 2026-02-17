


# Machine Learning & Analysis Pipeline

This document details the scripts used for statistical validation, model training, and feature analysis.

## 1. Main Benchmark (`ML_Main.py`)

**Script:** `src/ML_Main.py`

This script performs the comprehensive evaluation of standard Machine Learning models on spectral power features. It utilizes a **Nested Cross-Validation** scheme to prevent data leakage and ensure robust performance estimates.

### Experimental Design

The benchmark iterates through a $3 \times 3$ matrix:

| **Condition**             | **Scenarios**                                 |
| :------------------------------ | :-------------------------------------------------- |
| **1. EC** (Eyes Closed)   | **1. TDBrain Pure:** Healthy vs. Pain        |
| **2. EO** (Eyes Open)     | **2. TDBrain Extended:** + Unknown Indication |
| **3. COMBINED** (Stacked) | **3. Merged:** TDBrain + External CP          |

> **Note:** In Scenario 3 (Merged), the **Delta Band** is explicitly removed to mitigate scanner-specific noise (site effects).

### Models Evaluated

| Model                         | Type       | Rationale                                                   |
| :---------------------------- | :--------- | :---------------------------------------------------------- |
| **Logistic Regression** | Linear     | High interpretability; Baseline for linear separability.    |
| **LDA**                 | Linear     | Standard baseline in BCI/EEG literature.                    |
| **SVM (RBF)**           | Non-Linear | Captures complex boundaries; robust in high dimensions.     |
| **Random Forest**       | Ensemble   | Robust against overfitting; handles noise well.             |
| **XGBoost**             | Ensemble   | State-of-the-art for tabular data; handles class imbalance. |
| **MLP**                 | Neural Net | Benchmarking simple Deep Learning capacity.                 |
| **Dummy**               | Baseline   | Theoretical lower bound (Random guessing).                  |

### Output

* `results/final_benchmark_mega.csv`: Raw metrics for all folds.
* `results/hyperparameter_report.txt`: Best parameters found via GridSearch.
* `figures/detailed_metrics/`: ROC Curves and Barplots.

---

## 2. Riemannian Geometry (`ML_Riemann.py`)

**Script:** `src/ML_Riemann.py`

This pipeline tests a geometric approach to classification. Instead of using pre-calculated power vectors, it operates directly on the covariance matrices of the epochs using **Tangent Space Mapping (TSM)**.

* **Goal:** To determine if the high performance in the Merged dataset is driven by physiology or site-specific artifacts (Delta noise).
* **Comparison:**
  1. **Broadband (1-45 Hz):** Includes Delta (prone to site effects).
  2. **High-Pass (>4 Hz):** Excludes Delta (Physiological check).
* **Pipeline:** `Covariance (OAS)` $\rightarrow$ `Tangent Space` $\rightarrow$ `Logistic Regression`.

---

## 3. Model Inspection & Explainability

### A. Bias & Variance Analysis (`ML_bias_variance.py`)

Checks if the best performing model (typically Logistic Regression on Merged Data) is overfitting.

* **Method:** Compares Training Score vs. Test Score using 8-fold CV.
* **Visual:** Feature Coefficients plot (Red = Associated with Pain, Blue = Associated with Health).

### B. Feature Ablation (`ablation_analysis.py`)

Determines the importance of specific frequency bands by systematically removing them.

* **Method:** *Leave-One-Band-Out*. Trains the model using all bands *except* one (e.g., No-Delta, No-Alpha).
* **Insight:** If removing a band *improves* performance, that band contained noise. If it *hurts* performance, that band contained signal.

---

## 4. Physiological Validation

Before ML training, these scripts validate that the data contains genuine biological signals.

### A. Alpha Reactivity (Berger Effect)

**Script:** `src/RF_validation_EO_EC.py` & `src/validate_physiology.py`

* **Test:** Can we classify Eyes Open vs. Eyes Closed?
* **Expectation:** $\approx 90\%$ accuracy. Occipital Alpha should be the most important feature.
* **Result:** Confirms that the EEG signal quality is sufficient to detect state changes.

### B. Global Statistics

**Script:** `src/stats_global_powers.py`

* **Test:** T-Tests/Mann-Whitney U tests on global band power between Healthy and Chronic Pain groups.
* **Correction:** Uses False Discovery Rate (FDR/Benjamini-Hochberg) to correct for multiple testing.
