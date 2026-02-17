# Preprocessing & Data Management Pipeline

**Description:** This document outlines the step-by-step procedures for auditing the raw dataset, managing participant metadata, executing the core signal processing pipeline, and merging the final feature sets for analysis.

---

## 1: Data Audit & Completeness Check

**Script:** `src/check_data_completeness.py`

**Objective:** To verify that every subject listed in the metadata has a corresponding data folder containing all required file types (`.npy`, `.txt`, `.csv`, `.pdf`) for both experimental conditions (Eyes Closed `EC` / Eyes Open `EO`).

### Methodology

1. **Input:** Reads participant IDs from 5 sources:
   * TDBrain Healthy (Excel)
   * TDBrain Chronic Pain (Excel)
   * TDBrain Unknown Indication (Excel)
   * TDBrain Unknown/NaN (Excel)
   * External Chronic Pain Dataset (TSV)
2. **Audit Logic:** Iterates through every subject ID and checks the `results/` directory.
3. **Validation:** Flags subjects as "MISSING FOLDER" or "INCOMPLETE" if specific files are absent.
4. **Output:** Generates `full_dataset_audit.csv` containing a status report for every subject.

---

## 2: Metadata Splitting (TDBrain)

**Script:** `src/split_participants_TDBRAIN.py`

**Objective:** To segregate the monolithic TDBrain metadata file (`TDBRAIN_participants_V2.xlsx`) into four mutually exclusive subgroups to prevent data leakage and ensure clean labeling.

### Logic

The script filters subjects based on `formal_status` and `indication` columns:

1. **Healthy:** `formal_status == 'HEALTHY'`
2. **Chronic Pain:** `formal_status == 'CHRONIC PAIN'`
3. **Unknown (Informal):** `formal_status == 'UNKNOWN'` AND `indication` is present.
4. **Unknown (NaN):** `formal_status == 'UNKNOWN'` AND `indication` is missing/NaN.

**Outputs:**

* `TDBRAIN_participants_HEALTHY.xlsx`
* `TDBRAIN_participants_CHRONIC_PAIN.xlsx`
* `TDBRAIN_participants_UNKNOWN.xlsx`
* `TDBRAIN_participants_UNKNOWN_NaNs.xlsx`

---

## 3: Core Signal Processing Pipeline

**Script:** `src/preprocess_pipeline.py`

**Objective:** To transform raw EEG data (BrainVision `.vhdr`) into clean, artifact-free epochs and extract spectral power features.

### Processing Steps

1. **Loading & Renaming:** * Loads raw data using MNE-Python.
   * Standardizes channel names to the 10-20 system (e.g., `fp1` $\rightarrow$ `Fp1`).
   * **Scaling Fix:** Detects if data is in Volts and rescales to Microvolts ($\mu V$) if necessary.
2. **Filtering:**
   * **Notch Filter:** Removes power line noise at 50Hz (and harmonics).
   * **Bandpass Filter:** 0.5 - 100 Hz.
3. **RANSAC & AutoReject:**
   * Applies **RANSAC** (Random Sample Consensus) to robustly identify and interpolate bad channels.
   * Applies **AutoReject** to repair or discard bad epochs based on peak-to-peak amplitude thresholds.
4. **Feature Extraction:**
   * Computes Relative Power Spectral Density (PSD) using Welch's method.
   * Bands: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-100Hz).
5. **Reporting:**
   * Generates a PDF report for visual inspection (Raw vs Cleaned signal, PSD plot).
   * Saves cleaned data as `.npy` (3D array) and features as `.csv`.

**Dependencies:** `preprocessing_plotting.py` (Handles generation of visual reports without opening GUI windows).

---

## 4: Final Data Merge

**Script:** `src/final_prep.py`

**Objective:** To combine the processed feature files (`_features.csv`) with demographic metadata into a single "Master Dataset".

### Workflow

1. **Inclusion Criteria:**
   * Subject must be $\ge$ 18 years old.
   * Subject must have **BOTH** EC and EO processed files available.
2. **Labeling:**
   * **Healthy:** Label `0`
   * **Chronic Pain (Internal + External):** Label `1`
   * **Unknown (Informal):** Label `-1`
   * **Unknown (NaN):** Label `-2`
3. **Merging:**
   * Iterates through all valid feature files.
   * Appends Age, Gender, and Diagnosis columns.
   * Concatenates into one large DataFrame.
4. **Output:** `results/final_dataset.csv` (The input for all Machine Learning models).

---

## 5: Validation & Descriptive Statistics

### A. Global Power Validation

**Script:** `src/global_powers.py`

* Calculates the **Global Average Power** (average across all 20 channels) for every frequency band.
* Used to generate "Site Effect" plots and validate physiological consistency (e.g., Alpha peak).
* **Output:** `validation_global_powers.csv`

### B. Demographics Table

**Script:** `src/general_info.py`

* Generates the "Table 1" for the thesis.
* Calculates Mean Age ($\pm$ SD) and Gender distribution (% Female) per group.
* Ensures strict consistency by only counting subjects included in the final analysis (Unique IDs, >18y, Complete Data).

### C. Sensor Visualization

**Script:** `src/visualize_sensors.py`

* Produces 2D and 3D topographic maps of the channel montage (Standard 10-20 system).
* Used to verify channel locations and generate figures for the methodology section.
