# Data Harmonization: Chronic Pain Dataset

**Objective:** To harmonize the external "Chronic Pain" dataset so that its file structure and file formats exactly match the internal TDBrain dataset. This alignment is crucial to ensure that both datasets can be processed through the unified MNE-Python pipeline without requiring dataset-specific exception handling.

---

## 📂 Step 1: Directory Restructuring (BIDS-like)

**Rationale:** The original dataset had a flat or inconsistent structure. To maintain compatibility with the TDBrain pipeline, files were reorganized into a BIDS-like hierarchy: `sub-XXX/ses-1/eeg/`.

**Target Structure:**

```text
/data/Chronicpainset/derivatives/
├── sub-001/
│   └── ses-1/
│       └── eeg/
│           ├── sub-001.vhdr
│           ├── sub-001.eeg
│           └── sub-001.vmrk
└── ...
```


### Python Implementation

This script creates the target directory, iterates through the source subjects, and copies all relevant EEG files (`.vhdr`, `.vmrk`, `.eeg`, `.json`, `.csv`, `.tsv`) to the standardized location.

**Python**

```
import os
import shutil

# --- Configuration ---
project_root = r"C:\Users\Jasmyne\Documents\Thesis\thesis-eeg"
source_root = os.path.join(project_root, "data", "Chronicpainset")
target_root = os.path.join(source_root, "derivatives")

# --- Execution ---
os.makedirs(target_root, exist_ok=True)
subjects = [d for d in os.listdir(source_root) if d.startswith("sub-")]

print(f"Found {len(subjects)} subjects. Starting restructuring...")

for sub in subjects:
    eeg_dir = os.path.join(source_root, sub, "eeg")
  
    # Skip if no EEG folder exists
    if not os.path.exists(eeg_dir):
        print(f"[SKIP] No EEG folder found for {sub}")
        continue

    # Define BIDS target path
    target_eeg = os.path.join(target_root, sub, "ses-1", "eeg")
    os.makedirs(target_eeg, exist_ok=True)

    # Filter relevant extensions
    eeg_files = [
        f for f in os.listdir(eeg_dir)
        if f.endswith((".csv", ".json", ".vhdr", ".vmrk", ".eeg", ".tsv", ".fif"))
    ]

    # Copy files
    for f in eeg_files:
        src_path = os.path.join(eeg_dir, f)
        dst_path = os.path.join(target_eeg, f)
        shutil.copy2(src_path, dst_path)
        print(f"✅ {sub}: Moved {f}")

print("\n🎉 Restructuring complete.")
```

---

## 2: Data Cleaning (NaN Handling)

**Rationale:** Preliminary inspection revealed that several `.csv` files within the dataset contained `NaN` (Not a Number) values. These missing values cause runtime errors during Spectral Power Density (PSD) calculations and machine learning training.

**Method:** A zero-filling strategy was applied (`NaN` **$\rightarrow$** `0`), as these missing values often represented non-recorded channels or artifactual gaps.

### Python Implementation

The script recursively searches the new `derivatives` folder, replaces NaNs, and generates a log file (`nans_fixed_log.csv`) detailing the modifications.

**Python**

```
import os
import pandas as pd
from pathlib import Path

# --- Configuration ---
base_path = Path(r"C:\Users\Jasmyne\Documents\Thesis\data\Chronicpainset\derivatives")
log_rows = []

# --- Execution ---
print("Starting NaN detection and repair...")

for csv_file in base_path.rglob("*.csv"):
    try:
        df = pd.read_csv(csv_file)
        n_nans = df.isna().sum().sum()
      
        if n_nans > 0:
            # Replace NaN with 0
            df = df.fillna(0)
            df.to_csv(csv_file, index=False)
          
            # Log statistics
            perc = (n_nans / df.size) * 100
            log_rows.append([str(csv_file), n_nans, df.size, perc])
            print(f"✅ Fixed {csv_file.name}: {n_nans} NaNs replaced ({perc:.2f}%)")
        else:
            print(f"✔️ {csv_file.name}: Clean")

    except Exception as e:
        print(f"⚠️ Error processing {csv_file.name}: {e}")

# --- Save Log ---
if log_rows:
    log_df = pd.DataFrame(log_rows, columns=["File", "NaN_Count", "Total_Values", "Percentage"])
    log_df.to_csv(base_path / "nans_fixed_log.csv", index=False)
    print(f"\n📄 Log saved to: {base_path / 'nans_fixed_log.csv'}")
```

---

## 3: Metadata Repair (BrainVision Headers)

**Rationale:** The original `.vhdr` (BrainVision Header) files lacked specific column definitions in the `[Channel Infos]` section. While older software might ignore this, modern versions of **MNE-Python** require strict adherence to the file specification. Loading these files resulted in `ValueError` due to missing impedance and filter settings.

**The Fix:** The header file is parsed as text, and the channel definition columns are expanded to include default values for:

* `Low Cutoff`: DC
* `High Cutoff`: 250 Hz
* `Notch Filter`: Off

### Python Implementation

This script reads the original headers, reformats the channel table to match the expected schema, and saves the corrected version (suffixed as `_new.vhdr`).

**Python**

```
import re
from pathlib import Path

# --- Configuration ---
base_path = Path(r"C:\Users\Jasmyne\Documents\Thesis\thesis-eeg\data\Chronicpainset")

def update_vhdr_channels(vhdr_file):
    """Parses and reconstructs the VHDR content with corrected columns."""
    with open(vhdr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_channels = False

    for line in lines:
        # Detect start of Channel Infos
        if line.strip().startswith("Channels"):
            new_lines.append(line)
            in_channels = True
            continue

        # Inject new, standardized header row
        if in_channels and line.strip().startswith("#"):
            new_header = (
                "#     Name    Phys. chn.   Resolution / Unit   "
                "Low Cutoff [s]   High Cutoff [Hz]   Notch [Hz]   "
                "Series Res. [kOhm]   Gradient   Offset\n"
            )
            new_lines.append(new_header)
            continue

        # Reformat individual channel rows
        if in_channels and re.match(r"^\d+", line.strip()):
            parts = line.split()
            # Construct new line with default values for missing technical params
            if len(parts) >= 4:
                chan_num, name, phys_chn = parts[0], parts[1], parts[2]
                resolution = " ".join(parts[3:])
              
                new_line = (
                    f"{chan_num:<5} {name:<8} {phys_chn:<10} {resolution:<15} "
                    f"DC               250                Off          0                    0         0\n"
                )
                new_lines.append(new_line)
            else:
                new_lines.append(line) # Fallback
            continue

        # End of section detection
        if in_channels and line.strip() == "":
            in_channels = False
            new_lines.append(line)
            continue

        new_lines.append(line)

    return "".join(new_lines)

# --- Execution ---
print("Updating BrainVision Headers...")

for vhdr_file in base_path.rglob("*.vhdr"):
    # Avoid double processing
    if vhdr_file.stem.endswith("_new"): continue

    # Generate new filename
    new_stem = vhdr_file.stem.replace(" copy", "") + "_new"
    new_vhdr_file = vhdr_file.with_name(new_stem + ".vhdr")

    # Process and Save
    new_content = update_vhdr_channels(vhdr_file)
    with open(new_vhdr_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Created fixed header: {new_vhdr_file.name}")
```

---

## ✅ Summary of Changes

After executing this pipeline, the **Chronic Pain Dataset** is:

1. **Structurally Harmonized:** Aligned with `sub-XXX/ses-1/eeg` BIDS-convention.
2. **Numerically Clean:** Free of `NaN` values in feature CSVs.
3. **Readable:** `.vhdr` headers are now compatible with MNE-Python v1.10+.

The data is now ready for merging with TDBrain.
