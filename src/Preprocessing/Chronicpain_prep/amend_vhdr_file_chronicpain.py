"""
=============================================================================
🛠️ BRAINVISION HEADER REPAIR: ADD CHANNEL METADATA
=============================================================================
Objective:
    Amend raw BrainVision Header (.vhdr) files to include additional required 
    columns in the '[Channels]' section.

    Some analysis tools require specific metadata columns (Low Cutoff, 
    High Cutoff, Notch, etc.) to validly parse the file. This script 
    injects these columns with default values.

Transformation:
    - Input:  Original .vhdr file.
    - Output: *_new.vhdr (with expanded channel definitions).

Scope:
    - Dataset: Chronicpainset (Inside the new 'derivatives' folder)
    - Recursively processes all subjects.

Execution:
    python ./FM_thesis_ML/src/Preprocessing/Chronicpain_prep/amend_vhdr_file_chronicpain.py
=============================================================================
"""

import re
import sys
import shutil
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================

current_dir = Path(__file__).resolve()
sys.path.append(str(current_dir.parents[2]))

from config import CHRONIC_PAIN_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================
# Base path is now the new 'derivatives' folder!
BASE_PATH = CHRONIC_PAIN_DIR / "derivatives"

def update_vhdr_channels(vhdr_file):
    # (De functie blijft exact hetzelfde)
    with open(vhdr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_channels = False
    
    for line in lines:
        if line.strip().startswith("Channels"):
            new_lines.append(line)
            in_channels = True
            continue

        if in_channels and line.strip().startswith("#"):
            new_header = (
                "#     Name    Phys. chn.   Resolution / Unit   "
                "Low Cutoff [s]   High Cutoff [Hz]   Notch [Hz]   "
                "Series Res. [kOhm]   Gradient   Offset\n"
            )
            new_lines.append(new_header)
            continue

        if in_channels and re.match(r"^\d+", line.strip()):
            parts = line.split()
            if len(parts) >= 4:
                chan_num = parts[0]
                name = parts[1]
                phys_chn = parts[2]
                resolution = " ".join(parts[3:])
                
                new_line = (
                    f"{chan_num:<5} {name:<8} {phys_chn:<10} {resolution:<15} "
                    f"DC               250                Off          0                    0         0\n"
                )
                new_lines.append(new_line)
            else:
                new_lines.append(line)
            continue

        if in_channels and line.strip() == "":
            in_channels = False
            new_lines.append(line)
            continue

        new_lines.append(line)

    return "".join(new_lines)

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    print(f"🚀 Starting VHDR Header Repair in: {BASE_PATH}")
    
    if not BASE_PATH.exists():
         print(f"❌ Error: The directory {BASE_PATH} does not exist. Did you run moving_files.py?")
         sys.exit(1)

    for subject_dir in sorted(BASE_PATH.glob("sub-*")):
        # UPDATE: Voeg 'ses-1' toe aan het pad om overeen te komen met je nieuwe mappenstructuur
        eeg_path = subject_dir / "ses-1" / "eeg"

        if not eeg_path.exists():
            continue

        for vhdr_file in eeg_path.glob("*.vhdr"):
            if vhdr_file.stem.endswith("_new"):
                continue

            if " copy" in vhdr_file.stem:
                new_stem = vhdr_file.stem.replace(" copy", "") + "_new"
            else:
                new_stem = vhdr_file.stem + "_new"

            new_vhdr_file = vhdr_file.with_name(new_stem + ".vhdr")

            new_content = update_vhdr_channels(vhdr_file)

            with open(new_vhdr_file, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"   ✅ Created new header: {new_vhdr_file.name}")
            
    print("\n🎉 VHDR Repair Complete.")