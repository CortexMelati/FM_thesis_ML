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
    - Dataset: Chronicpainset
    - Recursively processes all subjects.

Execution:
    python ./FM_thesis_ML/src/Preprocessing/Chronicpain prep/amend_vhdr_file_chronicpain.py
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
# Base path to all subjects (from Config)
BASE_PATH = CHRONIC_PAIN_DIR

def update_vhdr_channels(vhdr_file):
    """
    Reads a .vhdr file and reconstructs it with expanded channel metadata.
    """
    with open(vhdr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_channels = False
    
    for line in lines:
        # Detect start of Channels section
        if line.strip().startswith("Channels"):
            new_lines.append(line)
            in_channels = True
            continue

        # Modify the Header Line within Channels section
        if in_channels and line.strip().startswith("#"):
            new_header = (
                "#     Name    Phys. chn.   Resolution / Unit   "
                "Low Cutoff [s]   High Cutoff [Hz]   Notch [Hz]   "
                "Series Res. [kOhm]   Gradient   Offset\n"
            )
            new_lines.append(new_header)
            continue

        # Rewrite Channel Lines
        if in_channels and re.match(r"^\d+", line.strip()):
            parts = line.split()
            if len(parts) >= 4:
                chan_num = parts[0]
                name = parts[1]
                phys_chn = parts[2]
                resolution = " ".join(parts[3:])  # e.g., '0.0993411 µV'
                
                # Format new line with fixed default values for the added columns
                new_line = (
                    f"{chan_num:<5} {name:<8} {phys_chn:<10} {resolution:<15} "
                    f"DC               250                Off          0                    0         0\n"
                )
                new_lines.append(new_line)
            else:
                # Keep original if parsing fails
                new_lines.append(line)
            continue

        # Detect End of Channels section (empty line)
        if in_channels and line.strip() == "":
            in_channels = False
            new_lines.append(line)
            continue

        # All other lines remain unchanged
        new_lines.append(line)

    return "".join(new_lines)

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    print(f"🚀 Starting VHDR Header Repair in: {BASE_PATH}")
    
    if not BASE_PATH.exists():
         print(f"❌ Error: The directory {BASE_PATH} does not exist. Check config.py.")
         sys.exit(1)

    # Iterate over all subject directories
    # Note: Using sorted() on path generator
    for subject_dir in sorted(BASE_PATH.glob("sub-*")):
        eeg_path = subject_dir / "eeg"

        # Check if eeg folder exists before globbing
        if not eeg_path.exists():
            continue

        # Find all .vhdr files
        for vhdr_file in eeg_path.glob("*.vhdr"):
            # Skip files that have already been processed
            if vhdr_file.stem.endswith("_new"):
                continue

            # Construct new filename
            if " copy" in vhdr_file.stem:
                new_stem = vhdr_file.stem.replace(" copy", "") + "_new"
            else:
                new_stem = vhdr_file.stem + "_new"

            new_vhdr_file = vhdr_file.with_name(new_stem + ".vhdr")

            # Generate new content
            new_content = update_vhdr_channels(vhdr_file)

            # Save the new file
            with open(new_vhdr_file, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"   ✅ Created new header: {new_vhdr_file.name}")
            
    print("\n🎉 VHDR Repair Complete.")