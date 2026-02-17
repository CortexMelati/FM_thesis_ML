import sys
from pathlib import Path

# Zorg dat we src kunnen vinden
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from config import RESULTS_DIR, TDBRAIN_DIR
    import pandas as pd
    import os

    print("✅ Config geladen!")
    print(f"📂 Project zoekt resultaten in: {RESULTS_DIR}")
    
    # Check 1: Bestaat de map?
    if RESULTS_DIR.exists():
        print("   -> Map bestaat! 👍")
    else:
        print("   -> ❌ Map bestaat NIET. Check config.py")

    # Check 2: Kunnen we de CSV vinden?
    csv_file = RESULTS_DIR / "final_dataset.csv"
    if csv_file.exists():
        print(f"   -> final_dataset.csv gevonden! ({os.path.getsize(csv_file) / 1024:.2f} KB)")
        # Even snel openen
        df = pd.read_csv(csv_file)
        print(f"   -> Data ingeladen: {len(df)} rijen.")
    else:
        print(f"   -> ❌ final_dataset.csv NIET gevonden in {RESULTS_DIR}")

    # Check 3: Check een map met .npy files
    npy_path = RESULTS_DIR / "TDBrain"
    if npy_path.exists():
        print(f"   -> TDBrain results map gevonden.")
    else:
        print(f"   -> ⚠️ TDBrain map niet gevonden in results.")

except ImportError as e:
    print("❌ Kon config.py niet vinden of importeren.")
    print(e)
except Exception as e:
    print(f"❌ Er ging iets fout: {e}")