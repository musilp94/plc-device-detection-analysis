"""Verify the chosen ZIP matches metadata.csv folder names and the loader works
end-to-end. Also confirms ML libraries import.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATASET_DIR = Path("dataset")
sys.path.insert(0, str(DATASET_DIR))
from load_dataset import Dataset, ROOT_FOLDER  # noqa: E402

# Canonical dataset = the version published on Zenodo (dataset/v1).
ZIP = DATASET_DIR / "v1" / "VZ6_039_021_VUT_D_0000_v1.zip"

meta = pd.read_csv(DATASET_DIR / "metadata.csv", index_col="id")

with zipfile.ZipFile(ZIP, "r") as zf:
    names = set(zf.namelist())

prefix = ROOT_FOLDER + "/"
present_folders = {n[len(prefix):].split("/")[0] for n in names
                   if n.startswith(prefix) and "/" in n[len(prefix):]}

print(f"metadata rows: {len(meta)}  | folders in zip: {len(present_folders)}")
missing = [(i, fn) for i, fn in meta["folder_name"].items() if fn not in present_folders]
print(f"metadata folder_names MISSING from zip: {len(missing)}")
for i, fn in missing:
    print(f"   id={i}: {fn!r}")

# Try a full load of a few devices, incl. the renamed UPS (42/43) and a PC (57)
ds = Dataset(zip_path=ZIP)
for dev in [0, 42, 43, 57]:
    try:
        df = ds.load_device(dev, topology=2)
        print(f"  device {dev:>2} ({meta.loc[dev,'folder_name']!r}): "
              f"{len(df)} rows, reps={sorted(df['repetition'].unique())[:3]}..., "
              f"channels={sorted(df['channel'].unique())}")
    except Exception as e:
        print(f"  device {dev}: ERROR {e}")

print("\n--- library versions ---")
import importlib.metadata as m
for p in ["scikit-learn", "catboost", "lightgbm", "xgboost", "pandas", "numpy"]:
    try:
        print(f"  {p} = {m.version(p)}")
    except Exception:
        print(f"  {p} = MISSING")
