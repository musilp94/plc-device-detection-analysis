"""
Extract full-resolution 2-channel noise spectra (on the common REF_F grid)
for the 1D-CNN and for the leakage-free normalization audit.

Output: analysis/cache/spectra_full.npz
  X            (n, 2, 3201) float32  spectra on REF_F (channels 1,2)
  device_id, category, topology, repetition   aligned label arrays
"""
from __future__ import annotations

import zipfile
import numpy as np
import pandas as pd

import features as F  # same folder


def main():
    meta = pd.read_csv(F.META_PATH, index_col="id")
    X = []
    dev_ids, cats, topos, reps = [], [], [], []

    with zipfile.ZipFile(F.ZIP_PATH, "r") as zf:
        names = set(zf.namelist())
        for dev in meta.index:
            folder = meta.loc[dev, "folder_name"]
            cat = meta.loc[dev, "category"]
            cat = "other" if pd.isna(cat) else cat
            print(f"dev {dev}: {folder}", flush=True)
            for topo_id, topo_name in F.TOPOLOGIES.items():
                for rep in F.list_reps(names, folder, topo_name):
                    m = F.read_measurement(zf, names, folder, topo_name, rep)
                    if m is None:
                        continue
                    X.append(np.stack([m[1], m[2]]).astype(np.float32))
                    dev_ids.append(dev); cats.append(cat)
                    topos.append(topo_id); reps.append(int(rep))

    out = F.OUT_DIR / "spectra_full.npz"
    F.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.array(X, dtype=np.float32),
        device_id=np.array(dev_ids), category=np.array(cats),
        topology=np.array(topos), repetition=np.array(reps),
    )
    print(f"saved {out}  X shape {np.array(X).shape}")


if __name__ == "__main__":
    main()
