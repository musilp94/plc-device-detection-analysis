"""
Feature extraction for the PLC device-detection dataset (Zenodo v1).
===================================================================

Reads the published dataset ZIP (dataset/v1/VZ6_039_021_VUT_D_0000_v1.zip)
once, interpolates every noise spectrum onto a common frequency grid, and
produces two cached feature matrices:

  analysis/cache/features_raw.csv   — features on the measured spectrum
  analysis/cache/features_norm.csv  — features on the *baseline-normalized*
                                      spectrum  (device − mean empty-grid
                                      spectrum for the same topology+channel)

One row per (device_id, topology, repetition); features for both MIMO
channels are concatenated, plus inter-channel band differences.

Feature design (per channel)
----------------------------
Per 5-MHz band (16 bands, 1.8–80 MHz): mean, std, min, max, median, iqr   -> 96
Global shape descriptors: mean, std, median, p5, p95, dyn_range, skew,
    kurtosis, spectral_centroid, spectral_flatness, spectral_slope,
    peak_count                                                            -> 12
Per channel total: 108 ; two channels: 216 ; + 16 inter-channel band-mean
differences = 232 features.

Run:
    python analysis/features.py            # full extraction (~1-2 min)
    python analysis/features.py --limit 4  # smoke test on first 4 devices
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "dataset"
ZIP_PATH = DATASET_DIR / "v1" / "VZ6_039_021_VUT_D_0000_v1.zip"
META_PATH = DATASET_DIR / "metadata.csv"
OUT_DIR = REPO / "analysis" / "cache"

ROOT_FOLDER = "dataset - measured data, graphs, and photos"
TOPOLOGIES: dict[int, str] = {
    1: "1 - 50 m–isolation transformer–50 m",
    2: "2 - direct connection",
    3: "3 - 50 m",
}
CHANNELS = (1, 2)
CSV_TEMPLATE = "NOISE from 46 to B1(CH{ch}).csv"

# Common frequency grid for interpolation (MHz)
FREQ_MIN, FREQ_MAX, N_GRID = 1.8, 80.0, 3201
REF_F = np.linspace(FREQ_MIN, FREQ_MAX, N_GRID)

# 5-MHz bands
BANDS: list[tuple[float, float]] = [
    (1.8, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
    (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60),
    (60, 65), (65, 70), (70, 75), (75, 80),
]
BAND_MASKS = [((REF_F >= lo) & (REF_F < hi)) for lo, hi in BANDS]
# ensure the last band includes 80.0
BAND_MASKS[-1] = (REF_F >= BANDS[-1][0]) & (REF_F <= BANDS[-1][1])

BAND_STATS = ("mean", "std", "min", "max", "median", "iqr")


# --------------------------------------------------------------------------
# Spectrum parsing & interpolation
# --------------------------------------------------------------------------
def parse_csv(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column (freq[MHz], amp[dBm/Hz]) CSV. Returns (freq, amp)."""
    txt = data.decode("utf-8", errors="replace")
    rows = [r for r in txt.replace("\r", "").split("\n") if r.strip()]
    sep = "," if rows[0].count(",") >= rows[0].count(";") else ";"
    arr = np.array([r.split(sep) for r in rows], dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def to_grid(freq: np.ndarray, amp: np.ndarray) -> np.ndarray:
    """Interpolate amplitude onto the common REF_F grid."""
    order = np.argsort(freq)
    return np.interp(REF_F, freq[order], amp[order])


# --------------------------------------------------------------------------
# Feature computation
# --------------------------------------------------------------------------
def channel_features(amp: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute band + global shape features for one spectrum on REF_F."""
    f: dict[str, float] = {}

    # Per-band statistics
    for (lo, hi), mask in zip(BANDS, BAND_MASKS):
        seg = amp[mask]
        tag = f"{prefix}_b{lo:g}_{hi:g}"
        f[f"{tag}_mean"] = float(seg.mean())
        f[f"{tag}_std"] = float(seg.std())
        f[f"{tag}_min"] = float(seg.min())
        f[f"{tag}_max"] = float(seg.max())
        f[f"{tag}_median"] = float(np.median(seg))
        f[f"{tag}_iqr"] = float(np.percentile(seg, 75) - np.percentile(seg, 25))

    # Global shape descriptors
    p5, p95 = np.percentile(amp, [5, 95])
    f[f"{prefix}_g_mean"] = float(amp.mean())
    f[f"{prefix}_g_std"] = float(amp.std())
    f[f"{prefix}_g_median"] = float(np.median(amp))
    f[f"{prefix}_g_p5"] = float(p5)
    f[f"{prefix}_g_p95"] = float(p95)
    f[f"{prefix}_g_dynrange"] = float(p95 - p5)
    f[f"{prefix}_g_skew"] = float(stats.skew(amp))
    f[f"{prefix}_g_kurt"] = float(stats.kurtosis(amp))

    # Spectral centroid / flatness on linear power (relative scale is fine)
    power = np.power(10.0, amp / 10.0)
    psum = power.sum()
    f[f"{prefix}_g_centroid"] = float((REF_F * power).sum() / psum) if psum > 0 else 0.0
    gmean = np.exp(np.mean(np.log(power + 1e-30)))
    f[f"{prefix}_g_flatness"] = float(gmean / (power.mean() + 1e-30))

    # Spectral slope (dB per MHz) via linear fit
    slope = np.polyfit(REF_F, amp, 1)[0]
    f[f"{prefix}_g_slope"] = float(slope)

    # Peak count: points exceeding mean + 2*std
    thr = amp.mean() + 2 * amp.std()
    f[f"{prefix}_g_peaks"] = float((amp > thr).sum())
    return f


def build_row_features(amp_ch: dict[int, np.ndarray]) -> dict[str, float]:
    """Features for one measurement: per-channel + inter-channel band diffs."""
    feats: dict[str, float] = {}
    for ch in CHANNELS:
        feats.update(channel_features(amp_ch[ch], prefix=f"ch{ch}"))
    # Inter-channel band-mean differences (MIMO information)
    for (lo, hi), mask in zip(BANDS, BAND_MASKS):
        d = float(amp_ch[1][mask].mean() - amp_ch[2][mask].mean())
        feats[f"diff_b{lo:g}_{hi:g}_mean"] = d
    return feats


# --------------------------------------------------------------------------
# Dataset traversal
# --------------------------------------------------------------------------
def list_reps(names: set[str], folder: str, topo_name: str) -> list[str]:
    prefix = f"{ROOT_FOLDER}/{folder}/{topo_name}/"
    reps = sorted({
        n[len(prefix):].split("/")[0]
        for n in names
        if n.startswith(prefix) and "/" in n[len(prefix):]
        and n[len(prefix):].split("/")[0].isdigit()
    })
    return reps


def read_measurement(zf: zipfile.ZipFile, names: set[str], folder: str,
                     topo_name: str, rep: str) -> dict[int, np.ndarray] | None:
    """Read both channels of one repetition, return {ch: amp_on_grid} or None."""
    out: dict[int, np.ndarray] = {}
    for ch in CHANNELS:
        entry = f"{ROOT_FOLDER}/{folder}/{topo_name}/{rep}/{CSV_TEMPLATE.format(ch=ch)}"
        if entry not in names:
            return None
        freq, amp = parse_csv(zf.read(entry))
        out[ch] = to_grid(freq, amp)
    return out


def compute_empty_grid_baseline(zf, names, meta) -> dict[tuple[int, int], np.ndarray]:
    """Mean empty-grid (device 0) spectrum per (topology, channel)."""
    folder = meta.loc[0, "folder_name"]
    baseline: dict[tuple[int, int], np.ndarray] = {}
    for topo_id, topo_name in TOPOLOGIES.items():
        reps = list_reps(names, folder, topo_name)
        acc = {ch: [] for ch in CHANNELS}
        for rep in reps:
            m = read_measurement(zf, names, folder, topo_name, rep)
            if m is None:
                continue
            for ch in CHANNELS:
                acc[ch].append(m[ch])
        for ch in CHANNELS:
            baseline[(topo_id, ch)] = np.mean(acc[ch], axis=0)
    return baseline


def extract(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(META_PATH, index_col="id")
    rows_raw: list[dict] = []
    rows_norm: list[dict] = []

    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = set(zf.namelist())
        print("Computing empty-grid baseline (device 0)...", flush=True)
        baseline = compute_empty_grid_baseline(zf, names, meta)

        dev_ids = list(meta.index)
        if limit:
            dev_ids = dev_ids[:limit]

        for k, dev in enumerate(dev_ids):
            folder = meta.loc[dev, "folder_name"]
            print(f"  [{k+1}/{len(dev_ids)}] dev {dev}: {folder}", flush=True)
            base_meta = {
                "device_id": dev,
                "device": meta.loc[dev, "device"],
                "state": meta.loc[dev, "state"],
                "category": meta.loc[dev, "category"],
                "device_group_id": meta.loc[dev, "device_group_id"],
                "rated_power_w": meta.loc[dev, "rated_power_w"],
            }
            for topo_id, topo_name in TOPOLOGIES.items():
                for rep in list_reps(names, folder, topo_name):
                    m = read_measurement(zf, names, folder, topo_name, rep)
                    if m is None:
                        continue
                    ctx = {**base_meta, "topology": topo_id, "repetition": int(rep)}

                    # raw
                    rows_raw.append({**ctx, **build_row_features(m)})

                    # baseline-normalized (subtract empty-grid spectrum)
                    m_norm = {ch: m[ch] - baseline[(topo_id, ch)] for ch in CHANNELS}
                    rows_norm.append({**ctx, **build_row_features(m_norm)})

    df_raw = pd.DataFrame(rows_raw)
    df_norm = pd.DataFrame(rows_norm)
    print(f"\nExtracted {len(df_raw)} samples × "
          f"{df_raw.shape[1] - 7} features in {time.time()-t0:.1f}s")
    return df_raw, df_norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limit #devices (smoke test)")
    args = ap.parse_args()

    df_raw, df_norm = extract(limit=args.limit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    df_raw.to_csv(OUT_DIR / f"features_raw{suffix}.csv", index=False)
    df_norm.to_csv(OUT_DIR / f"features_norm{suffix}.csv", index=False)
    print(f"Saved to {OUT_DIR}")
    # quick sanity print
    feat_cols = [c for c in df_raw.columns if c not in (
        "device_id", "device", "state", "category", "device_group_id",
        "rated_power_w", "topology", "repetition")]
    print(f"feature columns: {len(feat_cols)}")
    print(df_raw.groupby('topology').size().to_dict(), "samples per topology")


if __name__ == "__main__":
    main()
