# PLC Device Detection — analysis and reproduction code

Analysis code accompanying the paper:

> **Power Line Communication as a Sensor: Machine Learning-Based Spectral
> Fingerprinting for Non-Invasive Device Detection**
> P. Musil, R. Macho, P. Mlýnek, J. Mišurec.
> IEEE Open Journal of the Communications Society.

The study investigates whether the noise spectrum a broadband power line
communication (BB-PLC) modem already measures as part of its normal channel
estimation can be used to fingerprint and identify the electrical devices
connected nearby. This repository contains the feature extraction, model
evaluation, independent audit, anomaly detection and figure-generation scripts
that reproduce the reported numbers and figures.

## Dataset

The measurement dataset is published separately on Zenodo (CC BY 4.0):

> DOI: [10.5281/zenodo.20541725](https://doi.org/10.5281/zenodo.20541725)

It contains the noise spectra of 75 device configurations (38 physical devices
in various operating states) plus one empty-grid baseline, recorded over a live
230 V / 50 Hz grid in three cable configurations, 20 repetitions each, on two
MIMO channels, over the 1.8–80 MHz band.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your environment of choice
pip install -r requirements.txt
```

Download the dataset archive from Zenodo and lay it out as follows (the scripts
read the archive directly and also expect the companion CSVs alongside it):

```
dataset/
├── v1/
│   └── VZ6_039_021_VUT_D_v1.zip      # the archive downloaded from Zenodo
├── metadata.csv                       # extracted from the archive
├── summary_statistics.csv             # extracted from the archive
└── load_dataset.py                    # extracted from the archive
```

`metadata.csv`, `summary_statistics.csv` and `load_dataset.py` are inside the
archive (under `dataset - measured data, graphs, and photos/`); copy them up
into `dataset/`.

## Reproduction pipeline

Run from the repository root, in order:

```bash
python analysis/features.py          # -> analysis/cache/features_{raw,norm}.csv
python analysis/extract_spectra.py   # -> analysis/cache/spectra_full.npz (CNN, audit, figures)
python analysis/ml_evaluation.py     # -> analysis/results/model_comparison.csv, predictions_*.npz, feature_importance.csv
python analysis/cnn_eval.py          # -> appends the 1D-CNN rows to model_comparison.csv
python analysis/critical_audit.py    # -> analysis/results/critical_audit.md, audit_*.csv, figures/audit_*.pdf
python analysis/anomaly_detection.py # -> analysis/results/anomaly_*.csv, figures/anomaly_detection.pdf
python analysis/make_figures.py      # -> figures/*.pdf
python analysis/gen_device_table.py  # -> analysis/results/tab_device_spectral.tex
```

`analysis/check_data.py` verifies that the archive matches `metadata.csv` and
that the loader works end-to-end.

Pre-computed result tables are provided under `analysis/results/` so the figures
can be regenerated without rerunning the full evaluation.

## Layout

```
analysis/            evaluation and figure scripts
  features.py          spectral feature extraction (232 features per measurement)
  extract_spectra.py   full-resolution two-channel spectra for the 1D-CNN
  ml_evaluation.py     model comparison, two protocols, feature importance
  cnn_eval.py          1D-CNN baseline
  critical_audit.py    independent audit (leakage, generalization, error structure)
  anomaly_detection.py per-installation unauthorized-device detection
  make_figures.py      publication figures
  gen_device_table.py  per-device spectral characterization table
  figstyle.py          shared figure style
  check_data.py        dataset integrity check
  results/             pre-computed result tables
figures/             generated figures (created on demand)
```

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0 (see the Zenodo record).

## Citation

See `CITATION.cff`. Please cite both the article and the Zenodo dataset record.
