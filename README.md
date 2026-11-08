# PLC Device Detection — historical Version 1 analysis

## Data and code for the accepted article

The expanded article **Broadband Power-Line Communication as a Sensor: Device
Recognition and Topology Transfer** was accepted for publication in the IEEE
Open Journal of the Communications Society on 31 October 2026. Acceptance is
separate from publication; an article DOI is not assigned by this repository.

For the data and exact reproduction code accompanying that article, use
[the article-specific Zenodo release, DOI 10.5281/zenodo.22653610](https://doi.org/10.5281/zenodo.22653610).
Download all three files from that record:

- `VZ6_039_021_VUT_D_v2.zip` — analytical inputs, code, results and documentation;
- `VZ6_039_021_VUT_D_legacy-inventory_v2.zip` — native inventory and retained photographs;
- `ReadMeFile_v2.txt` — the authoritative delivery guide and checksum clarifications.

Extract both ZIP files into the same destination and follow the standalone
README, then each analytical component's instructions. This release contains
the historical analyses and the later controlled measurement and validation
components. Its scope is documented analytical inputs and reproduction code;
some original operational logs and private identity mappings are retained
separately.

The [original EcoStor inventory, DOI 10.5281/zenodo.20541725](https://doi.org/10.5281/zenodo.20541725),
is a separate source dataset. The `v2` delivery label does not make the
article-specific record a new version in the original record's Zenodo version
series.

## Scope of this repository

This repository preserves the analysis and reproduction code for the historical
Version 1 device/state study, **Power Line Communication as a Sensor: Machine
Learning-Based Spectral Fingerprinting for Non-Invasive Device Detection**,
associated with the 6 July 2026 manuscript submission. That submission was not
accepted. The later accepted article has the expanded scope and archived code
linked above. The historical numerical code and result files in this repository
are retained for reproducibility.

The study investigates whether the noise spectrum a broadband power line
communication (BB-PLC) modem already measures as part of its normal channel
estimation can be used to fingerprint and identify the electrical devices
connected nearby. This repository contains the feature extraction, model
evaluation, independent audit, anomaly detection and figure-generation scripts
that reproduce the reported dataset-derived numbers and figures in the
historically submitted article. The repository does not derive authority from
that submission status.

## Dataset

The measurement dataset is published separately on Zenodo (CC BY 4.0):

> Version 1 DOI: [10.5281/zenodo.20541725](https://doi.org/10.5281/zenodo.20541725)
>
> All-versions concept DOI: [10.5281/zenodo.20541724](https://doi.org/10.5281/zenodo.20541724)

It contains the noise spectra of 75 device configurations (38 physical devices
in various operating states) plus one empty-grid baseline, recorded over a live
230 V / 50 Hz grid in three cable configurations, 20 repetitions each, on two dependent
host-export views (CH1 and CH2), over the 1.8–80 MHz band. There are 9,120
spectrum CSV files and two additional CSV files.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your environment of choice
pip install -r requirements.txt
```

For this historical pipeline, download `VZ6_039_021_VUT_D_v1.zip` from the
original Zenodo record. Save a local copy under the historical filename
`VZ6_039_021_VUT_D_0000_v1.zip` expected below; this changes only the local
filename, not the archive contents. The scripts read the archive directly and
also expect the companion CSVs alongside it:

```
dataset/
├── v1/
│   └── VZ6_039_021_VUT_D_0000_v1.zip # the archive downloaded from Zenodo
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
python analysis/confusion_fig.py     # -> figures/confusion_category.pdf
python analysis/gen_device_table.py  # -> analysis/results/tab_device_spectral.tex
```

`analysis/check_data.py` verifies that the archive matches `metadata.csv` and
that the loader works end-to-end.

Pre-computed result files are provided under `analysis/results/` so the main
tables and confusion figures can be checked without rerunning the full model
evaluation:

- `model_comparison.csv`: all classification metrics, confidence intervals and
  macro/balanced scores used in the manuscript tables.
- `feature_importance.csv`: band-level importance source for
  `ml_band_importance.pdf`.
- `audit_*.csv` and `critical_audit.md`: the critical-analysis outputs used in
  Section VI.
- `anomaly_*.csv`: unauthorized-device detection metrics.
- `predictions_category.npz`: out-of-fold LightGBM predictions used for the
  manuscript category-confusion figure.
- `predictions_device.npz`: out-of-fold LightGBM predictions used for the
  supplemental full 76-class per-device confusion matrices mentioned in the
  manuscript.
- `tab_device_spectral.tex`: the full per-device spectral-characterization
  table accompanying the shorter representative table in the manuscript.

The scripts regenerate the dataset-derived manuscript figures:
`div_accuracy.pdf`, `spectra_archetypes.pdf`, `feature_embedding.pdf`,
`confusion_category.pdf`, `channel_coloring.pdf`, `ml_norm_ablation.pdf`,
`strong_devices.pdf`, `audit_recall_vs_strength.pdf`,
`ml_band_importance.pdf`, and `anomaly_detection.pdf`. The measurement
validation figures and the preliminary 24-hour office example use separate
non-release source files and are intentionally not part of this analysis
repository.

## Layout

```
analysis/            evaluation and figure scripts
  features.py          spectral feature extraction (232 features per measurement)
  extract_spectra.py   full-resolution two-channel spectra for the 1D-CNN
  ml_evaluation.py     model comparison, two protocols, feature importance
  cnn_eval.py          1D-CNN baseline
  critical_audit.py    independent audit (leakage, generalization, error structure)
  anomaly_detection.py per-installation unauthorized-device detection
  make_figures.py      dataset-derived manuscript and supplemental figures
  confusion_fig.py     manuscript category-confusion figure
  gen_device_table.py  per-device spectral characterization table
  figstyle.py          shared figure style
  check_data.py        dataset integrity check
  results/             pre-computed result tables
figures/             generated figures (created on demand)
```

## Repository role and interpretation boundary

This repository reproduces the historical Version 1 analyses and their stated
evaluation targets. For the later experiments and the accepted article, the
reproduction entry point is the archived data and code at
[DOI 10.5281/zenodo.22653610](https://doi.org/10.5281/zenodo.22653610).
Access to the authors' private working repositories is not required.

`CH1` and `CH2` are preserved dataset and host-export labels. Current evidence
does not identify them as particular conductor pairs, physical ports, spatial
streams, common-mode quantities or differential-mode quantities. Analyses may
use them as dependent, potentially complementary feature views without making
that hardware interpretation.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0 (see the Zenodo record).

## Citation

See `CITATION.cff`. When reusing this historical code or source inventory, cite
the exact original dataset release, DOI 10.5281/zenodo.20541725. When reusing
the expanded article's data or code, cite DOI 10.5281/zenodo.22653610 as well
as the original inventory where applicable. Cite the associated article for
its methods and interpretation once its publication details are available.
The software, dataset and article are distinct citation objects.
