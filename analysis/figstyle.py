"""Shared figure style for all generated manuscript figures.

One place to control font, sizes, palette and grid so every plot in the paper
looks the same and matches the (Times) body text. Import and call apply() at
the top of any figure script.
"""
from __future__ import annotations
import matplotlib as mpl

# Unified qualitative palette (colour-blind friendly, consistent roles).
PRIMARY = "#1f4e79"   # deep blue  -> topology-matched / first series
ACCENT  = "#c0392b"   # red        -> leave-one-topology-out / contrast
GOOD    = "#2e7d32"   # green      -> normalized / "good" case
MUTED   = "#7f8c8d"   # grey       -> raw / baseline
PURPLE  = "#6a51a3"   # purple     -> importance / third series
GOLD    = "#d68910"   # amber      -> fourth series
CYCLE   = [PRIMARY, ACCENT, GOOD, PURPLE, GOLD, MUTED]
SEQ_CMAP = "viridis"  # sequential maps (confusion matrices, heatmaps)


def apply():
    mpl.rcParams.update({
        # --- font: match the manuscript (Times) ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",          # serif math to match Times
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # --- consistent axes / grid ---
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "lines.linewidth": 1.2,
        # --- output ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "pdf.fonttype": 42,                  # embed real fonts, not Type-3
    })
