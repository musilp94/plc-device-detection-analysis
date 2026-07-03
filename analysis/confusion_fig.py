"""Category-level confusion matrices (topology-matched vs LOTO).

Reads the out-of-fold predictions saved by ml_evaluation.py
(analysis/results/predictions_category.npz; LightGBM, per-site-calibrated
features) and renders the two row-normalized 5x5 matrices side by side into
figures/confusion_category.pdf. This reproduces the per-category confusion
figure shown in the manuscript.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

import figstyle

figstyle.apply()

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "analysis" / "results"
FIGS = REPO / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DISPLAY = {"light_source": "light", "motor": "motor", "other": "other",
           "smps": "SMPS", "ups": "UPS"}


def main() -> None:
    d = np.load(RESULTS / "predictions_category.npz", allow_pickle=True)
    labels = list(d["labels"])
    names = [DISPLAY[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, proto, title in [
        (axes[0], "A", "(a) topology-matched"),
        (axes[1], "B", "(b) leave-one-topology-out (calibrated)"),
    ]:
        yt, yp = d[f"y_true_{proto}"], d[f"y_pred_{proto}"]
        acc = accuracy_score(yt, yp)
        cm = confusion_matrix(yt, yp, labels=labels, normalize="true") * 100
        ax.imshow(cm, cmap=figstyle.SEQ_CMAP, vmin=0, vmax=100)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm[i, j]
                if v < 0.5:
                    continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 55 else "black")
        ax.set_xticks(range(len(labels)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), names)
        ax.set_xlabel("predicted category")
        ax.set_title(f"{title}, acc. {100*acc:.1f}\\%"
                     if plt.rcParams["text.usetex"] else
                     f"{title}, acc. {100*acc:.1f}%")
        ax.grid(False)
    axes[0].set_ylabel("true category")

    fig.tight_layout()
    fig.savefig(FIGS / "confusion_category.pdf")
    plt.close(fig)
    print(f"saved {FIGS / 'confusion_category.pdf'}")


if __name__ == "__main__":
    main()
