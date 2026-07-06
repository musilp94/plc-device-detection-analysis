"""
Dataset-derived manuscript figures from evaluation results.
Reads analysis/results/* and analysis/cache/* and writes PDF figures into
figures/. All figures share analysis/figstyle.py (Times font, unified palette,
no in-figure titles -- captions carry the description).

Only figures derived from the published Zenodo dataset are included here. The
measurement-validation and 24-hour office figures in the manuscript use
separate non-release source files and are intentionally outside this repository.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import figstyle
figstyle.apply()
P = figstyle

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "analysis" / "results"
CACHE = REPO / "analysis" / "cache"
FIGS = REPO / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def _freq(npts):
    return np.linspace(1.8, 80, npts)


def _vector_colorbar(fig, axes, cmap, norm, label, ticks=None, width=0.018, pad=0.02):
    """Draw a PDF-vector colour bar instead of Matplotlib's image gradient."""
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    boxes = [ax.get_position() for ax in axes]
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    cax = fig.add_axes([x1 + pad, y0, width, y1 - y0])
    cmap_obj = plt.get_cmap(cmap)
    levels = np.linspace(norm.vmin, norm.vmax, 33)
    for lo, hi in zip(levels[:-1], levels[1:]):
        mid = (lo + hi) / 2
        cax.add_patch(plt.Rectangle((0, lo), 1, hi - lo,
                                    facecolor=cmap_obj(norm(mid)),
                                    edgecolor="none"))
    cax.set_xlim(0, 1); cax.set_ylim(norm.vmin, norm.vmax)
    cax.set_xticks([])
    if ticks is not None:
        cax.set_yticks(ticks)
    cax.yaxis.tick_right(); cax.yaxis.set_label_position("right")
    cax.set_ylabel(label)
    for spine in cax.spines.values():
        spine.set_linewidth(0.6)
    return cax


def fig_norm_ablation():
    df = pd.read_csv(RESULTS / "model_comparison.csv")
    primary = "Random Forest"
    rows = []
    for task in ["device", "category"]:
        for featset in ["raw", "norm"]:
            for proto, lbl in [("A", "Topology-matched"), ("B", "LOTO")]:
                r = df[(df.task == task) & (df.featset == featset) &
                       (df.model == primary) & (df.protocol.str.startswith(proto))]
                if len(r):
                    rows.append((f"per-{task}\n{lbl}", featset, r["accuracy_mean"].iloc[0]))
    d = pd.DataFrame(rows, columns=["cfg", "featset", "acc"])
    cfgs = d["cfg"].unique()
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(7, 3.0))
    for i, (fs, col, lbl) in enumerate([("raw", P.MUTED, "Raw spectrum"),
                                        ("norm", P.GOOD, "Baseline-normalized")]):
        vals = [d[(d.cfg == c) & (d.featset == fs)]["acc"].iloc[0] for c in cfgs]
        ax.bar(x + (i - 0.5) * 0.4, vals, width=0.4, label=lbl, color=col)
    ax.set_xticks(x); ax.set_xticklabels(cfgs)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(axis="y")
    fig.tight_layout(); fig.savefig(FIGS / "ml_norm_ablation.pdf"); plt.close(fig)
    print("  ml_norm_ablation.pdf")


def fig_device_confusion_supplement():
    """Full 76x76 per-device confusion matrices mentioned in the manuscript."""
    d = np.load(RESULTS / "predictions_device.npz", allow_pickle=True)
    for proto in ["A", "B"]:
        yt, yp = d[f"y_true_{proto}"], d[f"y_pred_{proto}"]
        labels = np.unique(np.concatenate([yt, yp]))
        cm = confusion_matrix(yt, yp, labels=labels)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        x = np.arange(len(labels) + 1)
        y = np.arange(len(labels) + 1)
        ax.pcolormesh(x, y, cmn, cmap=P.SEQ_CMAP, norm=norm,
                      edgecolors="white", linewidth=0.2, rasterized=False)
        ax.set_xlim(0, len(labels)); ax.set_ylim(len(labels), 0)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        title = "topology-matched" if proto == "A" else "leave-one-topology-out"
        ax.set_title(f"Per-device confusion ({title})", fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.90, 1])
        _vector_colorbar(fig, ax, P.SEQ_CMAP, norm, "Row-normalized fraction")
        name = "device_confusion_topology_matched.pdf" if proto == "A" else "device_confusion_loto.pdf"
        fig.savefig(FIGS / name); plt.close(fig)
        print(f"  {name}")


def fig_spectra_archetypes():
    d = np.load(CACHE / "spectra_full.npz", allow_pickle=True)
    X, dev, topo = d["X"], d["device_id"], d["topology"]
    freq = _freq(X.shape[2])
    sel = {0: "empty grid (flat)", 5: "LED bulb ON (light)",
           67: "drill ON (motor)", 46: "ozone gen. ON (SMPS)", 57: "PC ON (SMPS)"}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for did, lbl in sel.items():
        m = (dev == did) & (topo == 2)
        if not m.any():
            continue
        ax.plot(freq, X[m, 0, :].mean(axis=0), label=f"{did}: {lbl}")
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Mean amplitude (dBm/Hz)")
    ax.set_xlim(1.8, 80); ax.legend(ncol=2); ax.grid(True)
    fig.tight_layout(); fig.savefig(FIGS / "spectra_archetypes.pdf"); plt.close(fig)
    print("  spectra_archetypes.pdf")


def fig_strong_devices():
    """Empty grid vs strong emitters."""
    d = np.load(CACHE / "spectra_full.npz", allow_pickle=True)
    X, dev, topo = d["X"], d["device_id"], d["topology"]
    freq = _freq(X.shape[2])
    sel = [(0, "empty power grid", P.PRIMARY),
           (44, "assured power supply unit (ON)", P.GOOD),
           (57, "personal computer (ON)", P.PURPLE)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for did, lbl, col in sel:
        m = (dev == did) & (topo == 2)
        if not m.any():
            m = (dev == did)
        ax.plot(freq, X[m, 0, :].mean(axis=0), label=lbl, color=col)
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Mean amplitude (dBm/Hz)")
    ax.set_xlim(1.8, 80); ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(FIGS / "strong_devices.pdf"); plt.close(fig)
    print("  strong_devices.pdf")


def fig_band_importance():
    imp = pd.read_csv(RESULTS / "feature_importance.csv")
    band = imp.dropna(subset=["band_freq_mhz"]).groupby("band_freq_mhz")["importance"].sum()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(band.index, band.values, width=4, color=P.PURPLE, edgecolor="white", linewidth=0.4)
    ax.axvline(30, color=P.ACCENT, ls="--", lw=0.9)
    ax.text(30.8, ax.get_ylim()[1] * 0.92, "30 MHz", color=P.ACCENT, fontsize=7, va="top")
    ax.set_xlabel("Frequency band centre (MHz)")
    ax.set_ylabel("Summed importance")
    ax.set_xlim(0, 82); ax.grid(axis="y")
    fig.tight_layout(); fig.savefig(FIGS / "ml_band_importance.pdf"); plt.close(fig)
    print("  ml_band_importance.pdf")


def fig_div_accuracy():
    """Device accuracy vs number of 1.8-80 MHz sub-bands."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    d = np.load(CACHE / "spectra_full.npz", allow_pickle=True)
    X, dev = d["X"], d["device_id"]
    nbands = [1, 2, 4, 8, 16, 32, 64]
    accs = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for nb in nbands:
        edges = np.linspace(0, X.shape[2], nb + 1).astype(int)
        cols = []
        for ch in range(X.shape[1]):
            for b in range(nb):
                seg = X[:, ch, edges[b]:edges[b + 1]]
                cols += [seg.mean(1), seg.std(1), seg.min(1), seg.max(1),
                         np.median(seg, 1), seg.max(1) - seg.min(1)]
        Xf = np.column_stack(cols)
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        acc = cross_val_score(clf, Xf, dev, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        accs.append(acc)
        print(f"    nbands={nb:3d}  acc={acc:.3f}")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(nbands, accs, marker="o", color=P.PRIMARY)
    ax.axvline(16, color=P.ACCENT, ls="--", lw=0.9, label="16 sub-bands (used)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(nbands); ax.set_xticklabels(nbands)
    ax.set_xlabel("Number of sub-bands"); ax.set_ylabel("Per-device accuracy")
    ax.legend(loc="lower right"); ax.grid(True)
    fig.tight_layout(); fig.savefig(FIGS / "div_accuracy.pdf"); plt.close(fig)
    print("  div_accuracy.pdf")


def fig_feature_embedding():
    """2-D PCA of the 232-D features: (a) separable by category, (b) shifted by topology."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    df = pd.read_csv(CACHE / "features_norm.csv")
    meta = ["device_id", "device", "state", "category", "device_group_id",
            "rated_power_w", "topology", "repetition"]
    Xcols = [c for c in df.columns if c not in meta]
    X = np.nan_to_num(df[Xcols].to_numpy(dtype=float))
    Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    cats = df["category"].fillna("other").replace("electronics", "other").to_numpy()
    topo = df["topology"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    for i, c in enumerate(sorted(set(cats))):
        m = cats == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=6, color=P.CYCLE[i % len(P.CYCLE)],
                        alpha=0.6, linewidths=0, label=str(c))
    axes[0].legend(fontsize=6, markerscale=1.8, loc="best")
    axes[0].set_title("(a) coloured by device category", fontsize=9)
    for i, t in enumerate(sorted(set(topo))):
        m = topo == t
        axes[1].scatter(Z[m, 0], Z[m, 1], s=6, color=P.CYCLE[i % len(P.CYCLE)],
                        alpha=0.6, linewidths=0, label=f"topology {int(t)}")
    axes[1].legend(fontsize=7, markerscale=1.8, loc="best")
    axes[1].set_title("(b) coloured by installation (topology)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("PC 1"); ax.grid(True)
    axes[0].set_ylabel("PC 2")
    fig.tight_layout(); fig.savefig(FIGS / "feature_embedding.pdf"); plt.close(fig)
    print("  feature_embedding.pdf")


def fig_channel_coloring():
    """Same strong emitter measured in three topologies -> the channel colours the fingerprint."""
    d = np.load(CACHE / "spectra_full.npz", allow_pickle=True)
    X, dev, topo = d["X"], d["device_id"], d["topology"]
    freq = _freq(X.shape[2])
    did = 46  # ozone generator ON -- strong, distinctive emitter
    fig, ax = plt.subplots(figsize=(7, 3.0))
    for i, t in enumerate(sorted(set(topo))):
        m = (dev == did) & (topo == t)
        if m.any():
            ax.plot(freq, X[m, 0, :].mean(axis=0), color=P.CYCLE[i], label=f"topology {int(t)}")
    m0 = (dev == 0)
    ax.plot(freq, X[m0, 0, :].mean(axis=0), color=P.MUTED, ls="--", label="empty grid (ref.)")
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Mean amplitude (dBm/Hz)")
    ax.set_xlim(1.8, 80); ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(FIGS / "channel_coloring.pdf"); plt.close(fig)
    print("  channel_coloring.pdf")


def fig_recall_vs_strength():
    """Cold-LOTO per-device recall vs emission strength, in the unified style
    (the audit writes audit_recall_vs_strength.csv)."""
    csv = RESULTS / "audit_recall_vs_strength.csv"
    if not csv.exists():
        print("  (skip recall_vs_strength: csv not found)"); return
    t = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.scatter(t["emission_strength_dB"], t["loto_recall"], s=22, alpha=0.75,
               color=P.PRIMARY, edgecolors="none")
    ax.set_xlabel(r"Emission strength $\Delta_d$ (dB)")
    ax.set_ylabel("Cold-LOTO per-device recall")
    ax.set_ylim(-0.03, 1.03); ax.grid(True)
    fig.tight_layout(); fig.savefig(FIGS / "audit_recall_vs_strength.pdf"); plt.close(fig)
    print("  audit_recall_vs_strength.pdf")


if __name__ == "__main__":
    print("Generating figures into figures/ ...")
    fig_spectra_archetypes()
    fig_strong_devices()
    fig_norm_ablation()
    fig_band_importance()
    fig_div_accuracy()
    fig_feature_embedding()
    fig_channel_coloring()
    fig_recall_vs_strength()
    fig_device_confusion_supplement()
    print("done.")
