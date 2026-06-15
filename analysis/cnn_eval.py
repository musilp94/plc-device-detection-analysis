"""
1D-CNN comparison on downsampled noise spectra (baseline-normalized).
Appends results to analysis/results/model_comparison.csv as model "1D-CNN".

Same protocols as ml_evaluation.py:
  A  topology-matched : 5-fold stratified CV (pooled topologies)
  B  leave-one-topology-out
Tasks: per-device, per-category.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "analysis" / "cache"
RESULTS = REPO / "analysis" / "results"
SEED = 13
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cpu"


class CNN1D(nn.Module):
    def __init__(self, n_classes, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def standardize(tr, te):
    mu = tr.mean(axis=(0, 2), keepdims=True)
    sd = tr.std(axis=(0, 2), keepdims=True) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd


def train_eval(Xtr, ytr, Xte, n_classes, epochs=60):
    Xtr, Xte = standardize(Xtr, Xte)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    model = CNN1D(n_classes).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    return pred


def metrics(yt, yp):
    return {
        "accuracy": accuracy_score(yt, yp),
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
        "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
        "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
    }


def agg(folds):
    from scipy import stats as st
    out = {}
    n = len(folds)
    for k in folds[0]:
        v = np.array([f[k] for f in folds])
        out[f"{k}_mean"] = v.mean()
        out[f"{k}_std"] = v.std(ddof=1) if n > 1 else 0.0
        out[f"{k}_ci95"] = (st.t.ppf(0.975, n-1) * v.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0
    return out


def main():
    d = np.load(CACHE / "spectra_full.npz", allow_pickle=True)
    Xf = d["X"]                       # (n,2,3201) full-res raw spectra
    topo = d["topology"]; dev = d["device_id"]

    # baseline-normalize (subtract per-topology empty-grid mean; matches the
    # engineered 'norm' feature set) then downsample to 512 points for the CNN
    base = {t: Xf[(dev == 0) & (topo == t)].mean(axis=0) for t in np.unique(topo)}
    Xn = np.stack([Xf[i] - base[topo[i]] for i in range(len(Xf))]).astype(np.float32)
    L = 512
    idx = np.linspace(0, Xf.shape[2] - 1, L).astype(int)
    X = Xn[:, :, idx]

    cat = np.array(["other" if (c is None or str(c) in ("nan", "electronics")) else c
                    for c in d["category"]])
    rows = []
    for task, raw_y in [("device", dev), ("category", cat)]:
        y = LabelEncoder().fit_transform(raw_y)
        n_classes = len(np.unique(y))

        # Protocol A: 5-fold stratified
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        foldsA = []
        for tr, te in skf.split(X, y):
            yp = train_eval(X[tr], y[tr], X[te], n_classes)
            foldsA.append(metrics(y[te], yp))
        rows.append({"task": task, "featset": "norm", "protocol": "A_topology_matched",
                     "model": "1D-CNN", "n_samples": len(y), "n_classes": n_classes,
                     **agg(foldsA)})
        print(f"[{task}|A|1D-CNN] acc={rows[-1]['accuracy_mean']:.3f}", flush=True)

        # Protocol B: LOTO
        foldsB = []
        for held in sorted(np.unique(topo)):
            tr = topo != held; te = topo == held
            yp = train_eval(X[tr], y[tr], X[te], n_classes)
            foldsB.append(metrics(y[te], yp))
        rows.append({"task": task, "featset": "norm", "protocol": "B_leave_one_topo_out",
                     "model": "1D-CNN", "n_samples": len(y), "n_classes": n_classes,
                     **agg(foldsB)})
        print(f"[{task}|B|1D-CNN] acc={rows[-1]['accuracy_mean']:.3f}", flush=True)

    # append to existing comparison
    comp = RESULTS / "model_comparison.csv"
    new = pd.DataFrame(rows)
    if comp.exists():
        old = pd.read_csv(comp)
        old = old[old["model"] != "1D-CNN"]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(comp, index=False)
    print(f"appended 1D-CNN rows to {comp}")


if __name__ == "__main__":
    main()
