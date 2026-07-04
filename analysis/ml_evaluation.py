"""
Rigorous ML evaluation for PLC device detection.
================================================

Reads cached feature matrices (analysis/cache/features_{raw,norm}.csv) and runs
a full comparative study:

Tasks        : per-device (76 classes incl. empty grid) ; per-category (5)
Feature sets : raw spectrum  vs  baseline-normalized (device − empty grid)
Models       : Dummy, LogReg, SVM-RBF, RandomForest, HistGB*,
               CatBoost*, LightGBM, MLP           (*category task only)
Protocols    :
  A  topology-matched : stratified 5-fold CV over pooled topologies
                        (optimistic — deployment topology seen in training)
  B  leave-one-topology-out : train on 2 topologies, test on the 3rd, rotate
                        (honest cold-deployment generalization)

Outputs (analysis/results/):
  model_comparison.csv        — every (task, featset, protocol, model) × metrics
  predictions_*.npz           — out-of-fold y_true/y_pred for confusion matrices
  feature_importance.csv      — CatBoost importance, aggregated by band & stat
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "analysis" / "cache"
RESULTS = REPO / "analysis" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

META_COLS = ["device_id", "device", "state", "category", "device_group_id",
             "rated_power_w", "topology", "repetition"]
RANDOM_STATE = 13


# --------------------------------------------------------------------------
def load_features(featset: str) -> pd.DataFrame:
    df = pd.read_csv(CACHE / f"features_{featset}.csv")
    # five load types: the lone "electronics" sample (speakers) belongs to the
    # catch-all "other" class, as described in the manuscript ("electronics/other").
    df["category"] = df["category"].fillna("other").replace("electronics", "other")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def get_models(task: str, n_jobs: int = -1) -> dict:
    scaler = lambda est: make_pipeline(StandardScaler(), est)
    models = {
        "Dummy (stratified)": DummyClassifier(strategy="stratified",
                                              random_state=RANDOM_STATE),
        "Logistic Regression": scaler(LogisticRegression(max_iter=3000, C=1.0,
                                                          n_jobs=n_jobs)),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=n_jobs,
                                                random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=300, n_jobs=n_jobs,
                                   random_state=RANDOM_STATE, verbose=-1),
        "MLP": scaler(MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                                    random_state=RANDOM_STATE, early_stopping=True)),
    }
    # CatBoost and HistGB are evaluated on the 5-class category task only:
    # one tree per class per boosting round makes them slow at 76 classes on
    # CPU, and LightGBM already represents gradient boosting on the per-device
    # task. The RBF SVM is evaluated on both tasks (a 76-class fit takes only
    # a few seconds despite the one-vs-one expansion).
    if task == "category":
        models["CatBoost"] = CatBoostClassifier(iterations=300, depth=6,
                                                learning_rate=0.1, verbose=0,
                                                random_state=RANDOM_STATE,
                                                thread_count=-1)
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=300, random_state=RANDOM_STATE)
    models["SVM (RBF)"] = scaler(SVC(kernel="rbf", C=10, gamma="scale"))
    return models


def metric_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def aggregate(per_fold: list[dict]) -> dict:
    out = {}
    n = len(per_fold)
    for key in per_fold[0]:
        vals = np.array([d[key] for d in per_fold], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        ci = (sstats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
        out[f"{key}_ci95"] = ci
    return out


def stamp(msg: str) -> None:
    """Timestamped progress line -> console + analysis/results/_progress.log."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "_progress.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------
def protocol_A(model_factory, X, y, n_splits=5, n_repeats=1) -> list[dict]:
    """Topology-matched stratified 5-fold CV."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE)
    per_fold = []
    for tr, te in rskf.split(X, y):
        m = model_factory()
        m.fit(X[tr], y[tr])
        per_fold.append(metric_dict(y[te], m.predict(X[te])))
    return per_fold


def protocol_B(model_factory, X, y, topo) -> list[dict]:
    """Leave-one-topology-out."""
    per_fold = []
    for held in sorted(np.unique(topo)):
        tr = topo != held
        te = topo == held
        m = model_factory()
        m.fit(X[tr], y[tr])
        per_fold.append(metric_dict(y[te], m.predict(X[te])))
    return per_fold


# --------------------------------------------------------------------------
def run_comparison():
    rows = []
    out_csv = RESULTS / "model_comparison.csv"

    for task, label_col in [("device", "device_id"), ("category", "category")]:
        for featset in ["raw", "norm"]:
            df = load_features(featset)
            feat_cols = feature_columns(df)
            X = df[feat_cols].to_numpy()
            y = df[label_col].to_numpy()
            topo = df["topology"].to_numpy()
            models = get_models(task)
            stamp(f"START  {task}/{featset}  "
                  f"({len(df)} samples, {df[label_col].nunique()} classes, "
                  f"{len(models)} models)")

            for mname, template in models.items():
                # fresh estimator each fold via clone-like re-instantiation
                from sklearn.base import clone
                factory = (lambda t=template: clone(t))

                for proto_name, runner in [("A_topology_matched", protocol_A),
                                           ("B_leave_one_topo_out", protocol_B)]:
                    t0 = time.time()
                    try:
                        if proto_name.startswith("A"):
                            pf = protocol_A(factory, X, y)
                        else:
                            pf = protocol_B(factory, X, y, topo)
                        agg = aggregate(pf)
                        row = {"task": task, "featset": featset,
                               "protocol": proto_name, "model": mname,
                               "n_samples": len(y), "n_classes": len(np.unique(y)),
                               "seconds": round(time.time() - t0, 1), **agg}
                    except Exception as e:
                        row = {"task": task, "featset": featset,
                               "protocol": proto_name, "model": mname,
                               "error": str(e)[:200]}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)  # incremental save
                    acc = row.get("accuracy_mean", float("nan"))
                    f1 = row.get("f1_macro_mean", float("nan"))
                    stamp(f"  {task}/{featset} | {proto_name[0]} | {mname}: "
                          f"acc={acc:.3f} f1={f1:.3f} ({row.get('seconds','?')}s)")
    print(f"\nSaved {out_csv}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def save_predictions_for_confusion():
    """Out-of-fold predictions for the primary model (LightGBM, norm features)."""
    from sklearn.base import clone
    for task, label_col in [("device", "device_id"), ("category", "category")]:
        df = load_features("norm")
        feat_cols = feature_columns(df)
        X = df[feat_cols].to_numpy()
        y = df[label_col].to_numpy()
        topo = df["topology"].to_numpy()
        base = LGBMClassifier(n_estimators=300, n_jobs=-1,
                              random_state=RANDOM_STATE, verbose=-1)

        # Protocol A: out-of-fold via 5-fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        yp_A = np.empty_like(y)
        for tr, te in skf.split(X, y):
            m = clone(base); m.fit(X[tr], y[tr]); yp_A[te] = m.predict(X[te]).ravel()

        # Protocol B: LOTO test predictions
        yt_B, yp_B = [], []
        for held in sorted(np.unique(topo)):
            tr = topo != held; te = topo == held
            m = clone(base); m.fit(X[tr], y[tr])
            yt_B.append(y[te]); yp_B.append(m.predict(X[te]).ravel())
        yt_B = np.concatenate(yt_B); yp_B = np.concatenate(yp_B)

        np.savez(RESULTS / f"predictions_{task}.npz",
                 y_true_A=y, y_pred_A=yp_A, y_true_B=yt_B, y_pred_B=yp_B,
                 labels=np.unique(y).astype(str))
        stamp(f"saved predictions_{task}.npz  "
              f"(A acc={accuracy_score(y, yp_A):.3f}, "
              f"B acc={accuracy_score(yt_B, yp_B):.3f})")


# --------------------------------------------------------------------------
def feature_importance():
    df = load_features("norm")
    feat_cols = feature_columns(df)
    X = df[feat_cols].to_numpy()
    y = df["device_id"].to_numpy()
    m = LGBMClassifier(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
                       verbose=-1, importance_type="gain")
    m.fit(X, y)
    imp = pd.DataFrame({"feature": feat_cols, "importance": m.feature_importances_})

    # aggregate by band centre frequency and by stat
    def band_freq(feat):
        # ch1_b5_10_mean -> 7.5 ; diff_b5_10_mean -> 7.5
        parts = feat.split("_")
        try:
            i = [p.startswith("b") for p in parts].index(True)
            lo = float(parts[i][1:]); hi = float(parts[i + 1])
            return (lo + hi) / 2
        except Exception:
            return np.nan

    imp["band_freq_mhz"] = imp["feature"].map(band_freq)
    imp["stat"] = imp["feature"].map(lambda f: f.rsplit("_", 1)[-1])
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(RESULTS / "feature_importance.csv", index=False)
    print("saved feature_importance.csv  | top 8:")
    print(imp.head(8).to_string(index=False))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (RESULTS / "_progress.log").write_text("", encoding="utf-8")
    stamp(f"=== START {start_str} : model comparison ===")
    run_comparison()
    stamp("=== model comparison DONE -> predictions for confusion matrices ===")
    save_predictions_for_confusion()
    stamp("=== predictions DONE -> feature importance ===")
    feature_importance()
    stamp("=== feature importance DONE ===")
    end_dt = datetime.now()
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    mins = (time.time() - t0) / 60
    msg = f"ALL DONE at {end_str}  (elapsed {mins:.1f} min, started {start_str})"
    print(f"\n{msg}")
    (RESULTS / "_DONE.txt").write_text(msg + "\n", encoding="utf-8")
