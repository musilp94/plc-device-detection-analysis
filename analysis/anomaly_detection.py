"""
Per-installation unauthorized-device detection (reanalysis of the existing
dataset).

Operationalizes the honest conclusion of the paper --- that the method is a
dependable *per-installation* technique --- as the application named in the
abstract: detection of unauthorized / anomalous loads (physical-layer security
of a fixed installation). Within a single installation we train a ONE-CLASS
detector on "normal" traffic only --- the empty grid plus a set of A authorized
devices --- and measure how well it flags the connection of a previously-unseen,
*unauthorized* device as an anomaly. No anomaly is ever seen during training.

Anomaly score = Mahalanobis distance of the observed spectrum from the normal
class in a PCA subspace. This is the same Mahalanobis-type distance eta_d by
which the channel-aware model (Section III) measures device separability, so the
detector is the model's own decision rule. Isolation Forest is reported as a
model-agnostic cross-check.

Protocol = per-installation (topology-matched): normal and test come from the
SAME topology. Authorized/unauthorized is split at the physical-device level
(device_group_id) and rotated over many random splits and all three
installations; we sweep the size A of the authorized set, because a realistic
monitored installation (cabinet, substation) has a small, fixed set of
legitimate loads. Mean +/- 95% CI over topology x split.

Outputs:
  analysis/results/anomaly_vs_authsize.csv     (metrics vs authorized-set size)
  analysis/results/anomaly_per_device.csv      (per-device detection vs strength)
  figures/anomaly_detection.pdf     (AUC vs A ; detection vs Delta_d)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "analysis" / "cache"
RESULTS = REPO / "analysis" / "results"
FIGS = REPO / "figures"
SEED = 13
N_SPLITS = 25
TRAIN_FRAC = 0.6
PCA_VAR = 0.99
AUTH_SIZES = [3, 5, 8, 12, 16, 20]
A_DEVICE = 5                # authorized-set size used for the per-device panel
META = ["device_id", "device", "state", "category", "device_group_id",
        "rated_power_w", "topology", "repetition"]


def maha_scorer(Xtr):
    pca = PCA(n_components=PCA_VAR, svd_solver="full").fit(Xtr)
    Z = pca.transform(Xtr)
    VI = np.linalg.pinv(LedoitWolf().fit(Z).covariance_)
    mu = Z.mean(0)
    return lambda A: (lambda D: np.einsum("ij,jk,ik->i", D, VI, D))(pca.transform(A) - mu)


def tpr_at_fpr(y, s, q):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(q, fpr, tpr))


def ci(v):
    v = np.asarray(v, float); n = len(v)
    return v.mean(), (sstats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)


def main():
    df = pd.read_csv(CACHE / "features_raw.csv")
    feat = [c for c in df.columns if c not in META]
    X = df[feat].to_numpy(float)
    topo = df["topology"].to_numpy(); grp = df["device_group_id"].to_numpy()
    phys = np.array(sorted(g for g in np.unique(grp) if g != 0))
    rng = np.random.default_rng(SEED)

    rows = []
    det_hit = {g: 0 for g in phys}; det_tot = {g: 0 for g in phys}

    for A in AUTH_SIZES:
        for t in (1, 2, 3):
            in_t = topo == t
            for _ in range(N_SPLITS):
                perm = rng.permutation(phys)
                auth, unauth = set(perm[:A]), set(perm[A:])
                ni = np.where(in_t & ((grp == 0) | np.isin(grp, list(auth))))[0]
                ai = np.where(in_t & np.isin(grp, list(unauth)))[0]
                rng.shuffle(ni); c = int(TRAIN_FRAC * len(ni)); tr, neg = ni[:c], ni[c:]
                sc = StandardScaler().fit(X[tr])
                Xtr, Xneg, Xpos = sc.transform(X[tr]), sc.transform(X[neg]), sc.transform(X[ai])
                y = np.r_[np.zeros(len(neg)), np.ones(len(ai))]

                f = maha_scorer(Xtr); sM = np.r_[f(Xneg), f(Xpos)]
                sIF = -IsolationForest(n_estimators=200, random_state=SEED).fit(Xtr)\
                        .decision_function(np.r_[Xneg, Xpos])
                for name, s in (("Mahalanobis", sM), ("IsolationForest", sIF)):
                    rows.append({"A": A, "topology": t, "detector": name,
                                 "auc": roc_auc_score(y, s),
                                 "ap": average_precision_score(y, s),
                                 "tpr5": tpr_at_fpr(y, s, 0.05)})
                # per-device detection rate (Mahalanobis, FPR=5% threshold) at A_DEVICE
                if A == A_DEVICE:
                    thr = np.quantile(sM[:len(neg)], 0.95)
                    pos = sM[len(neg):]
                    for g in unauth:
                        m = grp[ai] == g
                        det_hit[g] += int((pos[m] > thr).sum()); det_tot[g] += int(m.sum())

    res = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    agg = (res.groupby(["A", "detector"])
              .agg(auc_m=("auc", "mean"), auc_h=("auc", lambda v: ci(v)[1]),
                   ap_m=("ap", "mean"), ap_h=("ap", lambda v: ci(v)[1]),
                   tpr5_m=("tpr5", "mean"), tpr5_h=("tpr5", lambda v: ci(v)[1]))
              .reset_index())
    agg.to_csv(RESULTS / "anomaly_vs_authsize.csv", index=False)
    print("=== Unauthorized-device detection vs authorized-set size A (mean +/- 95% CI) ===")
    for _, r in agg[agg.detector == "Mahalanobis"].iterrows():
        print(f"  A={int(r.A):2d}  AUC={r.auc_m:.3f}+/-{r.auc_h:.3f}  "
              f"AP={r.ap_m:.3f}  TPR@5%FAR={r.tpr5_m:.3f}")

    # per-device detection rate vs emission strength
    strength = pd.read_csv(RESULTS / "audit_recall_vs_strength.csv")
    s_by_dev = strength.set_index("device_id")["emission_strength_dB"]
    did2grp = df.drop_duplicates("device_id").set_index("device_id")["device_group_id"]
    grp_strength = s_by_dev.groupby(did2grp).max()
    name_map = df.drop_duplicates("device_group_id").set_index("device_group_id")["device"]
    pdv = pd.DataFrame({"device_group_id": list(det_hit),
                        "det_rate": [det_hit[g] / max(det_tot[g], 1) for g in det_hit]})
    pdv["emission_strength_dB"] = pdv["device_group_id"].map(grp_strength)
    pdv["device"] = pdv["device_group_id"].map(name_map)
    pdv = pdv.dropna(subset=["emission_strength_dB"]).sort_values("emission_strength_dB")
    pdv.to_csv(RESULTS / "anomaly_per_device.csv", index=False)
    rho, p = sstats.spearmanr(pdv.emission_strength_dB, pdv.det_rate)
    med = pdv.emission_strength_dB.median()
    print(f"\nper-device detection (A={A_DEVICE}) vs Delta_d: Spearman rho={rho:.2f} (p={p:.1e})")
    print(f"  mean detection @5%FAR  strong={pdv[pdv.emission_strength_dB>=med].det_rate.mean():.2f}"
          f"  weak={pdv[pdv.emission_strength_dB<med].det_rate.mean():.2f}")

    # ---------- figure ----------
    import sys; sys.path.insert(0, str(REPO / "analysis"))
    import figstyle; figstyle.apply()
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.85))
    for det, col, mk in (("Mahalanobis", figstyle.PRIMARY, "o"),
                         ("IsolationForest", figstyle.MUTED, "s")):
        d = agg[agg.detector == det]
        axA.errorbar(d.A, d.auc_m, yerr=d.auc_h, color=col, marker=mk, ms=4, lw=1.5,
                     capsize=2, label=det.replace("Isolation", "Isolation "))
    axA.axhline(0.5, ls=":", color=figstyle.MUTED, lw=1)
    axA.set_xlabel("Number of authorized devices $A$")
    axA.set_ylabel("Detection ROC-AUC")
    axA.set_ylim(0.45, 1.0); axA.legend(loc="upper right", frameon=False)
    axA.set_title("(a) unauthorized-device detection")
    axB.scatter(pdv.emission_strength_dB, pdv.det_rate, s=16, color=figstyle.PRIMARY,
                alpha=0.8, edgecolor="none")
    axB.set_xlabel(r"Emission strength $\Delta_d$ (dB)")
    axB.set_ylabel("Detection rate @ 5% FAR")
    axB.set_ylim(-0.03, 1.03)
    axB.set_title(f"(b) by device, $A={A_DEVICE}$ ($\\rho={rho:.2f}$)")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "anomaly_detection.pdf", bbox_inches="tight")
    print(f"\nwrote {FIGS/'anomaly_detection.pdf'}")


if __name__ == "__main__":
    main()
