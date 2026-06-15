"""Generate a per-device spectral-characterization LaTeX longtable
(-> analysis/results/tab_device_spectral.tex), to accompany the dataset table."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
stats = pd.read_csv(REPO / "dataset" / "summary_statistics.csv")
audit = pd.read_csv(REPO / "analysis" / "results" / "audit_recall_vs_strength.csv")
meta = pd.read_csv(REPO / "dataset" / "metadata.csv", index_col="id")

# mean level at the direct-connection topology, channel 1
lvl = (stats[(stats.topology == 2) & (stats.channel == 1)]
       .set_index("device_id")["mean_dBm"])

rows = []
for _, r in audit.sort_values("device_id").iterrows():
    did = int(r.device_id)
    state = meta.loc[did, "state"] if did in meta.index else ""
    state = "--" if pd.isna(state) else str(state)
    name = str(r.device).replace("&", "and")
    rows.append((did, name, state,
                 lvl.get(did, np.nan), r.emission_strength_dB, r.loto_recall))

lines = [
    r"\begin{longtable}{|c|p{.30\textwidth}|c|c|c|c|}",
    r"\caption{Per-device spectral characterization measured at the direct-connection topology (channel~1): mean noise level, spectral prominence $\Delta$ (mean absolute deviation of the band means from the empty grid), and the cross-installation (leave-one-topology-out) recall. Strong, distinctive emitters (large $\Delta$) are those that remain detectable across installations.\label{tab:device_spectral}}\\",
    r"\hline",
    r"ID & Device & State & Level [dBm/Hz] & $\Delta$ [dB] & LOTO recall \\ \hline\hline \endhead",
]
for did, name, state, level, delta, recall in rows:
    lvl_s = "--" if pd.isna(level) else f"{level:.1f}"
    lines.append(f"{did} & {name} & {state} & {lvl_s} & {delta:.1f} & {recall:.2f} \\\\ \\hline")
lines.append(r"\end{longtable}")

out = REPO / "analysis" / "results" / "tab_device_spectral.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(rows)} device rows")
