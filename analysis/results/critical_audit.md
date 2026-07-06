# Critical, independent audit of the PLC device-detection method
_Dataset: Zenodo v1; model: LightGBM; seed 13. Generated automatically._

## Test 1 — Pseudo-replication / temporal leakage

- Per-device accuracy, RANDOM 30% holdout      : **0.898**
- Per-device accuracy, TEMPORAL holdout (late reps): **0.886**
- Drop random→temporal: +0.012
- Predict early-vs-late repetition half from spectrum (balanced acc, chance=0.5): **0.495**

  - Temporal and random holdout are similar → repetitions are stable; within-session split is not the main inflator (the topology gap, Test 3, dominates).
  - Repetition order is ~not predictable → little intra-session temporal confound.

## Test 2 — What actually generalizes (cold LOTO): strong-emitter condition

- Cold-LOTO per-device recall vs emission strength: Spearman rho=0.49 (p=6.5e-06)
- Cold-LOTO per-device recall vs rated power:       Spearman rho=0.22 (p=7.5e-02)
- Mean LOTO recall, strong emitters (≥median dev): **0.268**  | weak emitters (<median): **0.083**
- Best-generalizing devices (cold LOTO):
    - 57 personal computer 1: recall=1.00, dev=7.2 dB
    - 70 electricity meter 3: recall=1.00, dev=5.4 dB
    - 46 ozone generator: recall=0.97, dev=29.3 dB
    - 69 electricity meter 2: recall=0.95, dev=7.0 dB
    - 19 power adapter: recall=0.80, dev=6.9 dB
    - 47 Power Quality monitor: recall=0.73, dev=23.1 dB
    - 44 assured power supply unit: recall=0.70, dev=15.6 dB
    - 29 coffee grinder 3: recall=0.67, dev=1.6 dB

## Test 3 — Leakage-free (fold-safe) normalization under LOTO

- device: cold-LOTO accuracy  raw=0.171  fold-safe-normalized=0.182  (delta=+0.011)
- category: cold-LOTO accuracy  raw=0.484  fold-safe-normalized=0.466  (delta=-0.018)

  Interpretation: if fold-safe Δ is clearly positive, a *leakage-free* per-site empty-grid calibration genuinely helps cold deployment; if it collapses vs the global-baseline number, the earlier gain was partly leakage.

## Test 4 — Error structure: are cold-LOTO errors at least within-type?

- Of cold-LOTO device misclassifications, 28% land in the CORRECT category.
  → the model captures device *type* far better than device *identity* even when topology is unseen.

## Test 5 — Leave-one-physical-device-out: category of a never-seen device

- Category accuracy on never-seen devices (LOPDO): **0.531** (balanced: 0.355); topology-matched reference with known devices is ~0.99
- Mean per-device recall by category:
    - light_source: 0.64  (11 physical devices)
    - motor: 0.65  (8 physical devices)
    - other: 0.00  (2 physical devices)
    - smps: 0.54  (15 physical devices)
    - ups: 0.00  (2 physical devices)

  Interpretation: types with several distinct exemplars in the data generalize partially; categories represented by only two physical devices do not generalize at all — the category fingerprint is partly device-specific, and the topology-matched per-category accuracy should be read as performance over a *known* device inventory.

## Test 6 — Installation (topology) separability from the readout

- Topology (3-class) predicted from the 232-D features, all samples: **0.991** (stratified 5-fold CV)
- Topology predicted from the empty-grid measurements alone: **0.967**

  Interpretation: the installation is (all but) perfectly identifiable from the readout itself — the three iterations are as separable in the feature space as the devices are, i.e. they realize genuinely distinct channels, and this covariate shift is what a pre-trained classifier meets in an unseen installation.

## Verdict — usability and conditions

See the per-test interpretations above. Key conditions for reliable use are summarized in the submitted manuscript. Raw numbers in audit_*.csv.
