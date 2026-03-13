# Hawkes Process Seizure Early Warning System

**Author:** Rishika Singh  
**Dataset:** [CHB-MIT Scalp EEG](https://physionet.org/content/chbmit/1.0.0/)  
**Status:** Ongoing research with expanded dataset validation and continued algorithmic refinement

---

## Overview

This project models EEG spike trains as Hawkes processes to detect rising neural excitability before seizure onset. The core insight is that the **branching ratio η = α/β** — a single parameter capturing how much each spike excites future spikes — rises measurably before seizures in patients with focal epilepsy and stable inter-ictal baselines.

The system uses a **two-stage probabilistic detection framework**:

- **Stage 1 — Hypothesis:** adaptive rolling baseline detects sustained η elevation personalised to each patient, issuing a probabilistic early warning
- **Stage 2 — Verification:** fine-resolution window updates P(seizure) continuously, confirming or rejecting the hypothesis automatically

This eliminates false positives that single-stage threshold methods cannot distinguish from genuine pre-seizure dynamics.

---

## Key Results

| Recording | Channel | Hypothesis Lead | P(seizure) | Status |
|---|---|---|---|---|
| chb01_03 | F7-T7 | 46s before seizure | 0.89 | confirmed |
| chb01_03 | FP1-F7 | 96s before seizure | — | supercritical |
| chb01_03 | C3-P3 | t=1750 false rise | 0.09 | **rejected** ✓ |
| chb01_18 | FP2-F4 | 70s before seizure | 0.89 | confirmed |

The two-stage system correctly suppressed a false positive on C3-P3 that single-stage adaptive detection would have reported 1246 seconds early — demonstrating the clinical value of probabilistic verification.

---

## Repository Structure

```
hawkes-seizure-detection/
│
├── hawkes_final_clean.py     # complete pipeline — run this
│
├── docs/
│   ├── THEORY.md             # mathematical background
│   ├── ALGORITHM.md          # system architecture and pseudocode
│   └── PHENOTYPES.md         # patient phenotype analysis
│
├── figures/                  # η timeseries plots (add after running)
│
├── requirements.txt          # dependencies
└── README.md
```

---

## Quick Start

```bash
pip install mne numpy scipy matplotlib
```

1. Download a recording from [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/)
2. Open `hawkes_final_clean.py`
3. Update **Cell 2 only**:

```python
EDF_FILE      = '/path/to/your/recording.edf'
SEIZURE_START = 2996    # from CHB-MIT summary file
SEIZURE_END   = 3036
PATIENT_ID    = 'chb01_03'
```

4. Run all cells in order. All other cells stay fixed across recordings and patients.

---

## How It Works

### 1. EEG to Spike Train
Each EEG channel is bandpass filtered (80–120 Hz) to isolate high-frequency neural activity, normalised in rolling 60-second windows, and thresholded at 3 standard deviations to extract spike times.

### 2. Sliding Window MLE
A 200-second window slides across the spike train in 50-second steps. At each position, Hawkes parameters (μ, α, β) are fitted by maximum likelihood estimation using L-BFGS-B with 10 random restarts. This produces a timeseries of η estimates tracking neural excitability over time.

### 3. Two-Stage Detection
**Stage 1** raises a hypothesis when three adaptive conditions hold for two consecutive windows: η is elevated above the patient's own rolling baseline, η is actively rising, and no recent suppression period is present. **Stage 2** switches to a 100-second fine window with 20-second steps and updates P(seizure) using a probabilistic accumulation rule, confirming when P ≥ 0.85 and rejecting when P ≤ 0.15.

### 4. Supercritical Safety Net
Independently checks whether η ≥ 1.0 — the mathematical threshold at which the Hawkes process becomes supercritical. This is a Level 2 alert complementing the hypothesis stage.

---

## Alert Levels

| Level | Trigger | Clinical Response |
|---|---|---|
| Level 1 — Hypothesis | η rises above rolling baseline | Advisory: nurse checks patient |
| Level 2 — Confirmed | P(seizure) ≥ 0.85 | Intervention alert: full clinical response |
| Level 2 — Supercritical | η ≥ 1.0 | Mathematical instability confirmed |
| Rejected | P falls below 0.15 | Transient spike — no alert issued |

---

## Patient Eligibility

This method is validated for patients with **stable inter-ictal baselines and large pre-ictal η rises**. Based on CHB-MIT analysis, three η phenotypes exist with distinct detection requirements. See [docs/PHENOTYPES.md](docs/PHENOTYPES.md).

---

## Dependencies

```
mne >= 1.0
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.4
```

---

## Citation

If you use this code please cite:

> Singh, R. (2025). Hawkes Process Seizure Early Warning System. GitHub.

---

## Acknowledgements

CHB-MIT Scalp EEG Database: Shoeb, A. H. (2009). Application of machine learning to epileptic seizure onset detection and treatment. PhD Thesis, MIT.
