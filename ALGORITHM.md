# System Architecture and Algorithm

**Author:** Rishika Singh

---

## System Overview

```
EEG Recording (.edf)
        │
        ▼
┌───────────────────────┐
│   EEG Preprocessing   │  bandpass 80-120Hz → normalise → threshold
│   (per channel)       │  output: spike times {t₁, t₂, ..., tₙ}
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Sliding Window MLE   │  window=200s, step=50s
│                       │  fit μ, α, β at each position
│                       │  output: η timeseries {(cᵢ, ηᵢ)}
└───────────┬───────────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
┌─────────┐  ┌─────────────┐
│ Stage 1 │  │Supercritical│
│Adaptive │  │ Safety Net  │
│Hypothesis  │  η ≥ 1.0    │
└────┬────┘  └──────┬──────┘
     │               │
     │ hypothesis     │ Level 2 alert
     ▼               │
┌─────────────┐      │
│  Stage 2    │      │
│Probabilistic│      │
│Verification │      │
│ P(seizure)  │      │
└──────┬──────┘      │
       │             │
  ┌────┴────┐        │
  │         │        │
  ▼         ▼        ▼
CONFIRMED REJECTED SUPERCRITICAL
Level 2   No alert  Level 2
alert     issued    alert
```

---

## Stage 1 — Adaptive Hypothesis

**Purpose:** Detect sustained η elevation personalised to each patient's own baseline.

**Input:** η timeseries {(cᵢ, ηᵢ)}, i = baseline_window to N

**Algorithm:**

```
count = 0

for i in baseline_window to N:

    recent    = η[i-20 : i]                     # last 20 windows = 1000s
    baseline  = mean(recent)
    std       = std(recent)
    threshold = baseline + 1.5 × max(std, 0.05)  # personalised threshold

    cond_1 = η[i] > threshold                    # elevated above baseline
    cond_2 = η[i] > η[i-2]                       # actively rising
    cond_3 = min(η[i-10:i]) > 0.3               # no recent suppression

    if cond_1 AND cond_2 AND cond_3:
        count += 1
    else:
        count = 0

    if count >= 2:                               # 2 consecutive windows
        raise HYPOTHESIS at c[i-1]
        switch to Stage 2
```

**Output:** hypothesis_time (seconds before seizure)

**Why three conditions:**

```
Condition 1 alone → would fire on any η spike above mean
Condition 2 alone → would fire on any upward movement
Condition 3 alone → prevents post-suppression recovery triggering
All three together → requires sustained directional rise from stable baseline
```

---

## Stage 2 — Probabilistic Verification

**Purpose:** Distinguish sustained pre-seizure rises from transient neural fluctuations.

**Trigger:** Immediately after hypothesis raised.

**Window switch:** normal (200s/50s) → fine (100s/20s)

**Algorithm:**

```
P = 0.5                                   # initial probability at hypothesis

for each window after hypothesis_time:

    if η > threshold:
        P = P + (1 - P) × 0.4            # pull toward 1.0
    else:
        P = P - P × 0.4                  # pull toward 0.0

    print t, η, P

    if P ≥ 0.85:  CONFIRM → Level 2 alert
    if P ≤ 0.15:  REJECT  → transient, reset Stage 1
```

**Probability update intuition:**

```
Starting P = 0.50

After 1 window above threshold:   P = 0.50 + (0.50)(0.4) = 0.70
After 2 windows above threshold:  P = 0.70 + (0.30)(0.4) = 0.82
After 3 windows above threshold:  P = 0.82 + (0.18)(0.4) = 0.89 ← CONFIRMED

After 1 window above then below:  P = 0.70 - (0.70)(0.4) = 0.42
After 2 windows below:            P = 0.42 - (0.42)(0.4) = 0.25
After 3 windows below:            P = 0.25 - (0.25)(0.4) = 0.15 ← REJECTED
```

Genuine pre-seizure rises confirm in 3 windows (60 seconds at step=20s).
Transient spikes reject in 3–5 windows (60–100 seconds).

---

## Supercritical Safety Net

**Purpose:** Detect absolute instability regardless of baseline.

**Algorithm:**

```
for each window:
    if η ≥ 1.0:
        fire SUPERCRITICAL alert
```

**Interpretation:** η = 1.0 is the mathematical boundary of stability for a Hawkes process. Above this threshold each spike generates more than one descendant on average — excitation grows without bound. This is not a pre-seizure prediction but a statement about the current dynamical state.

**Clinical role:** Complementary to adaptive hypothesis. May detect patients or channels where η crosses 1.0 before the adaptive rise is large enough to trigger the hypothesis.

---

## Multi-Channel Architecture

The pipeline runs on all EEG channels in parallel using ThreadPoolExecutor.

```
for each channel:
    1. eeg_to_spikes(channel)
    2. sliding_window_eta(spikes)
    3. adaptive_window_detection(spikes)   → hypothesis, P, status
    4. critical_threshold_warning(etas)   → supercritical time
    5. compute lead times vs seizure_start
    6. store result

report:
    → all channels with valid detections
    → best channel = earliest hypothesis lead time
    → plot η timeseries for best channel
    → save full results to text file
```

**No channel pre-selection:** All channels run through the same pipeline. Post-hoc reporting of which channels detected is informational — the best result is the maximum lead time across all channels.

---

## EEG Preprocessing

```
Input:  raw EEG channel (voltage timeseries at 256 Hz)

Step 1: Bandpass filter 80–120 Hz
        → isolates high gamma band
        → captures fast neural oscillations associated with spike activity
        → removes slow drift, EMG artifacts, line noise

Step 2: Rolling normalisation (60-second windows)
        for each 60s chunk:
            normalised = (chunk - mean) / std
        → removes amplitude variation between windows
        → standardises threshold across recording duration

Step 3: Threshold detection at 3 standard deviations
        crossings = where |normalised| > 3.0
        → detects significant amplitude events

Step 4: Refractory period 50ms
        remove spikes < 50ms apart
        → prevents double-counting of single neural events
        → 50ms matches physiological refractory period

Output: spike times (seconds)
```

---

## Parameter Summary

All parameters are fixed across all patients and recordings. They are not tuned per patient.

| Parameter | Value | Justification |
|---|---|---|
| window_size | 200s | sufficient spikes for reliable MLE at typical rates |
| step_size | 50s | temporal resolution adequate for minutes-scale dynamics |
| min_spikes | 15 | minimum for L-BFGS-B convergence |
| n_restarts | 10 | sufficient to avoid local minima in MLE |
| baseline_window | 20 | 1000s rolling baseline captures inter-ictal dynamics |
| z_score | 1.5 | balances sensitivity and specificity |
| consecutive | 2 | requires 100s sustained rise to raise hypothesis |
| suppression_floor | 0.3 | blocks post-suppression recovery artefacts |
| alert_window | 100s | finer resolution for verification stage |
| alert_step | 20s | fast enough to verify within typical pre-ictal period |
| p_confirm | 0.85 | high confidence before Level 2 alert |
| p_reject | 0.15 | decisive evidence of transient before reset |

---

## Computational Complexity

**Per channel per recording:**

```
sliding_window_eta:
    N_windows = (T - window_size) / step_size ≈ 68 windows for 3600s
    per window: MLE with 10 restarts, n² matrix operations
    ≈ 5–8 minutes on standard CPU

adaptive_window_detection:
    Stage 1: O(N_windows) — fast
    Stage 2: additional fine-window MLE on subset
    ≈ 1–2 additional minutes if hypothesis raised

Total per channel: ≈ 6–10 minutes
Total for 23 channels in parallel: ≈ 15–25 minutes (ThreadPoolExecutor)
```

---

## Originality

A Google Scholar search for the combination of terms "adaptive threshold", "Hawkes process", "seizure detection", "rolling baseline", "branching ratio", and "EEG warning" returned no prior work using this specific combination. The two-stage probabilistic verification framework applied to Hawkes branching ratio dynamics is, to the best of the author's knowledge, a novel contribution.
