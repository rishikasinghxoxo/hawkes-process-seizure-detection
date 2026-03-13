# Patient Phenotype Analysis

**Author:** Rishika Singh

---

## Overview

Multi-patient analysis of CHB-MIT recordings revealed that the effectiveness of η-based seizure detection depends critically on the patient's inter-ictal η dynamics. Three distinct phenotypes were identified, each with different detection requirements and biological interpretations.

This phenotype analysis is itself a contribution — no existing Hawkes seizure detection paper characterises inter-patient variability in η dynamics or proposes patient stratification as a prerequisite for method selection.

---

## Phenotype A — Stable Baseline with Clear Pre-ictal Rise

**η characteristics:**
```
Inter-ictal baseline:  η ≈ 0.4–0.6
Baseline variability:  std ≈ 0.15–0.25
Pre-seizure rise:      Δη > 0.25 above baseline
Rise duration:         150–300+ seconds
```

**Example patients:** chb01

**Detection result:** Adaptive hypothesis works reliably. Two-stage system achieves 46–96 second warning lead times.

**Biological interpretation:** Neural network operates in a stable sub-critical regime between seizures. In the minutes before seizure, self-excitatory dynamics gradually increase — reflected in rising η — until the network crosses into a seizure state. This is consistent with the gradual pre-ictal transition described in focal epilepsy literature.

**η timeseries characteristics:**
- Clear oscillations around stable mean
- Returns to baseline between oscillations
- Unambiguous directional rise 50–100 seconds before seizure
- No chronic elevation

---

## Phenotype B — Chronically Elevated with Small Pre-ictal Rise

**η characteristics:**
```
Inter-ictal baseline:  η ≈ 0.75–0.90
Baseline variability:  std ≈ 0.04–0.10
Pre-seizure rise:      Δη < 0.15
Rise duration:         variable
```

**Example patients:** chb02

**Detection result:** Adaptive hypothesis fails. The pre-seizure rise exists (Δη ≈ 0.09–0.11 on best channel) but is smaller than the detection sensitivity of relative threshold methods at these baseline levels.

**Why adaptive fails:**
```
baseline_mean = 0.85
z_score       = 1.5
std           = 0.05
threshold     = 0.85 + 1.5×0.05 = 0.93

pre-seizure η = 0.82 → below threshold
```

Even reducing z_score to 0.5 would barely catch the rise and would generate excessive false positives during normal inter-ictal fluctuations.

**Biological interpretation:** The neural network is chronically hyperexcitable — possibly due to structural lesions, medication effects, or intrinsic network properties. The seizure trigger may involve a secondary mechanism (e.g. inhibitory failure, neuromodulatory change) superimposed on a already high-excitability baseline. The small pre-ictal η rise suggests the transition to seizure involves a different process than gradual excitability build-up.

**What would work:** Higher sensitivity detection requiring population-level calibration. Multivariate Hawkes models capturing cross-channel interactions. Cross-channel synchronisation measures (phase-locking value, coherence).

---

## Phenotype C — Plateau-Collapse Pattern

**η characteristics:**
```
Inter-ictal baseline:  η ≈ 0.90–1.00 throughout recording
Baseline variability:  variable
Seizure timing:        occurs during η decline from plateau
Pre-seizure η:         decreasing, not increasing
```

**Example patients:** chb05

**η timeseries characteristics:**
- η remains near or above 1.0 for extended periods (1000+ seconds)
- Seizure occurs when η declines from plateau, not at η peak
- Supercritical method fires early (1000+ seconds) correctly identifying chronic instability
- No focal pre-ictal build-up

**Biological interpretation:** Two possible mechanisms:

*Post-excitation collapse:* The brain reaches peak excitability (η ≈ 1.0), inhibitory mechanisms engage to suppress it, η begins declining, inhibitory suppression fails suddenly and seizure fires during the declining phase. The high-η plateau is the build-up; the dip is failed suppression; the seizure is the collapse of that suppression.

*Measurement artefact:* The sliding window at seizure onset averages pre-ictal and ictal spikes, producing an apparent η decrease as firing pattern changes character from self-exciting to fully synchronised.

**Detection result:** Adaptive hypothesis correctly does not fire on decline. Supercritical method fires early on the plateau, providing a very long-lead warning that the brain is in a dangerous state — though not a precise seizure prediction.

**What would work:** Duration-based detection of sustained supercritical states. Inhibitory/excitatory balance models. The 1067-second supercritical alert is technically correct — the brain was genuinely unstable for that entire period.

---

## Screening Criterion

Based on this phenotype analysis, patient eligibility for the adaptive two-stage method can be assessed from the recording itself:

**Retrospective screening** (requires seizure label for pre-seizure window):
```
Condition 1: at least 2 channels with baseline_mean < 0.6 AND baseline_std < 0.25
Condition 2: at least 1 channel with pre-seizure rise > 0.25 above baseline
→ if both: CANDIDATE for adaptive detection
→ if not:  use alternative method based on phenotype
```

**Real-time eligibility** (no seizure label needed):
```
detection_space = 1.0 - η_inter-ictal_mean
if detection_space > 0.4:  sufficient range for rise detection
if detection_space < 0.2:  insufficient range, adaptive unreliable
```

The detection_space criterion directly captures whether the patient's η dynamics have enough room to produce a detectable pre-seizure rise before reaching the supercritical threshold.

---

## Summary Table

| Phenotype | η baseline | Δη pre-seizure | Adaptive | Supercritical | Recommended |
|---|---|---|---|---|---|
| A — Stable | 0.4–0.6 | > 0.25 | ✓ works | Complementary | Two-stage adaptive |
| B — Elevated | 0.75–0.90 | < 0.15 | ✗ fails | May fire early | Multivariate / synchronisation |
| C — Plateau | 0.90–1.00 | negative | ✗ fails | Very early (1000s+) | Duration-based supercritical |

---

## Clinical Implication

Patient phenotyping based on inter-ictal η distribution should precede deployment of any η-based detection method. This is analogous to receptor subtyping before pharmacological treatment — the same intervention produces different outcomes in different patient subtypes.

A practical clinical workflow:
1. Record 30–60 minutes of inter-ictal EEG
2. Compute sliding window η across all channels
3. Classify phenotype based on baseline_mean and detection_space
4. Select detection method accordingly
5. Begin monitoring with appropriate method

This stratification approach is more principled than applying a single universal algorithm to all patients and reporting average performance across phenotypes — a practice that obscures the heterogeneity of seizure dynamics.
