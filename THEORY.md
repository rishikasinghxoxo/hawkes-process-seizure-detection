# Mathematical Background

**Author:** Rishika Singh

---

## 1. The Hawkes Process

A Hawkes process is a self-exciting point process — a model where each event increases the probability of future events. This makes it a natural model for neural spike trains, where each action potential can trigger subsequent firing through synaptic connections.

The process is defined by its conditional intensity function:

```
λ(t) = μ + Σᵢ: tᵢ < t  α · exp(-β · (t - tᵢ))
```

**Parameters:**
- `μ > 0` — baseline firing rate (spontaneous activity)
- `α > 0` — excitation strength (how much each spike contributes)
- `β > 0` — decay rate (how quickly excitation fades)
- `tᵢ`    — times of all past spikes

**Interpretation:** At any moment t, the firing rate is the baseline μ plus the sum of decaying contributions from all past spikes. Each spike adds a pulse of amplitude α that decays exponentially with rate β.

---

## 2. The Branching Ratio η

The most important derived parameter is the branching ratio:

```
η = α / β
```

η measures the **average number of additional spikes triggered by each spike**. It is the key quantity for seizure detection.

**Regime classification:**

```
η < 1.0  →  subcritical    each spike triggers < 1 descendant on average
                            excitation dies out, process is stable
                            healthy inter-ictal state: η ≈ 0.2–0.5

η = 1.0  →  critical       boundary of instability
                            sustained oscillation is possible

η > 1.0  →  supercritical  each spike triggers > 1 descendant on average
                            excitation grows without bound
                            seizure state: η ≥ 0.9–1.0
```

The hypothesis of this work is that η rises from a stable inter-ictal baseline toward 1.0 in the minutes before seizure onset — reflecting the gradual build-up of neural excitability that precedes the transition to seizure.

---

## 3. Maximum Likelihood Estimation

Given observed spike times {t₁, t₂, ..., tₙ} in interval [0, T], the log-likelihood of the Hawkes model is:

```
ℓ(μ, α, β) = Σᵢ log λ(tᵢ) − Λ(T)
```

where the compensator Λ(T) is the expected total number of events:

```
Λ(T) = μT + (α/β) · Σᵢ [1 − exp(−β · (T − tᵢ))]
```

Expanding the log-intensity term:

```
log λ(tᵢ) = log[ μ + Σⱼ: tⱼ < tᵢ  α · exp(−β · (tᵢ − tⱼ)) ]
```

The full log-likelihood is therefore:

```
ℓ = Σᵢ log[ μ + Σⱼ<ᵢ α · exp(−β(tᵢ − tⱼ)) ]
    − μT
    − (α/β) · Σᵢ [1 − exp(−β(T − tᵢ))]
```

**Maximisation:** L-BFGS-B with bound constraints μ, α, β > 0. Ten random restarts are used to avoid local minima. The vectorised implementation computes the double sum using an n×n matrix of pairwise time differences, with exponential clipping to prevent numerical overflow.

**MLE validation** (5 trials per regime):

```
Healthy regime   true η=0.20  →  recovered η=0.206 ± 0.039
Epileptic regime true η=0.80  →  recovered η=0.796 ± 0.040
```

Bias and variance are small relative to the detection signal (Δη ≈ 0.3–0.4), confirming the estimator is reliable for this application.

---

## 4. Sliding Window η Estimation

Rather than fitting a single model to the entire recording, η is estimated in a sliding window to track how neural dynamics evolve over time.

**Window parameters:**
```
window_size = 200 seconds   (sufficient spikes for reliable MLE)
step_size   = 50 seconds    (temporal resolution of η timeseries)
min_spikes  = 15            (minimum for MLE stability)
```

At each window position [t, t + 200s], the spike times within the window are extracted (shifted to start at 0), Hawkes MLE is fitted, and η is recorded at the window centre t + 100s. This produces a timeseries {(cᵢ, ηᵢ)} tracking neural excitability across the recording.

---

## 5. Adaptive Hypothesis Detection

The key challenge is distinguishing genuine pre-seizure η rises from transient neural fluctuations. A fixed threshold (e.g. η > 0.7) fails because baseline η varies substantially between patients.

Instead, each patient's own η history is used as the reference. At window position i, the adaptive threshold is:

```
baseline  = mean(η[i−20 : i])       rolling mean of last 20 windows (1000s)
std       = std(η[i−20 : i])
threshold = baseline + 1.5 × max(std, 0.05)
```

A hypothesis is raised when all three conditions hold for two consecutive windows:

```
Condition 1: η[i] > threshold            (genuinely elevated above baseline)
Condition 2: η[i] > η[i−2]              (actively rising not plateauing)
Condition 3: min(η[i−10:i]) > 0.3       (no recent suppression recovery)
```

This is entirely personalised — the same parameters work across patients because the threshold adapts to each patient's baseline level and variability.

---

## 6. Probabilistic Verification

After a hypothesis is raised, the system switches to a fine-resolution window (100s window, 20s step) and updates P(seizure) at each subsequent window:

```
if η > threshold:  P ← P + (1 − P) × 0.4     (pull toward 1.0)
if η < threshold:  P ← P − P × 0.4            (pull toward 0.0)
```

Starting from P = 0.5 at hypothesis time:

```
P ≥ 0.85  →  CONFIRMED  (seizure alert)
P ≤ 0.15  →  REJECTED   (transient, reset)
```

**Why this works:**
- Genuine pre-seizure rises produce sustained η elevation → P rises monotonically
- Transient spikes produce brief elevation then rapid decline → P rises then falls
- The probability trajectory shape is itself informative — a rising P signals genuine pre-ictal dynamics even before the confirmation threshold

---

## 7. Supercritical Safety Net

Independently of the hypothesis system, the supercritical condition fires when η ≥ 1.0:

```
critical_threshold_warning fires when η ≥ 1.0 for 1 consecutive window
```

This is mathematically motivated: η ≥ 1.0 means the process is supercritical — runaway excitation is occurring or imminent. Unlike the hypothesis stage (which detects relative rise), this detects absolute instability.

**Clinical interpretation:** This is a Level 2 alert — not a prediction but a state notification. It confirms the brain is in a mathematically unstable regime.

---

## 8. References

- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83–90.
- Ogata, Y. (1978). The asymptotic behaviour of maximum likelihood estimators for stationary point processes. *Annals of the Institute of Statistical Mathematics*, 30(1), 243–261.
- Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Transactions on Information Theory*, 27(1), 23–31.
- Truccolo, W., Eden, U. T., Fellows, M. R., Donoghue, J. P., & Brown, E. N. (2005). A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects. *Journal of Neurophysiology*, 93(2), 1074–1089.
- Gerhard, F., Deger, M., & Truccolo, W. (2017). On the stability and dynamics of stochastic spiking neuron models. *PLOS Computational Biology*, 13(12).
