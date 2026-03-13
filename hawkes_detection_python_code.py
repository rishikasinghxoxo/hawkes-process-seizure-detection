# HAWKES PROCESS SEIZURE EARLY WARNING SYSTEM
# Author: Rishika Singh
#
# OVERVIEW:
# Models EEG spike trains as Hawkes processes.
# Tracks branching ratio η over time using sliding window MLE.
# Two-stage probabilistic detection:
#   Stage 1 — Adaptive hypothesis (personalised rolling baseline)
#   Stage 2 — Probabilistic fine-window verification
#
# CELL ORDER:
# Cell 1  — Imports
# Cell 2  — Configuration  ← only cell to change per recording
# Cell 3  — Hawkes Simulator
# Cell 4  — MLE (vectorised)
# Cell 5  — Simulation Validation
# Cell 6  — Sliding Window
# Cell 7  — Detection Functions
# Cell 8  — EEG Preprocessing
# Cell 9  — Plotting
# Cell 10 — Multi-Channel Pipeline
# Cell 11 — Run

# CELL 1 — IMPORTS

# !pip install mne -q  # uncomment if mne not installed

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
import mne


# CELL 2 — CONFIGURATION
# Change these values for each new recording.
# Everything else stays fixed.

EDF_FILE      = '/content/chb01_03.edf'
SEIZURE_START = 2996        # seconds — from CHB-MIT summary file
SEIZURE_END   = 3036        # seconds — from CHB-MIT summary file
PATIENT_ID    = 'chb01_03'

# For control recordings with no seizure:
# SEIZURE_START = None
# SEIZURE_END   = None


# CELL 3 — HAWKES SIMULATOR
# Ogata thinning algorithm.
# Used for simulation validation only — not for real EEG.

def hawkes_simulator(mu, alpha, beta, T):
    """
    Simulate Hawkes process using Ogata thinning algorithm.

    Intensity: λ(t) = μ + Σ α·exp(-β·(t - tᵢ))

    Parameters:
        mu    : baseline firing rate
        alpha : excitation strength per spike
        beta  : decay rate of excitation
        T     : total duration (seconds)

    Returns:
        numpy array of spike times
    """
    t      = 0
    events = []

    while t < T:
        upper_bound = mu + alpha * len(events)
        t          += np.random.exponential(1 / upper_bound)

        if t > T:
            break

        intensity   = mu + np.sum(
            alpha * np.exp(-beta * (t - np.array(events)))
        )
        probability = intensity / upper_bound

        if np.random.uniform(0, 1) < probability:
            events.append(t)

    return np.array(events)


def simulate_preseizure(T_healthy=300, T_preseizure=300,
                        T_seizure=100):
    """
    Simulate brain transitioning healthy → pre-seizure → seizure.

    Phase 1: healthy     η=0.20 (stable baseline)
    Phase 2: pre-seizure η rises 0.20 → 0.80 gradually
    Phase 3: seizure     η=0.90 (supercritical)

    Returns:
        all_events      : full spike train
        T_total         : total duration
        preseizure_start: start of pre-seizure phase
        seizure_start   : start of seizure phase
    """
    mu, beta = 0.5, 2.0

    phase1 = hawkes_simulator(mu=mu, alpha=0.4, beta=beta,
                               T=T_healthy)

    phase2_events = []
    chunk_size    = T_preseizure / 6
    current_t     = T_healthy

    for alpha in np.linspace(0.4, 1.6, 6):
        chunk  = hawkes_simulator(mu=mu, alpha=alpha,
                                   beta=beta, T=chunk_size)
        chunk += current_t
        phase2_events.extend(chunk)
        current_t += chunk_size

    phase3  = hawkes_simulator(mu=mu, alpha=1.8, beta=beta,
                                T=T_seizure)
    phase3 += T_healthy + T_preseizure

    all_events = np.sort(np.concatenate(
        [phase1, np.array(phase2_events), phase3]
    ))
    T_total = T_healthy + T_preseizure + T_seizure

    return all_events, T_total, T_healthy, T_healthy + T_preseizure

print("Cell 3 ready — Hawkes simulator defined")


# CELL 4 — MLE (VECTORISED)
# Maximum likelihood estimation for Hawkes parameters.
# Vectorised for efficiency on high spike-rate recordings.

def hawkes_log_likelihood(params, events, T):
    """
    Negative log likelihood of a Hawkes process.
    Vectorised using numpy matrix operations.

    L = Σ log λ(tᵢ) - [μT + (α/β)·Σ(1 - exp(-β(T-tᵢ)))]

    Returns negative LL because scipy.minimize finds minimums.
    """
    mu, alpha, beta = params

    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e10

    events = np.array(events)

    # vectorised intensity computation
    dt          = events[:, None] - events[None, :]
    mask        = dt > 0
    exponent    = np.where(mask, -beta * dt, -np.inf)
    exponent    = np.clip(exponent, -500, 0)
    excitation  = np.where(mask, alpha * np.exp(exponent), 0)
    intensities = mu + excitation.sum(axis=1)

    if np.any(intensities <= 0):
        return 1e10

    part1 = np.sum(np.log(intensities))
    part2 = mu * T + (alpha / beta) * np.sum(
        1 - np.exp(-beta * (T - events))
    )

    return -(part1 - part2)


def fit_hawkes(events, T, n_restarts=10):
    """
    Fit Hawkes parameters (μ, α, β) using MLE with random restarts.
    Multiple restarts avoid local minima.

    Returns dict with mu, alpha, beta, eta=alpha/beta
    """
    best_result = None
    best_ll     = np.inf

    for _ in range(n_restarts):
        x0 = [
            np.random.uniform(0.1, 2.0),
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.5, 5.0)
        ]

        result = minimize(
            hawkes_log_likelihood, x0,
            args=(events, T),
            method='L-BFGS-B',
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
            options={'maxiter': 1000}
        )

        if result.fun < best_ll:
            best_ll     = result.fun
            best_result = result

    mu, alpha, beta = best_result.x
    return {'mu': mu, 'alpha': alpha, 'beta': beta,
            'eta': alpha / beta}

print("Cell 4 ready — MLE defined")


# CELL 5 — SIMULATION VALIDATION
# Verifies MLE correctly recovers known parameters.
# Run once to confirm estimator works before real data.

def validate_mle(mu, alpha, beta, T=500, n_trials=5):
    """
    Run n_trials simulations and check η recovery.
    Mean recovered η should be within 0.05 of true η.
    """
    true_eta  = alpha / beta
    recovered = []

    for _ in range(n_trials):
        events = hawkes_simulator(mu=mu, alpha=alpha,
                                   beta=beta, T=T)
        if len(events) > 10:
            result = fit_hawkes(events, T)
            recovered.append(result['eta'])

    print(f"True η: {true_eta:.2f}  →  "
          f"Recovered: mean={np.mean(recovered):.3f}  "
          f"std={np.std(recovered):.3f}  "
          f"bias={np.mean(recovered)-true_eta:+.3f}")
    return recovered


print("Healthy regime (true η=0.20):")
validate_mle(mu=0.5, alpha=0.4, beta=2.0)

print("\nEpileptic regime (true η=0.80):")
validate_mle(mu=0.5, alpha=1.6, beta=2.0)


# CELL 6 — SLIDING WINDOW
# Fits MLE at each window position to track η over time.

def sliding_window_eta(events, T_total,
                       window_size=200, step_size=50,
                       min_spikes=15):
    """
    Slide a window across the spike train.
    Fit Hawkes MLE at each position.

    Parameters:
        events      : spike times (numpy array)
        T_total     : total recording duration (seconds)
        window_size : width of each window (seconds)
        step_size   : step between windows (seconds)
        min_spikes  : minimum spikes required for MLE

    Returns:
        window_centers : centre time of each window
        recovered_etas : η estimate at each window
    """
    window_centers = []
    recovered_etas = []
    start          = 0

    while start + window_size <= T_total:
        end           = start + window_size
        mask          = (events >= start) & (events < end)
        window_events = events[mask] - start

        if len(window_events) >= min_spikes:
            result = fit_hawkes(window_events, T=window_size)
            recovered_etas.append(result['eta'])
            window_centers.append(start + window_size / 2)

        start += step_size

    return np.array(window_centers), np.array(recovered_etas)

print("Cell 6 ready — sliding window defined")


# CELL 7 — DETECTION FUNCTIONS

def adaptive_warning(window_centers, recovered_etas,
                     baseline_window=20,
                     z_score=1.5,
                     consecutive=2,
                     suppression_floor=0.3):
    """
    METHOD 1 — Adaptive Gradual Rise Detection

    Detects sustained η elevation above patient's own rolling baseline.
    Personalises threshold to each patient — no fixed η threshold.

    Three conditions must all hold for consecutive windows:
    1. η > rolling mean + z_score×std  (genuinely elevated)
    2. η > η from 2 windows ago        (actively rising)
    3. min(last 10 windows) > floor    (not recovering from suppression)

    Parameters are fixed across all patients and recordings.
    Do not tune per patient.
    """
    count = 0

    for i in range(baseline_window, len(recovered_etas)):

        recent        = recovered_etas[i - baseline_window:i]
        baseline      = np.mean(recent)
        std           = np.std(recent)
        effective_std = max(std, 0.05)
        threshold     = baseline + z_score * effective_std

        is_elevated    = recovered_etas[i] > threshold
        is_rising      = recovered_etas[i] > recovered_etas[i - 2]
        recent_short   = recovered_etas[max(0, i - 10):i]
        not_recovering = np.min(recent_short) > suppression_floor

        if is_elevated and is_rising and not_recovering:
            count += 1
        else:
            count = 0

        if count >= consecutive:
            return window_centers[i - consecutive + 1]

    return None


def critical_threshold_warning(window_centers, recovered_etas,
                                critical_eta=1.0,
                                consecutive=1):
    """
    METHOD 2 — Supercritical Safety Net

    Detects when η ≥ 1.0 — the mathematical instability threshold.
    η ≥ 1.0 means each spike triggers ≥ 1 new spike on average.
    Process is supercritical — runaway excitation is occurring.

    This is a LEVEL 2 ALERT — confirms crisis is happening.
    Not a pre-seizure prediction — a state notification.
    Complements adaptive_warning rather than replacing it.
    """
    count = 0

    for i in range(1, len(recovered_etas)):
        if recovered_etas[i] >= critical_eta:
            count += 1
        else:
            count = 0

        if count >= consecutive:
            return window_centers[i - consecutive + 1]

    return None


def probabilistic_verification(alert_centers, alert_etas,
                                alert_threshold,
                                p_initial=0.5,
                                p_confirm=0.85,
                                p_reject=0.15):
    """
    Update P(seizure) after hypothesis is raised.

    Each window above threshold pulls P toward 1.0.
    Each window below threshold pulls P toward 0.0.

    Update rule:
        above threshold: P = P + (1-P) × 0.4
        below threshold: P = P - P × 0.4

    Parameters:
        p_initial : starting probability at hypothesis
        p_confirm : threshold to confirm seizure alert
        p_reject  : threshold to reject as transient

    Returns:
        (time, status, final_probability)
        status = 'confirmed', 'rejected', or 'uncertain'
    """
    p = p_initial

    for i in range(len(alert_etas)):
        t = alert_centers[i]
        e = alert_etas[i]

        if e > alert_threshold:
            p = p + (1 - p) * 0.4
        else:
            p = p - p * 0.4

        print(f"    t={t:.0f}s  η={e:.3f}  P(seizure)={p:.2f}")

        if p >= p_confirm:
            return t, 'confirmed', p
        if p <= p_reject:
            return t, 'rejected', p

    return None, 'uncertain', p


def adaptive_window_detection(events, T_total,
                               normal_window=200,
                               normal_step=50,
                               alert_window=100,
                               alert_step=20,
                               baseline_window=20,
                               z_score=1.5,
                               hypothesis_consecutive=2,
                               suppression_floor=0.3,
                               p_confirm=0.85,
                               p_reject=0.15):
    """
    TWO-STAGE PROBABILISTIC DETECTION — Primary Detection Method

    Stage 1 — HYPOTHESIS (adaptive gradual rise)
        Normal resolution sliding window.
        Raises hypothesis when adaptive conditions met.
        Output: "possible seizure" + lead time.

    Stage 2 — PROBABILISTIC VERIFICATION
        Switches to fine resolution window after hypothesis.
        Updates P(seizure) each window.
        Confirmed when P ≥ p_confirm.
        Rejected when P ≤ p_reject.
        Output: probability trace + final status.

    Clinical interpretation:
        Hypothesis → Level 1 advisory alert (low-cost response)
        Confirmed  → Level 2 intervention alert (full response)
        Rejected   → false alarm suppressed, no alert issued

    Returns:
        hypothesis_time : when possibility first raised (seconds)
        confirm_time    : when confirmed (None if rejected/uncertain)
        status          : 'confirmed', 'rejected', 'uncertain'
        final_prob      : final P(seizure)
        rejected_times  : list of rejected hypothesis times
    """

    # Stage 1 — normal monitoring
    centers_normal, etas_normal = sliding_window_eta(
        events, T_total,
        window_size=normal_window,
        step_size=normal_step
    )

    valid_mask    = centers_normal > 300
    valid_centers = centers_normal[valid_mask]
    valid_etas    = etas_normal[valid_mask]

    hypothesis_time = None
    in_hypothesis   = False
    hyp_count       = 0
    rejected_times  = []

    for i in range(baseline_window, len(valid_etas)):

        recent        = valid_etas[i - baseline_window:i]
        baseline      = np.mean(recent)
        std           = np.std(recent)
        effective_std = max(std, 0.05)
        threshold     = baseline + z_score * effective_std

        is_elevated    = valid_etas[i] > threshold
        is_rising      = valid_etas[i] > valid_etas[i - 2]
        recent_short   = valid_etas[max(0, i - 10):i]
        not_recovering = np.min(recent_short) > suppression_floor
        conditions_met = is_elevated and is_rising and not_recovering

        if not in_hypothesis:
            if conditions_met:
                hyp_count += 1
            else:
                hyp_count = 0

            if hyp_count >= hypothesis_consecutive:
                hypothesis_time = valid_centers[
                    i - hypothesis_consecutive + 1
                ]
                in_hypothesis = True
                t_hypothesis  = valid_centers[i]

                print(f"  HYPOTHESIS at t={hypothesis_time:.0f}s"
                      f" — possible seizure detected")

                # Stage 2 — fine window verification
                centers_alert, etas_alert = sliding_window_eta(
                    events, T_total,
                    window_size=alert_window,
                    step_size=alert_step
                )

                alert_mask    = centers_alert > t_hypothesis
                alert_centers = centers_alert[alert_mask]
                alert_etas    = etas_alert[alert_mask]

                if len(alert_centers) < 2:
                    print(f"  Not enough windows to verify")
                    return (hypothesis_time, None,
                            'uncertain', 0.5, rejected_times)

                verify_time, status, final_p = probabilistic_verification(
                    alert_centers, alert_etas,
                    threshold,
                    p_confirm=p_confirm,
                    p_reject=p_reject
                )

                if status == 'confirmed':
                    print(f"  CONFIRMED at t={verify_time:.0f}s"
                          f" — P(seizure)={final_p:.2f}")
                    return (hypothesis_time, verify_time,
                            status, final_p, rejected_times)

                elif status == 'rejected':
                    print(f"  REJECTED at t={verify_time:.0f}s"
                          f" — P(seizure)={final_p:.2f} transient")
                    rejected_times.append(hypothesis_time)
                    hypothesis_time = None
                    in_hypothesis   = False
                    hyp_count       = 0

                else:
                    print(f"  UNCERTAIN — P(seizure)={final_p:.2f}")
                    return (hypothesis_time, None,
                            status, final_p, rejected_times)

    return hypothesis_time, None, 'uncertain', 0.5, rejected_times

print("Cell 7 ready — detection functions defined")


# CELL 8 — EEG PREPROCESSING

def load_eeg(filepath):
    """Load EDF file using MNE."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    print(f"Duration:    {raw.times[-1]:.0f} seconds")
    print(f"Channels:    {len(raw.ch_names)}")
    print(f"Sample rate: {raw.info['sfreq']} Hz")
    return raw


def eeg_to_spikes(raw, channel, low_freq=80, high_freq=120,
                  threshold_std=3.0):
    """
    Convert one EEG channel to spike times.

    Pipeline:
    1. Bandpass filter 80-120 Hz (high gamma band)
       — isolates high-frequency neural activity
    2. Rolling normalisation in 60s windows
       — removes slow drift, standardises amplitude
    3. Threshold crossings at 3 standard deviations
       — detects significant amplitude events
    4. Refractory period 50ms minimum between spikes
       — prevents double-counting of single events

    Parameters:
        raw           : MNE Raw object
        channel       : channel name string
        low_freq      : bandpass lower bound (Hz)
        high_freq     : bandpass upper bound (Hz)
        threshold_std : detection threshold (std deviations)

    Returns:
        numpy array of spike times (seconds)
    """
    sfreq        = raw.info['sfreq']
    raw_filtered = raw.copy().filter(
        low_freq, high_freq, picks=[channel], verbose=False
    )
    data = raw_filtered.get_data()[raw.ch_names.index(channel)]

    # rolling normalisation
    window_samples = int(60 * sfreq)
    normalized     = np.zeros_like(data)

    for i in range(0, len(data), window_samples):
        chunk = data[i:i + window_samples]
        if np.std(chunk) > 0:
            normalized[i:i + window_samples] = (
                (chunk - np.mean(chunk)) / np.std(chunk)
            )

    # threshold crossings
    above     = np.abs(normalized) > threshold_std
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(crossings) == 0:
        return np.array([])

    spike_times = crossings / sfreq

    # refractory period 50ms
    filtered = [spike_times[0]]
    for t in spike_times[1:]:
        if t - filtered[-1] > 0.05:
            filtered.append(t)

    return np.array(filtered)


def spike_density_report(spikes, T_total,
                         seizure_start=None,
                         seizure_end=None):
    """
    Report spike statistics.
    Seizure times used for post-hoc validation only.
    Never passed to detection functions.
    """
    print(f"  Total spikes:  {len(spikes)}")
    print(f"  Overall rate:  {len(spikes)/T_total:.3f} spikes/s")

    if seizure_start and seizure_end:
        pre         = spikes[spikes < seizure_start]
        during      = spikes[(spikes >= seizure_start) &
                             (spikes <= seizure_end)]
        dur         = seizure_end - seizure_start
        rate_pre    = len(pre) / seizure_start
        rate_during = len(during) / dur if dur > 0 else 0
        ratio       = rate_during / rate_pre if rate_pre > 0 else 0

        print(f"  Pre-seizure:   {len(pre)} spikes ({rate_pre:.3f}/s)")
        print(f"  During:        {len(during)} spikes "
              f"({rate_during:.3f}/s)")
        print(f"  Density ratio: {ratio:.1f}x")
        return ratio

    return None

print("Cell 8 ready — EEG preprocessing defined")


# ============================================================
# CELL 9 — PLOTTING
# ============================================================

def plot_eta_timeseries(window_centers, recovered_etas, T_total,
                        seizure_start=None,
                        hypothesis_time=None,
                        confirm_time=None,
                        patient_id='recording',
                        channel=''):
    """
    Plot η timeseries with hypothesis and seizure markers.
    Seizure marker added post-hoc for validation only.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(window_centers, recovered_etas,
            'b-', linewidth=2, label='Recovered η')

    ax.axhline(y=1.0, color='gray', linestyle=':',
               alpha=0.5, label='Supercritical (η=1.0)')

    if hypothesis_time:
        ax.axvline(x=hypothesis_time, color='orange',
                   linewidth=2, linestyle='--',
                   label=f'Hypothesis (t={hypothesis_time:.0f}s)')

    if confirm_time:
        ax.axvline(x=confirm_time, color='red',
                   linewidth=2, linestyle='--',
                   label=f'Confirmed (t={confirm_time:.0f}s)')

    if seizure_start:
        ax.axvline(x=seizure_start, color='darkred',
                   linewidth=2, linestyle='-',
                   label=f'Seizure onset (t={seizure_start}s)')

    ax.set_title(f'Hawkes η — {patient_id} {channel}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('η (branching ratio)')
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper left')

    plt.tight_layout()
    fname = f'eta_{patient_id}_{channel}.png'.replace(' ', '_')
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

print("Cell 9 ready — plotting defined")


# ============================================================
# CELL 10 — MULTI-CHANNEL PIPELINE
# Runs two-stage detection on ALL channels in parallel.
# No channel selection needed — reports best result.
# ============================================================

def run_single_channel(args):
    """
    Run full two-stage detection pipeline on one channel.
    Designed for parallel execution via ThreadPoolExecutor.
    """
    raw, channel, T_total, seizure_start = args

    try:
        spikes = eeg_to_spikes(raw, channel=channel)
        rate   = len(spikes) / T_total

        if rate < 0.1 or len(spikes) < 50:
            return None

        # two-stage probabilistic detection
        hyp_time, conf_time, status, prob, rejected = \
            adaptive_window_detection(spikes, T_total)

        # supercritical safety net
        centers, etas = sliding_window_eta(
            spikes, T_total, window_size=200, step_size=50
        )
        mask          = centers > 300
        valid_centers = centers[mask]
        valid_etas    = etas[mask]

        if len(valid_centers) < 20:
            return None

        critical_time = critical_threshold_warning(
            valid_centers, valid_etas
        )

        # compute lead times
        hyp_lead      = None
        critical_lead = None

        if hyp_time and seizure_start:
            lead = seizure_start - hyp_time
            if lead > 0:
                hyp_lead = lead

        if critical_time and seizure_start:
            lead = seizure_start - critical_time
            if lead > 0:
                critical_lead = lead

        # best result for this channel
        best_lead   = None
        best_method = None
        best_time   = None

        if hyp_lead and critical_lead:
            if hyp_lead >= critical_lead:
                best_lead   = hyp_lead
                best_method = 'hypothesis'
                best_time   = hyp_time
            else:
                best_lead   = critical_lead
                best_method = 'supercritical'
                best_time   = critical_time
        elif hyp_lead:
            best_lead, best_method, best_time = (
                hyp_lead, 'hypothesis', hyp_time
            )
        elif critical_lead:
            best_lead, best_method, best_time = (
                critical_lead, 'supercritical', critical_time
            )

        return {
            'channel':       channel,
            'rate':          rate,
            'eta_mean':      float(valid_etas.mean()),
            'eta_std':       float(valid_etas.std()),
            'eta_max':       float(valid_etas.max()),
            'hypothesis':    hyp_time,
            'hyp_lead':      hyp_lead,
            'status':        status,
            'probability':   prob,
            'rejected':      rejected,
            'critical':      critical_time,
            'critical_lead': critical_lead,
            'best_time':     best_time,
            'best_lead':     best_lead,
            'best_method':   best_method,
            'centers':       valid_centers,
            'etas':          valid_etas
        }

    except Exception as e:
        return None


def run_all_channels(raw, T_total, seizure_start=None,
                     seizure_end=None, patient_id='recording'):
    """
    Run two-stage detection on all EEG channels in parallel.
    Reports all channels that detect before seizure.
    Best result = earliest valid hypothesis across all channels.
    """
    print(f"\n{'='*60}")
    print(f"Multi-Channel Detection — {patient_id}")
    if seizure_start:
        print(f"Seizure ground truth: t={seizure_start}s")
    else:
        print(f"Control recording — no seizure")
    print(f"{'='*60}\n")

    args = [
        (raw, ch, T_total, seizure_start)
        for ch in raw.ch_names
    ]

    print(f"Running {len(raw.ch_names)} channels in parallel...\n")
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(run_single_channel, args))

    results = [r for r in results if r is not None]
    print(f"\nValid channels: {len(results)}\n")

    # summary table
    print(f"{'Channel':<15} {'Rate':<8} {'η mean':<8} {'η std':<8} "
          f"{'Hypothesis':<12} {'P':<6} {'Critical':<12} {'Best'}")
    print("-" * 80)

    detected = []
    for r in sorted(results,
                    key=lambda x: x['best_lead'] or 0,
                    reverse=True):

        hyp_str  = f"{r['hyp_lead']:.0f}s" \
                   if r['hyp_lead'] else "—"
        crit_str = f"{r['critical_lead']:.0f}s" \
                   if r['critical_lead'] else "—"
        p_str    = f"{r['probability']:.2f}" \
                   if r['probability'] else "—"

        if r['best_lead']:
            best_str = (f"{r['best_lead']:.0f}s "
                        f"({r['best_method']})")
            detected.append(r)
        else:
            best_str = "—"

        print(f"{r['channel']:<15} {r['rate']:<8.3f} "
              f"{r['eta_mean']:<8.3f} {r['eta_std']:<8.3f} "
              f"{hyp_str:<12} {p_str:<6} "
              f"{crit_str:<12} {best_str}")

    print(f"\n{'='*60}")

    if detected:
        best = max(detected, key=lambda x: x['best_lead'])
        print(f"→ {len(detected)} channels detected before onset")
        print(f"→ Best: {best['best_lead']:.0f}s on "
              f"{best['channel']} via {best['best_method']}")

        # plot best channel
        plot_eta_timeseries(
            best['centers'], best['etas'], T_total,
            seizure_start=seizure_start,
            hypothesis_time=best['hypothesis'],
            confirm_time=None,
            patient_id=patient_id,
            channel=best['channel']
        )

        # save results
        _save_results(results, patient_id, seizure_start)

    else:
        if seizure_start:
            print(f"→ No channel detected before onset")
        else:
            print(f"→ Control recording — no warnings ✓")

    return results


def _save_results(results, patient_id, seizure_start):
    """Save detection results to text file."""
    fname = f"results_{patient_id}.txt"
    lines = [
        f"RESULTS — {patient_id}",
        f"Seizure: t={seizure_start}s",
        "=" * 60, ""
    ]

    detected = [r for r in results if r['best_lead']]
    lines.append(f"Channels detecting before seizure: {len(detected)}\n")

    for r in sorted(results,
                    key=lambda x: x['best_lead'] or 0,
                    reverse=True):
        lines += [
            f"Channel: {r['channel']}",
            f"  Rate:          {r['rate']:.3f}/s",
            f"  η mean ± std:  {r['eta_mean']:.3f} ± {r['eta_std']:.3f}",
            f"  η max:         {r['eta_max']:.3f}",
            f"  Hypothesis:    {r['hypothesis']}s "
            f"(lead={r['hyp_lead']}s)",
            f"  Status:        {r['status']} "
            f"P={r['probability']:.2f}",
            f"  Rejected:      {r['rejected']}",
            f"  Critical:      {r['critical']}s "
            f"(lead={r['critical_lead']}s)",
            f"  Best:          {r['best_lead']}s "
            f"via {r['best_method']}", ""
        ]

    with open(fname, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Results saved to {fname}")

print("Cell 10 ready — multi-channel pipeline defined")


# CELL 11 — RUN
# Update Cell 2 configuration then run this cell.
# All other cells stay fixed.

raw     = load_eeg(EDF_FILE)
T_total = raw.times[-1]

results = run_all_channels(
    raw, T_total,
    seizure_start=SEIZURE_START,
    seizure_end=SEIZURE_END,
    patient_id=PATIENT_ID
)
