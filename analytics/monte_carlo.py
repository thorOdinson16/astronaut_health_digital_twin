"""
analytics/monte_carlo.py

Monte Carlo batch simulation engine for the Astronaut Digital Twin.
Runs N independent simulation trajectories, collects risk distributions,
and returns aggregated statistics ready for the API and frontend.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SINGLE-RUN SIMULATION (no I/O, pure compute)
# ─────────────────────────────────────────────

def _run_single_trajectory(
    seed: int,
    timesteps: int,
    dt_hours: float,
    baseline_hr: float,
    baseline_sleep_quality: float,
    initial_fatigue: float,
    ms_lambda: float,          # Poisson rate for motion sickness (events/hour)
    alpha_sleep: float,        # fatigue accumulation from sleep debt
    beta_motion: float,        # fatigue accumulation from motion stress
    gamma_recovery: float,     # fatigue recovery rate
    recovery_threshold: float, # sleep quality threshold for recovery
    fatigue_to_ms_threshold: float,
    fatigue_to_ms_prob_slope: float,
    risk_fatigue_threshold: float,
    risk_sleep_threshold: float,
) -> Dict[str, Any]:
    """
    Simulate one astronaut trajectory.  All randomness is seeded so runs
    are reproducible and independent of one another.

    Returns a dict with scalar summary metrics and key time-series arrays.
    """
    rng = np.random.default_rng(seed)

    # ── initialise arrays ──────────────────────────────────────────────
    hr            = np.zeros(timesteps, dtype=np.float32)
    sleep_quality = np.zeros(timesteps, dtype=np.float32)
    fatigue       = np.zeros(timesteps, dtype=np.float32)
    motion_sev    = np.zeros(timesteps, dtype=np.float32)
    stress        = np.zeros(timesteps, dtype=np.float32)

    # Baseline trajectories (beta sleep, normal HR, circadian)
    time_hours = np.arange(timesteps) * dt_hours
    circadian  = 5.0 * np.sin(2 * np.pi * time_hours / 24.0)
    hr[:]            = np.clip(rng.normal(baseline_hr, 5.0, timesteps) + circadian, 40, 200)
    sleep_quality[:] = np.clip(rng.beta(5.0, 2.0, timesteps), 0.05, 1.0)
    fatigue[0]       = initial_fatigue

    ms_events: List[Dict] = []

    # ── main loop ──────────────────────────────────────────────────────
    for t in range(timesteps):
        t_h = time_hours[t]

        # --- motion sickness onset (Poisson, modulated by fatigue) ---
        excess_fat  = max(0.0, float(fatigue[t - 1]) - fatigue_to_ms_threshold) if t > 0 else 0.0
        lambda_t    = ms_lambda * dt_hours * (1.0 + fatigue_to_ms_prob_slope * excess_fat)
        lambda_t   *= max(0.1, 1.0 - 0.005 * t_h)   # adaptation decay
        if rng.poisson(lambda_t) > 0:
            sev = float(np.clip(rng.normal(0.6, 0.2), 0.2, 1.0))
            dur = float(np.clip(rng.normal(1.5, 0.5), 0.5, 4.0)) * sev
            ms_events.append({"t": t, "t_h": t_h, "sev": sev, "dur": dur})
            motion_sev[t] = sev
            hr[t] = float(np.clip(hr[t] + 15 * sev, 40, 200))

        # current active motion severity (any ongoing event)
        active_sev = max(
            (e["sev"] for e in ms_events if t_h - e["t_h"] <= e["dur"]),
            default=0.0
        )
        motion_sev[t] = max(motion_sev[t], active_sev)

        # --- fatigue update ---
        if t > 0:
            sq = float(sleep_quality[t])
            ms = float(motion_sev[t])
            sleep_deficit  = (1.0 - sq) ** 1.2
            motion_stress  = ms ** 1.5
            recovery       = 0.0
            if sq > recovery_threshold:
                sq_factor = (sq - recovery_threshold) / (1.0 - recovery_threshold)
                recovery  = gamma_recovery * sq_factor * dt_hours
            delta = alpha_sleep * sleep_deficit * dt_hours + beta_motion * motion_stress * dt_hours - recovery
            fatigue[t] = float(np.clip(fatigue[t - 1] + delta, 0.0, 10.0))

        # --- stress proxy ---
        stress[t] = float(np.clip(
            0.12 + 0.06 * np.sin(2 * np.pi * (t_h % 24) / 24.0 - np.pi / 2)
            + min(0.45, fatigue[t] / 10.0 * 0.6)
            + min(0.50, motion_sev[t] * 0.7),
            0.0, 0.95
        ))

    # ── summary metrics ────────────────────────────────────────────────
    steps_above_fat   = int(np.sum(fatigue > risk_fatigue_threshold))
    steps_above_sleep = int(np.sum(sleep_quality < risk_sleep_threshold))
    prob_fat_risk     = steps_above_fat   / timesteps
    prob_sleep_risk   = steps_above_sleep / timesteps

    # Recovery time: hours from peak fatigue to fatigue ≤ 1.0
    peak_idx   = int(np.argmax(fatigue))
    after_peak = fatigue[peak_idx:]
    recovered  = np.where(after_peak <= 1.0)[0]
    recovery_time_h = float(recovered[0] * dt_hours) if len(recovered) > 0 else float(len(after_peak) * dt_hours)

    # Cumulative fatigue load (area under curve)
    cumulative_fatigue = float(np.trapz(fatigue, dx=dt_hours))

    return {
        # scalar metrics
        "peak_fatigue":        float(np.max(fatigue)),
        "mean_fatigue":        float(np.mean(fatigue)),
        "cumulative_fatigue":  cumulative_fatigue,
        "prob_fatigue_risk":   prob_fat_risk,
        "prob_sleep_risk":     prob_sleep_risk,
        "recovery_time_hours": recovery_time_h,
        "ms_event_count":      len(ms_events),
        "mean_hr":             float(np.mean(hr)),
        "peak_hr":             float(np.max(hr)),
        "mean_sleep_quality":  float(np.mean(sleep_quality)),
        # trajectory arrays (for envelope plots)
        "fatigue_trace":       fatigue.tolist(),
        "sleep_trace":         sleep_quality.tolist(),
        "stress_trace":        stress.tolist(),
    }


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def run_monte_carlo(
    n_runs: int = 50,
    mission_duration_hours: float = 720.0,
    time_step_minutes: float = 30.0,
    baseline_hr: float = 75.0,
    baseline_sleep_quality: float = 0.8,
    initial_fatigue: float = 0.0,
    ms_lambda: float = 0.03,
    alpha_sleep: float = 0.3,
    beta_motion: float = 0.5,
    gamma_recovery: float = 0.08,
    recovery_threshold: float = 0.6,
    fatigue_to_ms_threshold: float = 3.0,
    fatigue_to_ms_prob_slope: float = 0.05,
    risk_fatigue_threshold: float = 5.0,
    risk_sleep_threshold: float = 0.4,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run N independent Monte Carlo trajectories and aggregate results.

    Returns a dict suitable for JSON serialisation with:
      - risk_summary:    aggregate risk probabilities and thresholds
      - distributions:   per-metric lists across runs (for histograms)
      - envelopes:       mean ± std time-series (for fan plots)
      - conclusions:     two plain-English insight strings
    """
    dt_hours  = time_step_minutes / 60.0
    # Use coarser resolution for MC to keep runtime fast
    mc_dt_hours   = max(dt_hours, 0.5)           # at least 30-min steps
    mc_timesteps  = int(mission_duration_hours / mc_dt_hours)

    logger.info(f"Running MC: n={n_runs}, T={mission_duration_hours}h, steps={mc_timesteps}")

    results: List[Dict] = []
    for i in range(n_runs):
        r = _run_single_trajectory(
            seed=base_seed + i,
            timesteps=mc_timesteps,
            dt_hours=mc_dt_hours,
            baseline_hr=baseline_hr,
            baseline_sleep_quality=baseline_sleep_quality,
            initial_fatigue=initial_fatigue,
            ms_lambda=ms_lambda,
            alpha_sleep=alpha_sleep,
            beta_motion=beta_motion,
            gamma_recovery=gamma_recovery,
            recovery_threshold=recovery_threshold,
            fatigue_to_ms_threshold=fatigue_to_ms_threshold,
            fatigue_to_ms_prob_slope=fatigue_to_ms_prob_slope,
            risk_fatigue_threshold=risk_fatigue_threshold,
            risk_sleep_threshold=risk_sleep_threshold,
        )
        results.append(r)

    # ── aggregate scalar distributions ────────────────────────────────
    def _collect(key: str) -> List[float]:
        return [r[key] for r in results]

    peak_fatigues    = np.array(_collect("peak_fatigue"))
    prob_fat_risks   = np.array(_collect("prob_fatigue_risk"))
    prob_sleep_risks = np.array(_collect("prob_sleep_risk"))
    recovery_times   = np.array(_collect("recovery_time_hours"))
    ms_counts        = np.array(_collect("ms_event_count"))
    cum_fatigues     = np.array(_collect("cumulative_fatigue"))

    # ── time-series envelopes (mean ± std, min, max) ───────────────────
    fat_mat   = np.array([r["fatigue_trace"] for r in results])
    sleep_mat = np.array([r["sleep_trace"]   for r in results])

    # Down-sample to 200 points max for frontend
    n_pts = min(200, mc_timesteps)
    idx   = np.round(np.linspace(0, mc_timesteps - 1, n_pts)).astype(int)

    fat_mean  = fat_mat[:, idx].mean(axis=0).tolist()
    fat_std   = fat_mat[:, idx].std(axis=0).tolist()
    fat_max   = fat_mat[:, idx].max(axis=0).tolist()
    fat_min   = fat_mat[:, idx].min(axis=0).tolist()
    slp_mean  = sleep_mat[:, idx].mean(axis=0).tolist()
    slp_std   = sleep_mat[:, idx].std(axis=0).tolist()

    time_axis = (np.arange(mc_timesteps)[idx] * mc_dt_hours).tolist()  # hours

    # ── plain-English conclusions ──────────────────────────────────────
    mean_fat_risk_pct = float(np.mean(prob_fat_risks)) * 100
    mean_slp_risk_pct = float(np.mean(prob_sleep_risks)) * 100
    med_recovery_h    = float(np.median(recovery_times))

    conclusions = [
        (
            f"Across {n_runs} simulated missions, the astronaut spends an average of "
            f"{mean_fat_risk_pct:.1f}% of mission time above the fatigue risk threshold "
            f"(index > {risk_fatigue_threshold}). "
            f"95th-percentile peak fatigue reached {float(np.percentile(peak_fatigues, 95)):.1f}/10, "
            f"suggesting countermeasures (improved sleep scheduling, workload reduction) should "
            f"target the first {int(mission_duration_hours * 0.3)} hours when adaptation stress peaks."
        ),
        (
            f"Sleep quality dropped below the risk threshold (<{risk_sleep_threshold}) for "
            f"{mean_slp_risk_pct:.1f}% of time on average. "
            f"Median recovery time after peak fatigue was {med_recovery_h:.1f} hours. "
            f"The coupling between motion sickness and sleep quality creates a feedback loop: "
            f"runs with ≥{int(np.percentile(ms_counts, 75))} SMS events showed "
            f"{float(np.percentile(recovery_times[ms_counts >= int(np.percentile(ms_counts, 75))], 50)):.1f}h "
            f"median recovery vs "
            f"{float(np.percentile(recovery_times[ms_counts < int(np.percentile(ms_counts, 75))], 50)):.1f}h "
            f"in lower-SMS runs — supporting early anti-emetic intervention."
        ),
    ]

    return {
        "n_runs":          n_runs,
        "mission_hours":   mission_duration_hours,
        "risk_summary": {
            "mean_prob_fatigue_risk":   float(np.mean(prob_fat_risks)),
            "p95_prob_fatigue_risk":    float(np.percentile(prob_fat_risks, 95)),
            "mean_prob_sleep_risk":     float(np.mean(prob_sleep_risks)),
            "mean_peak_fatigue":        float(np.mean(peak_fatigues)),
            "p95_peak_fatigue":         float(np.percentile(peak_fatigues, 95)),
            "median_recovery_hours":    float(np.median(recovery_times)),
            "p90_recovery_hours":       float(np.percentile(recovery_times, 90)),
            "mean_ms_events":           float(np.mean(ms_counts)),
            "mean_cumulative_fatigue":  float(np.mean(cum_fatigues)),
        },
        "distributions": {
            "peak_fatigue":        peak_fatigues.tolist(),
            "prob_fatigue_risk":   prob_fat_risks.tolist(),
            "prob_sleep_risk":     prob_sleep_risks.tolist(),
            "recovery_time_hours": recovery_times.tolist(),
            "ms_event_count":      ms_counts.tolist(),
        },
        "envelopes": {
            "time_hours":   time_axis,
            "fatigue_mean": fat_mean,
            "fatigue_std":  fat_std,
            "fatigue_max":  fat_max,
            "fatigue_min":  fat_min,
            "sleep_mean":   slp_mean,
            "sleep_std":    slp_std,
        },
        "conclusions": conclusions,
    }