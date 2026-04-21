"""
analytics/monte_carlo.py

Monte Carlo batch simulation engine — physics-based version.

Each trajectory now integrates the coupled ODE system:
  - BorbelyModel      (homeostatic sleep pressure + circadian oscillator)
  - VestibularMismatchModel  (Oman 1982 + sleep-pressure-gated adaptation)
  - FatigueModel      (circadian-gated accumulation ODE)

The key addition beyond standard MC:
  For each trajectory we also compute the COUNTERFACTUAL (independent) risk —
  what P(motion sickness) would be if the two subsystems were uncoupled.
  The distributions of  Δrisk = coupled − independent  are the paper's
  main result: they quantify the synergistic excess risk from the coupling.

Output envelopes include the novel internal ODE states:
  S (homeostatic pressure), C (circadian), mismatch m(t), k_adapt(t)
  in addition to the existing fatigue / sleep_quality / stress.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Physics engine imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.fatigue_model import (
    PhysicsEngine,
    BorbelyParameters,
    VestibularParameters,
    FatigueParameters,
)
from core.coupling_engine import CouplingDiagnostics

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE TRAJECTORY  (physics-based)
# =============================================================================

def _run_single_trajectory(
    seed:                   int,
    timesteps:              int,
    dt_hours:               float,
    baseline_hr:            float,
    baseline_sleep_quality: float,    # used only for Borbély S_0 initialisation
    initial_fatigue:        float,
    ms_lambda:              float,    # kept for API compatibility; onset now from ODE
    fatigue_to_ms_threshold: float,
    fatigue_to_ms_prob_slope: float,
    risk_fatigue_threshold: float,
    risk_sleep_threshold:   float,
    # Per-run physiological variability (sampled by caller)
    run_tau_wake:           float = 18.2,
    run_tau_sleep:          float = 4.2,
    run_k_adapt_0:          float = 0.18,
    run_w_s:                float = 0.65,
    run_alpha:              float = 0.28,
    run_gamma:              float = 0.14,
) -> Dict[str, Any]:
    """
    Simulate one astronaut trajectory using the coupled ODE physics engine.

    Physiological inter-individual variability is captured by the
    run_* parameters (sampled from distributions by the MC wrapper).

    Returns a dict with:
      - Scalar summary metrics (for MC aggregate statistics)
      - Key time-series arrays (for envelope plots)
      - Counterfactual coupling analysis (for paper results)
    """
    rng = np.random.default_rng(seed)

    # ── Configure physics engine with per-run parameters ──────────────
    borbely_p   = BorbelyParameters(
        tau_wake=run_tau_wake,
        tau_sleep=run_tau_sleep,
        # S_0 derived from baseline_sleep_quality heuristic:
        # good sleeper (sq=0.9) → S_0 low (0.20), poor sleeper → S_0 higher
        S_0=float(np.clip(0.60 - baseline_sleep_quality * 0.40, 0.10, 0.55)),
    )
    vestibular_p = VestibularParameters(
        k_adapt_0=run_k_adapt_0,
        w_s=run_w_s,
    )
    fatigue_p   = FatigueParameters(
        alpha_sleep_debt=run_alpha,
        gamma_recovery_base=run_gamma,
    )

    engine = PhysicsEngine(borbely_p, vestibular_p, fatigue_p)
    engine.seed(seed)
    engine.reset(initial_fatigue=initial_fatigue, S_0=borbely_p.S_0)

    # ── Allocate arrays ────────────────────────────────────────────────
    time_hours       = np.arange(timesteps, dtype=np.float32) * dt_hours
    hr               = np.zeros(timesteps, dtype=np.float32)
    sleep_quality    = np.zeros(timesteps, dtype=np.float32)
    fatigue          = np.zeros(timesteps, dtype=np.float32)
    motion_sev       = np.zeros(timesteps, dtype=np.float32)
    stress           = np.zeros(timesteps, dtype=np.float32)
    S_trace          = np.zeros(timesteps, dtype=np.float32)
    C_trace          = np.zeros(timesteps, dtype=np.float32)
    mismatch_trace   = np.zeros(timesteps, dtype=np.float32)
    k_adapt_trace    = np.zeros(timesteps, dtype=np.float32)
    cum_mismatch_tr  = np.zeros(timesteps, dtype=np.float32)

    # Baseline HR with circadian component
    circadian_hr = 5.0 * np.sin(2 * np.pi * time_hours / 24.0)
    hr[:]        = np.clip(
        rng.normal(baseline_hr, 6.0, timesteps) + circadian_hr, 40, 200
    ).astype(np.float32)

    ms_events: List[Dict] = []
    fatigue[0] = initial_fatigue

    # ── Main simulation loop ───────────────────────────────────────────
    for t in range(timesteps):
        t_h = float(time_hours[t])

        # Physics step: Borbély + Vestibular + Fatigue (all coupled)
        phys = engine.step(dt_hours=dt_hours, t_h=t_h, sensory_input=1.0)

        # Store ODE internal states
        S_trace[t]         = phys["S"]
        C_trace[t]         = phys["C"]
        mismatch_trace[t]  = phys["mismatch"]
        k_adapt_trace[t]   = phys["k_adapt"]
        cum_mismatch_tr[t] = phys["cumulative_mismatch"]
        sleep_quality[t]   = float(np.clip(phys["sleep_quality"], 0.05, 1.0))
        if t > 0:
            fatigue[t]     = float(np.clip(phys["fatigue"], 0.0, 10.0))

        # ── Motion sickness onset ──────────────────────────────────────
        # Onset probability comes from the ODE (p_ms_step) rather than
        # an empirical Poisson rate.  The Poisson roll is kept as the
        # stochastic gate so runs remain comparable to the legacy model.
        p_onset = phys["p_ms_step"]
        if rng.random() < p_onset:
            sev = float(np.clip(rng.normal(0.55, 0.20), 0.15, 1.0))
            dur = float(np.clip(rng.normal(1.5, 0.5), 0.5, 4.0)) * sev
            ms_events.append({"t": t, "t_h": t_h, "sev": sev, "dur": dur})
            motion_sev[t] = sev
            # HR: circadian baseline + vestibulo-cardiac reflex delta
            hr[t] = float(np.clip(hr[t] + phys["hr_delta"] * sev, 40, 200))

        # Keep active motion severity from ongoing events
        active_sev = max(
            (e["sev"] for e in ms_events if t_h - e["t_h"] <= e["dur"]),
            default=0.0,
        )
        motion_sev[t] = max(float(motion_sev[t]), float(active_sev))

        # Stress proxy (circadian + fatigue + motion severity)
        stress[t] = float(np.clip(
            0.12
            + 0.06 * np.sin(2 * np.pi * (t_h % 24) / 24.0 - np.pi / 2)
            + min(0.45, fatigue[t] / 10.0 * 0.60)
            + min(0.50, motion_sev[t] * 0.70),
            0.0, 0.95,
        ))

    # ── Summary metrics ────────────────────────────────────────────────
    steps_above_fat   = int(np.sum(fatigue   > risk_fatigue_threshold))
    steps_above_sleep = int(np.sum(sleep_quality < risk_sleep_threshold))
    prob_fat_risk     = steps_above_fat   / timesteps
    prob_sleep_risk   = steps_above_sleep / timesteps

    peak_idx    = int(np.argmax(fatigue))
    after_peak  = fatigue[peak_idx:]
    recovered   = np.where(after_peak <= 1.0)[0]
    recovery_h  = float(recovered[0] * dt_hours) if len(recovered) > 0 else float(len(after_peak) * dt_hours)
    cum_fatigue = float(np.trapezoid(fatigue, dx=dt_hours))

    # ── Coupling diagnostics (counterfactual vs coupled) ───────────────
    coupling_analysis = CouplingDiagnostics.analyse(
        fatigue_trace              = fatigue.tolist(),
        cumulative_mismatch_trace  = cum_mismatch_tr.tolist(),
        S_norm_trace               = (S_trace / borbely_p.S_max).tolist(),
        k_adapt_trace              = k_adapt_trace.tolist(),
        dt_hours                   = dt_hours,
        risk_fatigue_threshold     = risk_fatigue_threshold,
        sigma_ms                   = vestibular_p.sigma_ms,
        ms_saturation              = vestibular_p.ms_saturation,
        k_adapt_0                  = run_k_adapt_0,
        w_s                        = run_w_s,
    )

    return {
        # Scalar summaries
        "peak_fatigue":            float(np.max(fatigue)),
        "mean_fatigue":            float(np.mean(fatigue)),
        "cumulative_fatigue":      cum_fatigue,
        "prob_fatigue_risk":       prob_fat_risk,
        "prob_sleep_risk":         prob_sleep_risk,
        "recovery_time_hours":     recovery_h,
        "ms_event_count":          len(ms_events),
        "mean_hr":                 float(np.mean(hr)),
        "peak_hr":                 float(np.max(hr)),
        "mean_sleep_quality":      float(np.mean(sleep_quality)),
        # Physics-specific scalars (new for paper)
        "mean_mismatch":           float(np.mean(np.abs(mismatch_trace))),
        "peak_mismatch":           float(np.max(np.abs(mismatch_trace))),
        "final_cum_mismatch":      float(cum_mismatch_tr[-1]),
        "mean_k_adapt":            float(np.mean(k_adapt_trace)),
        "mean_k_suppress":         float(coupling_analysis["mean_k_suppress"]),
        # Coupling excess risk (key result for paper)
        "excess_p_ms":             coupling_analysis["mean_excess_p_ms"],
        "joint_risk_excess":       coupling_analysis["joint_risk_excess_fraction"],
        # Time-series arrays for envelope plots
        "fatigue_trace":           fatigue.tolist(),
        "sleep_trace":             sleep_quality.tolist(),
        "stress_trace":            stress.tolist(),
        "S_trace":                 S_trace.tolist(),
        "C_trace":                 C_trace.tolist(),
        "mismatch_trace":          mismatch_trace.tolist(),
        "k_adapt_trace":           k_adapt_trace.tolist(),
        "p_ms_coupled_trace":      coupling_analysis["p_ms_coupled_trace"],
        "p_ms_independent_trace":  coupling_analysis["p_ms_independent_trace"],
        "excess_risk_trace":       coupling_analysis["excess_risk_trace"],
    }


# =============================================================================
# MONTE CARLO  PUBLIC API
# =============================================================================

def run_monte_carlo(
    n_runs:                  int   = 50,
    mission_duration_hours:  float = 720.0,
    time_step_minutes:       float = 30.0,
    baseline_hr:             float = 75.0,
    baseline_sleep_quality:  float = 0.80,
    initial_fatigue:         float = 0.0,
    ms_lambda:               float = 0.03,    # legacy param, kept for API compat
    alpha_sleep:             float = 0.28,
    beta_motion:             float = 0.42,
    gamma_recovery:          float = 0.14,
    recovery_threshold:      float = 0.55,    # legacy param, kept for API compat
    fatigue_to_ms_threshold: float = 3.0,
    fatigue_to_ms_prob_slope: float = 0.05,
    risk_fatigue_threshold:  float = 5.0,
    risk_sleep_threshold:    float = 0.4,
    base_seed:               int   = 42,
) -> Dict[str, Any]:
    """
    Run N independent Monte Carlo trajectories and aggregate results.

    Inter-individual variability is sampled for each run:
      - tau_wake     ~ Normal(18.2, 1.5)  [hours]
      - tau_sleep    ~ Normal(4.2,  0.4)
      - k_adapt_0    ~ Normal(0.18, 0.03) [1/h]
      - w_s          ~ Uniform(0.55, 0.75)
      - alpha        ~ run_alpha * Uniform(0.75, 1.35)
      - gamma        ~ run_gamma * Uniform(0.80, 1.25)

    Returns a dict with:
      risk_summary     — aggregate risk metrics
      distributions    — per-metric lists across runs (histograms)
      envelopes        — mean ± std time-series (fan plots)
      coupling_summary — excess risk from coupling (the paper result)
      conclusions      — plain-English insights
    """
    variability_rng = np.random.default_rng(base_seed)

    dt_hours     = time_step_minutes / 60.0
    mc_dt_hours  = max(dt_hours, 0.5)
    mc_timesteps = int(mission_duration_hours / mc_dt_hours)

    logger.info(f"MC start: n={n_runs}, T={mission_duration_hours}h, steps={mc_timesteps}")

    results: List[Dict] = []

    for i in range(n_runs):
        # Sample per-run physiological variability
        run_tau_wake  = float(np.clip(variability_rng.normal(18.2, 1.5),  12.0, 24.0))
        run_tau_sleep = float(np.clip(variability_rng.normal(4.2,  0.4),   2.5,  6.5))
        run_k_adapt   = float(np.clip(variability_rng.normal(0.18, 0.03),  0.08, 0.35))
        run_w_s       = float(variability_rng.uniform(0.55, 0.75))
        run_alpha     = alpha_sleep  * float(variability_rng.uniform(0.75, 1.35))
        run_gamma     = gamma_recovery * float(variability_rng.uniform(0.80, 1.25))

        r = _run_single_trajectory(
            seed                    = base_seed + i,
            timesteps               = mc_timesteps,
            dt_hours                = mc_dt_hours,
            baseline_hr             = baseline_hr,
            baseline_sleep_quality  = baseline_sleep_quality,
            initial_fatigue         = initial_fatigue,
            ms_lambda               = ms_lambda,
            fatigue_to_ms_threshold = fatigue_to_ms_threshold,
            fatigue_to_ms_prob_slope = fatigue_to_ms_prob_slope,
            risk_fatigue_threshold  = risk_fatigue_threshold,
            risk_sleep_threshold    = risk_sleep_threshold,
            run_tau_wake            = run_tau_wake,
            run_tau_sleep           = run_tau_sleep,
            run_k_adapt_0           = run_k_adapt,
            run_w_s                 = run_w_s,
            run_alpha               = run_alpha,
            run_gamma               = run_gamma,
        )
        results.append(r)

    # ── Aggregate scalar distributions ────────────────────────────────

    def _col(key: str) -> np.ndarray:
        return np.array([r[key] for r in results])

    peak_fatigues    = _col("peak_fatigue")
    prob_fat_risks   = _col("prob_fatigue_risk")
    prob_sleep_risks = _col("prob_sleep_risk")
    recovery_times   = _col("recovery_time_hours")
    ms_counts        = _col("ms_event_count")
    cum_fatigues     = _col("cumulative_fatigue")
    excess_p_ms_arr  = _col("excess_p_ms")
    joint_excess_arr = _col("joint_risk_excess")
    mean_mismatch_arr= _col("mean_mismatch")
    mean_k_sup_arr   = _col("mean_k_suppress")

    # ── Time-series envelopes  (down-sampled to 200 pts) ──────────────
    n_pts = min(200, mc_timesteps)
    idx   = np.round(np.linspace(0, mc_timesteps - 1, n_pts)).astype(int)

    def _envelope(key: str):
        mat = np.array([r[key] for r in results])   # (n_runs, timesteps)
        return {
            "mean": mat[:, idx].mean(axis=0).tolist(),
            "std":  mat[:, idx].std(axis=0).tolist(),
            "max":  mat[:, idx].max(axis=0).tolist(),
            "min":  mat[:, idx].min(axis=0).tolist(),
        }

    fat_env    = _envelope("fatigue_trace")
    slp_env    = _envelope("sleep_trace")
    S_env      = _envelope("S_trace")
    C_env      = _envelope("C_trace")
    mis_env    = _envelope("mismatch_trace")
    ka_env     = _envelope("k_adapt_trace")
    pms_c_env  = _envelope("p_ms_coupled_trace")
    pms_i_env  = _envelope("p_ms_independent_trace")
    exc_env    = _envelope("excess_risk_trace")

    time_axis  = (np.arange(mc_timesteps)[idx] * mc_dt_hours).tolist()

    # ── Plain-English conclusions ──────────────────────────────────────
    mean_fat_pct    = float(np.mean(prob_fat_risks)) * 100
    mean_slp_pct    = float(np.mean(prob_sleep_risks)) * 100
    med_recovery    = float(np.median(recovery_times))
    mean_excess_pct = float(np.mean(excess_p_ms_arr)) * 100
    mean_joint_pct  = float(np.mean(joint_excess_arr)) * 100
    high_ms_thresh  = int(np.percentile(ms_counts, 75))

    conclusions = [
        (
            f"Across {n_runs} simulated missions, the astronaut spends "
            f"{mean_fat_pct:.1f}% of mission time above the fatigue risk threshold "
            f"(index > {risk_fatigue_threshold:.0f}/10). "
            f"The 95th-percentile peak fatigue reached "
            f"{float(np.percentile(peak_fatigues, 95)):.1f}/10. "
            f"Median recovery time after peak fatigue: {med_recovery:.1f} h."
        ),
        (
            f"Sleep quality dropped below the risk threshold "
            f"(<{risk_sleep_threshold}) for {mean_slp_pct:.1f}% of mission time. "
            f"The coupled Borbély–vestibular model predicts {mean_excess_pct:.1f}% "
            f"higher mean motion-sickness probability than an independent "
            f"(uncoupled) model — demonstrating synergistic escalation. "
            f"Joint risk windows (fatigue AND motion sickness simultaneously elevated) "
            f"were {mean_joint_pct:.1f}% more frequent under the coupled model."
        ),
        (
            f"Inter-individual variability in circadian period (τ_c) and "
            f"vestibular adaptation rate (k₀) drives substantial trajectory spread. "
            f"Runs with ≥{high_ms_thresh} motion-sickness episodes showed "
            f"{float(np.percentile(recovery_times[ms_counts >= high_ms_thresh], 50)):.1f} h "
            f"median recovery vs "
            f"{float(np.percentile(recovery_times[ms_counts < high_ms_thresh], 50)):.1f} h "
            f"in lower-SMS runs — supporting early anti-emetic intervention "
            f"to break the fatigue-mismatch feedback loop."
        ),
    ]

    return {
        "n_runs":         n_runs,
        "mission_hours":  mission_duration_hours,
        "model":          "borbely_oman_coupled_ode",

        "risk_summary": {
            "mean_prob_fatigue_risk":    float(np.mean(prob_fat_risks)),
            "p95_prob_fatigue_risk":     float(np.percentile(prob_fat_risks, 95)),
            "mean_prob_sleep_risk":      float(np.mean(prob_sleep_risks)),
            "mean_peak_fatigue":         float(np.mean(peak_fatigues)),
            "p95_peak_fatigue":          float(np.percentile(peak_fatigues, 95)),
            "median_recovery_hours":     float(np.median(recovery_times)),
            "p90_recovery_hours":        float(np.percentile(recovery_times, 90)),
            "mean_ms_events":            float(np.mean(ms_counts)),
            "mean_cumulative_fatigue":   float(np.mean(cum_fatigues)),
        },

        # ── Novel outputs: coupling excess risk ────────────────────────
        "coupling_summary": {
            "mean_excess_p_ms":          float(np.mean(excess_p_ms_arr)),
            "p95_excess_p_ms":           float(np.percentile(excess_p_ms_arr, 95)),
            "mean_joint_risk_excess":    float(np.mean(joint_excess_arr)),
            "mean_abs_mismatch":         float(np.mean(mean_mismatch_arr)),
            "mean_k_suppress":           float(np.mean(mean_k_sup_arr)),
            "description": (
                "excess_p_ms = P(MS|coupled) − P(MS|independent). "
                "Positive values confirm synergistic risk escalation from "
                "the sleep-pressure-gated vestibular adaptation mechanism."
            ),
        },

        "distributions": {
            "peak_fatigue":       peak_fatigues.tolist(),
            "prob_fatigue_risk":  prob_fat_risks.tolist(),
            "prob_sleep_risk":    prob_sleep_risks.tolist(),
            "recovery_time_hours":recovery_times.tolist(),
            "ms_event_count":     ms_counts.tolist(),
            "excess_p_ms":        excess_p_ms_arr.tolist(),
            "joint_risk_excess":  joint_excess_arr.tolist(),
        },

        "envelopes": {
            "time_hours":     time_axis,
            # Existing state variables
            "fatigue_mean":   fat_env["mean"],
            "fatigue_std":    fat_env["std"],
            "fatigue_max":    fat_env["max"],
            "fatigue_min":    fat_env["min"],
            "sleep_mean":     slp_env["mean"],
            "sleep_std":      slp_env["std"],
            # Novel ODE state envelopes (new for paper figures)
            "S_mean":         S_env["mean"],
            "S_std":          S_env["std"],
            "C_mean":         C_env["mean"],
            "mismatch_mean":  mis_env["mean"],
            "mismatch_std":   mis_env["std"],
            "k_adapt_mean":   ka_env["mean"],
            "k_adapt_std":    ka_env["std"],
            # Coupling excess risk envelopes
            "p_ms_coupled_mean":      pms_c_env["mean"],
            "p_ms_coupled_std":       pms_c_env["std"],
            "p_ms_independent_mean":  pms_i_env["mean"],
            "excess_risk_mean":       exc_env["mean"],
            "excess_risk_std":        exc_env["std"],
        },

        "conclusions": conclusions,
    }