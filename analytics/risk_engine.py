"""
analytics/risk_engine.py

Derives risk metrics, trend statistics, and at-risk window flags
from a completed AstronautState trajectory (as returned by state.to_dict()).

All functions are pure (no I/O) — they accept plain Python dicts/lists
and return JSON-serialisable dicts.
"""

import numpy as np
from typing import Dict, List, Any, Tuple


# ─────────────────────────────────────────────
# RISK THRESHOLDS  (can be overridden per call)
# ─────────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "fatigue_mild":      2.0,
    "fatigue_moderate":  4.0,
    "fatigue_severe":    6.0,
    "fatigue_critical":  8.0,
    "hr_elevated":     100.0,
    "hr_high":         120.0,
    "hr_critical":     150.0,
    "sleep_poor":        0.5,
    "sleep_critical":    0.3,
    "stress_high":       0.6,
    "ms_moderate":       2.0,
    "ms_severe":         3.5,
}


# ─────────────────────────────────────────────
# THRESHOLD EXCEEDANCE
# ─────────────────────────────────────────────

def compute_threshold_metrics(
    state: Dict[str, Any],
    thresholds: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Compute time-above/below-threshold probabilities for every key metric.

    Args:
        state: dict from AstronautState.to_dict()
        thresholds: optional overrides for DEFAULT_THRESHOLDS

    Returns:
        dict with probability and duration metrics per threshold.
    """
    th  = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    n   = len(state["fatigue"])
    fat = np.array(state["fatigue"])
    hr  = np.array(state["hr"])
    slp = np.array(state["sleep_quality"])
    sts = np.array(state.get("stress", np.zeros(n)))
    ms  = np.array(state.get("motion_severity", np.zeros(n)))
    dt  = state.get("metadata", {}).get("dt_minutes", 5.0) / 60.0  # hours per step

    def _prob(arr: np.ndarray, threshold: float, above: bool = True) -> float:
        mask = arr > threshold if above else arr < threshold
        return float(mask.sum()) / n

    def _duration_hours(arr: np.ndarray, threshold: float, above: bool = True) -> float:
        mask = arr > threshold if above else arr < threshold
        return float(mask.sum()) * dt

    return {
        "fatigue": {
            "prob_mild":     _prob(fat, th["fatigue_mild"]),
            "prob_moderate": _prob(fat, th["fatigue_moderate"]),
            "prob_severe":   _prob(fat, th["fatigue_severe"]),
            "prob_critical": _prob(fat, th["fatigue_critical"]),
            "hours_mild":    _duration_hours(fat, th["fatigue_mild"]),
            "hours_moderate":_duration_hours(fat, th["fatigue_moderate"]),
            "hours_severe":  _duration_hours(fat, th["fatigue_severe"]),
            "peak":          float(fat.max()),
            "mean":          float(fat.mean()),
        },
        "heart_rate": {
            "prob_elevated": _prob(hr, th["hr_elevated"]),
            "prob_high":     _prob(hr, th["hr_high"]),
            "prob_critical": _prob(hr, th["hr_critical"]),
            "peak":          float(hr.max()),
            "mean":          float(hr.mean()),
        },
        "sleep_quality": {
            "prob_poor":     _prob(slp, th["sleep_poor"],    above=False),
            "prob_critical": _prob(slp, th["sleep_critical"], above=False),
            "mean":          float(slp.mean()),
            "hours_poor":    _duration_hours(slp, th["sleep_poor"], above=False),
        },
        "stress": {
            "prob_high":  _prob(sts, th["stress_high"]),
            "peak":       float(sts.max()),
            "mean":       float(sts.mean()),
        },
        "motion_sickness": {
            "prob_moderate": _prob(ms, th["ms_moderate"]),
            "prob_severe":   _prob(ms, th["ms_severe"]),
            "peak":          float(ms.max()),
        },
    }


# ─────────────────────────────────────────────
# RISK WINDOWS (flagging)
# ─────────────────────────────────────────────

def find_risk_windows(
    state: Dict[str, Any],
    thresholds: Dict[str, float] = None,
    min_window_hours: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Identify contiguous time windows where the astronaut is 'at risk'.

    A window is opened when ANY two of the following are simultaneously true:
      - fatigue > fatigue_moderate
      - sleep_quality < sleep_poor
      - stress > stress_high
      - motion_severity > ms_moderate

    Args:
        state: AstronautState.to_dict()
        thresholds: optional overrides
        min_window_hours: minimum window length to include in output

    Returns:
        List of window dicts with start/end times and peak values.
    """
    th  = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    n   = len(state["fatigue"])
    fat = np.array(state["fatigue"])
    slp = np.array(state["sleep_quality"])
    sts = np.array(state.get("stress", np.zeros(n)))
    ms  = np.array(state.get("motion_severity", np.zeros(n)))
    dt  = state.get("metadata", {}).get("dt_minutes", 5.0) / 60.0
    time_h = np.arange(n) * dt

    # Boolean risk flags
    fat_risk = fat > th["fatigue_moderate"]
    slp_risk = slp < th["sleep_poor"]
    sts_risk = sts > th["stress_high"]
    ms_risk  = ms  > th["ms_moderate"]

    # Combined: at least 2 flags active
    combined = (fat_risk.astype(int) + slp_risk.astype(int)
                + sts_risk.astype(int) + ms_risk.astype(int)) >= 2

    windows = []
    in_window = False
    start_idx = 0

    for i in range(n):
        if combined[i] and not in_window:
            in_window  = True
            start_idx  = i
        elif not combined[i] and in_window:
            in_window = False
            duration  = (i - start_idx) * dt
            if duration >= min_window_hours:
                seg = slice(start_idx, i)
                windows.append({
                    "start_hour":     float(time_h[start_idx]),
                    "end_hour":       float(time_h[i]),
                    "duration_hours": duration,
                    "peak_fatigue":   float(fat[seg].max()),
                    "min_sleep":      float(slp[seg].min()),
                    "peak_stress":    float(sts[seg].max()),
                    "peak_ms":        float(ms[seg].max()),
                    "risk_level":     _classify_window(fat[seg], slp[seg], th),
                })

    # close an open window at end of run
    if in_window:
        duration = (n - start_idx) * dt
        if duration >= min_window_hours:
            seg = slice(start_idx, n)
            windows.append({
                "start_hour":     float(time_h[start_idx]),
                "end_hour":       float(time_h[-1]),
                "duration_hours": duration,
                "peak_fatigue":   float(fat[seg].max()),
                "min_sleep":      float(slp[seg].min()),
                "peak_stress":    float(sts[seg].max()),
                "peak_ms":        float(ms[seg].max()),
                "risk_level":     _classify_window(fat[seg], slp[seg], th),
            })

    return windows


def _classify_window(
    fat_seg: np.ndarray,
    slp_seg: np.ndarray,
    th: Dict[str, float],
) -> str:
    if fat_seg.max() > th["fatigue_critical"] or slp_seg.min() < th["sleep_critical"]:
        return "CRITICAL"
    if fat_seg.max() > th["fatigue_severe"]:
        return "HIGH"
    return "MODERATE"


# ─────────────────────────────────────────────
# CUMULATIVE LOAD
# ─────────────────────────────────────────────

def compute_cumulative_load(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute cumulative physiological load metrics:
      - cumulative fatigue (area under fatigue curve, fatigue·hour)
      - cumulative sleep debt (hours of sleep quality below 0.5)
      - cumulative stress exposure

    Useful for longitudinal comparisons and recovery planning.
    """
    n   = len(state["fatigue"])
    fat = np.array(state["fatigue"])
    slp = np.array(state["sleep_quality"])
    sts = np.array(state.get("stress", np.zeros(n)))
    dt  = state.get("metadata", {}).get("dt_minutes", 5.0) / 60.0

    sleep_debt = np.maximum(0.0, 0.8 - slp)  # deficit vs 0.8 baseline

    return {
        "cumulative_fatigue_load":   float(np.trapz(fat,        dx=dt)),
        "cumulative_sleep_debt":     float(np.trapz(sleep_debt, dx=dt)),
        "cumulative_stress_exposure":float(np.trapz(sts,        dx=dt)),
        "fatigue_integral_per_day":  float(np.trapz(fat, dx=dt) / max(1, len(fat) * dt / 24)),
    }


# ─────────────────────────────────────────────
# COMPOSITE RISK SCORE (per timestep)
# ─────────────────────────────────────────────

def compute_risk_score_trace(
    state: Dict[str, Any],
    thresholds: Dict[str, float] = None,
) -> List[float]:
    """
    Return a per-timestep composite risk score in [0, 1].

    Score = 0.3*(F/10) + 0.25*(1-S) + 0.25*(ST) + 0.2*(MS/5)
    where F=fatigue, S=sleep_quality, ST=stress, MS=motion_severity.
    A score > 0.6 → HIGH risk, > 0.8 → CRITICAL.
    """
    n   = len(state["fatigue"])
    fat = np.clip(np.array(state["fatigue"]) / 10.0,          0, 1)
    slp = np.clip(1.0 - np.array(state["sleep_quality"]),     0, 1)
    sts = np.clip(np.array(state.get("stress",       np.zeros(n))), 0, 1)
    ms  = np.clip(np.array(state.get("motion_severity", np.zeros(n))) / 5.0, 0, 1)

    score = 0.30 * fat + 0.25 * slp + 0.25 * sts + 0.20 * ms
    return score.tolist()


# ─────────────────────────────────────────────
# FULL RISK REPORT (single entry point)
# ─────────────────────────────────────────────

def compute_full_risk_report(
    state: Dict[str, Any],
    events: List[Dict[str, Any]] = None,
    thresholds: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Compute the complete risk report for one completed simulation run.

    Args:
        state:      AstronautState.to_dict()
        events:     scheduler.get_timeline() (optional, for event correlation)
        thresholds: optional threshold overrides

    Returns:
        JSON-serialisable dict with all risk metrics.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    threshold_metrics = compute_threshold_metrics(state, th)
    risk_windows      = find_risk_windows(state, th)
    cumulative        = compute_cumulative_load(state)
    risk_trace        = compute_risk_score_trace(state, th)

    # Overall risk level based on worst window
    levels    = [w["risk_level"] for w in risk_windows]
    if "CRITICAL" in levels:
        overall = "CRITICAL"
    elif "HIGH" in levels:
        overall = "HIGH"
    elif "MODERATE" in levels:
        overall = "MODERATE"
    else:
        overall = "LOW"

    # Event correlation: how many risk windows contain an event onset?
    correlated_events = 0
    if events:
        for w in risk_windows:
            for e in events:
                onset_h = e.get("onset_time", e.get("simulation_time", 0))
                if w["start_hour"] <= onset_h <= w["end_hour"]:
                    correlated_events += 1
                    break

    return {
        "overall_risk_level":  overall,
        "threshold_metrics":   threshold_metrics,
        "risk_windows":        risk_windows,
        "n_risk_windows":      len(risk_windows),
        "correlated_events":   correlated_events,
        "cumulative_load":     cumulative,
        "risk_score_trace":    risk_trace,   # downsampled to 500 pts on large runs
        "thresholds_used":     th,
    }