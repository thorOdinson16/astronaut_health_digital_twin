"""
analytics/trend_analysis.py

Trend analysis utilities for the Astronaut Digital Twin.
Computes:
  - rolling statistics (mean, std, slope) over a sliding window
  - time-to-recovery estimates after event onsets
  - mission-phase summaries (adaptation / stable / late)
  - linear trend detection per variable
"""

import numpy as np
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────
# ROLLING STATISTICS
# ─────────────────────────────────────────────

def rolling_stats(
    values: List[float],
    dt_hours: float,
    window_hours: float = 24.0,
) -> Dict[str, List[float]]:
    """
    Compute rolling mean, std, and linear slope over a sliding window.

    Args:
        values:       list of scalar values (one per timestep)
        dt_hours:     hours per timestep
        window_hours: sliding window width in hours

    Returns:
        dict with keys mean, std, slope (all same length as input, NaN-padded at start)
    """
    arr    = np.array(values, dtype=float)
    n      = len(arr)
    w      = max(2, int(window_hours / dt_hours))
    mean   = np.full(n, np.nan)
    std    = np.full(n, np.nan)
    slope  = np.full(n, np.nan)

    x = np.arange(w, dtype=float)

    for i in range(w - 1, n):
        seg      = arr[i - w + 1 : i + 1]
        mean[i]  = seg.mean()
        std[i]   = seg.std()
        # linear slope (units per hour)
        p        = np.polyfit(x * dt_hours, seg, 1)
        slope[i] = p[0]

    # Replace NaN with first valid value for cleaner frontend consumption
    first_valid = w - 1
    mean[:first_valid]  = mean[first_valid]  if not np.isnan(mean[first_valid])  else 0.0
    std[:first_valid]   = std[first_valid]   if not np.isnan(std[first_valid])   else 0.0
    slope[:first_valid] = slope[first_valid] if not np.isnan(slope[first_valid]) else 0.0

    return {
        "mean":  mean.tolist(),
        "std":   std.tolist(),
        "slope": slope.tolist(),
    }


# ─────────────────────────────────────────────
# TIME-TO-RECOVERY AFTER EVENTS
# ─────────────────────────────────────────────

def compute_recovery_times(
    state: Dict[str, Any],
    events: List[Dict[str, Any]],
    recovery_fatigue_target: float = 2.0,
    recovery_sleep_target: float = 0.65,
    max_recovery_hours: float = 96.0,
) -> List[Dict[str, Any]]:
    """
    For each event in the timeline, estimate how long the astronaut takes
    to recover (fatigue ≤ recovery_fatigue_target AND sleep ≥ target)
    after the event ends.

    Args:
        state:                    AstronautState.to_dict()
        events:                   scheduler.get_timeline()
        recovery_fatigue_target:  fatigue index considered 'recovered'
        recovery_sleep_target:    sleep quality considered 'recovered'
        max_recovery_hours:       cap on recovery search window

    Returns:
        List of dicts, one per event with onset_hour, type, recovery_hours.
    """
    fat   = np.array(state["fatigue"])
    slp   = np.array(state["sleep_quality"])
    dt    = state.get("metadata", {}).get("dt_minutes", 5.0) / 60.0
    n     = len(fat)

    out = []
    for ev in events:
        onset_h  = float(ev.get("onset_time", ev.get("simulation_time", 0)))
        duration = float(ev.get("duration", 0))
        end_h    = onset_h + duration

        end_idx  = min(n - 1, int(end_h / dt))
        max_idx  = min(n, end_idx + int(max_recovery_hours / dt))

        # Find first index where both criteria are met after event end
        recovered_idx = None
        for i in range(end_idx, max_idx):
            if fat[i] <= recovery_fatigue_target and slp[i] >= recovery_sleep_target:
                recovered_idx = i
                break

        if recovered_idx is not None:
            recovery_h = (recovered_idx - end_idx) * dt
        else:
            recovery_h = max_recovery_hours  # capped / not recovered

        out.append({
            "event_type":      ev.get("type", "unknown"),
            "onset_hour":      onset_h,
            "end_hour":        end_h,
            "severity":        float(ev.get("severity", 0)),
            "recovery_hours":  recovery_h,
            "recovered":       recovered_idx is not None,
        })

    return out


# ─────────────────────────────────────────────
# MISSION-PHASE SUMMARY
# ─────────────────────────────────────────────

def compute_phase_summary(
    state: Dict[str, Any],
    adaptation_hours: float = 72.0,    # first 3 days
    late_phase_hours: float = 168.0,   # last 7 days
) -> Dict[str, Any]:
    """
    Summarise health metrics across three mission phases:
      - adaptation  (0 → adaptation_hours)
      - stable      (adaptation_hours → mission_end - late_phase_hours)
      - late        (mission_end - late_phase_hours → mission_end)

    Returns per-phase mean, std, and peak for fatigue, sleep, HR, stress.
    """
    dt_h  = state.get("metadata", {}).get("dt_minutes", 5.0) / 60.0
    n     = len(state["fatigue"])
    total = n * dt_h

    adapt_idx = int(adaptation_hours / dt_h)
    late_idx  = max(adapt_idx + 1, n - int(late_phase_hours / dt_h))

    phases = {
        "adaptation": slice(0,          adapt_idx),
        "stable":     slice(adapt_idx,  late_idx),
        "late":       slice(late_idx,   n),
    }

    vars_ = {
        "fatigue":        np.array(state["fatigue"]),
        "sleep_quality":  np.array(state["sleep_quality"]),
        "hr":             np.array(state["hr"]),
        "stress":         np.array(state.get("stress", np.zeros(n))),
    }

    summary = {}
    for phase_name, sl in phases.items():
        summary[phase_name] = {}
        for var_name, arr in vars_.items():
            seg = arr[sl]
            if len(seg) == 0:
                summary[phase_name][var_name] = {"mean": None, "std": None, "peak": None}
            else:
                summary[phase_name][var_name] = {
                    "mean": float(seg.mean()),
                    "std":  float(seg.std()),
                    "peak": float(seg.max()),
                }
        summary[phase_name]["duration_hours"] = len(arr[sl]) * dt_h

    return {
        "mission_total_hours": total,
        "phases": summary,
    }


# ─────────────────────────────────────────────
# LINEAR TREND DETECTION
# ─────────────────────────────────────────────

def detect_trends(
    state: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Fit a linear trend to each key variable over the full mission.
    Returns slope (units/hour), R², and a plain-English direction label.

    Useful for spotting slow deterioration that threshold checks miss.
    """
    n    = len(state["fatigue"])
    dt_h = state.get("metadata", {}).get("dt_minutes", 5.0) / 60.0
    t    = np.arange(n) * dt_h

    results = {}
    for var in ("fatigue", "sleep_quality", "hr", "stress"):
        arr = np.array(state.get(var, np.zeros(n)), dtype=float)
        if len(arr) < 2:
            continue
        p     = np.polyfit(t, arr, 1)
        slope = p[0]
        # R² from residuals
        predicted = np.polyval(p, t)
        ss_res    = np.sum((arr - predicted) ** 2)
        ss_tot    = np.sum((arr - arr.mean()) ** 2)
        r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "increasing" if var != "sleep_quality" else "degrading"
        else:
            direction = "decreasing" if var != "sleep_quality" else "improving"

        results[var] = {
            "slope_per_hour": float(slope),
            "slope_per_day":  float(slope * 24),
            "r_squared":      float(r2),
            "direction":      direction,
            "significant":    bool(r2 > 0.1 and abs(slope) > 0.005),
        }

    return results


# ─────────────────────────────────────────────
# FULL TREND REPORT (single entry point)
# ─────────────────────────────────────────────

def compute_full_trend_report(
    state: Dict[str, Any],
    events: List[Dict[str, Any]] = None,
    window_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    Compute the complete trend report for one simulation run.

    Returns rolling stats, recovery times per event, phase summaries,
    and linear trend detection — all in a single JSON-serialisable dict.
    """
    dt_h = state.get("metadata", {}).get("dt_minutes", 5.0) / 60.0

    rolling = {
        var: rolling_stats(state.get(var, [0.0]), dt_h, window_hours)
        for var in ("fatigue", "sleep_quality", "hr", "stress")
    }

    recovery_times = []
    if events:
        recovery_times = compute_recovery_times(state, events)

    phase_summary = compute_phase_summary(state)
    trends        = detect_trends(state)

    # Downsample rolling arrays to 300 pts for frontend
    n = len(state["fatigue"])
    if n > 300:
        idx = np.round(np.linspace(0, n - 1, 300)).astype(int).tolist()
        for var in rolling:
            for stat in rolling[var]:
                arr = rolling[var][stat]
                rolling[var][stat] = [arr[i] for i in idx]

    return {
        "rolling_stats":    rolling,
        "recovery_times":   recovery_times,
        "phase_summary":    phase_summary,
        "trends":           trends,
    }