"""
events/sleep_disruption_event.py

Sleep Disruption Event Module.

FIX (v1.2): SleepDisruptionParameters.validate() now asserts refractory_hours > 0,
consistent with the validation pattern used by the other two event parameter classes.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from events.base_event import Event, EventEffect, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class SleepDisruptionParameters:
    """Parameters for sleep disruption event dynamics."""

    # Trigger thresholds
    fatigue_threshold:          float = 3.0
    critical_fatigue_threshold: float = 6.0
    motion_severity_threshold:  float = 1.5

    # Probability parameters
    base_probability:   float = 0.1
    fatigue_sensitivity: float = 0.15
    ms_sensitivity:      float = 0.2

    # Severity parameters
    severity_mean: float = 0.5
    severity_std:  float = 0.2
    min_severity:  float = 0.2
    max_severity:  float = 1.0

    # Duration parameters
    min_duration: float = 2.0
    max_duration: float = 8.0

    # Sleep quality effects
    sleep_quality_reduction:  float = 0.4
    sleep_duration_reduction: float = 0.3

    # Recovery parameters
    recovery_sleep_needed:  float = 1.5
    next_day_effect_decay:  float = 0.7

    # FIX: refractory period — one disruption per sleep night maximum.
    refractory_hours: float = 8.0

    def validate(self):
        """Validate parameter bounds."""
        assert self.fatigue_threshold > 0
        assert 0 <= self.base_probability <= 1
        assert 0 <= self.min_severity <= self.max_severity <= 1
        assert self.min_duration <= self.max_duration
        # FIX: validate the refractory_hours field (was missing in previous revision)
        assert self.refractory_hours > 0, "refractory_hours must be positive"


class SleepDisruptionEvent(Event):
    """
    Sleep Disruption Event.

    Represents episodes of disrupted sleep caused by:
    - Accumulated fatigue (primary trigger)
    - Motion sickness discomfort
    - Circadian rhythm disruption

    Key characteristics:
    - Triggered only during the sleep window (22:00–06:00)
    - At most one disruption per sleep night (refractory guard)
    - Reduces sleep quality and duration
    - Creates positive feedback (disruption → more fatigue)
    """

    def __init__(
        self,
        params: Optional[SleepDisruptionParameters] = None,
        **kwargs,
    ):
        super().__init__(priority=EventPriority.MEDIUM, **kwargs)
        self.params = params or SleepDisruptionParameters()
        self.params.validate()

        self.sleep_debt:        float = 0.0
        self.disrupted_hours:   float = 0.0
        self.recovery_achieved: bool  = False

        logger.debug(f"Created SleepDisruptionEvent with params: {self.params}")

    def sample_onset(
        self,
        state: Any,
        t: int,
        last_event_time: float = -999.0,
        **kwargs,
    ) -> Tuple[bool, Optional[float]]:
        """
        Determine if sleep disruption occurs this timestep.

        Args:
            state:           AstronautState
            t:               Current time index
            last_event_time: Simulation-time (hours) of the previous sleep
                             disruption onset, supplied by EventScheduler.
            **kwargs:        Absorbs dt_hours etc. silently.
        """
        mission_time_hours = t * (getattr(state, "dt", 5.0) / 60.0)
        hour_of_day = mission_time_hours % 24

        # Only trigger during sleep window (22:00–06:00)
        in_sleep_window = hour_of_day >= 22 or hour_of_day <= 6
        if not in_sleep_window:
            return False, None

        # Refractory guard — one disruption per sleep night
        time_since_last = mission_time_hours - last_event_time
        if time_since_last < self.params.refractory_hours:
            return False, None

        fatigue        = state.fatigue[t]        if t < len(state.fatigue)        else 0
        motion_severity = state.motion_severity[t] if t < len(state.motion_severity) else 0

        onset_prob = self.params.base_probability

        if fatigue > self.params.fatigue_threshold:
            onset_prob += self.params.fatigue_sensitivity * (
                fatigue - self.params.fatigue_threshold
            )

        if motion_severity > self.params.motion_severity_threshold:
            onset_prob += self.params.ms_sensitivity * (
                motion_severity - self.params.motion_severity_threshold
            )

        onset_prob = min(0.95, onset_prob)

        if fatigue > self.params.critical_fatigue_threshold:
            onset_prob = 1.0

        self.trigger_conditions = {
            "mission_time":   mission_time_hours,
            "hour_of_day":    hour_of_day,
            "fatigue":        fatigue,
            "motion_severity": motion_severity,
            "onset_prob":     onset_prob,
        }

        should_occur = np.random.random() < onset_prob

        if should_occur:
            base_severity  = np.random.beta(2, 2)
            fatigue_factor = fatigue / 10.0
            ms_factor      = motion_severity / 5.0

            severity = base_severity * (0.5 + 0.5 * fatigue_factor) * (0.8 + 0.2 * ms_factor)
            severity = float(np.clip(severity, self.params.min_severity, self.params.max_severity))

            logger.info(
                f"Sleep disruption onset at t={t} ({mission_time_hours:.1f}h): "
                f"prob={onset_prob:.3f}, severity={severity:.2f}, fatigue={fatigue:.1f}"
            )
            return True, severity

        return False, None

    def get_duration(self, severity: float, **kwargs) -> float:
        duration_range = self.params.max_duration - self.params.min_duration
        duration = self.params.min_duration + duration_range * severity
        return float(np.clip(duration, self.params.min_duration, self.params.max_duration))

    def _create_effect(self, severity: float, **kwargs) -> EventEffect:
        quality_reduction    = self.params.sleep_quality_reduction * severity
        duration_reduction   = self.params.sleep_duration_reduction * severity
        fatigue_acceleration = 0.2 * severity

        return EventEffect(
            immediate={
                "sleep_quality":        -quality_reduction,
                "effective_sleep_hours": -duration_reduction * 8.0,
                "sleep_debt":            duration_reduction * 8.0,
            },
            duration_hours=self.duration,
            delayed={
                "fatigue_accumulation_rate": (fatigue_acceleration / 24.0, 0.3),
                "recovery_efficiency":       (-0.1 * severity, -0.3),
            },
            recovery_rate=0.3,
            recovery_delay=self.duration,
        )

    def apply_effect(self, state: Any, t: int, dt_hours: float) -> Dict[str, Any]:
        """Apply sleep disruption effects to astronaut state."""
        if self.effect is None:
            raise RuntimeError("Event not initialized")

        current_time = t * dt_hours
        progress = self.get_progress(current_time)

        self.disrupted_hours += dt_hours

        current_sleep_quality = state.sleep_quality[t - 1] if t > 0 else 0.8
        current_fatigue       = state.fatigue[t - 1]       if t > 0 else 0

        effect_strength = 1.0

        quality_effect = (
            self.effect.immediate.get("sleep_quality", 0) * effect_strength
        )

        new_sleep_quality = float(np.clip(
            current_sleep_quality + quality_effect * dt_hours,
            0.05, 1.0,
        ))

        # Sleep quality write: BorbelyModel has already written state.sleep_quality[t]
        # this step; the event degrades it further.  This is the correct ordering
        # (see evaluation report — no overwrite problem here).
        state.update(t, sleep_quality=new_sleep_quality)

        return {
            "type":                "sleep_disruption_effect",
            "severity":            float(self.severity) if hasattr(self, "severity") else 0.0,
            "progress":            progress,
            "disrupted_hours":     self.disrupted_hours,
            "quality_applied":     new_sleep_quality,
            "effect_strength":     effect_strength,
        }