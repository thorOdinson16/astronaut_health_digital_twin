"""
events/sleep_disruption_event.py

Sleep Disruption Event Module.

BioGears integration (v1.3):
    get_biogears_perturbation() now returns type="sleep_deprivation" with
    duration_minutes equal to the event's disrupted sleep window.

    In scenario_runner._build_actions(), "sleep_deprivation" triggers:
        SleepData On → AdvanceTime → SleepData Off
        → PatientAssessmentRequestData Type="PsychomotorVigilanceTask"

    The PVT is NASA's standard ISS cognitive impairment metric.
    The adapter returns bio_response["pvt_score"] ∈ [0,1] where:
        0.0 = no impairment (full sleep achieved)
        1.0 = maximal impairment (no sleep)

    The simulation loop should write pvt_score into the fatigue ODE as
    additional forcing when it is available, or log it as a diagnostic.

FIX (v1.2):
    SleepDisruptionParameters.validate() now asserts refractory_hours > 0.
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
    base_probability:    float = 0.1
    fatigue_sensitivity: float = 0.15
    ms_sensitivity:      float = 0.2

    # Severity parameters
    severity_mean: float = 0.5
    severity_std:  float = 0.2
    min_severity:  float = 0.2
    max_severity:  float = 1.0

    # Duration parameters (hours of disrupted sleep)
    min_duration: float = 2.0
    max_duration: float = 8.0

    # Sleep quality effects
    sleep_quality_reduction:  float = 0.4
    sleep_duration_reduction: float = 0.3

    # Recovery parameters
    recovery_sleep_needed:  float = 1.5
    next_day_effect_decay:  float = 0.7

    # One disruption per sleep night maximum
    refractory_hours: float = 8.0

    def validate(self):
        """Validate parameter bounds."""
        assert self.fatigue_threshold > 0
        assert 0 <= self.base_probability <= 1
        assert 0 <= self.min_severity <= self.max_severity <= 1
        assert self.min_duration <= self.max_duration
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

    BioGears integration (v1.3):
    - get_biogears_perturbation() returns type="sleep_deprivation"
    - BioGears runs SleepData On/Off + PVT assessment
    - Caller can read bio_response["pvt_score"] for cognitive impairment logging
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

        # Stores PVT result from BioGears for external consumption
        self.last_pvt_score: float = 0.0

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

        fatigue         = state.fatigue[t]         if t < len(state.fatigue)         else 0
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
            "mission_time":    mission_time_hours,
            "hour_of_day":     hour_of_day,
            "fatigue":         fatigue,
            "motion_severity": motion_severity,
            "onset_prob":      onset_prob,
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
                "sleep_quality":         -quality_reduction,
                "effective_sleep_hours": -duration_reduction * 8.0,
                "sleep_debt":             duration_reduction * 8.0,
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
        quality_effect = self.effect.immediate.get("sleep_quality", 0)

        new_sleep_quality = float(np.clip(
            current_sleep_quality + quality_effect * dt_hours,
            0.05, 1.0,
        ))

        state.update(t, sleep_quality=new_sleep_quality)

        return {
            "type":              "sleep_disruption_effect",
            "severity":          float(self.severity) if hasattr(self, "severity") else 0.0,
            "progress":          progress,
            "disrupted_hours":   self.disrupted_hours,
            "quality_applied":   new_sleep_quality,
            "pvt_score":         self.last_pvt_score,
        }

    def get_biogears_perturbation(self) -> Dict[str, Any]:
        """
        Return perturbation dict for BioGears adapter.

        Routes to SleepData On/Off + PatientAssessmentRequestData PsychomotorVigilanceTask
        in scenario_runner._build_actions().

        duration_minutes is the disrupted sleep window duration.
        The adapter will return bio_response["pvt_score"] which can be stored
        in self.last_pvt_score by the simulation loop for diagnostic logging.
        """
        disrupted_minutes = (self.duration or 2.0) * 60.0
        return {
            "type":             "sleep_deprivation",
            "duration_minutes": disrupted_minutes,
        }

    def record_pvt_score(self, pvt_score: float) -> None:
        """
        Called by the simulation loop after BioGears returns bio_response.
        Stores the PVT neurocognitive impairment score for this disruption event.

        Usage in simulation loop:
            bio_response = await adapter.run_perturbation_async(perturbation)
            sleep_event.record_pvt_score(bio_response.get("pvt_score", 0.0))
        """
        self.last_pvt_score = float(np.clip(pvt_score, 0.0, 1.0))
        logger.info(f"Sleep disruption PVT score recorded: {self.last_pvt_score:.3f}")