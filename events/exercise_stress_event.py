"""
events/exercise_stress_event.py

EVA (Extravehicular Activity) / Exercise Stress Event.

Models the physiological response to spacewalk exertion or intense
in-habitat exercise. Triggered stochastically (like motion sickness)
with probability scaling from mission elapsed time and fatigue level.

BioGears action (v1.3):
    ExerciseData > GenericExercise > Intensity
    This is the correct action for metabolic workload — NOT AcuteStressData.
    BioGears models VO2 consumption, cardiac output increase, core temperature
    rise, respiratory rate increase, and tidal volume increase. The scenario
    terminates with Intensity=0.0 so BioGears captures the post-exercise
    recovery curve, which is physiologically distinct from the exercise itself.

Coupling:
    High fatigue → amplified BioGears output (handled in adapter._scale_to_twin_state)
    EVA raises HR acutely, depletes energy, accelerates fatigue accumulation
    Poor sleep → higher EVA perceived intensity (same workload = harder cardiovascular cost)

FIX (v1.2):
    Two bugs corrected:
    1. fatigue_acceleration now returned as "fatigue_forcing" for ODE injection.
    2. stress_delta returned for main loop formula, not written directly to state.

FIX (v1.3):
    get_biogears_perturbation() now returns type="stress" with exercise_intensity,
    which routes to ExerciseData in scenario_runner._build_actions().
    Previously it returned nausea_severity which routed to AcuteStressData — wrong.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from events.base_event import Event, EventEffect, EventPriority

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class ExerciseStressParameters:
    """
    Parameters for EVA / exercise stress event dynamics.

    EVA frequency reference: ISS averages ~3-4 EVAs per 6-month increment,
    each ~6-8 hours. We model shorter high-intensity bursts as the
    physiologically relevant unit.
    """

    # Trigger probability per timestep (Poisson-like)
    # ~2 EVA events per 30-day mission → λ ≈ 0.003/h
    base_rate_per_hour: float = 0.003

    # Fatigue amplifies trigger probability
    fatigue_rate_gain: float = 0.0004
    fatigue_threshold: float = 2.0

    # Exercise intensity (Beta distribution)
    intensity_alpha: float = 3.0
    intensity_beta:  float = 2.0
    min_intensity:   float = 0.3
    max_intensity:   float = 1.0

    # Duration
    min_duration:       float = 0.5
    max_duration:       float = 2.0
    duration_exponent:  float = 1.2

    # Physiological effects per unit intensity
    hr_increase_per_intensity:            float = 30.0   # bpm
    stress_increase_per_intensity:        float = 0.25   # stress units / h (additive to formula)
    fatigue_acceleration_per_intensity:   float = 0.3    # fatigue-units / h (into ODE forcing)

    # Refractory period
    refractory_hours: float = 4.0

    def validate(self):
        assert self.base_rate_per_hour > 0
        assert 0 < self.min_intensity <= self.max_intensity <= 1.0
        assert self.min_duration <= self.max_duration


# =============================================================================
# EXERCISE STRESS EVENT
# =============================================================================

class ExerciseStressEvent(Event):
    """
    EVA / Exercise Stress Episode.

    Onset: stochastic Poisson process, rate scaled by fatigue.
    Severity: exercise intensity sampled from Beta distribution.
    BioGears: fires ExerciseData > GenericExercise > Intensity so BioGears
              models actual metabolic workload (not a generic stress response).

    apply_effect() contract (v1.2):
    --------------------------------
    Returns a dict containing:
      "fatigue_forcing"  — rate (fatigue-units/h) to be passed into
                           PhysicsEngine.step(fatigue_forcing=…).
      "stress_delta"     — additive stress increment for the main-loop formula.
      "hr_applied"       — value written directly to state.hr[t].
    """

    def __init__(
        self,
        params: Optional[ExerciseStressParameters] = None,
        **kwargs,
    ):
        super().__init__(priority=EventPriority.MEDIUM, **kwargs)
        self.params = params or ExerciseStressParameters()
        self.params.validate()
        self.last_intensity: float = 0.0

    # ------------------------------------------------------------------
    # Onset decision
    # ------------------------------------------------------------------
    def sample_onset(
        self,
        state: Any,
        t: int,
        dt_hours: float = 0.5,
        last_event_time: float = -999.0,
        **kwargs,
    ) -> Tuple[bool, Optional[float]]:
        """
        Decide if an EVA / exercise stress episode begins this timestep.
        """
        mission_time_h = t * (getattr(state, "dt", 30.0) / 60.0)

        # Refractory check
        time_since_last = mission_time_h - last_event_time
        if time_since_last < self.params.refractory_hours:
            return False, None

        fatigue = float(state.fatigue[t]) if t < len(state.fatigue) else 0.0

        rate = self.params.base_rate_per_hour
        if fatigue > self.params.fatigue_threshold:
            rate += self.params.fatigue_rate_gain * (fatigue - self.params.fatigue_threshold)

        p_step = float(np.clip(rate * dt_hours, 0.0, 0.15))
        should_occur = np.random.random() < p_step

        if not should_occur:
            return False, None

        intensity = float(np.random.beta(
            self.params.intensity_alpha, self.params.intensity_beta
        ))
        intensity = float(np.clip(intensity, self.params.min_intensity, self.params.max_intensity))

        self.last_intensity = intensity
        logger.info(
            f"EVA/Exercise onset t={t} ({mission_time_h:.1f}h) "
            f"intensity={intensity:.3f} fatigue={fatigue:.1f}"
        )
        return True, intensity

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------
    def get_duration(self, severity: float, **kwargs) -> float:
        span = self.params.max_duration - self.params.min_duration
        return float(np.clip(
            self.params.min_duration + span * (severity ** self.params.duration_exponent),
            self.params.min_duration,
            self.params.max_duration,
        ))

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------
    def _create_effect(self, severity: float, **kwargs) -> EventEffect:
        hr_delta      = self.params.hr_increase_per_intensity * severity
        stress_delta  = self.params.stress_increase_per_intensity * severity
        fatigue_accel = self.params.fatigue_acceleration_per_intensity * severity

        self.metadata.update({
            "severity":             severity,
            "intensity":            severity,
            "hr_delta":             hr_delta,
            "stress_delta":         stress_delta,
            "fatigue_acceleration": fatigue_accel,
        })

        return EventEffect(
            immediate={
                "hr_delta":             hr_delta,
                "stress_delta":         stress_delta,
                "fatigue_acceleration": fatigue_accel,
            },
            duration_hours=float(self.duration or 0.0),
        )

    def apply_effect(self, state: Any, t: int, dt_hours: float) -> Dict[str, Any]:
        """
        Apply ongoing EVA effects for one timestep.

        ✅ WRITES state.hr[t]  — safe, nothing overwrites HR after the scheduler.
        ❌ does NOT write state.fatigue[t] — returns "fatigue_forcing" for ODE.
        ❌ does NOT write state.stress[t]  — returns "stress_delta" for formula.
        """
        if getattr(self, "severity", None) is None:
            return {}

        effect    = self.effect or self._create_effect(float(self.severity))
        immediate = effect.immediate if effect else {}

        hr_delta      = float(immediate.get("hr_delta", 0.0))
        stress_delta  = float(immediate.get("stress_delta", 0.0))
        fatigue_accel = float(immediate.get("fatigue_acceleration", 0.0))

        new_hr = float(np.clip(state.hr[t] + hr_delta, 40, 200))
        state.update(t, hr=new_hr)

        return {
            "type":            "exercise_stress_effect",
            "severity":        float(self.severity),
            "hr_applied":      new_hr,
            "fatigue_forcing": fatigue_accel,
            "stress_delta":    stress_delta,
        }

    def get_biogears_perturbation(self) -> Dict[str, Any]:
        """
        Return perturbation dict for BioGears adapter.

        FIX (v1.3): type is now "stress" (not "exercise_stress") and the
        intensity is passed as exercise_intensity. This routes to
        ExerciseData > GenericExercise in scenario_runner._build_actions(),
        replacing the previous incorrect AcuteStressData routing.
        """
        return {
            "type":               "stress",
            "exercise_intensity": self.last_intensity,   # → ExerciseData Intensity
            "duration_minutes":   (self.duration or 1.0) * 60.0,
        }