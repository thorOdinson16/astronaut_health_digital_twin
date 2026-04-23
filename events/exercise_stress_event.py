"""
events/exercise_stress_event.py

EVA (Extravehicular Activity) / Exercise Stress Event.

Models the physiological response to spacewalk exertion or intense
in-habitat exercise. Triggered stochastically (like motion sickness)
with probability scaling from mission elapsed time and fatigue level.

BioGears stressor: "stress" with exercise_intensity — already supported
by the adapter and scenario_runner.

Coupling:
    High fatigue → amplified BioGears response (handled in adapter)
    EVA raises HR acutely, depletes energy, accelerates fatigue accumulation
    Poor sleep → higher EVA risk (higher intensity felt at same workload)

FIX (v1.2):
    Two bugs corrected from the previous revision:

    1. fatigue_acceleration was written to state.fatigue[t] in apply_effect()
       but PhysicsEngine.step() overwrites that slot every step because
       FatigueModel._fatigue_state is self-contained.  The fix moves forcing
       into the ODE: apply_effect() now returns the forcing rate via the dict
       key "fatigue_forcing", which the simulation loop collects and passes to
       engine.step(fatigue_forcing=…).  apply_effect() no longer writes
       state.fatigue at all.

    2. new_stress computed in apply_effect() was immediately overwritten by the
       post-scheduler stress formula in the main loop:
           total_stress = clip(0.12 + circadian + fat_term + ms_term, …)
           state.update(t, stress=total_stress)
       The fix: apply_effect() no longer writes state.stress either.  Instead
       it returns "stress_delta" in the effect dict so that the main loop can
       incorporate it into the formula:
           eva_stress_bonus = sum of stress_delta from active EVA events
           total_stress = clip(0.12 + circadian + fat_term + ms_term + eva_bonus, …)
       See api/routes/simulation.py for the updated loop.

    HR write from apply_effect() is retained — no formula overwrites it after
    the scheduler runs, so it is safe to write directly.
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
    BioGears: fires "stress" perturbation so BioGears models the
              full cardiovascular response to exertion.

    apply_effect() contract (v1.2):
    --------------------------------
    Returns a dict containing:
      "fatigue_forcing"  — rate (fatigue-units/h) to be passed into
                           PhysicsEngine.step(fatigue_forcing=…).
                           The simulation loop is responsible for summing
                           contributions from all active EVA events each step.
      "stress_delta"     — additive stress increment to be folded into the
                           main-loop stress formula (not written to state here).
      "hr_applied"       — value written directly to state.hr[t] (safe: nothing
                           overwrites HR after the scheduler runs).
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

        Args:
            state           : AstronautState
            t               : Current time index
            dt_hours        : Step duration [hours] — forwarded by check_triggers
            last_event_time : Simulation-time (hours) of the previous EVA onset
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
                "stress_delta":         stress_delta,       # rate, used by stress formula
                "fatigue_acceleration": fatigue_accel,      # rate, forwarded to ODE
            },
            duration_hours=float(self.duration or 0.0),
        )

    def apply_effect(self, state: Any, t: int, dt_hours: float) -> Dict[str, Any]:
        """
        Apply ongoing EVA effects for one timestep.

        IMPORTANT — what this method writes and what it does NOT write:

        ✅ WRITES state.hr[t]
           Nothing in the main loop overwrites HR after the scheduler, so
           writing it directly is safe and correct.

        ❌ does NOT write state.fatigue[t]
           Fatigue is owned by PhysicsEngine._fatigue_state.  Writing to
           state.fatigue[t] here would be overwritten by the very next
           engine.step() call.  Instead, the forcing rate is returned as
           "fatigue_forcing" (fatigue-units/hour) and the simulation loop
           passes it to engine.step(fatigue_forcing=…).

        ❌ does NOT write state.stress[t]
           The main loop computes total_stress from a formula and calls
           state.update(t, stress=total_stress) after the scheduler returns,
           which would silently overwrite any write made here.  Instead,
           "stress_delta" is returned so the loop can add it into the formula.

        Returns:
            dict with keys:
                "fatigue_forcing"  — rate to pass to PhysicsEngine.step()
                "stress_delta"     — additive increment for the stress formula
                "hr_applied"       — new HR value written to state
                "type", "severity" — bookkeeping
        """
        if getattr(self, "severity", None) is None:
            return {}

        effect    = self.effect or self._create_effect(float(self.severity))
        immediate = effect.immediate if effect else {}

        hr_delta      = float(immediate.get("hr_delta", 0.0))
        stress_delta  = float(immediate.get("stress_delta", 0.0))
        fatigue_accel = float(immediate.get("fatigue_acceleration", 0.0))

        # HR: safe to write directly
        new_hr = float(np.clip(state.hr[t] + hr_delta, 40, 200))
        state.update(t, hr=new_hr)

        # Fatigue forcing and stress delta — returned to caller, NOT written here
        return {
            "type":            "exercise_stress_effect",
            "severity":        float(self.severity),
            "hr_applied":      new_hr,
            "fatigue_forcing": fatigue_accel,   # caller passes to engine.step()
            "stress_delta":    stress_delta,    # caller adds to stress formula
        }

    def get_biogears_perturbation(self) -> Dict[str, Any]:
        """Return perturbation dict for BioGears adapter."""
        return {
            "type":               "stress",
            "exercise_intensity": self.last_intensity,
            "duration_minutes":   (self.duration or 1.0) * 60.0,
        }