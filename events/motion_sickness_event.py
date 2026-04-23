"""
events/motion_sickness_event.py

Space Motion Sickness Event — physics-based onset.

PREVIOUS:  Poisson process with empirical lambda, modulated by scalar fatigue multipliers.

NEW:       Onset probability is read directly from VestibularMismatchModel.p_ms_step,
           which is derived from the integrated sensory conflict ODE:

               P(onset/step) = σ · ∫|m(τ)|dτ  /  (ξ + ∫|m(τ)|dτ)  × dt

           This means:
             - Onset probability rises continuously with unresolved mismatch
             - Sleep-deprived astronauts (high S_norm) adapt slower → higher ∫|m|dt
               → higher onset probability automatically, with no magic multipliers
             - The refractory period is implicit: immediately after onset the
               internal model ê jumps toward s(t), reducing mismatch and therefore
               suppressing the next onset probability naturally

           The event still samples severity and duration stochastically
           (these are observable, not ODE-derived), but now severity is
           modulated by k_adapt (slower adaptation → longer/worse episodes).

FIX (v1.3):
    apply_effect() no longer writes state.stress[t].  It returns "stress_delta"
    so the main simulation loop can incorporate it into the stress formula,
    following the same pattern as ExerciseStressEvent.

References
----------
Oman (1982) Acta Otolaryngol Suppl 392:44.
Heer & Paloski (2006) Auton Neurosci 129:77-79.
Reschke et al. (2018) J Vestib Res 28:99-109.
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
class MotionSicknessParameters:
    """
    Parameters for motion sickness event dynamics.
    Onset probability is now supplied by the physics engine;
    these parameters govern the severity and duration sampling.
    """

    # Severity (Beta distribution — right-skewed, most episodes mild-moderate)
    severity_alpha: float = 2.0
    severity_beta: float = 3.0
    min_severity: float = 0.15
    max_severity: float = 1.0

    # Duration scales with severity (nonlinear)
    min_duration: float = 0.5  # hours
    max_duration: float = 4.0  # hours
    duration_exponent: float = 1.5  # duration ∝ severity^exponent

    # Severity amplification from slow adaptation (k_suppress ∈ [0,1])
    # Severely sleep-deprived astronaut: k_suppress ≈ 0.65 → severity × 1.33
    severity_k_suppress_gain: float = 0.50

    # Physiological effects per unit severity
    hr_increase_per_severity: float = 15.0  # bpm
    stress_increase_per_severity: float = 0.30
    sleep_degradation_per_severity: float = 0.20

    def validate(self):
        assert 0 < self.severity_alpha
        assert 0 < self.severity_beta
        assert 0 <= self.min_severity <= self.max_severity <= 1.0
        assert 0 < self.min_duration <= self.max_duration


# =============================================================================
# MOTION SICKNESS EVENT
# =============================================================================


class MotionSicknessEvent(Event):
    """
    Space Motion Sickness Episode.

    Onset probability is driven by the vestibular mismatch ODE (PhysicsEngine);
    this class handles:
      - Receiving p_ms_step from the engine and deciding if onset occurs
      - Sampling severity (amplified by adaptation suppression)
      - Sampling duration
      - Computing physiological effects on HR, stress, sleep quality

    apply_effect() contract (v1.3):
    --------------------------------
    Returns a dict containing:
      "stress_delta"     — additive stress increment for the main-loop formula.
                           Does NOT write state.stress[t] directly — the main
                           loop handles that via _collect_event_stress_deltas().
      "sleep_delta"      — sleep quality degradation (written directly to
                           state.sleep_quality[t], which is safe because the
                           Borbély model only writes state.sleep_quality once
                           per step and the event degrades it further).
      "hr_applied"       — value written directly to state.hr[t] (safe: nothing
                           overwrites HR after the scheduler runs).
    """

    def __init__(
        self,
        params: Optional[MotionSicknessParameters] = None,
        **kwargs,
    ):
        super().__init__(priority=EventPriority.HIGH, **kwargs)
        self.params = params or MotionSicknessParameters()
        self.params.validate()
        # Tracks ongoing adaptation state for analysis
        self.last_k_adapt: float = 0.18
        self.last_k_suppress: float = 0.0

    # ------------------------------------------------------------------
    # Onset decision  (replaces the old Poisson sample_onset)
    # ------------------------------------------------------------------
    def sample_onset(
        self,
        state: Any,
        t: int,
        # Physics engine outputs passed in from simulation loop:
        p_ms_step: float = 0.0,  # P(onset this step) from VestibularMismatchModel
        k_suppress: float = 0.0,  # adaptation suppression fraction [0,1]
        k_adapt: float = 0.18,  # current adaptation rate
        fatigue_multiplier: float = 1.0,  # legacy kwarg, kept for API compat
        **kwargs,
    ) -> Tuple[bool, Optional[float]]:
        """
        Decide if a motion sickness episode begins this time step.

        Args:
            state       : AstronautState
            t           : Current time index
            p_ms_step   : Onset probability from physics engine (ODE-derived)
            k_suppress  : Fraction of adaptation capacity lost to sleep debt
            k_adapt     : Current vestibular adaptation rate
            fatigue_multiplier : Legacy; ignored (physics handles this via k_suppress)

        Returns:
            (should_occur, severity_if_occurs)
        """
        self.last_k_adapt = k_adapt
        self.last_k_suppress = k_suppress

        # Stochastic gate on ODE-derived probability
        should_occur = np.random.random() < p_ms_step

        if not should_occur:
            return False, None

        # ── Sample severity ────────────────────────────────────────────
        # Base severity from right-skewed Beta (mostly mild–moderate)
        base_sev = float(
            np.random.beta(self.params.severity_alpha, self.params.severity_beta)
        )

        # Amplify by adaptation suppression: slow adapters have worse episodes
        # Because ê is farther from s(t), the vestibular conflict is larger.
        sev = base_sev * (1.0 + self.params.severity_k_suppress_gain * k_suppress)
        sev = float(np.clip(sev, self.params.min_severity, self.params.max_severity))

        mission_time_h = t * (getattr(state, "dt", 30.0) / 60.0)
        logger.info(
            f"MS onset t={t} ({mission_time_h:.1f}h) "
            f"p={p_ms_step:.4f} sev={sev:.3f} k_sup={k_suppress:.3f}"
        )
        return True, sev

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------
    def get_duration(self, severity: float, **kwargs) -> float:
        span = self.params.max_duration - self.params.min_duration
        return float(
            np.clip(
                self.params.min_duration
                + span * (severity**self.params.duration_exponent),
                self.params.min_duration,
                self.params.max_duration,
            )
        )

    # ------------------------------------------------------------------
    # Physiological effects
    # ------------------------------------------------------------------
    def _create_effect(self, severity: float, **kwargs) -> EventEffect:
        """
        Compute HR, stress, and sleep-quality effects.
        Effect magnitudes scale with severity.
        HR delta is additionally informed by vestibulo-cardiac reflex
        (hr_delta from PhysicsEngine is additive).
        """
        hr_delta = self.params.hr_increase_per_severity * severity
        stress_delta = self.params.stress_increase_per_severity * severity
        sleep_delta = self.params.sleep_degradation_per_severity * severity

        self.metadata.update(
            {
                "severity": severity,
                "k_suppress": self.last_k_suppress,
                "k_adapt": self.last_k_adapt,
                "hr_delta": hr_delta,
                "stress_delta": stress_delta,
                "sleep_delta": sleep_delta,
            }
        )

        return EventEffect(
            immediate={
                "hr_delta": hr_delta,
                "stress_delta": stress_delta,
                "sleep_quality_delta": -sleep_delta,
                "motion_severity": severity * 5.0,
            },
            duration_hours=float(self.duration or 0.0),
        )

    def apply_effect(self, state: Any, t: int, dt_hours: float) -> Dict[str, Any]:
        """
        Apply ongoing event effects to state at timestep t.

        FIX (v1.3): Does NOT write state.stress[t] — returns "stress_delta"
        for the main loop to fold into the stress formula, following the
        same pattern as ExerciseStressEvent.

        Writes state.hr[t], state.sleep_quality[t], and state.motion_severity[t]
        directly — these are safe because nothing in the main loop overwrites
        them after the scheduler runs.
        """
        if getattr(self, "severity", None) is None:
            return {}

        effect = self.effect or self._create_effect(float(self.severity))
        immediate = effect.immediate if effect else {}
        hr_delta = float(immediate.get("hr_delta", 0.0))
        stress_delta = float(immediate.get("stress_delta", 0.0))
        sleep_quality_delta = float(immediate.get("sleep_quality_delta", 0.0))
        motion_severity = float(immediate.get("motion_severity", 0.0))

        # HR: clamp to physiological bounds — safe to write directly
        new_hr = float(np.clip(state.hr[t] + hr_delta * dt_hours, 40, 200))
        state.update(t, hr=new_hr)

        # Sleep quality: degradation — safe to write directly
        # (Borbély model writes once per step; the event degrades it further)
        new_sleep = float(
            np.clip(state.sleep_quality[t] + sleep_quality_delta * dt_hours, 0.05, 1.0)
        )
        state.update(t, sleep_quality=new_sleep)

        # Motion severity — safe to write directly
        new_ms = float(np.clip(motion_severity, 0.0, 5.0))
        state.update(t, motion_severity=new_ms)

        # FIX (v1.3): return stress_delta for main loop — do NOT write state.stress[t]
        # The main loop computes total_stress from the formula and calls
        # state.update(t, stress=total_stress) after the scheduler returns,
        # which would silently overwrite any write made here.
        return {
            "type": "motion_sickness_effect",
            "severity": float(self.severity),
            "hr_applied": new_hr,
            "sleep_quality_applied": new_sleep,
            "motion_severity_applied": new_ms,
            "stress_delta": stress_delta,  # ← returned, not written
        }

    # Legacy shim so EventScheduler can call check_triggers normally
    def _check_refractory(self, state: Any, current_time: float) -> float:
        """
        In the ODE model refractory effects emerge naturally (mismatch drops
        after onset), but we keep a small hard floor to prevent back-to-back
        events within 1 h which are physiologically implausible.
        """
        refractory_window = 1.0
        for event in reversed(state.event_log):
            if event["type"] == "motion_sickness":
                event_time = event.get("simulation_time", 0)
                time_since = current_time - (event_time / 60.0)  # convert min→h
                if time_since < refractory_window:
                    return time_since / refractory_window
        return 1.0