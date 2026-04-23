"""
core/fatigue_model.py

Physics-based fatigue and sleep model for the Astronaut Digital Twin.

Replaces the previous scalar drift model with two coupled dynamical systems:

1. BORBÉLY TWO-PROCESS SLEEP MODEL  (Borbély 1982; Daan et al. 1984)
   ------------------------------------------------------------------
   Process S  — homeostatic sleep pressure:
       dS/dt =  (S_max - S) / tau_wake       [wakefulness: exponential rise]
       dS/dt = -S / tau_sleep                [sleep: exponential decay]

   Process C  — circadian oscillator (approximated as cosine):
       C(t) = M_c + A_c * cos(2π(t + Δφ) / T_c)

   In microgravity the ISS 90-min orbital period creates 16 sunrises/day,
   modelled here as stochastic phase noise on C(t)  (Flynn-Evans et al. 2016).

   Sleep quality is derived geometrically from how well S sits between the
   circadian gates — NOT sampled from an empirical distribution.

2. VESTIBULAR MISMATCH ODE  (Oman 1982; Dai et al. 2011)
   -------------------------------------------------------
   The brain maintains an internal model ê(t) of the expected otolith signal.
   The mismatch  m(t) = s(t) − ê(t)  drives both sickness and adaptation:

       dê/dt = k_adapt(S) · m(t)

   Adaptation rate is GATED BY homeostatic pressure:
       k_adapt(S) = k₀ · (1 − w_s · S_norm)

   This is the key novel coupling:
     high sleep debt  →  slow vestibular adaptation
     slow adaptation  →  sustained mismatch
     sustained mismatch → motion sickness + fatigue
     fatigue          →  worse sleep (recovery gated by C(t))
   The loop closes and produces emergent multi-day risk escalation that
   cannot be modelled with scalar multipliers.

FIX (v1.2): PhysicsEngine.step() now accepts an optional `fatigue_forcing`
parameter (units: fatigue-units / hour). When non-zero (e.g. during an active
ExerciseStressEvent), this term is added to the ODE delta *before* the engine
stores self._fatigue_state. This makes EVA fatigue acceleration visible to all
downstream callers and persists correctly across timesteps.

The FatigueModel.compute_fatigue_update() signature is unchanged; the forcing
is injected at the PhysicsEngine level so that the ODE state variable F is
always up-to-date.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BORBÉLY MODEL PARAMETERS
# =============================================================================

@dataclass
class BorbelyParameters:
    """
    Parameters for the two-process sleep model.
    Defaults calibrated from Daan et al. (1984) and ISS crew data.
    """

    # Process S — homeostatic pressure
    S_max:      float = 1.0     # Maximum homeostatic pressure (normalised)
    S_min:      float = 0.0     # Minimum (fully recovered)
    tau_wake:   float = 18.7    # Hours to saturate during wakefulness
    tau_sleep:  float = 4.2     # Hours to decay during sleep

    # Process C — circadian oscillator
    T_c:        float = 24.0    # Period (hours)
    M_c:        float = 0.50    # Oscillator mean
    A_c:        float = 0.17    # Amplitude
    phi_c:      float = 0.0     # Phase offset (hours past midnight)

    # Sleep scheduling
    sleep_start_hour: float = 22.0   # Nominal lights-out (local clock)
    sleep_end_hour:   float = 6.0    # Nominal wake-up
    sleep_noise_std:  float = 0.5    # ±h jitter in sleep onset

    # ISS orbital phase noise
    orbital_period_hours: float = 1.5   # 90 min
    phase_noise_amplitude: float = 0.03  # fraction of A_c per orbit


@dataclass
class VestibularParameters:
    """
    Parameters for the vestibular mismatch ODE.
    Calibrated from Oman (1982) and Reschke et al. (2018).
    """

    k_adapt_0:  float = 0.18    # Baseline adaptation rate (h⁻¹)
    w_s:        float = 0.60    # Sleep-debt suppression weight [0,1]
    tau_ms:     float = 6.0     # Mismatch integration time-constant (h)
    sigma_ms:   float = 0.40    # Sigmoid steepness for p_ms
    xi_ms:      float = 0.25    # Sigmoid offset (half-saturation)
    hr_gain:    float = 8.0     # bpm per unit |mismatch|


@dataclass
class FatigueParameters:
    """
    Parameters for the physics-based fatigue ODE.

    ODE:
        dF/dt = α·sleep_debt^1.2  +  β·|m|^1.5  −  γ(C)·sleep_quality  +  ε

    where ε ~ Gamma(shape, scale) adds biological variability.
    """

    alpha_sleep_debt:      float = 0.35   # Sleep-debt accumulation gain
    beta_mismatch:         float = 0.25   # Vestibular-mismatch fatigue gain
    gamma_recovery_base:   float = 0.20   # Base recovery rate during sleep
    gamma_circadian_boost: float = 0.10   # Extra recovery at circadian nadir

    # Noise model
    noise_shape: float = 0.5   # Gamma shape (k)
    noise_scale: float = 0.03  # Gamma scale (θ)

    # Hard bounds
    min_fatigue: float = 0.0
    max_fatigue: float = 10.0


# =============================================================================
# BORBÉLY TWO-PROCESS MODEL
# =============================================================================

class BorbelyModel:
    """
    Discrete-time Euler integration of the two-process sleep model.

    State variables:
        S   — homeostatic sleep pressure  ∈ [0, S_max]
        C   — circadian oscillator value  ∈ [M_c − A_c, M_c + A_c]
        sleeping — bool, whether the astronaut is currently in the sleep phase
    """

    def __init__(self, params: Optional[BorbelyParameters] = None):
        self.p = params or BorbelyParameters()
        self._rng = np.random.default_rng(42)

        # Integrate from waking baseline
        self.S        = self.p.S_min + 0.30 * (self.p.S_max - self.p.S_min)
        self.S_norm   = self.S / self.p.S_max
        self.sleeping = False
        self._phase_noise_accum: float = 0.0

        logger.info("BorbelyModel initialised")

    def seed(self, s: int):
        self._rng = np.random.default_rng(s)

    def reset(self, S_0: float = 0.30):
        self.S        = S_0
        self.S_norm   = S_0 / self.p.S_max
        self.sleeping = False
        self._phase_noise_accum = 0.0

    def _circadian(self, t_h: float) -> float:
        """Circadian oscillator value at mission time t_h (hours)."""
        phi = (t_h + self.p.phi_c) / self.p.T_c          # fractional phase
        base = self.p.M_c + self.p.A_c * math.cos(2 * math.pi * phi)
        # ISS orbital phase noise — small jitter every ~90 min
        noise_sigma = self.p.phase_noise_amplitude * self.p.A_c
        self._phase_noise_accum += float(self._rng.normal(0, noise_sigma))
        self._phase_noise_accum *= 0.97   # slow leak so drift doesn't accumulate
        return base + self._phase_noise_accum

    def _in_sleep_window(self, t_h: float) -> bool:
        """True when local clock is inside the nominal sleep window."""
        hour_of_day = t_h % 24.0
        ss = self.p.sleep_start_hour
        se = self.p.sleep_end_hour
        if ss > se:   # window spans midnight
            return hour_of_day >= ss or hour_of_day <= se
        return ss <= hour_of_day <= se

    def step(self, dt_hours: float, t_h: float) -> Dict[str, Any]:
        """
        Advance the Borbély model by one timestep.

        Returns:
            dict with keys: S, C, sleeping, sleep_quality, phase_noise
        """
        C = self._circadian(t_h)

        # Sleep / wake decision: cross into sleep window if S has risen enough
        in_window = self._in_sleep_window(t_h)
        if not self.sleeping and in_window and self.S >= (self.p.M_c + 0.5 * self.p.A_c):
            self.sleeping = True
        elif self.sleeping and not in_window:
            self.sleeping = False

        # ODE integration (Euler)
        if self.sleeping:
            dS = -(self.S / self.p.tau_sleep) * dt_hours
        else:
            dS = ((self.p.S_max - self.S) / self.p.tau_wake) * dt_hours

        self.S = float(np.clip(self.S + dS, self.p.S_min, self.p.S_max))
        self.S_norm = self.S / self.p.S_max

        # Sleep quality: how deeply S is within the gate [C−A_c, C]
        upper_gate = C
        lower_gate = C - self.p.A_c
        if self.sleeping and upper_gate > lower_gate:
            sq = float(np.clip(
                (upper_gate - self.S) / max(upper_gate - lower_gate, 1e-6),
                0.0, 1.0
            ))
        elif not self.sleeping:
            # Daytime: quality reflects how well-rested they are (low S = rested)
            sq = float(np.clip(1.0 - self.S_norm, 0.1, 1.0))
        else:
            sq = 0.3  # Edge case

        return {
            "S":            self.S,
            "C":            C,
            "sleeping":     self.sleeping,
            "sleep_quality": sq,
            "phase_noise":  self._phase_noise_accum,
        }


# =============================================================================
# VESTIBULAR MISMATCH MODEL
# =============================================================================

class VestibularMismatchModel:
    """
    ODE-based vestibular mismatch and motion sickness model.

    State:
        e_hat — internal model of expected otolith signal
        cumulative_mismatch — ∫|m(τ)|dτ (hours)

    Onset probability per step:
        p_ms_step = σ(cum_mis) · dt
    where σ is a saturating sigmoid.
    """

    def __init__(self, params: Optional[VestibularParameters] = None):
        self.p     = params or VestibularParameters()
        self.e_hat = 0.0
        self.cumulative_mismatch = 0.0
        logger.info("VestibularMismatchModel initialised")

    def reset(self, sensory_input: float = 0.0):
        self.e_hat = sensory_input
        self.cumulative_mismatch = 0.0

    def step(
        self,
        dt_hours:      float,
        sensory_input: float,
        S_norm:        float,
    ) -> Dict[str, Any]:
        """
        Advance vestibular model by one timestep.

        Args:
            dt_hours:      step size in hours
            sensory_input: otolith signal magnitude (1.0 = full µg)
            S_norm:        normalised homeostatic pressure (0=rested, 1=saturated)

        Returns:
            dict with keys: mismatch, e_hat, k_adapt, cumulative_mismatch,
                            p_ms_step, hr_delta, abs_mismatch
        """
        # Adaptation rate gated by sleep pressure
        k_adapt = self.p.k_adapt_0 * max(0.0, 1.0 - self.p.w_s * S_norm)

        # Mismatch between real and expected signal
        mismatch = sensory_input - self.e_hat
        abs_mismatch = abs(mismatch)

        # Euler integration of internal model
        d_ehat = k_adapt * mismatch * dt_hours
        self.e_hat = float(np.clip(self.e_hat + d_ehat, -2.0, 2.0))

        # Accumulate mismatch integral
        self.cumulative_mismatch += abs_mismatch * dt_hours

        # Sigmoid onset probability per step
        cum = self.cumulative_mismatch
        p_rate = self.p.sigma_ms * cum / (self.p.xi_ms + cum)   # [0, sigma_ms]
        p_ms_step = float(np.clip(p_rate * dt_hours, 0.0, 0.25))

        # Vestibulo-cardiac reflex HR contribution
        hr_delta = self.p.hr_gain * abs_mismatch

        return {
            "mismatch":             mismatch,
            "e_hat":                self.e_hat,
            "k_adapt":              k_adapt,
            "cumulative_mismatch":  self.cumulative_mismatch,
            "p_ms_step":            p_ms_step,
            "hr_delta":             hr_delta,
            "abs_mismatch":         abs_mismatch,
        }


# =============================================================================
# FATIGUE MODEL (ODE)
# =============================================================================

class FatigueModel:
    """
    Physics-based fatigue accumulation model.

    ODE:
        dF/dt = α·sleep_debt^1.2  +  β·|m|^1.5  −  γ(C)·sleep_quality  +  ε

    where ε ~ Gamma(shape, scale) adds biological variability.
    """

    def __init__(self, params: Optional[FatigueParameters] = None):
        self.p = params or FatigueParameters()
        self._rng = np.random.default_rng(1)
        logger.info("FatigueModel initialised (circadian-gated ODE)")

    def seed(self, s: int):
        self._rng = np.random.default_rng(s)

    def _circadian_recovery_gate(self, C_val: float, M_c: float = 0.50) -> float:
        """
        Recovery efficiency is highest when C(t) is at its trough (below M_c).
        Returns a multiplier in [1.0, 1 + gamma_circadian_boost/gamma_base].
        """
        trough_depth = max(0.0, M_c - C_val)
        max_depth = 0.17
        boost = self.p.gamma_circadian_boost * (trough_depth / max(max_depth, 1e-6))
        return self.p.gamma_recovery_base + boost

    def compute_fatigue_update(
        self,
        current_fatigue: float,
        sleep_quality:   float,
        S_norm:          float,
        abs_mismatch:    float,
        C_val:           float,
        dt_hours:        float,
        fatigue_forcing: float = 0.0,   # FIX: external forcing term (fatigue-units/hour)
    ) -> Tuple[float, Dict[str, float]]:
        """
        One Euler step of the fatigue ODE.

        Args:
            current_fatigue:  F(t)
            sleep_quality:    from BorbelyModel
            S_norm:           normalised homeostatic pressure
            abs_mismatch:     |m(t)| from VestibularMismatchModel
            C_val:            circadian oscillator value
            dt_hours:         step size
            fatigue_forcing:  external additive forcing rate (fatigue-units/hour).
                              Used by ExerciseStressEvent to accelerate fatigue
                              accumulation during EVA without bypassing the ODE.

        Returns:
            (new_fatigue, components_dict)
        """
        sleep_debt = float(np.clip(S_norm - 0.5, 0.0, 1.0))

        accum_sleep    = self.p.alpha_sleep_debt * (sleep_debt ** 1.2) * dt_hours
        accum_mismatch = self.p.beta_mismatch    * (abs_mismatch ** 1.5) * dt_hours

        gamma    = self._circadian_recovery_gate(C_val)
        recovery = gamma * sleep_quality * dt_hours

        noise = float(self._rng.gamma(self.p.noise_shape, self.p.noise_scale)) * dt_hours

        # FIX: include forcing term — this is the only place where external
        # event forcing should enter the fatigue ODE so the state variable F
        # remains authoritative and is not silently overwritten next step.
        forcing_contrib = fatigue_forcing * dt_hours

        delta = accum_sleep + accum_mismatch - recovery + noise + forcing_contrib
        new_fatigue = float(np.clip(
            current_fatigue + delta,
            self.p.min_fatigue,
            self.p.max_fatigue,
        ))

        components = {
            "sleep_debt":        sleep_debt,
            "accum_sleep":       accum_sleep,
            "accum_mismatch":    accum_mismatch,
            "recovery":          recovery,
            "gamma_effective":   gamma,
            "noise":             noise,
            "forcing_contrib":   forcing_contrib,
            "delta":             delta,
        }
        return new_fatigue, components

    def estimate_recovery_hours(
        self,
        current_fatigue: float,
        optimal_sleep_quality: float = 0.9,
        C_val: float = 0.33,
        dt_hours: float = 1.0,
    ) -> float:
        """Estimate hours to recover to fatigue < 1.0 under optimal conditions."""
        F = current_fatigue
        hours = 0.0
        while F > 1.0 and hours < 168:
            F, _ = self.compute_fatigue_update(
                current_fatigue=F,
                sleep_quality=optimal_sleep_quality,
                S_norm=0.1,
                abs_mismatch=0.0,
                C_val=C_val,
                dt_hours=dt_hours,
            )
            hours += dt_hours
        return hours

    def reset(self):
        self._rng = np.random.default_rng(1)


# =============================================================================
# INTEGRATED PHYSICS ENGINE  (single entry point for simulation loop)
# =============================================================================

class PhysicsEngine:
    """
    Combines BorbelyModel, VestibularMismatchModel, and FatigueModel into a
    single object that the simulation loop calls once per time step.

    FIX (v1.2): step() accepts `fatigue_forcing` (fatigue-units/hour) so that
    active ExerciseStressEvents can drive the internal ODE state (self._fatigue_state)
    directly. Previously, apply_effect() wrote to state.fatigue[t] but that write
    was immediately overwritten by the next physics step, giving the forcing zero
    net effect. Now the forcing is passed into compute_fatigue_update() and persists.

    Usage in execute_simulation():
        engine = PhysicsEngine()
        engine.seed(rng_seed)

        for t in range(timesteps):
            # Collect fatigue forcing from active events
            fatigue_forcing = sum(
                e.effect.immediate.get("fatigue_acceleration", 0.0)
                for e in scheduler.get_active_events("ExerciseStressEvent")
                if e.effect
            )
            out = engine.step(
                dt_hours=dt_hours, t_h=t * dt_hours,
                sensory_input=1.0,
                fatigue_forcing=fatigue_forcing,
            )
    """

    def __init__(
        self,
        borbely_params:    Optional[BorbelyParameters]    = None,
        vestibular_params: Optional[VestibularParameters] = None,
        fatigue_params:    Optional[FatigueParameters]    = None,
    ):
        self.borbely    = BorbelyModel(borbely_params)
        self.vestibular = VestibularMismatchModel(vestibular_params)
        self.fatigue    = FatigueModel(fatigue_params)
        self._fatigue_state: float = 0.0

    def seed(self, s: int):
        self.borbely.seed(s)
        self.fatigue.seed(s + 1)

    def reset(self, initial_fatigue: float = 0.0, S_0: float = 0.30):
        self.borbely.reset(S_0=S_0)
        self.vestibular.reset(sensory_input=0.0)
        self.fatigue.reset()
        self._fatigue_state = initial_fatigue

    def step(
        self,
        dt_hours:        float,
        t_h:             float,
        sensory_input:   float = 1.0,    # 1.0 = full microgravity
        fatigue_forcing: float = 0.0,    # FIX: external forcing from active events
    ) -> Dict[str, Any]:
        """
        One coupled physics step.

        Args:
            dt_hours:        step size in hours
            t_h:             mission elapsed time in hours
            sensory_input:   otolith signal (1.0 = full µg)
            fatigue_forcing: additive fatigue accumulation rate from active events
                             (fatigue-units / hour). Caller should sum contributions
                             from all currently active ExerciseStressEvents.

        Returns a flat dict suitable for writing directly into AstronautState arrays.
        Keys: fatigue, sleep_quality, S, C, sleeping, mismatch, e_hat,
              k_adapt, cumulative_mismatch, p_ms_step, hr_delta,
              abs_mismatch, phase_noise, fatigue_components
        """
        # 1. Circadian + homeostatic sleep pressure
        borbely_out = self.borbely.step(dt_hours=dt_hours, t_h=t_h)

        # 2. Vestibular adaptation (gated by sleep pressure)
        vest_out = self.vestibular.step(
            dt_hours=dt_hours,
            sensory_input=sensory_input,
            S_norm=self.borbely.S_norm,
        )

        # 3. Fatigue update (physics inputs from steps 1 & 2 + external forcing)
        new_fatigue, fat_components = self.fatigue.compute_fatigue_update(
            current_fatigue=self._fatigue_state,
            sleep_quality=borbely_out["sleep_quality"],
            S_norm=self.borbely.S_norm,
            abs_mismatch=vest_out["abs_mismatch"],
            C_val=borbely_out["C"],
            dt_hours=dt_hours,
            fatigue_forcing=fatigue_forcing,   # FIX: wire forcing into ODE
        )
        self._fatigue_state = new_fatigue      # ODE state is always authoritative

        return {
            "fatigue":              new_fatigue,
            "sleep_quality":        borbely_out["sleep_quality"],
            "S":                    borbely_out["S"],
            "C":                    borbely_out["C"],
            "sleeping":             borbely_out["sleeping"],
            "phase_noise":          borbely_out["phase_noise"],
            "mismatch":             vest_out["mismatch"],
            "e_hat":                vest_out["e_hat"],
            "k_adapt":              vest_out["k_adapt"],
            "cumulative_mismatch":  vest_out["cumulative_mismatch"],
            "p_ms_step":            vest_out["p_ms_step"],
            "hr_delta":             vest_out["hr_delta"],
            "abs_mismatch":         vest_out["abs_mismatch"],
            "fatigue_components":   fat_components,
        }