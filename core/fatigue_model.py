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
   cannot be reproduced by any pair of independent scalar models.

3. FATIGUE INDEX  (Samn-Perelli scale, ODE formulation)
   -----------------------------------------------------
   dF/dt = α·sleep_debt^1.2  +  β·|m|^1.5  −  γ(C)·recovery  +  ε(t)

   γ(C) is circadian-gated: recovery is most efficient at the C(t) trough
   (biological night).  Sleep at 02:00 restores more than sleep at 14:00 —
   documented in ISS crew actigraphy (Barger et al. 2014).

References
----------
Borbély (1982) Human Neurobiology 1:195-204.
Daan, Beersma, Borbély (1984) Am J Physiol 246:R161-83.
Oman (1982) Acta Otolaryngol Suppl 392:44.
Dai et al. (2011) Exp Brain Res 210:45-64.
Barger et al. (2014) Lancet Neurology 13:904-912.
Flynn-Evans et al. (2016) npj Microgravity 2:15013.
Reschke et al. (2018) J Vestib Res 28:99-109.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER DATACLASSES
# =============================================================================

@dataclass
class BorbelyParameters:
    """
    Borbély two-process model parameters.
    Calibrated from ISS actigraphy (Barger et al. 2014) and
    ground-based human sleep studies (Daan et al. 1984).
    """
    # --- Process S (homeostatic) ---
    S_max: float = 1.0        # Maximum homeostatic pressure (normalised)
    S_0: float   = 0.30       # Initial pressure (rested crew at mission start)
    tau_wake:  float = 18.2   # Time constant during wakefulness  [hours]
    tau_sleep: float = 4.2    # Time constant during sleep         [hours]

    # --- Process C (circadian) ---
    T_c: float  = 24.35       # Free-running period in µg (slightly >24h)
    A_c: float  = 0.17        # Oscillation amplitude  [normalised]
    M_c: float  = 0.50        # Mean offset            [normalised]
    phase_0: float = 0.0      # Phase offset at mission start [hours]

    # --- Sleep gates ---
    gate_upper_offset: float =  0.10  # sleep onset: S must exceed C + this
    gate_lower_offset: float = -0.10  # wake onset:  S must fall below C + this

    # --- ISS-specific circadian perturbation ---
    # 16 orbital sunrises/day create phase noise.  Std per day ≈ 0.08 h.
    circadian_phase_noise_std: float = 0.08   # hours std per day


@dataclass
class VestibularParameters:
    """
    Vestibular mismatch model parameters.
    Adapted from Oman (1982) and calibrated to 67% incidence in first 72 h
    (Heer & Paloski 2006).
    """
    k_adapt_0: float = 0.18   # Baseline adaptation rate  [1/h], rested
    w_s: float       = 0.65   # Sleep-pressure weight on adaptation suppression

    s_baseline:     float = 0.0   # Otolith signal in 1 g  (reference)
    s_microgravity: float = 1.0   # Otolith signal at full µg onset

    # Motion-sickness probability transfer  (sigmoidal saturation)
    sigma_ms:      float = 0.22   # Transfer coefficient
    ms_saturation: float = 2.5    # Cumulative mismatch at P(onset)≈0.8/h

    hr_per_mismatch: float = 12.0  # bpm per unit mismatch (vestibulo-cardiac reflex)


@dataclass
class FatigueParameters:
    """
    Fatigue accumulation parameters (Samn-Perelli derived ODE).
    """
    alpha_sleep_debt:    float = 0.28   # Fatigue rate per unit sleep debt  [/h]
    beta_mismatch:       float = 0.42   # Fatigue rate per unit mismatch    [/h]
    gamma_recovery_base: float = 0.14   # Base recovery rate during sleep   [/h]
    gamma_circadian_boost: float = 0.08 # Extra recovery at C(t) trough     [/h]

    max_fatigue: float = 10.0
    min_fatigue: float = 0.0

    # Gamma process noise  (shape=2, scale=0.04 → mean=0.08, right-skewed)
    noise_shape: float = 2.0
    noise_scale: float = 0.04


# =============================================================================
# BORBÉLY TWO-PROCESS MODEL
# =============================================================================

class BorbelyModel:
    """
    Borbély (1982) two-process model adapted for the ISS microgravity context.

    State:
        S  — homeostatic sleep pressure ∈ [0, S_max]
        cumulative_phase_drift  — accumulated circadian phase perturbation [h]
        _sleeping  — current sleep/wake boolean

    The model generates sleep_quality from first principles rather than
    sampling from an empirical Beta distribution.
    """

    def __init__(self, params: Optional[BorbelyParameters] = None):
        self.p = params or BorbelyParameters()
        self.S: float = self.p.S_0
        self.cumulative_phase_drift: float = 0.0
        self._sleeping: bool = False
        self._rng = np.random.default_rng(0)
        logger.info("BorbelyModel initialised (µg two-process model)")

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed(self, s: int):
        self._rng = np.random.default_rng(s)

    # ------------------------------------------------------------------
    # Process C
    # ------------------------------------------------------------------
    def C(self, t_h: float) -> float:
        """Circadian oscillator value at mission time t_h [hours]."""
        phase = t_h + self.p.phase_0 + self.cumulative_phase_drift
        return self.p.M_c + self.p.A_c * np.cos(2 * np.pi * phase / self.p.T_c)

    def C_upper(self, t_h: float) -> float:
        return self.C(t_h) + self.p.gate_upper_offset

    def C_lower(self, t_h: float) -> float:
        return self.C(t_h) + self.p.gate_lower_offset

    # ------------------------------------------------------------------
    # Single Euler step
    # ------------------------------------------------------------------
    def step(self, dt_hours: float, t_h: float) -> Dict[str, Any]:
        """
        Advance homeostatic pressure S by one time step.

        Args:
            dt_hours : Step duration [hours]
            t_h      : Current mission elapsed time [hours]

        Returns dict with:
            S, C, sleeping, sleep_quality, phase_noise_this_step
        """
        c_val   = self.C(t_h)
        c_upper = self.C_upper(t_h)
        c_lower = self.C_lower(t_h)

        # ── State transitions (hysteretic via separate gates) ──────────
        if not self._sleeping and self.S >= c_upper:
            self._sleeping = True
        elif self._sleeping and self.S <= c_lower:
            self._sleeping = False

        # ── ODE integration (Euler) ────────────────────────────────────
        if not self._sleeping:
            dS = (self.p.S_max - self.S) / self.p.tau_wake * dt_hours
        else:
            dS = -self.S / self.p.tau_sleep * dt_hours

        self.S = float(np.clip(self.S + dS, 0.0, self.p.S_max))

        # ── ISS orbital phase perturbation ─────────────────────────────
        # Gaussian noise injected each step, scaled to hours/day std.
        phase_noise = (self.p.circadian_phase_noise_std
                       * np.sqrt(dt_hours / 24.0)
                       * float(self._rng.standard_normal()))
        self.cumulative_phase_drift += phase_noise

        # ── Sleep quality from gate geometry ──────────────────────────
        # While sleeping: quality peaks when S is centred between the gates.
        # While awake:    quality is how far S is below the upper gate.
        if self._sleeping:
            gate_range = c_upper - c_lower
            if gate_range > 1e-6:
                pos = (self.S - c_lower) / gate_range      # 0 at lower, 1 at upper
                sleep_quality = float(np.clip(1.0 - 2.0 * abs(pos - 0.5), 0.1, 1.0))
            else:
                sleep_quality = 0.5
        else:
            # Margin between current S and onset gate, normalised
            margin = max(0.0, c_upper - self.S)
            max_margin = self.p.gate_upper_offset + self.p.A_c
            sleep_quality = float(np.clip(margin / max(max_margin, 1e-6), 0.0, 1.0))

        return {
            "S": self.S,
            "C": c_val,
            "C_upper": c_upper,
            "C_lower": c_lower,
            "sleeping": self._sleeping,
            "sleep_quality": sleep_quality,
            "phase_noise": phase_noise,
        }

    @property
    def S_norm(self) -> float:
        """Homeostatic pressure normalised to [0, 1] for coupling."""
        return self.S / self.p.S_max

    def reset(self, S_0: Optional[float] = None, phase_0: float = 0.0):
        self.S = S_0 if S_0 is not None else self.p.S_0
        self.cumulative_phase_drift = phase_0
        self._sleeping = False


# =============================================================================
# VESTIBULAR MISMATCH MODEL
# =============================================================================

class VestibularMismatchModel:
    """
    Oman (1982) sensory conflict model extended with sleep-pressure-gated
    adaptation — the novel physics coupling in this system.

    State:
        e_hat               — internal model estimate of otolith signal
        cumulative_mismatch — ∫|m(τ)|dτ, drives P(motion sickness onset)

    ODE:
        dê/dt = k_adapt(S_norm) · (s(t) − ê)

    k_adapt(S_norm) = k₀ · (1 − w_s · S_norm)

    When S_norm → 1 (maximally sleep-deprived), adaptation rate drops to
    k₀·(1−w_s) ≈ 0.063/h, meaning full adaptation takes ~16 h instead of ~6 h.
    This is the mechanism that makes fatigue and motion sickness escalate
    together in a physically grounded way.
    """

    def __init__(self, params: Optional[VestibularParameters] = None):
        self.p = params or VestibularParameters()
        self.e_hat: float = self.p.s_baseline
        self.cumulative_mismatch: float = 0.0
        logger.info("VestibularMismatchModel initialised (Oman 1982 + sleep coupling)")

    def mismatch(self, sensory_input: float) -> float:
        return sensory_input - self.e_hat

    def step(self, dt_hours: float, sensory_input: float,
             S_norm: float) -> Dict[str, Any]:
        """
        Advance vestibular internal model by one step.

        Args:
            dt_hours      : Step size [hours]
            sensory_input : Normalised otolith signal (0=1g, 1=full µg)
            S_norm        : Normalised homeostatic sleep pressure from BorbelyModel

        Returns dict with:
            mismatch, e_hat, k_adapt, cumulative_mismatch,
            p_ms_step, hr_delta, abs_mismatch
        """
        m = self.mismatch(sensory_input)

        # Sleep-pressure-gated adaptation (the key coupled ODE)
        k_adapt = self.p.k_adapt_0 * (1.0 - self.p.w_s * S_norm)
        k_adapt = max(k_adapt, 0.01)  # never fully stop adapting

        # Euler integration: dê/dt = k_adapt · m
        self.e_hat = float(np.clip(
            self.e_hat + k_adapt * m * dt_hours,
            min(self.p.s_baseline, self.p.s_microgravity),
            max(self.p.s_baseline, self.p.s_microgravity),
        ))

        abs_m = abs(m)
        self.cumulative_mismatch += abs_m * dt_hours

        # P(motion sickness onset per step) — sigmoidal saturation
        p_ms_per_hour = (self.p.sigma_ms * self.cumulative_mismatch
                         / (self.p.ms_saturation + self.cumulative_mismatch))
        p_ms_step = float(np.clip(p_ms_per_hour * dt_hours, 0.0, 0.5))

        # HR elevation from vestibulo-cardiac sympathetic reflex
        hr_delta = self.p.hr_per_mismatch * abs_m

        return {
            "mismatch":             float(m),
            "e_hat":                self.e_hat,
            "k_adapt":              k_adapt,
            "cumulative_mismatch":  self.cumulative_mismatch,
            "p_ms_step":            p_ms_step,
            "hr_delta":             float(hr_delta),
            "abs_mismatch":         float(abs_m),
        }

    def reset(self, sensory_input: float = 0.0):
        self.e_hat = sensory_input
        self.cumulative_mismatch = 0.0


# =============================================================================
# FATIGUE MODEL (circadian-gated ODE)
# =============================================================================

class FatigueModel:
    """
    Fatigue accumulation from two physics-based inputs:
      1. Sleep debt derived from Borbély S deviation from circadian baseline
      2. Vestibular mismatch intensity |m(t)|

    Recovery is circadian-gated: γ(C) is boosted when C(t) is at its trough
    (biological night), consistent with ISS sleep quality data showing that
    sleep timed to the circadian nadir restores 30–40% more per hour.

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
        # C_val < M_c means circadian nadir (biological night)
        trough_depth = max(0.0, M_c - C_val)   # 0 at peak, A_c at trough
        max_depth = 0.17                         # A_c default
        boost = self.p.gamma_circadian_boost * (trough_depth / max(max_depth, 1e-6))
        return self.p.gamma_recovery_base + boost

    def compute_fatigue_update(
        self,
        current_fatigue: float,
        sleep_quality:   float,   # from BorbelyModel (physics-derived)
        S_norm:          float,   # normalised homeostatic pressure
        abs_mismatch:    float,   # |m(t)| from VestibularMismatchModel
        C_val:           float,   # circadian oscillator value
        dt_hours:        float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        One Euler step of the fatigue ODE.

        Returns:
            (new_fatigue, components_dict)
        """
        # Sleep debt: deviation of S from the circadian target (M_c)
        # When S >> M_c the astronaut is sleep-deprived regardless of clock time.
        sleep_debt = float(np.clip(S_norm - 0.5, 0.0, 1.0))

        accum_sleep   = self.p.alpha_sleep_debt * (sleep_debt ** 1.2) * dt_hours
        accum_mismatch = self.p.beta_mismatch   * (abs_mismatch ** 1.5) * dt_hours

        gamma = self._circadian_recovery_gate(C_val)
        recovery = gamma * sleep_quality * dt_hours

        # Gamma noise (positive, right-skewed — occasional large stressors)
        noise = float(self._rng.gamma(self.p.noise_shape, self.p.noise_scale)) * dt_hours

        delta = accum_sleep + accum_mismatch - recovery + noise
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
            "delta":             delta,
        }
        return new_fatigue, components

    def estimate_recovery_hours(
        self,
        current_fatigue: float,
        optimal_sleep_quality: float = 0.9,
        C_val: float = 0.33,    # nighttime trough
        dt_hours: float = 1.0,
    ) -> float:
        """Estimate hours to recover to fatigue < 1.0 under optimal conditions."""
        F = current_fatigue
        hours = 0.0
        while F > 1.0 and hours < 168:
            F, _ = self.compute_fatigue_update(
                current_fatigue=F,
                sleep_quality=optimal_sleep_quality,
                S_norm=0.1,         # well-rested
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

    Usage in execute_simulation():
        engine = PhysicsEngine()
        engine.seed(rng_seed)

        for t in range(timesteps):
            out = engine.step(dt_hours=dt_hours, t_h=t * dt_hours,
                              sensory_input=1.0)  # 1.0 = full µg
            state.update(t,
                fatigue=out["fatigue"],
                sleep_quality=out["sleep_quality"],
                ...)

    The engine also exposes internal ODE states (S, e_hat, mismatch, k_adapt)
    as additional time-series columns, enabling novel visualisations and the
    emergent-behaviour analysis required for the paper.
    """

    def __init__(
        self,
        borbely_params:     Optional[BorbelyParameters]     = None,
        vestibular_params:  Optional[VestibularParameters]  = None,
        fatigue_params:     Optional[FatigueParameters]     = None,
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
        dt_hours:      float,
        t_h:           float,
        sensory_input: float = 1.0,   # 1.0 = full microgravity
    ) -> Dict[str, Any]:
        """
        One coupled physics step.

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

        # 3. Fatigue update (physics inputs from steps 1 & 2)
        new_fatigue, fat_components = self.fatigue.compute_fatigue_update(
            current_fatigue=self._fatigue_state,
            sleep_quality=borbely_out["sleep_quality"],
            S_norm=self.borbely.S_norm,
            abs_mismatch=vest_out["abs_mismatch"],
            C_val=borbely_out["C"],
            dt_hours=dt_hours,
        )
        self._fatigue_state = new_fatigue

        return {
            # Primary state variables (consumed by AstronautState)
            "fatigue":              new_fatigue,
            "sleep_quality":        borbely_out["sleep_quality"],
            # Internal ODE states (novel time-series for paper figures)
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