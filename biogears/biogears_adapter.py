"""
biogears/biogears_adapter.py

The bridge between the Digital Twin and BioGears.

Responsibilities:
  - Modify scenario parameters from digital twin event data
  - Inject stressors into BioGears
  - Scale BioGears output back to twin state variables
  - Synchronise time axes
  - Async interface for use inside FastAPI background tasks

Data flow:
  DigitalTwin Event
      │
      ▼
  BioGearsAdapter.run_perturbation_async(perturbation: dict)
      │  builds BioGearsStressor
      ▼
  ScenarioRunner.run(stressor)          ← calls bg-cli subprocess
      │  returns CSV path
      ▼
  OutputParser.parse(csv_path)          ← numpy arrays
      │
      ▼
  _scale_to_twin_state()                ← normalise to twin units
      │
      ▼
  returns dict  →  state_manager.update(t, **bio_response)

Action mapping (v1.3):
  motion_sickness   → AcuteStressData (nausea_severity)
  stress / EVA      → ExerciseData > GenericExercise (exercise_intensity)
  sleep_deprivation → SleepData On/Off + PsychomotorVigilanceTask assessment
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

from .scenario_runner import BioGearsScenarioRunner, BioGearsStressor
from .output_parser   import BioGearsOutputParser, BioGearsOutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PERTURBATION SCHEMA
# ─────────────────────────────────────────────

"""
Expected perturbation dict keys (all optional with defaults):

    type              : str   "motion_sickness" | "stress" | "sleep_deprivation"
    nausea_severity   : float [0-1]   motion_sickness events
    exercise_intensity: float [0-1]   EVA/exercise stress events
    duration_minutes  : float         how long to run BioGears for
    baseline_hr       : float         astronaut's personal baseline HR (bpm)
    baseline_map      : float         astronaut's personal baseline MAP (mmHg)
    fatigue_level     : float [0-10]  current fatigue — amplifies BioGears response
"""


# ─────────────────────────────────────────────
# ADAPTER
# ─────────────────────────────────────────────

class BioGearsAdapter:
    """
    High-level async bridge between the digital twin and BioGears.

    Usage in simulation loop (from simulation.py):

        adapter = BioGearsAdapter()

        # On motion sickness event:
        perturbation = ms_event.get_biogears_perturbation()
        bio_response = await adapter.run_perturbation_async(perturbation)
        state.update(t, hr=bio_response["hr"], spo2=bio_response["spo2"])

        # On EVA event:
        perturbation = eva_event.get_biogears_perturbation()
        bio_response = await adapter.run_perturbation_async(perturbation)
        # bio_response now contains exercise-specific outputs:
        #   peak_hr, core_temp, tidal_volume, achieved_work_rate (if available)

        # On sleep disruption event:
        perturbation = sleep_event.get_biogears_perturbation()
        bio_response = await adapter.run_perturbation_async(perturbation)
        # bio_response["pvt_score"] contains the PVT neurocognitive impairment metric
    """

    def __init__(
        self,
        bg_cli_path: str = r"C:\Users\AbhiDS\biogears\bin",
        working_dir: Optional[str] = None,
        timeout_seconds: int = 300,
    ):
        self.runner = BioGearsScenarioRunner(
            bg_cli_path=bg_cli_path,
            working_dir=working_dir,
            timeout_seconds=timeout_seconds,
        )
        self.parser = BioGearsOutputParser(skip_initial_seconds=60.0)

        self._last_output: Optional[BioGearsOutput] = None
        self._call_count: int = 0

        logger.info(
            f"BioGearsAdapter initialised | "
            f"mock={self.runner._mock_mode} | "
            f"cli={bg_cli_path}"
        )

    # ── PRIMARY INTERFACE ────────────────────

    async def run_perturbation_async(
        self,
        perturbation: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Async wrapper — runs BioGears in a thread pool so it doesn't
        block the FastAPI event loop.

        Returns:
            Dict of state variables ready to inject into AstronautState.
            Keys depend on stressor type — see _scale_to_twin_state() for full list.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.run_perturbation,
            perturbation
        )
        return result

    def run_perturbation(
        self,
        perturbation: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Synchronous version. Builds stressor → runs BioGears → returns scaled output.

        Steps:
          1. Build BioGearsStressor from perturbation dict
          2. Apply fatigue amplification (coupling between supervisory layer and BioGears)
          3. Run scenario_runner → get CSV path
          4. Parse CSV → BioGearsOutput
          5. Scale output to twin state units (type-aware)
          6. Return scaled dict
        """
        self._call_count += 1
        stressor = self._build_stressor(perturbation)

        logger.info(
            f"[BioGears call #{self._call_count}] "
            f"type={stressor.stressor_type} "
            f"nausea={stressor.nausea_severity:.2f} "
            f"exercise={stressor.exercise_intensity:.2f} "
            f"duration={stressor.duration_minutes:.1f}min"
        )

        try:
            csv_path = self.runner.run(stressor)
            output   = self.parser.parse(csv_path)
        except (RuntimeError, FileNotFoundError, TimeoutError) as e:
            logger.error(f"BioGears failed: {e}. Using degraded fallback.")
            return self._fallback_response(perturbation)

        self._last_output = output

        scaled = self._scale_to_twin_state(output, perturbation)

        logger.info(
            f"BioGears response → "
            f"HR={scaled['hr']:.1f} bpm | "
            f"MAP={scaled['map']:.1f} mmHg | "
            f"SpO2={scaled['spo2']:.1f}% | "
            f"severity={scaled['severity']:.3f}"
        )
        return scaled

    # ── STRESSOR BUILDER ─────────────────────

    def _build_stressor(self, p: Dict[str, Any]) -> BioGearsStressor:
        """
        Convert a perturbation dict → BioGearsStressor.

        Applies fatigue amplification: higher fatigue → stronger physiological response.
        This implements coupling between the supervisory layer and BioGears.

        Amplification is applied differently per stressor type:
          motion_sickness:   nausea_severity × amplification (autonomic response worsens)
          stress/exercise:   exercise_intensity × amplification (less efficient → same work = more strain)
                             duration extended (fatigue slows recovery)
          sleep_deprivation: no amplitude amplification (sleep physiology is what it is),
                             but duration is extended to model the extra time spent in
                             disrupted sleep due to fatigue-driven fragmentation.
        """
        stressor_type = p.get("type", "motion_sickness")
        nausea        = float(p.get("nausea_severity", 0.3))
        exercise      = float(p.get("exercise_intensity", 0.0))
        duration      = float(p.get("duration_minutes", 10.0))
        fatigue       = float(p.get("fatigue_level", 0.0))  # [0-10]

        fatigue_norm  = np.clip(fatigue / 10.0, 0.0, 1.0)
        amplification = 1.0 + 0.4 * fatigue_norm   # up to 40% amplification at max fatigue

        if stressor_type == "motion_sickness":
            nausea_amplified   = float(np.clip(nausea * amplification, 0.0, 1.0))
            exercise_amplified = 0.0
            duration_extended  = duration * (1.0 + 0.2 * fatigue_norm)

        elif stressor_type == "stress":
            # For exercise: fatigue makes the same workload feel harder →
            # cardiovascular response is amplified even at the same objective intensity.
            # We do NOT increase the Intensity value (that would change the scenario),
            # instead the adapter scales the output in _scale_to_twin_state.
            # Here we only extend duration to model slower post-exercise recovery.
            nausea_amplified   = 0.0
            exercise_amplified = float(np.clip(exercise, 0.0, 1.0))   # no amplitude change
            duration_extended  = duration * (1.0 + 0.3 * fatigue_norm)  # recovery takes longer

        else:  # sleep_deprivation
            nausea_amplified   = 0.0
            exercise_amplified = 0.0
            duration_extended  = duration * (1.0 + 0.15 * fatigue_norm)  # fragmented longer

        logger.debug(
            f"Stressor build: type={stressor_type} fatigue={fatigue:.1f} "
            f"amp={amplification:.2f} dur={duration:.1f}→{duration_extended:.1f}"
        )

        return BioGearsStressor(
            stressor_type      = stressor_type,
            duration_minutes   = duration_extended,
            nausea_severity    = nausea_amplified,
            exercise_intensity = exercise_amplified,
            patient_file       = p.get("patient_file", "StandardMale.xml"),
        )

    # ── OUTPUT SCALER ────────────────────────

    def _scale_to_twin_state(
        self,
        output: BioGearsOutput,
        perturbation: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Convert raw BioGears physiology → digital twin state variables.

        Scaling is type-aware:

        motion_sickness:
          - Stress proxy: HR delta / 40 (40 bpm elevation = stress 1.0)
          - Hypotension risk: if MAP drops below 70 mmHg
          - SpO2 desaturation captured

        stress/exercise (EVA):
          - Peak HR is the primary metric (can exceed 150 bpm at high intensity)
          - Core temperature rise is significant and tracked separately
          - Tidal volume increase (deeper breathing during exertion)
          - Fatigue amplification applied here to output (not to input intensity):
              exercise_output_amplification = 1 + 0.25 * fatigue_norm
            This means a fatigued astronaut's cardiovascular system reacts harder
            to the same objective exercise load.

        sleep_deprivation:
          - HR is suppressed (normal during sleep)
          - PVT score derived from sleep duration vs. expected (longer = worse)
          - No stress from sleep itself; stress comes from next-day fatigue
        """
        stressor_type = perturbation.get("type", "motion_sickness")
        baseline_hr   = float(perturbation.get("baseline_hr",  75.0))
        baseline_map  = float(perturbation.get("baseline_map", 93.0))
        fatigue       = float(perturbation.get("fatigue_level", 0.0))
        fatigue_norm  = float(np.clip(fatigue / 10.0, 0.0, 1.0))

        deltas = self.parser.summarise_perturbation_response(
            output,
            baseline_hr  = baseline_hr,
            baseline_map = baseline_map,
        )

        base_result = {
            # Direct physiology — common to all types
            "hr":               output.mean_hr,
            "peak_hr":          output.peak_hr,
            "map":              output.mean_map,
            "spo2":             float(np.mean(output.spo2)) * 100.0,
            "min_spo2":         output.min_spo2 * 100.0,
            "respiration_rate": float(np.mean(output.respiration_rate)),
            "core_temp":        float(np.mean(output.core_temp_celsius)),
            "tidal_volume":     float(np.mean(output.tidal_volume_ml)),

            # Deltas vs baseline
            "delta_hr":         deltas["delta_hr"],
            "delta_map":        deltas["delta_map"],
            "delta_spo2":       deltas["delta_spo2"],

            # Hypotension risk
            "hypotension_risk": float(np.clip((70.0 - output.map_mmhg.min()) / 20.0, 0.0, 1.0)),

            # Pass-through
            "duration_minutes": output.duration_minutes,
            "stressor_type":    stressor_type,
        }

        if stressor_type == "motion_sickness":
            # Stress from HR elevation (40 bpm = max stress)
            stress = float(np.clip(deltas["delta_hr"] / 40.0, 0.0, 1.0))
            base_result["stress"]   = stress
            base_result["severity"] = deltas["severity"]

        elif stressor_type == "stress":
            # EVA: amplify output by fatigue factor (same work, harder physiological cost)
            exercise_output_amp = 1.0 + 0.25 * fatigue_norm
            effective_peak_hr   = float(np.clip(output.peak_hr * exercise_output_amp, 0, 220))
            effective_delta_hr  = max(0.0, effective_peak_hr - baseline_hr)

            base_result["hr"]       = float(np.clip(output.mean_hr * exercise_output_amp, 0, 220))
            base_result["peak_hr"]  = effective_peak_hr
            base_result["delta_hr"] = effective_delta_hr

            # Peak core temp rise is the key exercise metric
            base_result["peak_core_temp"] = float(np.max(output.core_temp_celsius))
            base_result["core_temp_rise"] = float(np.max(output.core_temp_celsius)) - 37.0

            # Stress from HR elevation (exercise can push delta_hr > 60 bpm → normalise to 80)
            stress = float(np.clip(effective_delta_hr / 80.0, 0.0, 1.0))
            base_result["stress"]   = stress
            base_result["severity"] = float(np.clip(
                (effective_delta_hr / 80.0) * 0.5 +
                (base_result["core_temp_rise"] / 2.0) * 0.3 +
                (abs(min(0.0, deltas["delta_spo2"])) / 5.0) * 0.2,
                0.0, 1.0
            ))

        else:  # sleep_deprivation
            # During sleep physiology is suppressed — no stress from the sleep itself.
            # PVT score: proxy for neurocognitive impairment.
            # NASA PVT: reaction time increases linearly with sleep loss.
            # We estimate it from how long the patient was in disrupted sleep.
            # Longer disruption → higher impairment score [0-1].
            sleep_minutes      = output.duration_minutes
            expected_sleep_min = 480.0  # 8 hours
            pvt_score = float(np.clip(
                (expected_sleep_min - sleep_minutes) / expected_sleep_min,
                0.0, 1.0
            ))
            base_result["pvt_score"] = pvt_score
            base_result["stress"]    = 0.0   # stress comes from next-day fatigue, not sleep itself
            base_result["severity"]  = pvt_score  # severity = degree of cognitive impairment

        return base_result

    # ── ALIGNMENT HELPER ─────────────────────

    def align_response_to_timeline(
        self,
        output: BioGearsOutput,
        sim_time_hours: np.ndarray,
        event_start_hour: float,
    ) -> Dict[str, np.ndarray]:
        """
        Align BioGears output arrays to the simulation timeline.
        Use this when you want per-timestep BioGears values, not just averages.

        Returns:
            Dict of signal → array (same length as sim_time_hours, NaN outside event window)
        """
        return self.parser.align_to_simulation_time(
            output,
            sim_time_hours,
            event_start_hour,
        )

    # ── UTILITY ──────────────────────────────

    def get_version(self) -> str:
        return self.runner.get_version()

    def get_last_output(self) -> Optional[BioGearsOutput]:
        return self._last_output

    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count":  self._call_count,
            "mock_mode":   self.runner._mock_mode,
            "cli_path":    str(self.runner.bg_cli_path),
            "working_dir": str(self.runner.working_dir),
            "version":     self.get_version(),
        }

    # ── FALLBACK ─────────────────────────────

    def _fallback_response(self, perturbation: Dict[str, Any]) -> Dict[str, float]:
        """
        If BioGears fails, return a synthetic response derived from
        the perturbation parameters alone. Type-aware fallback.
        """
        stressor_type = perturbation.get("type", "motion_sickness")
        baseline      = float(perturbation.get("baseline_hr", 75.0))
        fatigue       = float(perturbation.get("fatigue_level", 0.0))
        fat_norm      = np.clip(fatigue / 10.0, 0.0, 1.0)

        if stressor_type == "motion_sickness":
            nausea   = float(perturbation.get("nausea_severity", 0.3))
            delta_hr = nausea * 25.0 * (1.0 + 0.3 * fat_norm)
            severity = float(np.clip(nausea * (1.0 + 0.4 * fat_norm), 0.0, 1.0))
            return {
                "hr": baseline + delta_hr, "peak_hr": baseline + delta_hr * 1.3,
                "map": 93.0 + nausea * 12.0, "spo2": 98.0 - nausea * 1.5,
                "min_spo2": 97.0 - nausea * 2.0, "respiration_rate": 15.0 + nausea * 4.0,
                "core_temp": 37.0 + nausea * 0.2, "tidal_volume": 500.0,
                "delta_hr": delta_hr, "delta_map": nausea * 12.0,
                "delta_spo2": -nausea * 1.5,
                "stress": float(np.clip(delta_hr / 40.0, 0.0, 1.0)),
                "severity": severity, "hypotension_risk": 0.0,
                "duration_minutes": float(perturbation.get("duration_minutes", 10.0)),
                "stressor_type": stressor_type,
            }

        elif stressor_type == "stress":
            intensity = float(perturbation.get("exercise_intensity", 0.5))
            delta_hr  = intensity * 60.0 * (1.0 + 0.25 * fat_norm)
            temp_rise = intensity * 1.5
            return {
                "hr": baseline + delta_hr, "peak_hr": baseline + delta_hr * 1.1,
                "map": 93.0 + intensity * 20.0, "spo2": 98.0 - intensity * 0.5,
                "min_spo2": 97.5 - intensity * 1.0, "respiration_rate": 15.0 + intensity * 20.0,
                "core_temp": 37.0 + temp_rise, "tidal_volume": 500.0 + intensity * 700.0,
                "peak_core_temp": 37.0 + temp_rise, "core_temp_rise": temp_rise,
                "delta_hr": delta_hr, "delta_map": intensity * 20.0, "delta_spo2": -intensity * 0.5,
                "stress": float(np.clip(delta_hr / 80.0, 0.0, 1.0)),
                "severity": float(np.clip(intensity * (1.0 + 0.3 * fat_norm), 0.0, 1.0)),
                "hypotension_risk": 0.0,
                "duration_minutes": float(perturbation.get("duration_minutes", 10.0)),
                "stressor_type": stressor_type,
            }

        else:  # sleep_deprivation
            sleep_min = float(perturbation.get("duration_minutes", 10.0))
            pvt_score = float(np.clip((480.0 - sleep_min) / 480.0, 0.0, 1.0))
            return {
                "hr": 58.0, "peak_hr": 62.0, "map": 80.0,
                "spo2": 97.0, "min_spo2": 96.5, "respiration_rate": 12.0,
                "core_temp": 36.7, "tidal_volume": 450.0,
                "delta_hr": 0.0, "delta_map": 0.0, "delta_spo2": 0.0,
                "pvt_score": pvt_score, "stress": 0.0, "severity": pvt_score,
                "hypotension_risk": 0.0,
                "duration_minutes": sleep_min,
                "stressor_type": stressor_type,
            }