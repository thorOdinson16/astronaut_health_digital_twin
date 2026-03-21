"""
biogears/biogears_adapter.py

The bridge between the Digital Twin and BioGears.

Responsibilities (per guidelines):
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

Guidelines reference:
  "This demonstrates real integration."
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
# (what the simulation loop sends to the adapter)
# ─────────────────────────────────────────────

"""
Expected perturbation dict keys (all optional with defaults):

    type              : str   "motion_sickness" | "stress" | "sleep_deprivation"
    nausea_severity   : float [0-1]   from motion_sickness_event.get_biogears_perturbation()
    exercise_intensity: float [0-1]   for stress events
    duration_minutes  : float         how long to run BioGears for
    baseline_hr       : float         astronaut's personal baseline HR (bpm)
    baseline_map      : float         astronaut's personal baseline MAP (mmHg)
    fatigue_level     : float [0-10]  current fatigue — used to amplify response
"""


# ─────────────────────────────────────────────
# ADAPTER
# ─────────────────────────────────────────────

class BioGearsAdapter:
    """
    High-level async bridge between the digital twin and BioGears.

    Usage in simulation loop (from simulation.py):

        adapter = BioGearsAdapter(bg_cli_path=r"C:\\BioGears\\bin\\bg-cli.exe")

        # On motion sickness event:
        perturbation = event_obj.get_biogears_perturbation()
        bio_response = await adapter.run_perturbation_async(perturbation)

        # Inject into state:
        state.update(t, hr=bio_response["hr"], spo2=bio_response["spo2"])
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

        # Cache last output for debugging / trend analysis
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

        Args:
            perturbation: Dict from event.get_biogears_perturbation()

        Returns:
            Dict of state variables ready to inject into AstronautState:
                hr, map, spo2, rr, core_temp,
                delta_hr, delta_map, severity
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
          2. Amplify stressor if fatigue is high (coupling logic)
          3. Run scenario_runner → get CSV path
          4. Parse CSV → BioGearsOutput
          5. Scale output to twin state units
          6. Summarise perturbation response (deltas)

        Args:
            perturbation: Dict (see schema at top of file)

        Returns:
            Dict with scaled state variables
        """
        self._call_count += 1
        stressor = self._build_stressor(perturbation)

        logger.info(
            f"[BioGears call #{self._call_count}] "
            f"type={stressor.stressor_type} "
            f"nausea={stressor.nausea_severity:.2f} "
            f"duration={stressor.duration_minutes:.1f}min"
        )

        try:
            csv_path = self.runner.run(stressor)
            output   = self.parser.parse(csv_path)
        except (RuntimeError, FileNotFoundError, TimeoutError) as e:
            logger.error(f"BioGears failed: {e}. Using degraded fallback.")
            return self._fallback_response(perturbation)

        self._last_output = output

        # Scale back to twin state
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
        """
        stressor_type     = p.get("type", "motion_sickness")
        nausea            = float(p.get("nausea_severity", 0.3))
        exercise          = float(p.get("exercise_intensity", 0.0))
        duration          = float(p.get("duration_minutes", 10.0))
        fatigue           = float(p.get("fatigue_level", 0.0))  # [0-10]

        # ── FATIGUE AMPLIFICATION ──────────────
        # High fatigue makes the body respond more severely to stressors.
        # Guidelines: "High fatigue → increases event probability"
        # We extend this: high fatigue also amplifies BioGears stressor severity.
        fatigue_norm  = np.clip(fatigue / 10.0, 0.0, 1.0)    # normalise to [0-1]
        amplification = 1.0 + 0.4 * fatigue_norm              # up to 40% amplification

        nausea_amplified   = float(np.clip(nausea   * amplification, 0.0, 1.0))
        exercise_amplified = float(np.clip(exercise * amplification, 0.0, 1.0))
        duration_extended  = duration * (1.0 + 0.2 * fatigue_norm)  # fatigued = longer recovery

        logger.debug(
            f"Stressor amplification: fatigue={fatigue:.1f} "
            f"→ ×{amplification:.2f} "
            f"nausea {nausea:.2f}→{nausea_amplified:.2f}"
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

        Guidelines:
          "Scale BioGears output — synchronize time axes"

        Scaling rationale:
          - HR:   BioGears gives bpm → twin uses bpm directly
          - MAP:  BioGears gives mmHg → twin uses mmHg directly
          - SpO2: BioGears gives fraction [0-1] → twin uses percentage [0-100]
          - Stress: derived from HR delta, normalised to [0-1]
          - Severity: composite from parser summary
        """
        baseline_hr  = float(perturbation.get("baseline_hr",  75.0))
        baseline_map = float(perturbation.get("baseline_map", 93.0))

        deltas = self.parser.summarise_perturbation_response(
            output,
            baseline_hr  = baseline_hr,
            baseline_map = baseline_map,
        )

        # Stress proxy: normalised HR elevation
        # HR delta of 40 bpm → stress = 1.0
        stress_from_hr = float(np.clip(deltas["delta_hr"] / 40.0, 0.0, 1.0))

        # Hypotension risk proxy (MAP drop below 70 mmHg = risk)
        hypotension_risk = float(np.clip((70.0 - output.map_mmhg.min()) / 20.0, 0.0, 1.0))

        return {
            # Direct physiology
            "hr":               output.mean_hr,
            "peak_hr":          output.peak_hr,
            "map":              output.mean_map,
            "spo2":             float(np.mean(output.spo2)) * 100.0,
            "min_spo2":         output.min_spo2 * 100.0,
            "respiration_rate": float(np.mean(output.respiration_rate)),
            "core_temp":        float(np.mean(output.core_temp_celsius)),

            # Deltas vs baseline
            "delta_hr":         deltas["delta_hr"],
            "delta_map":        deltas["delta_map"],
            "delta_spo2":       deltas["delta_spo2"],

            # Derived twin variables
            "stress":           stress_from_hr,
            "severity":         deltas["severity"],
            "hypotension_risk": hypotension_risk,

            # Pass-through for state_manager
            "duration_minutes": output.duration_minutes,
        }

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
        """Return the most recent parsed BioGears output (for debugging / analytics)."""
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
        the perturbation parameters alone. This keeps the simulation
        running in degraded mode rather than crashing.
        """
        nausea   = float(perturbation.get("nausea_severity", 0.3))
        baseline = float(perturbation.get("baseline_hr", 75.0))
        fatigue  = float(perturbation.get("fatigue_level", 0.0))
        fat_norm = np.clip(fatigue / 10.0, 0.0, 1.0)

        delta_hr  = nausea * 25.0 * (1.0 + 0.3 * fat_norm)
        severity  = float(np.clip(nausea * (1.0 + 0.4 * fat_norm), 0.0, 1.0))

        logger.warning(f"Using fallback response: delta_hr={delta_hr:.1f}, severity={severity:.3f}")

        return {
            "hr":               baseline + delta_hr,
            "peak_hr":          baseline + delta_hr * 1.3,
            "map":              93.0 + nausea * 12.0,
            "spo2":             98.0 - nausea * 1.5,
            "min_spo2":         97.0 - nausea * 2.0,
            "respiration_rate": 15.0 + nausea * 4.0,
            "core_temp":        37.0 + nausea * 0.2,
            "delta_hr":         delta_hr,
            "delta_map":        nausea * 12.0,
            "delta_spo2":       -nausea * 1.5,
            "stress":           float(np.clip(delta_hr / 40.0, 0.0, 1.0)),
            "severity":         severity,
            "hypotension_risk": 0.0,
            "duration_minutes": float(perturbation.get("duration_minutes", 10.0)),
        }