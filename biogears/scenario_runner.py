"""
biogears/scenario_runner.py

Responsible for:
  - Building BioGears XML scenario files from stressor parameters
  - Executing bg-cli via subprocess
  - Managing temp scenario files and output paths
  - Returning raw CSV path for output_parser to consume

Guidelines reference:
  "Even if you just call: bg-cli scenario.xml — wrap it inside Python using subprocess."
"""

import subprocess
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STRESSOR DEFINITION
# ─────────────────────────────────────────────

@dataclass
class BioGearsStressor:
    """
    Represents a physiological stressor to inject into a BioGears scenario.
    Built by the adapter from digital twin event data.
    """
    stressor_type: str              # "motion_sickness" | "sleep_deprivation" | "stress"
    duration_minutes: float = 10.0  # How long to simulate
    nausea_severity: float = 0.3    # [0-1] maps to BioGears nausea action
    exercise_intensity: float = 0.0 # [0-1] optional physical stressor
    patient_file: str = "StandardMale.xml"
    output_frequency_seconds: float = 1.0  # How often BioGears samples output

    # Data requests (which physiology signals to capture)
    data_requests: list = field(default_factory=lambda: [
        ("HeartRate",             "1/min"),
        ("MeanArterialPressure",  "mmHg"),
        ("SystolicArterialPressure", "mmHg"),
        ("DiastolicArterialPressure","mmHg"),
        ("OxygenSaturation",     ""),
        ("RespirationRate",       "1/min"),
        ("TidalVolume",           "mL"),
        ("CoreTemperature",       "degC"),
    ])


# ─────────────────────────────────────────────
# SCENARIO RUNNER
# ─────────────────────────────────────────────

class BioGearsScenarioRunner:
    """
    Wraps the bg-cli executable.
    Builds scenario XML → calls bg-cli → returns output CSV path.

    Usage:
        runner = BioGearsScenarioRunner(bg_cli_path=r"C:\\BioGears\\bin\\bg-cli.exe")
        csv_path = runner.run(stressor)
    """

    def __init__(
        self,
        bg_cli_path: str = r"C:\Program Files\BioGears\bin\bg-cli.exe",
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120,
    ):
        self.bg_cli_path   = Path(bg_cli_path)
        self.working_dir = Path(working_dir) if working_dir else Path(r"C:\Program Files\BioGears\bin")
        self.timeout       = timeout_seconds
        self.working_dir.mkdir(parents=True, exist_ok=True)

        if not self.bg_cli_path.exists():
            logger.warning(
                f"bg-cli not found at {self.bg_cli_path}. "
                "ScenarioRunner will operate in MOCK mode."
            )
            self._mock_mode = True
        else:
            self._mock_mode = False
            logger.info(f"BioGearsScenarioRunner ready. CLI: {self.bg_cli_path}")

    # ── PUBLIC ──────────────────────────────

    def run(self, stressor: BioGearsStressor) -> str:
        """
        Build scenario XML, call bg-cli, return path to results CSV.

        Args:
            stressor: BioGearsStressor with all scenario parameters

        Returns:
            Absolute path to the output CSV file

        Raises:
            RuntimeError: If bg-cli exits with non-zero code
            TimeoutError: If simulation exceeds timeout
        """
        if self._mock_mode:
            logger.warning("MOCK MODE: returning synthetic CSV path")
            return self._mock_run(stressor)

        # Write scenario XML
        xml_path = self._write_scenario_xml(stressor)
        csv_path = str(xml_path).replace(".xml", "Results.csv")

        logger.info(f"Running BioGears scenario: {xml_path.name}")

        try:
            result = subprocess.run(
                [str(self.bg_cli_path), "Scenario", str(xml_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.working_dir),
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"BioGears scenario timed out after {self.timeout}s. "
                "Try reducing duration_minutes or increasing timeout."
            )

        if result.returncode != 0:
            logger.error(f"bg-cli stderr:\n{result.stderr}")
            raise RuntimeError(
                f"bg-cli failed (exit {result.returncode}):\n{result.stderr[:500]}"
            )

        # BioGears writes CSV next to scenario XML
        if not os.path.exists(csv_path):
            # Some versions write to working_dir
            alt = self.working_dir / (xml_path.stem + "Results.csv")
            if alt.exists():
                csv_path = str(alt)
            else:
                raise FileNotFoundError(
                    f"BioGears ran successfully but output CSV not found at:\n"
                    f"  {csv_path}\n  {alt}"
                )

        logger.info(f"BioGears output: {csv_path}")
        return csv_path

    def get_version(self) -> str:
        """Query bg-cli version string."""
        if self._mock_mode:
            return "MOCK-8.0.0"
        try:
            r = subprocess.run(
                [str(self.bg_cli_path), "--version"],
                capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip() or r.stderr.strip() or "unknown"
        except Exception as e:
            return f"error: {e}"

    # ── XML BUILDER ─────────────────────────

    def _write_scenario_xml(self, s: BioGearsStressor) -> Path:
        """
        Build a BioGears Scenario XML file from the stressor definition.
        Supports: nausea action, exercise action, advance-time.
        """
        xml_path = self.working_dir / f"scenario_{s.stressor_type}_{id(s)}.xml"

        # Build action blocks
        actions = self._build_actions(s)

        # Build data request blocks
        data_reqs = "\n    ".join([
            self._data_request_xml(name, unit)
            for name, unit in s.data_requests
        ])

        xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Scenario xmlns="uri:biogears.biogears.scenario"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Name>AstronautDigitalTwin_{s.stressor_type}</Name>
  <Description>
    Stressor: {s.stressor_type} | Duration: {s.duration_minutes}min |
    Nausea: {s.nausea_severity:.2f} | Exercise: {s.exercise_intensity:.2f}
  </Description>

  <InitialParameters>
    <PatientFile>{s.patient_file}</PatientFile>
  </InitialParameters>

  <DataRequestManager>
    <SamplesPerSecond>{1.0 / max(s.output_frequency_seconds, 0.1):.2f}</SamplesPerSecond>
    <DataRequests>
    {data_reqs}
    </DataRequests>
  </DataRequestManager>

  <Actions>
    {actions}
  </Actions>

</Scenario>
"""
        xml_path.write_text(xml, encoding="utf-8")
        logger.debug(f"Scenario XML written: {xml_path}")
        return xml_path

    def _build_actions(self, s: BioGearsStressor) -> str:
        """Generate the <Actions> block based on stressor type."""
        actions = []

        # Stabilisation period — let BioGears reach steady state
        actions.append(self._advance_time_xml(minutes=1.0))

        if s.stressor_type == "motion_sickness" and s.nausea_severity > 0:
            # Nausea onset
            severity = max(0.0, min(1.0, s.nausea_severity))
            actions.append(f"""
    <Action xsi:type="NauseaData">
      <Severity>
        <Scalar0To1>
          <Value>{severity:.3f}</Value>
        </Scalar0To1>
      </Severity>
    </Action>""")
            # Run stressor duration
            actions.append(self._advance_time_xml(minutes=s.duration_minutes))
            # Clear nausea
            actions.append("""
    <Action xsi:type="NauseaData">
      <Severity><Scalar0To1><Value>0.0</Value></Scalar0To1></Severity>
    </Action>""")
            # Recovery window
            actions.append(self._advance_time_xml(minutes=2.0))

        elif s.stressor_type == "stress" and s.exercise_intensity > 0:
            intensity = max(0.0, min(1.0, s.exercise_intensity))
            actions.append(f"""
    <Action xsi:type="ExerciseData">
      <Intensity>
        <Scalar0To1>
          <Value>{intensity:.3f}</Value>
        </Scalar0To1>
      </Intensity>
    </Action>""")
            actions.append(self._advance_time_xml(minutes=s.duration_minutes))
            # Stop exercise
            actions.append("""
    <Action xsi:type="ExerciseData">
      <Intensity><Scalar0To1><Value>0.0</Value></Scalar0To1></Intensity>
    </Action>""")
            actions.append(self._advance_time_xml(minutes=5.0))

        else:
            # Baseline advance — just simulate physiology at rest
            actions.append(self._advance_time_xml(minutes=s.duration_minutes))

        return "\n".join(actions)

    @staticmethod
    def _advance_time_xml(minutes: float) -> str:
        return f"""
    <Action xsi:type="AdvanceTimeData">
      <Time value="{minutes:.2f}" unit="min"/>
    </Action>"""

    @staticmethod
    def _data_request_xml(name: str, unit: str) -> str:
        unit_attr = f' Unit="{unit}"' if unit else ""
        return (
            f'<DataRequest xsi:type="PhysiologyDataRequestData"'
            f' Name="{name}"{unit_attr}/>'
        )

    # ── MOCK MODE ───────────────────────────

    def _mock_run(self, stressor: BioGearsStressor) -> str:
        """
        Generate a synthetic CSV when bg-cli is unavailable.
        Allows full pipeline testing without BioGears installed.
        """
        import numpy as np
        import csv

        n = int(stressor.duration_minutes * 60 / stressor.output_frequency_seconds)
        t = np.linspace(0, stressor.duration_minutes * 60, n)

        # Simulate nausea effect on HR: peak then recover
        nausea_profile = stressor.nausea_severity * np.exp(-((t - t[n//3]) ** 2) / (2 * (t[-1] * 0.2) ** 2))
        hr   = 75 + 25 * nausea_profile + np.random.normal(0, 1.5, n)
        map_ = 93 + 15 * nausea_profile + np.random.normal(0, 2.0, n)
        spo2 = np.clip(0.98 - 0.02 * nausea_profile + np.random.normal(0, 0.001, n), 0.90, 1.0)
        rr   = 15 + 5  * nausea_profile + np.random.normal(0, 0.5, n)

        csv_path = self.working_dir / f"mock_{stressor.stressor_type}_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time(s)",
                "HeartRate(1/min)",
                "MeanArterialPressure(mmHg)",
                "SystolicArterialPressure(mmHg)",
                "DiastolicArterialPressure(mmHg)",
                "OxygenSaturation",
                "RespirationRate(1/min)",
                "TidalVolume(mL)",
                "CoreTemperature(degC)",
            ])
            for i in range(n):
                writer.writerow([
                    round(t[i], 2),
                    round(hr[i], 2),
                    round(map_[i], 2),
                    round(map_[i] + 20 + np.random.normal(0, 1), 2),
                    round(map_[i] - 20 + np.random.normal(0, 1), 2),
                    round(float(spo2[i]), 4),
                    round(rr[i], 2),
                    round(500 + 50 * nausea_profile[i], 2),
                    round(37.0 + 0.3 * nausea_profile[i], 3),
                ])

        logger.info(f"Mock CSV generated: {csv_path} ({n} rows)")
        return str(csv_path)