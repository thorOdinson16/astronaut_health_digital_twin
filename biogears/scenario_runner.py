"""
biogears/scenario_runner.py

Responsible for:
  - Building BioGears XML scenario files from stressor parameters
  - Executing bg-cli via subprocess
  - Managing temp scenario files and output paths
  - Returning raw CSV path for output_parser to consume
"""

import subprocess
import uuid
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
    stressor_type: str               # "motion_sickness" | "sleep_deprivation" | "stress"
    duration_minutes: float = 10.0
    nausea_severity: float  = 0.3    # [0-1] — used for motion_sickness severity
    exercise_intensity: float = 0.0  # [0-1] — used for stress severity
    patient_file: str = "StandardMale.xml"
    output_frequency_seconds: float = 1.0

    # (name, unit, xsi:type) triples — confirmed against shipped scenario files
    data_requests: list = field(default_factory=lambda: [
        ("HeartRate",                "1/min",    "PhysiologyDataRequestData"),
        ("MeanArterialPressure",      "mmHg",     "PhysiologyDataRequestData"),
        ("SystolicArterialPressure",  "mmHg",     "PhysiologyDataRequestData"),
        ("DiastolicArterialPressure", "mmHg",     "PhysiologyDataRequestData"),
        ("OxygenSaturation",          "unitless", "PhysiologyDataRequestData"),
        ("RespirationRate",           "1/min",    "PhysiologyDataRequestData"),
        ("TidalVolume",               "mL",       "PhysiologyDataRequestData"),
        ("CoreTemperature",           "degC",     "PhysiologyDataRequestData"),
    ])


# ─────────────────────────────────────────────
# SCENARIO RUNNER
# ─────────────────────────────────────────────

class BioGearsScenarioRunner:
    """
    Wraps the bg-cli executable.
    Builds scenario XML → calls bg-cli → returns output CSV path.
    """

    def __init__(
        self,
        bg_cli_path: str = r"C:\Users\AbhiDS\biogears\bin",
        working_dir: Optional[str] = None,
        timeout_seconds: int = 300,
    ):
        self.bg_cli_path = Path(bg_cli_path)
        self.bg_cli_exe  = self.bg_cli_path / "bg-cli.exe"
        self.working_dir = Path(working_dir) if working_dir else self.bg_cli_path
        self.timeout     = timeout_seconds
        self.working_dir.mkdir(parents=True, exist_ok=True)

        if not self.bg_cli_exe.exists():
            logger.warning(
                f"bg-cli.exe not found at {self.bg_cli_exe}. "
                "ScenarioRunner will operate in MOCK mode."
            )
            self._mock_mode = True
        else:
            self._mock_mode = False
            logger.info(f"BioGearsScenarioRunner ready. CLI: {self.bg_cli_exe}")

    # ── PUBLIC ──────────────────────────────

    def run(self, stressor: BioGearsStressor) -> str:
        """
        Build scenario XML, call bg-cli, return path to results CSV.

        Raises:
            RuntimeError:      bg-cli reported errors in its output
            FileNotFoundError: bg-cli ran cleanly but CSV is missing
            TimeoutError:      simulation exceeded timeout
        """
        if self._mock_mode:
            logger.warning("MOCK MODE: returning synthetic CSV path")
            return self._mock_run(stressor)

        xml_path, scenario_name = self._write_scenario_xml(stressor)

        logger.info(f"Running BioGears scenario: {xml_path.name}")

        try:
            result = subprocess.run(
                [str(self.bg_cli_exe), "Scenario", xml_path.name],
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

        # FIX: bg-cli always exits with code 0, even on XML parse errors.
        # The only reliable failure signal is the word "error" in its output.
        # Confirmed by running a broken XML and observing:
        #   exit code = 0, stdout contains ":23:58 error: attribute ... not declared"
        combined_output = result.stdout + result.stderr
        if "error" in combined_output.lower() and "completed" not in combined_output.lower():
            logger.error(f"bg-cli output:\n{combined_output}")
            raise RuntimeError(
                f"bg-cli reported errors:\n{combined_output[:500]}"
            )

        # FIX: BioGears 8.x writes output to:
        #   <cwd>/Scenarios/<ScenarioName>Results.csv
        # where <ScenarioName> is the value of the <Name> element in the XML,
        # NOT the XML filename. Confirmed from OverrideTest.xml run which produced
        # cwd/Scenarios/Scenarios/OverrideTestResults.csv when a relative path
        # was passed, and cwd/Scenarios/<Name>Results.csv with an absolute path.
        csv_path = self._find_output_csv(scenario_name)
        if csv_path is None:
            raise FileNotFoundError(
                f"BioGears ran successfully but output CSV not found.\n"
                f"Scenario <Name> tag: {scenario_name}\n"
                f"Searched under:      {self.working_dir}\n"
                f"bg-cli output:\n{combined_output[:300]}"
            )

        logger.info(f"BioGears output: {csv_path}")
        return str(csv_path)

    def get_version(self) -> str:
        """Query bg-cli version string."""
        if self._mock_mode:
            return "MOCK-8.0.0"
        try:
            r = subprocess.run(
                [str(self.bg_cli_exe), "--version"],
                capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip() or r.stderr.strip() or "unknown"
        except Exception as e:
            return f"error: {e}"

    # ── OUTPUT FINDER ────────────────────────

    def _find_output_csv(self, scenario_name: str) -> Optional[Path]:
        """
        Locate the Results CSV that bg-cli wrote.

        BioGears 8.x (confirmed): writes to
            <working_dir>/Scenarios/<ScenarioName>Results.csv
        """
        csv_name = f"{scenario_name}Results.csv"

        candidates = [
            self.working_dir / "Scenarios" / csv_name,   # confirmed primary location
            self.working_dir / csv_name,                  # fallback: root
            self.working_dir / "results" / csv_name,      # fallback: older builds
        ]

        for path in candidates:
            if path.exists():
                logger.debug(f"Found CSV at: {path}")
                return path

        # Last resort: recursive mtime scan
        import time
        now = time.time()
        for p in self.working_dir.rglob(f"*{scenario_name}Results.csv"):
            if now - p.stat().st_mtime < 60:
                logger.warning(f"Found CSV via filesystem scan (unexpected path): {p}")
                return p

        return None

    # ── XML BUILDER ─────────────────────────

    def _write_scenario_xml(self, s: BioGearsStressor) -> tuple:
        """
        Build a BioGears Scenario XML file from the stressor definition.

        Returns:
            (xml_path, scenario_name) — path written and the <Name> used,
            because BioGears names the output CSV after <Name>, not the filename.
        """
        unique_id     = uuid.uuid4().hex[:12]
        scenario_name = f"AstronautTwin_{s.stressor_type}_{unique_id}"
        xml_path      = self.working_dir / f"scenario_{unique_id}.xml"

        actions   = self._build_actions(s)
        data_reqs = "\n    ".join(
            self._data_request_xml(name, unit, req_type)
            for name, unit, req_type in s.data_requests
        )

        # Cap at 1 sample/sec — sufficient for a 30-min timestep twin
        samples_per_sec = 0.05

        xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Scenario xmlns="uri:/mil/tatrc/physiology/datamodel"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          contentVersion="BioGears_6.3.0-beta"
          xsi:schemaLocation="">
  <Name>{scenario_name}</Name>
  <Description>Stressor: {s.stressor_type} | Duration: {s.duration_minutes:.1f}min | Severity: {s.nausea_severity:.2f}</Description>
  <EngineStateFile>states/StandardMale@0s.xml</EngineStateFile>
  <DataRequests SamplesPerSecond="{samples_per_sec:.2f}">
    {data_reqs}
  </DataRequests>
  <Actions>
    {actions}
  </Actions>
</Scenario>
"""
        xml_path.write_text(xml, encoding="utf-8")
        logger.debug(f"Scenario XML written: {xml_path} (Name={scenario_name})")
        return xml_path, scenario_name

    def _build_actions(self, s: BioGearsStressor) -> str:
        actions = []

        # Stabilise after loading state file — without this, AcuteStressData
        # can cause the cardiovascular solver to hang indefinitely.
        actions.append(self._advance_time_xml(seconds=30.0))

        if s.stressor_type in ("motion_sickness", "stress"):
            raw = s.nausea_severity if s.stressor_type == "motion_sickness" else s.exercise_intensity
            severity = round(max(0.0, min(1.0, raw)), 3)

            if severity > 0:
                actions.append(
                    f'<Action xsi:type="AcuteStressData">\n'
                    f'      <Severity value="{severity}"/>\n'
                    f'    </Action>'
                )
                # FIX: cap simulation to 10 minutes max regardless of event duration.
                # The full event duration (up to 200min) causes solver hangs.
                # We only need the physiological response snapshot, not the full timeline.
                sim_minutes = min(s.duration_minutes, 10.0)
                actions.append(self._advance_time_xml(minutes=sim_minutes))
                actions.append(
                    '<Action xsi:type="AcuteStressData">\n'
                    '      <Severity value="0.0"/>\n'
                    '    </Action>'
                )
                actions.append(self._advance_time_xml(seconds=60.0))
            else:
                actions.append(self._advance_time_xml(minutes=min(s.duration_minutes, 10.0)))

        elif s.stressor_type == "sleep_deprivation":
            actions.append(self._advance_time_xml(minutes=min(s.duration_minutes, 10.0)))
        else:
            actions.append(self._advance_time_xml(minutes=min(s.duration_minutes, 10.0)))

        return "\n    ".join(actions)

    @staticmethod
    def _advance_time_xml(minutes: float = 0.0, seconds: float = 0.0) -> str:
        if minutes > 0:
            return (f'<Action xsi:type="AdvanceTimeData">'
                    f'<Time value="{minutes:.2f}" unit="min"/></Action>')
        return (f'<Action xsi:type="AdvanceTimeData">'
                f'<Time value="{seconds:.2f}" unit="s"/></Action>')

    @staticmethod
    def _data_request_xml(name: str, unit: str,
                          req_type: str = "PhysiologyDataRequestData") -> str:
        unit_attr = f' Unit="{unit}"' if unit else ""
        return (f'<DataRequest xsi:type="{req_type}"'
                f' Name="{name}"{unit_attr} Precision="4"/>')

    # ── MOCK MODE ───────────────────────────

    def _mock_run(self, stressor: BioGearsStressor) -> str:
        """
        Generate a synthetic CSV when bg-cli is unavailable.
        Allows full pipeline testing without BioGears installed.
        """
        import numpy as np
        import csv

        n = max(2, int(stressor.duration_minutes * 60 / stressor.output_frequency_seconds))
        t = np.linspace(0, stressor.duration_minutes * 60, n)

        peak_t         = t[n // 3]
        sigma          = t[-1] * 0.2 if t[-1] > 0 else 1.0
        stress_profile = stressor.nausea_severity * np.exp(
            -((t - peak_t) ** 2) / (2 * sigma ** 2)
        )

        hr   = 75  + 25 * stress_profile + np.random.normal(0, 1.5, n)
        map_ = 93  + 15 * stress_profile + np.random.normal(0, 2.0, n)
        spo2 = np.clip(0.98 - 0.02 * stress_profile + np.random.normal(0, 0.001, n), 0.90, 1.0)
        rr   = 15  +  5 * stress_profile + np.random.normal(0, 0.5,  n)

        unique_id = uuid.uuid4().hex[:8]
        csv_path  = self.working_dir / f"mock_{stressor.stressor_type}_{unique_id}_results.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time(s)", "HeartRate(1/min)", "MeanArterialPressure(mmHg)",
                "SystolicArterialPressure(mmHg)", "DiastolicArterialPressure(mmHg)",
                "OxygenSaturation(unitless)", "RespirationRate(1/min)",
                "TidalVolume(mL)", "CoreTemperature(degC)",
            ])
            for i in range(n):
                writer.writerow([
                    round(float(t[i]),    2),
                    round(float(hr[i]),   2),
                    round(float(map_[i]), 2),
                    round(float(map_[i]) + 20 + float(np.random.normal(0, 1)), 2),
                    round(float(map_[i]) - 20 + float(np.random.normal(0, 1)), 2),
                    round(float(spo2[i]), 4),
                    round(float(rr[i]),   2),
                    round(500 + 50 * float(stress_profile[i]), 2),
                    round(37.0 + 0.3 * float(stress_profile[i]), 3),
                ])

        logger.info(f"Mock CSV generated: {csv_path} ({n} rows)")
        return str(csv_path)