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
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
load_dotenv()

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
        bg_cli_path: str = os.environ.get("BIOGEARS_PATH"),
        working_dir: Optional[str] = None,
        timeout_seconds: int = 300,
    ):
        self.bg_cli_path = Path(bg_cli_path)
        self.bg_cli_exe  = self.bg_cli_path / "bg-cli.exe"

        # FIX: working_dir must be bg_cli_path (the bin/ folder itself).
        # The BioGears install puts everything — states/, xsd/, Scenarios/ —
        # INSIDE bin/, not in the parent. bg-cli resolves all relative paths
        # (states/StandardMale@0s.xml, xsd/BioGearsDataModel.xsd) from its cwd,
        # which must be bin/.
        self.working_dir = Path(working_dir) if working_dir else self.bg_cli_path
        self.timeout = timeout_seconds
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

        # Check if the XSD is present — it controls whether we need a namespace
        self._xsd_path = self.working_dir / "xsd" / "BioGearsDataModel.xsd"
        if self._xsd_path.exists():
            logger.warning(
                f"BioGears XSD found at {self._xsd_path}. "
                "XML schema validation is ACTIVE. "
                "Run this once to disable it (avoids namespace issues):\n"
                f'  rename "{self._xsd_path}" BioGearsDataModel.xsd.bak'
            )
        else:
            logger.info("BioGears XSD not found — schema validation is disabled (good).")

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

        launch_time = time.time()

        try:
            result = subprocess.run(
                [str(self.bg_cli_exe), "Scenario", xml_path.name],  # ← relative name only
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

        combined_output = result.stdout + result.stderr
        logger.info(f"bg-cli full output:\n{combined_output}")

        # Detect XSD validation failure specifically — actionable error message
        if "no declaration found for element" in combined_output:
            raise RuntimeError(
                f"BioGears XSD validation failed — the schema is rejecting our XML.\n"
                f"Fix: rename the XSD to disable validation (run once in cmd.exe):\n"
                f'  rename "{self._xsd_path}" BioGearsDataModel.xsd.bak\n'
                f"Then restart uvicorn."
            )

        has_hard_error = (
            "error" in combined_output.lower()
            and "completed" not in combined_output.lower()
        )
        if has_hard_error:
            logger.error(f"bg-cli reported a hard error:\n{combined_output}")
            raise RuntimeError(
                f"bg-cli reported errors:\n{combined_output[:500]}"
            )

        csv_path = self._find_output_csv(scenario_name, launch_time, xml_path)
        if csv_path is None:
            raise FileNotFoundError(
                f"BioGears ran successfully but output CSV not found.\n"
                f"Scenario <n> tag: {scenario_name}\n"
                f"Searched under:   {self.working_dir}\n"
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

    def _find_output_csv(self, scenario_name: str, launch_time: float, xml_path: Path = None) -> Optional[Path]:
        """
        Locate the Results CSV that bg-cli wrote.

        BioGears writes to: <working_dir>/Scenarios/<ScenarioName>Results.csv
        where <ScenarioName> is the value of the <n> element in the XML.
        """
        # BioGears names the CSV after the <n> tag in the XML, NOT the xml filename.
        # e.g. <n>AstronautTwin_motion_sickness_abc123</n> → Scenarios/AstronautTwin_motion_sickness_abc123Results.csv
        # Also check xml_stem as a fallback (some BioGears versions use the filename).
        stems_to_check = [scenario_name]
        if xml_path is not None:
            stems_to_check.append(xml_path.stem)

        for stem in stems_to_check:
            csv_name = f"{stem}Results.csv"
            candidates = [
                self.working_dir / "Scenarios" / csv_name,       # ← most common (BioGears default)
                self.working_dir / csv_name,
                self.working_dir / "results" / csv_name,
                self.working_dir / "Scenarios" / "Scenarios" / csv_name,
            ]
            for path in candidates:
                if path.exists():
                    logger.debug(f"Found CSV at: {path}")
                    return path

        # Fallback: any *Results.csv written after this run started
        for p in self.working_dir.rglob("*Results.csv"):
            if p.stat().st_mtime >= launch_time:
                logger.warning(f"Found CSV via filesystem scan (unexpected path): {p}")
                return p

        return None

    # ── XML BUILDER ─────────────────────────

    def _write_scenario_xml(self, s: BioGearsStressor) -> tuple:
        """
        Build a BioGears Scenario XML file from the stressor definition.

        Namespace strategy:
          - XSD ABSENT (after rename to .bak): use bare <Scenario> root.
            bg-cli skips validation and runs fine with no namespace.
          - XSD PRESENT: we still use bare <Scenario> root, but the validation
            will fail. The fix is to rename the XSD (see __init__ warning).

        Returns:
            (xml_path, scenario_name)
        """
        unique_id     = uuid.uuid4().hex[:12]
        scenario_name = f"AstronautTwin_{s.stressor_type}_{unique_id}"

        # Write the XML into working_dir (bin/) so bg-cli can find it
        xml_path = self.working_dir / f"scenario_{unique_id}.xml"

        actions   = self._build_actions(s)
        data_reqs = "\n    ".join(
            self._data_request_xml(name, unit, req_type)
            for name, unit, req_type in s.data_requests
        )

        samples_per_sec = 0.05  # 1 sample per 20s

        # Use the full namespace + contentVersion format.
        # This matches the format BioGears expects when XSDs are present.
        # xmlns:xsi is declared once on the root so child elements inherit it.
        # xsi:schemaLocation="" tells the validator not to look for an external schema URL.
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

        actions.append(self._advance_time_xml(seconds=30.0))

        if s.stressor_type in ("motion_sickness", "stress"):
            raw      = s.nausea_severity if s.stressor_type == "motion_sickness" else s.exercise_intensity
            severity = round(max(0.0, min(1.0, raw)), 3)

            if severity > 0:
                actions.append(
                    f'<Action xsi:type="AcuteStressData">\n'
                    f'      <Severity value="{severity}"/>\n'
                    f'    </Action>'
                )
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
        """Generate a synthetic CSV when bg-cli is unavailable."""
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