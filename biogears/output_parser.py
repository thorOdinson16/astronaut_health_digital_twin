"""
biogears/output_parser.py

Responsible for:
  - Reading BioGears results CSV
  - Mapping BioGears column names → digital twin variable names
  - Interpolating BioGears time axis → simulation time axis
  - Returning clean numpy arrays ready for state_manager injection

Guidelines reference:
  "Parse: Heart rate, Blood pressure, Mean arterial pressure.
   Load CSV → convert to numpy arrays."
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PARSED OUTPUT CONTAINER
# ─────────────────────────────────────────────

@dataclass
class BioGearsOutput:
    """
    Clean, typed container for BioGears physiology output.
    All arrays are numpy, already aligned to the requested time axis.
    """
    time_seconds:       np.ndarray   # Original BioGears time axis
    heart_rate:         np.ndarray   # bpm
    map_mmhg:           np.ndarray   # Mean arterial pressure, mmHg
    systolic_bp:        np.ndarray   # mmHg
    diastolic_bp:       np.ndarray   # mmHg
    spo2:               np.ndarray   # fraction [0-1]
    respiration_rate:   np.ndarray   # breaths/min
    tidal_volume_ml:    np.ndarray   # mL
    core_temp_celsius:  np.ndarray   # °C

    # Summary scalars (computed on parse)
    peak_hr:            float = 0.0
    mean_hr:            float = 0.0
    min_spo2:           float = 1.0
    peak_map:           float = 0.0
    mean_map:           float = 0.0
    duration_minutes:   float = 0.0

    def __post_init__(self):
        if len(self.heart_rate):
            self.peak_hr          = float(np.max(self.heart_rate))
            self.mean_hr          = float(np.mean(self.heart_rate))
            self.min_spo2         = float(np.min(self.spo2))
            self.peak_map         = float(np.max(self.map_mmhg))
            self.mean_map         = float(np.mean(self.map_mmhg))
            self.duration_minutes = float(self.time_seconds[-1] / 60.0)

    def to_dict(self) -> dict:
        """Serialise summary scalars for API / state_manager consumption."""
        return {
            "hr":           self.mean_hr,
            "peak_hr":      self.peak_hr,
            "map":          self.mean_map,
            "peak_map":     self.peak_map,
            "spo2":         float(np.mean(self.spo2)) * 100,   # → percentage
            "min_spo2":     self.min_spo2 * 100,
            "rr":           float(np.mean(self.respiration_rate)),
            "core_temp":    float(np.mean(self.core_temp_celsius)),
            "duration_min": self.duration_minutes,
        }


# ─────────────────────────────────────────────
# COLUMN NAME MAP
# BioGears uses verbose column names with units.
# We map them to clean internal names.
# ─────────────────────────────────────────────

COLUMN_MAP = {
    # Heart rate
    "HeartRate(1/min)":                  "heart_rate",
    "HeartRate(bpm)":                    "heart_rate",
    "CardiovascularSystem-HeartRate(1/min)": "heart_rate",

    # Mean arterial pressure
    "MeanArterialPressure(mmHg)":        "map_mmhg",
    "CardiovascularSystem-MeanArterialPressure(mmHg)": "map_mmhg",

    # Systolic BP
    "SystolicArterialPressure(mmHg)":    "systolic_bp",
    "CardiovascularSystem-SystolicArterialPressure(mmHg)": "systolic_bp",

    # Diastolic BP
    "DiastolicArterialPressure(mmHg)":   "diastolic_bp",
    "CardiovascularSystem-DiastolicArterialPressure(mmHg)": "diastolic_bp",

    # SpO2
    "OxygenSaturation":                  "spo2",
    "BloodChemistry-OxygenSaturation":   "spo2",

    # Respiration
    "RespirationRate(1/min)":            "respiration_rate",
    "RespiratorySystem-RespirationRate(1/min)": "respiration_rate",

    # Tidal volume
    "TidalVolume(mL)":                   "tidal_volume_ml",
    "RespiratorySystem-TidalVolume(mL)": "tidal_volume_ml",

    # Core temp
    "CoreTemperature(degC)":             "core_temp_celsius",
    "EnergyMetabolism-CoreTemperature(degC)": "core_temp_celsius",

    # Time
    "Time(s)":  "time_seconds",
    "Time(min)": "time_minutes",
}

# Fallback defaults if a column is missing in CSV
DEFAULTS = {
    "heart_rate":       75.0,
    "map_mmhg":         93.0,
    "systolic_bp":      120.0,
    "diastolic_bp":     80.0,
    "spo2":             0.98,
    "respiration_rate": 15.0,
    "tidal_volume_ml":  500.0,
    "core_temp_celsius":37.0,
}


# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────

class BioGearsOutputParser:
    """
    Reads a BioGears results CSV and returns a clean BioGearsOutput object.

    Usage:
        parser = BioGearsOutputParser()
        output = parser.parse("/tmp/biogears_runs/scenarioResults.csv")
        print(output.mean_hr)
        aligned = parser.align_to_simulation_time(output, sim_time_hours)
    """

    def __init__(self, skip_initial_seconds: float = 60.0):
        """
        Args:
            skip_initial_seconds: Drop the first N seconds of BioGears output.
                BioGears stabilises physiology in the first ~60s; these are
                not meaningful perturbation responses.
        """
        self.skip_initial = skip_initial_seconds

    # ── PUBLIC ──────────────────────────────

    def parse(self, csv_path: str) -> BioGearsOutput:
        """
        Parse a BioGears results CSV into a BioGearsOutput.

        Args:
            csv_path: Absolute path to the BioGears output CSV

        Returns:
            BioGearsOutput with all physiology arrays populated

        Raises:
            FileNotFoundError: If CSV does not exist
            ValueError: If CSV is empty or missing critical columns
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"BioGears output CSV not found: {csv_path}")

        logger.info(f"Parsing BioGears output: {path.name}")

        # Read CSV — BioGears sometimes adds a header comment row starting with #
        df = self._read_csv_safely(path)

        if df.empty:
            raise ValueError(f"BioGears CSV is empty: {csv_path}")

        # Rename columns to internal names
        df = self._normalise_columns(df)

        # Resolve time axis → always seconds
        time_s = self._resolve_time_seconds(df)

        # Skip stabilisation period
        mask = time_s >= self.skip_initial
        if mask.sum() < 2:
            logger.warning("Skipping stabilisation filter — not enough rows remain")
            mask = np.ones(len(time_s), dtype=bool)

        time_s = time_s[mask]
        df     = df[mask].reset_index(drop=True)

        # Extract each signal (with fallback to default if missing)
        def extract(col: str) -> np.ndarray:
            if col in df.columns:
                arr = pd.to_numeric(df[col], errors='coerce').fillna(DEFAULTS[col]).values
                return arr.astype(np.float64)
            else:
                logger.warning(f"Column '{col}' missing from BioGears CSV — using default {DEFAULTS[col]}")
                return np.full(len(time_s), DEFAULTS[col], dtype=np.float64)

        output = BioGearsOutput(
            time_seconds      = time_s,
            heart_rate        = extract("heart_rate"),
            map_mmhg          = extract("map_mmhg"),
            systolic_bp       = extract("systolic_bp"),
            diastolic_bp      = extract("diastolic_bp"),
            spo2              = extract("spo2"),
            respiration_rate  = extract("respiration_rate"),
            tidal_volume_ml   = extract("tidal_volume_ml"),
            core_temp_celsius = extract("core_temp_celsius"),
        )

        if output.spo2.max() > 1.5:
            logger.warning(f"SpO2 appears to be percentage (max={output.spo2.max():.1f}) — normalising to fraction")
            output.spo2     = output.spo2 / 100.0
            output.min_spo2 = float(np.min(output.spo2))

        logger.info(
            f"Parsed {len(time_s)} rows | "
            f"HR: {output.mean_hr:.1f} bpm (peak {output.peak_hr:.1f}) | "
            f"MAP: {output.mean_map:.1f} mmHg | "
            f"SpO2: {output.min_spo2*100:.1f}% min"
        )
        return output

    def align_to_simulation_time(
        self,
        output: BioGearsOutput,
        sim_time_hours: np.ndarray,
        event_start_hour: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate BioGears output onto the simulation time axis.

        BioGears runs its own fine-grained time axis (seconds).
        The digital twin runs on a coarser axis (hours).
        This function resamples BioGears → twin time.

        Args:
            output:             Parsed BioGears output
            sim_time_hours:     Digital twin time axis in hours
            event_start_hour:   When the event starts in the twin timeline

        Returns:
            Dict of signal_name → array aligned to sim_time_hours
            Only covers the event window; values outside = NaN
        """
        # BioGears time relative to event start, converted to hours
        bg_time_hours = (output.time_seconds / 3600.0) + event_start_hour
        event_end_hour = bg_time_hours[-1]

        # Mask: which sim timesteps fall inside the BioGears event window
        in_window = (sim_time_hours >= event_start_hour) & (sim_time_hours <= event_end_hour)

        signals = {
            "hr":             output.heart_rate,
            "map":            output.map_mmhg,
            "systolic_bp":    output.systolic_bp,
            "diastolic_bp":   output.diastolic_bp,
            "spo2":           output.spo2 * 100,   # → percentage
            "respiration_rate": output.respiration_rate,
            "core_temp":      output.core_temp_celsius,
        }

        aligned = {}
        for name, arr in signals.items():
            out = np.full(len(sim_time_hours), np.nan)
            if in_window.any():
                out[in_window] = np.interp(
                    sim_time_hours[in_window],
                    bg_time_hours,
                    arr
                )
            aligned[name] = out

        return aligned

    def summarise_perturbation_response(
        self,
        output: BioGearsOutput,
        baseline_hr: float = 75.0,
        baseline_map: float = 93.0,
    ) -> Dict[str, float]:
        """
        Compute delta values vs. baseline — what the stressor actually changed.
        This is what the coupling engine uses to modify digital twin state.

        Returns:
            delta_hr:   HR change above baseline (bpm)
            delta_map:  MAP change above baseline (mmHg)
            delta_spo2: SpO2 drop (negative = desaturation)
            severity:   Composite severity score [0-1]
        """
        delta_hr   = max(0.0, output.peak_hr  - baseline_hr)
        delta_map  = output.peak_map - baseline_map
        delta_spo2 = (output.min_spo2 * 100) - 98.0  # negative if desaturating

        # Composite severity: normalised weighted sum
        severity = float(np.clip(
            (delta_hr / 40.0) * 0.5 +
            (abs(delta_map) / 30.0) * 0.3 +
            (abs(min(0, delta_spo2)) / 5.0) * 0.2,
            0.0, 1.0
        ))

        return {
            "delta_hr":   delta_hr,
            "delta_map":  delta_map,
            "delta_spo2": delta_spo2,
            "severity":   severity,
        }

    # ── PRIVATE ─────────────────────────────

    def _read_csv_safely(self, path: Path) -> pd.DataFrame:
        """Handle BioGears CSV quirks: comment lines, mixed headers."""
        try:
            # Try standard read first
            df = pd.read_csv(path, comment='#', skipinitialspace=True)
            if df.empty or len(df.columns) < 2:
                raise ValueError("Too few columns")
            return df
        except Exception:
            pass

        # Fallback: skip lines starting with # manually
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        data_lines = [l for l in lines if not l.strip().startswith('#') and l.strip()]
        if not data_lines:
            return pd.DataFrame()

        from io import StringIO
        return pd.read_csv(StringIO("\n".join(data_lines)), skipinitialspace=True)

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename BioGears verbose column names to internal short names."""
        rename = {}
        for col in df.columns:
            col_stripped = col.strip()
            if col_stripped in COLUMN_MAP:
                rename[col] = COLUMN_MAP[col_stripped]
            else:
                # Fuzzy match: check if any key is a substring of the column
                for bg_name, internal_name in COLUMN_MAP.items():
                    if bg_name.lower() in col_stripped.lower():
                        rename[col] = internal_name
                        break
        return df.rename(columns=rename)

    def _resolve_time_seconds(self, df: pd.DataFrame) -> np.ndarray:
        """Extract time in seconds regardless of whether CSV uses s or min."""
        if "time_seconds" in df.columns:
            return pd.to_numeric(df["time_seconds"], errors='coerce').fillna(0).values.astype(np.float64)
        elif "time_minutes" in df.columns:
            return (pd.to_numeric(df["time_minutes"], errors='coerce').fillna(0).values * 60.0).astype(np.float64)
        else:
            # No time column — manufacture one assuming 1s sample rate
            logger.warning("No time column found in BioGears CSV — assuming 1s sample rate")
            return np.arange(len(df), dtype=np.float64)