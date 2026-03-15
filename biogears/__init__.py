"""
biogears/ — BioGears Integration Layer

3-file bridge between the Digital Twin and the bg-cli executable.

  scenario_runner.py  — builds XML, calls bg-cli subprocess, returns CSV path
  output_parser.py    — parses CSV into numpy arrays, aligns to sim time axis
  biogears_adapter.py — high-level async bridge used by simulation.py
"""

from .biogears_adapter import BioGearsAdapter
from .scenario_runner  import BioGearsScenarioRunner, BioGearsStressor
from .output_parser    import BioGearsOutputParser, BioGearsOutput

__all__ = [
    "BioGearsAdapter",
    "BioGearsScenarioRunner",
    "BioGearsStressor",
    "BioGearsOutputParser",
    "BioGearsOutput",
]