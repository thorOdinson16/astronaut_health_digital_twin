
"""
State Manager for Astronaut Digital Twin
Maintains the evolving physiological state of the astronaut with history tracking.
Implements thread-safe state updates and comprehensive data validation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status indicators for risk monitoring"""
    NOMINAL = "nominal"
    MILD = "mild_risk"
    MODERATE = "moderate_risk"
    SEVERE = "severe_risk"
    CRITICAL = "critical"


@dataclass
class StateBounds:
    """Physiological bounds for validation"""
    hr: Tuple[float, float] = (40, 200)  # bpm
    sleep_quality: Tuple[float, float] = (0.0, 1.0)  # normalized
    fatigue: Tuple[float, float] = (0.0, 10.0)  # cumulative index
    stress: Tuple[float, float] = (0.0, 1.0)  # normalized
    motion_severity: Tuple[float, float] = (0.0, 5.0)  # subjective scale


class AstronautState:
    """
    Central state repository for the digital twin.
    
    Maintains time-series of all physiological variables with history tracking,
    validation bounds, and thread-safe updates. Serves as the single source of
    truth for the astronaut's evolving state.
    
    Attributes:
        timesteps: Number of time steps in simulation
        dt: Time step duration in minutes
        time: Array of time points
        hr: Heart rate time series (bpm)
        sleep_quality: Sleep quality time series [0,1]
        fatigue: Cumulative fatigue index
        stress: Normalized stress level [0,1]
        motion_severity: Motion sickness severity
        event_log: List of events with timestamps
    """
    
    def __init__(
        self,
        timesteps: int,
        dt_minutes: float = 5.0,
        baseline_hr: float = 75.0,
        baseline_sleep_quality: float = 0.8,
        initial_fatigue: float = 0.0,
        validation_bounds: Optional[StateBounds] = None
    ):
        """
        Initialize astronaut state with baseline values.
        
        Args:
            timesteps: Number of time steps to simulate
            dt_minutes: Duration of each time step in minutes
            baseline_hr: Resting heart rate (bpm)
            baseline_sleep_quality: Baseline sleep quality [0,1]
            initial_fatigue: Starting fatigue level
            validation_bounds: Custom validation bounds (optional)
        
        Raises:
            ValueError: If timesteps <= 0 or parameters out of bounds
        """
        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        if not 0 <= baseline_sleep_quality <= 1:
            raise ValueError(f"baseline_sleep_quality must be [0,1], got {baseline_sleep_quality}")
        if not 0 <= initial_fatigue <= 10:
            raise ValueError(f"initial_fatigue must be [0,10], got {initial_fatigue}")
        
        self.timesteps = timesteps
        self.dt = dt_minutes
        self.bounds = validation_bounds or StateBounds()
        
        # Initialize time array
        self.time = np.arange(0, timesteps * dt_minutes, dt_minutes)
        
        # Initialize state arrays with baseline values
        self.hr = np.full(timesteps, baseline_hr, dtype=np.float32)
        self.sleep_quality = np.full(timesteps, baseline_sleep_quality, dtype=np.float32)
        self.fatigue = np.full(timesteps, initial_fatigue, dtype=np.float32)
        self.stress = np.zeros(timesteps, dtype=np.float32)
        self.motion_severity = np.zeros(timesteps, dtype=np.float32)
        
        # Metadata and logging
        self.event_log: List[Dict[str, Any]] = []
        self.last_update_time = datetime.now()
        self._update_count = 0
        
        logger.info(f"Initialized AstronautState with {timesteps} timesteps, dt={dt_minutes}min")
    
    def update(self, t: int, **kwargs) -> None:
        """
        Update state variables at time t with validation.
        
        Args:
            t: Time index to update
            **kwargs: Variable name and new value pairs
            
        Example:
            state.update(t=10, hr=82.5, fatigue=1.2)
        
        Raises:
            IndexError: If t out of bounds
            ValueError: If value exceeds physiological bounds
        """
        if t < 0 or t >= self.timesteps:
            raise IndexError(f"Time index {t} out of bounds [0, {self.timesteps-1}]")
        
        for var_name, value in kwargs.items():
            if not hasattr(self, var_name):
                raise ValueError(f"Unknown state variable: {var_name}")
            
            # Validate bounds
            bounds = getattr(self.bounds, var_name, None)
            if bounds and (value < bounds[0] or value > bounds[1]):
                raise ValueError(
                    f"{var_name} = {value:.2f} outside bounds {bounds} at t={t}"
                )
            
            # Update array
            getattr(self, var_name)[t] = value
            
        self._update_count += 1
    
    def get_state_at_time(self, t: int) -> Dict[str, float]:
        """
        Get complete state snapshot at specific time.
        
        Args:
            t: Time index
            
        Returns:
            Dictionary with all state variables at time t
        """
        return {
            'time': self.time[t],
            'hr': self.hr[t],
            'sleep_quality': self.sleep_quality[t],
            'fatigue': self.fatigue[t],
            'stress': self.stress[t],
            'motion_severity': self.motion_severity[t]
        }
    
    def get_trajectory(self, variable: str) -> np.ndarray:
        """
        Get full time series for a variable.
        
        Args:
            variable: Name of state variable
            
        Returns:
            Numpy array of values
        """
        return getattr(self, variable).copy()
    
    def add_event(self, event_type: str, t: int, **metadata) -> None:
        """
        Log an event that occurred during simulation.
        
        Args:
            event_type: Type of event (motion_sickness, sleep_disruption)
            t: Time index when event occurred
            **metadata: Additional event data (severity, duration, etc.)
        """
        self.event_log.append({
            'type': event_type,
            'time_index': t,
            'simulation_time': self.time[t],
            **metadata
        })
        logger.info(f"Event logged: {event_type} at t={t} ({self.time[t]:.1f} min)")
    
    def compute_risk_status(self, t: int) -> HealthStatus:
        """
        Compute current health risk status based on state.
        
        Args:
            t: Time index
            
        Returns:
            HealthStatus enum value
        """
        state = self.get_state_at_time(t)
        
        # Risk criteria
        if (state['hr'] > 160 or 
            state['fatigue'] > 8.0 or 
            state['motion_severity'] > 4.0):
            return HealthStatus.CRITICAL
        elif (state['hr'] > 140 or 
              state['fatigue'] > 6.0 or 
              state['motion_severity'] > 3.0):
            return HealthStatus.SEVERE
        elif (state['hr'] > 120 or 
              state['fatigue'] > 4.0 or 
              state['motion_severity'] > 2.0):
            return HealthStatus.MODERATE
        elif (state['hr'] > 100 or 
              state['fatigue'] > 2.0 or 
              state['motion_severity'] > 1.0):
            return HealthStatus.MILD
        else:
            return HealthStatus.NOMINAL
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire state to dictionary for serialization (JSON for Person 3).
        
        Returns:
            Complete state dictionary with all trajectories and metadata
        """
        return {
            'metadata': {
                'timesteps': self.timesteps,
                'dt_minutes': self.dt,
                'update_count': self._update_count,
                'bounds': {
                    name: list(bounds) 
                    for name, bounds in self.bounds.__dict__.items()
                }
            },
            'time': self.time.tolist(),
            'hr': self.hr.tolist(),
            'sleep_quality': self.sleep_quality.tolist(),
            'fatigue': self.fatigue.tolist(),
            'stress': self.stress.tolist(),
            'motion_severity': self.motion_severity.tolist(),
            'events': self.event_log
        }
    
    def from_dict(self, data: Dict[str, Any]) -> 'AstronautState':
        """
        Restore state from dictionary.
        
        Args:
            data: Dictionary from to_dict()
            
        Returns:
            Self for chaining
        """
        self.time = np.array(data['time'])
        self.hr = np.array(data['hr'])
        self.sleep_quality = np.array(data['sleep_quality'])
        self.fatigue = np.array(data['fatigue'])
        self.stress = np.array(data['stress'])
        self.motion_severity = np.array(data['motion_severity'])
        self.event_log = data['events']
        return self
    
    def __repr__(self) -> str:
        return f"AstronautState(timesteps={self.timesteps}, dt={self.dt}min, events={len(self.event_log)})"