
"""
Motion Sickness Event Module
Implements space motion sickness episodes with vestibular mismatch physiology.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from events.base_event import Event, EventEffect, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class MotionSicknessParameters:
    """Parameters for motion sickness event dynamics."""
    
    # Onset parameters
    base_rate: float = 0.03  # events per hour (Poisson λ)
    adaptation_rate: float = 0.1  # per hour (decreasing probability over time)
    minimum_onset_prob: float = 0.01  # minimum probability after adaptation
    
    # Severity parameters
    severity_mean: float = 0.6
    severity_std: float = 0.2
    min_severity: float = 0.2
    max_severity: float = 1.0
    
    # Duration parameters
    min_duration: float = 0.5  # hours
    max_duration: float = 4.0  # hours
    duration_severity_exponent: float = 1.5  # duration ∝ severity^exponent
    
    # Physiological effects
    hr_increase_mean: float = 15.0  # bpm increase per severity unit
    hr_increase_std: float = 5.0
    stress_increase_mean: float = 0.3  # stress [0,1] per severity unit
    stress_increase_std: float = 0.1
    sleep_degradation: float = 0.3  # sleep quality reduction per severity unit
    
    # Vestibular parameters (for BioGears coupling)
    vestibular_stimulus_intensity: float = 0.8  # base stimulus
    otolith_asymmetry: float = 0.3  # otolith organ asymmetry factor
    semicircular_canal_sensitivity: float = 0.6  # canal sensitivity
    
    def validate(self):
        """Validate parameter bounds."""
        assert 0 <= self.base_rate <= 1, "base_rate must be [0,1]"
        assert 0 <= self.min_severity <= self.max_severity <= 1
        assert self.min_duration <= self.max_duration
        assert self.sleep_degradation >= 0


class MotionSicknessEvent(Event):
    """
    Space Motion Sickness Event
    
    Represents acute motion sickness episodes during microgravity adaptation.
    Caused by vestibular mismatch between visual cues and otolith organ signals.
    
    Key characteristics:
    - Stochastic onset (Poisson process)
    - Severity-dependent duration
    - Multiple physiological effects (HR, stress, sleep)
    - Adaptation over mission duration
    - Bidirectional coupling with fatigue
    """
    
    def __init__(
        self,
        params: Optional[MotionSicknessParameters] = None,
        **kwargs
    ):
        """
        Initialize motion sickness event.
        
        Args:
            params: MotionSicknessParameters object (uses defaults if None)
            **kwargs: Additional event parameters
        """
        super().__init__(priority=EventPriority.HIGH, **kwargs)
        self.params = params or MotionSicknessParameters()
        self.params.validate()
        
        # Event-specific state
        self.adaptation_level: float = 0.0  # 0 = no adaptation, 1 = fully adapted
        self.vestibular_conflict: float = 0.0  # Current mismatch level
        self.recovery_progress: float = 0.0
        
        logger.debug(f"Created MotionSicknessEvent with params: {self.params}")
    
    def sample_onset(
        self,
        state: Any,
        t: int,
        fatigue_multiplier: float = 1.0,
        **kwargs
    ) -> Tuple[bool, Optional[float]]:
        """
        Determine if motion sickness event occurs.
        
        Onset probability follows a Poisson process modified by:
        - Adaptation to microgravity (decreases over time)
        - Fatigue level (increases susceptibility)
        - Recent events (refractory period)
        
        Args:
            state: Current astronaut state
            t: Current time index
            fatigue_multiplier: Fatigue effect on probability (from coupling)
            **kwargs: Additional factors
            
        Returns:
            Tuple of (should_occur, severity if occurs)
        """
        # Get current mission time in hours
        mission_time_hours = t * (getattr(state, 'dt', 5.0) / 60.0)
        
        # Update adaptation level (increases with time)
        self.adaptation_level = min(
            1.0,
            mission_time_hours * self.params.adaptation_rate / 24.0  # per day adaptation
        )
        
        # Base probability from Poisson process
        dt_hours = getattr(state, 'dt', 5.0) / 60.0
        base_prob = self.params.base_rate * dt_hours
        
        # Adaptation reduces probability
        adaptation_factor = 1.0 - (self.adaptation_level * 0.7)  # Max 70% reduction
        base_prob *= adaptation_factor
        
        # Fatigue increases probability (from coupling engine)
        fatigue = state.fatigue[t] if t < len(state.fatigue) else 0
        fatigue_factor = 1.0 + (fatigue / 10.0) * 0.5  # Up to 50% increase
        
        # Refractory period (can't have another event too soon)
        refractory_factor = self._check_refractory(state, mission_time_hours)
        
        # Calculate final probability
        onset_prob = base_prob * fatigue_multiplier * fatigue_factor * refractory_factor
        onset_prob = max(self.params.minimum_onset_prob, min(0.5, onset_prob))
        
        # Store trigger conditions for analysis
        self.trigger_conditions = {
            'mission_time': mission_time_hours,
            'adaptation': self.adaptation_level,
            'fatigue': fatigue,
            'base_prob': base_prob,
            'final_prob': onset_prob,
            'refractory_factor': refractory_factor
        }
        
        # Sample onset
        should_occur = np.random.random() < onset_prob
        
        if should_occur:
            # Sample severity (beta distribution for bounded [0,1])
            severity = np.random.beta(2, 3)  # Right-skewed (mostly mild)
            
            # Fatigue amplifies severity
            severity *= (1.0 + (fatigue / 20.0))  # Up to 50% increase
            severity = min(self.params.max_severity, 
                          max(self.params.min_severity, severity))
            
            logger.info(
                f"Motion sickness onset at t={t} ({mission_time_hours:.1f}h): "
                f"prob={onset_prob:.3f}, severity={severity:.2f}, "
                f"adaptation={self.adaptation_level:.2f}"
            )
            
            return True, severity
        
        return False, None
    
    def _check_refractory(self, state: Any, current_time: float) -> float:
        """Check if in refractory period from recent events."""
        refractory_window = 4.0  # hours
        for event in reversed(state.event_log):
            if event['type'] == 'motion_sickness':
                event_time = event.get('simulation_time', 0)
                time_since = current_time - event_time
                if time_since < refractory_window:
                    # Linear ramp from 0 to 1 over refractory window
                    return time_since / refractory_window
        return 1.0
    
    def get_duration(self, severity: float, **kwargs) -> float:
        """
        Calculate event duration based on severity.
        
        Duration scales nonlinearly with severity:
        duration = min_duration + (max_duration - min_duration) * severity^exponent
        
        Args:
            severity: Event severity [0,1]
            **kwargs: Additional parameters
            
        Returns:
            Duration in hours
        """
        duration_range = self.params.max_duration - self.params.min_duration
        severity_component = severity ** self.params.duration_severity_exponent
        
        duration = self.params.min_duration + duration_range * severity_component
        return float(np.clip(duration, self.params.min_duration, self.params.max_duration))
    
    def _create_effect(self, severity: float, **kwargs) -> EventEffect:
        """
        Create physiological effects based on severity.
        
        Effects include:
        - Elevated heart rate (sympathetic activation)
        - Increased stress (vestibular stress response)
        - Sleep quality degradation (post-event)
        - Vestibular conflict (for BioGears)
        
        Args:
            severity: Event severity [0,1]
            **kwargs: Additional effect parameters
            
        Returns:
            Configured EventEffect object
        """
        # Heart rate effect
        hr_increase = np.random.normal(
            self.params.hr_increase_mean * severity,
            self.params.hr_increase_std * severity
        )
        
        # Stress effect
        stress_increase = np.random.normal(
            self.params.stress_increase_mean * severity,
            self.params.stress_increase_std * severity
        )
        stress_increase = np.clip(stress_increase, 0, 0.8)
        
        # Create effect object
        effect = EventEffect(
            immediate={
                'hr': hr_increase,
                'stress': stress_increase,
                'vestibular_conflict': self.params.vestibular_stimulus_intensity * severity
            },
            duration_hours=self.duration,
            delayed={
                'sleep_quality': (
                    -self.params.sleep_degradation * severity,  # rate of degradation
                    -0.5  # maximum degradation
                ),
                'fatigue_susceptibility': (
                    0.1 * severity,  # increased fatigue accumulation
                    0.3  # maximum increase
                )
            },
            recovery_rate=0.2,
            recovery_delay=0.5  # hours before recovery starts
        )
        
        # Store BioGears-specific parameters
        self.biogears_params = {
            'vestibular_stimulus': self.params.vestibular_stimulus_intensity * severity,
            'otolith_asymmetry': self.params.otolith_asymmetry,
            'canal_sensitivity': self.params.semicircular_canal_sensitivity,
            'duration': self.duration,
            'severity': severity
        }
        
        return effect
    
    def apply_effect(self, state: Any, t: int, dt_hours: float) -> Dict[str, Any]:
        """
        Apply motion sickness effects to astronaut state.
        
        Args:
            state: AstronautState instance
            t: Current time index
            dt_hours: Time step duration in hours
            
        Returns:
            Dictionary with effect metrics
        """
        if self.effect is None:
            raise RuntimeError("Event not initialized")
        
        current_time = t * dt_hours
        progress = self.get_progress(current_time)
        
        # Get current state values
        current_hr = state.hr[t-1] if t > 0 else 75
        current_stress = state.stress[t-1] if t > 0 else 0
        
        # Calculate effects (peak at middle, decay at ends)
        effect_strength = np.sin(progress * np.pi)  # Bell-shaped curve
        
        # Apply immediate effects
        hr_effect = self.effect.immediate.get('hr', 0) * effect_strength
        stress_effect = self.effect.immediate.get('stress', 0) * effect_strength
        
        # Update state
        state.update(
            t,
            hr=current_hr + hr_effect,
            stress=min(1.0, current_stress + stress_effect)
        )
        
        # Apply delayed effects to future state (stored for coupling engine)
        if not hasattr(state, 'pending_effects'):
            state.pending_effects = []
        
        state.pending_effects.append({
            'time': t,
            'type': 'motion_sickness',
            'sleep_degradation': self.effect.delayed.get('sleep_quality', (0,0))[0] * dt_hours,
            'fatigue_susceptibility': self.effect.delayed.get('fatigue_susceptibility', (0,0))[0] * dt_hours
        })
        
        # Record effect
        effect_metrics = {
            'time': current_time,
            'progress': progress,
            'effect_strength': effect_strength,
            'hr_effect': hr_effect,
            'stress_effect': stress_effect,
            'remaining_duration': self.get_remaining_duration(current_time)
        }
        
        self.effect_history.append(effect_metrics)
        
        return effect_metrics
    
    def get_biogears_perturbation(self) -> Dict[str, Any]:
        """
        Get perturbation parameters for BioGears integration.
        
        This method provides the data that Person 2's BioGears adapter needs
        to simulate the physiological response.
        
        Returns:
            Dictionary with BioGears scenario parameters
        """
        if not hasattr(self, 'biogears_params'):
            return {}
        
        return {
            'event_type': 'motion_sickness',
            'scenario': 'vestibular_stimulation',
            'duration_minutes': self.duration * 60,
            'parameters': self.biogears_params,
            'expected_outputs': ['HeartRate', 'MeanArterialPressure', 'StressLevel']
        }


# Factory function for creating motion sickness events
def create_motion_sickness_event(
    severity: Optional[float] = None,
    **kwargs
) -> MotionSicknessEvent:
    """
    Factory function to create a motion sickness event with optional preset severity.
    
    Args:
        severity: Optional preset severity (if None, will be sampled at onset)
        **kwargs: Additional event parameters
        
    Returns:
        Configured MotionSicknessEvent
    """
    event = MotionSicknessEvent(**kwargs)
    if severity is not None:
        event.severity = severity
        event.duration = event.get_duration(severity)
    return event