
"""
Sleep Disruption Event Module
Implements sleep disruption episodes triggered by fatigue and discomfort.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from events.base_event import Event, EventEffect, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class SleepDisruptionParameters:
    """Parameters for sleep disruption event dynamics."""
    
    # Trigger thresholds
    fatigue_threshold: float = 3.0  # Fatigue level that triggers risk
    critical_fatigue_threshold: float = 6.0  # Guaranteed disruption
    motion_severity_threshold: float = 1.5  # Motion sickness level that affects sleep
    
    # Probability parameters
    base_probability: float = 0.1  # Base probability per sleep period
    fatigue_sensitivity: float = 0.15  # Probability increase per fatigue unit
    ms_sensitivity: float = 0.2  # Probability increase per motion severity unit
    
    # Severity parameters
    severity_mean: float = 0.5
    severity_std: float = 0.2
    min_severity: float = 0.2
    max_severity: float = 1.0
    
    # Duration parameters
    min_duration: float = 2.0  # hours of disrupted sleep
    max_duration: float = 8.0  # hours (entire sleep period)
    
    # Sleep quality effects
    sleep_quality_reduction: float = 0.4  # Max reduction per severity unit
    sleep_duration_reduction: float = 0.3  # Max duration reduction per severity unit
    
    # Recovery parameters
    recovery_sleep_needed: float = 1.5  # Hours of quality sleep per hour disrupted
    next_day_effect_decay: float = 0.7  # Carryover to next day
    
    def validate(self):
        """Validate parameter bounds."""
        assert self.fatigue_threshold > 0
        assert 0 <= self.base_probability <= 1
        assert 0 <= self.min_severity <= self.max_severity <= 1
        assert self.min_duration <= self.max_duration


class SleepDisruptionEvent(Event):
    """
    Sleep Disruption Event
    
    Represents episodes of disrupted sleep caused by:
    - Accumulated fatigue (primary trigger)
    - Motion sickness discomfort
    - Circadian rhythm disruption
    
    Key characteristics:
    - Triggered by threshold crossings
    - Reduces sleep quality and duration
    - Creates positive feedback (disruption → more fatigue)
    - Recovery requires quality sleep
    """
    
    def __init__(
        self,
        params: Optional[SleepDisruptionParameters] = None,
        **kwargs
    ):
        """
        Initialize sleep disruption event.
        
        Args:
            params: SleepDisruptionParameters object (uses defaults if None)
            **kwargs: Additional event parameters
        """
        super().__init__(priority=EventPriority.MEDIUM, **kwargs)
        self.params = params or SleepDisruptionParameters()
        self.params.validate()
        
        # Event-specific state
        self.sleep_debt: float = 0.0
        self.disrupted_hours: float = 0.0
        self.recovery_achieved: bool = False
        
        logger.debug(f"Created SleepDisruptionEvent with params: {self.params}")
    
    def sample_onset(
        self,
        state: Any,
        t: int,
        **kwargs
    ) -> Tuple[bool, Optional[float]]:
        """
        Determine if sleep disruption occurs.
        
        Onset is triggered by a combination of:
        - Fatigue level exceeding threshold
        - Recent motion sickness events
        - Probabilistic component even at lower fatigue
        
        Args:
            state: Current astronaut state
            t: Current time index
            **kwargs: Additional factors
            
        Returns:
            Tuple of (should_occur, severity if occurs)
        """
        # Only check during sleep periods (simplified: every 24 hours at hour 22)
        mission_time_hours = t * (getattr(state, 'dt', 5.0) / 60.0)
        hour_of_day = mission_time_hours % 24
        
        # Only trigger during sleep window (10 PM to 6 AM)
        in_sleep_window = hour_of_day >= 22 or hour_of_day <= 6
        if not in_sleep_window:
            return False, None
        
        # Get current state values
        fatigue = state.fatigue[t] if t < len(state.fatigue) else 0
        motion_severity = state.motion_severity[t] if t < len(state.motion_severity) else 0
        
        # Calculate onset probability
        onset_prob = self.params.base_probability
        
        # Fatigue contribution
        if fatigue > self.params.fatigue_threshold:
            fatigue_contribution = self.params.fatigue_sensitivity * (
                fatigue - self.params.fatigue_threshold
            )
            onset_prob += fatigue_contribution
        
        # Motion sickness contribution
        if motion_severity > self.params.motion_severity_threshold:
            ms_contribution = self.params.ms_sensitivity * (
                motion_severity - self.params.motion_severity_threshold
            )
            onset_prob += ms_contribution
        
        # Cap probability
        onset_prob = min(0.95, onset_prob)
        
        # Critical fatigue guarantees disruption
        if fatigue > self.params.critical_fatigue_threshold:
            onset_prob = 1.0
        
        # Store trigger conditions
        self.trigger_conditions = {
            'mission_time': mission_time_hours,
            'hour_of_day': hour_of_day,
            'fatigue': fatigue,
            'motion_severity': motion_severity,
            'onset_prob': onset_prob
        }
        
        # Sample onset
        should_occur = np.random.random() < onset_prob
        
        if should_occur:
            # Severity increases with fatigue and motion sickness
            base_severity = np.random.beta(2, 2)  # Symmetric distribution
            fatigue_factor = fatigue / 10.0  # Normalize to [0,1]
            ms_factor = motion_severity / 5.0  # Normalize to [0,1]
            
            severity = base_severity * (0.5 + 0.5 * fatigue_factor) * (0.8 + 0.2 * ms_factor)
            severity = np.clip(severity, self.params.min_severity, self.params.max_severity)
            
            logger.info(
                f"Sleep disruption onset at t={t} ({mission_time_hours:.1f}h): "
                f"prob={onset_prob:.3f}, severity={severity:.2f}, "
                f"fatigue={fatigue:.1f}"
            )
            
            return True, severity
        
        return False, None
    
    def get_duration(self, severity: float, **kwargs) -> float:
        """
        Calculate disruption duration based on severity.
        
        Duration represents hours of disrupted sleep within the sleep period.
        
        Args:
            severity: Event severity [0,1]
            **kwargs: Additional parameters
            
        Returns:
            Duration in hours
        """
        duration_range = self.params.max_duration - self.params.min_duration
        duration = self.params.min_duration + duration_range * severity
        return float(np.clip(duration, self.params.min_duration, self.params.max_duration))
    
    def _create_effect(self, severity: float, **kwargs) -> EventEffect:
        """
        Create sleep disruption effects.
        
        Effects include:
        - Reduced sleep quality during the event
        - Reduced effective sleep duration
        - Increased next-day fatigue accumulation
        
        Args:
            severity: Event severity [0,1]
            **kwargs: Additional effect parameters
            
        Returns:
            Configured EventEffect object
        """
        # Sleep quality reduction
        quality_reduction = self.params.sleep_quality_reduction * severity
        
        # Duration reduction (fraction of sleep period lost)
        duration_reduction = self.params.sleep_duration_reduction * severity
        
        # Next-day fatigue acceleration
        fatigue_acceleration = 0.2 * severity  # 20% faster fatigue accumulation
        
        effect = EventEffect(
            immediate={
                'sleep_quality': -quality_reduction,
                'effective_sleep_hours': -duration_reduction * 8.0,  # Assuming 8h sleep period
                'sleep_debt': duration_reduction * 8.0  # Hours of sleep debt
            },
            duration_hours=self.duration,
            delayed={
                'fatigue_accumulation_rate': (
                    fatigue_acceleration / 24.0,  # per hour
                    0.3  # max increase
                ),
                'recovery_efficiency': (
                    -0.1 * severity,  # reduced recovery during next sleep
                    -0.3  # max reduction
                )
            },
            recovery_rate=0.3,  # Recovery during next sleep period
            recovery_delay=self.duration  # Recovery starts after disruption ends
        )
        
        return effect
    
    def apply_effect(self, state: Any, t: int, dt_hours: float) -> Dict[str, Any]:
        """
        Apply sleep disruption effects to astronaut state.
        
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
        
        # Track disrupted hours
        self.disrupted_hours += dt_hours
        
        # Get current state
        current_sleep_quality = state.sleep_quality[t-1] if t > 0 else 0.8
        current_fatigue = state.fatigue[t-1] if t > 0 else 0
        
        # Calculate effects (strongest during disruption)
        effect_strength = 1.0  # Constant during event (disruption is ongoing)
        
        # Apply sleep quality reduction
        quality_effect = self.effect.immediate.get('sleep_quality', 0) * effect_strength
        new_sleep_quality = max(0.1, current_sleep_quality + quality_effect)
        
        # Update state
        state.update(
            t,
            sleep_quality=new_sleep_quality
        )
        
        # Track sleep debt for recovery calculation
        sleep_debt_increment = self.effect.immediate.get('sleep_debt', 0) * dt_hours / self.duration
        self.sleep_debt += sleep_debt_increment
        
        # Store effect for fatigue model
        if not hasattr(state, 'sleep_disruption_effects'):
            state.sleep_disruption_effects = []
        
        state.sleep_disruption_effects.append({
            'time': t,
            'disrupted_hours': self.disrupted_hours,
            'sleep_debt': self.sleep_debt,
            'quality_reduction': quality_effect
        })
        
        # Record effect
        effect_metrics = {
            'time': current_time,
            'progress': progress,
            'effect_strength': effect_strength,
            'sleep_quality': new_sleep_quality,
            'disrupted_hours': self.disrupted_hours,
            'sleep_debt': self.sleep_debt,
            'remaining_duration': self.get_remaining_duration(current_time)
        }
        
        self.effect_history.append(effect_metrics)
        
        return effect_metrics
    
    def compute_recovery_needed(self) -> float:
        """
        Calculate hours of quality sleep needed for recovery.
        
        Returns:
            Hours of quality sleep required
        """
        return self.sleep_debt * self.params.recovery_sleep_needed
    
    def check_recovery(self, sleep_quality: float, duration_hours: float) -> bool:
        """
        Check if recovery has been achieved.
        
        Args:
            sleep_quality: Quality of sleep [0,1]
            duration_hours: Hours of sleep
            
        Returns:
            True if recovered
        """
        if sleep_quality < 0.7:  # Need quality sleep for recovery
            return False
        
        effective_recovery = duration_hours * (sleep_quality - 0.5) * 2  # Scale to [0,1]
        self.sleep_debt -= effective_recovery * self.params.next_day_effect_decay
        self.sleep_debt = max(0, self.sleep_debt)
        
        self.recovery_achieved = self.sleep_debt < 0.1
        return self.recovery_achieved


# Factory function for creating sleep disruption events
def create_sleep_disruption_event(
    severity: Optional[float] = None,
    **kwargs
) -> SleepDisruptionEvent:
    """
    Factory function to create a sleep disruption event.
    
    Args:
        severity: Optional preset severity
        **kwargs: Additional event parameters
        
    Returns:
        Configured SleepDisruptionEvent
    """
    event = SleepDisruptionEvent(**kwargs)
    if severity is not None:
        event.severity = severity
        event.duration = event.get_duration(severity)
    return event