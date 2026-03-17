
"""
Fatigue Accumulation Model for Astronaut Digital Twin
Implements the core fatigue equation with recovery dynamics and coupling effects.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FatigueParameters:
    """
    Parameters for fatigue accumulation model.
    
    The fatigue model follows:
    F(t) = F(t-1) + α*(1 - sleep_quality) + β*(motion_severity) - γ*recovery
    
    Where:
    - α: Sleep debt accumulation rate
    - β: Motion sickness stress multiplier
    - γ: Recovery rate during sleep
    """
    alpha_sleep_debt: float = 0.3  # Fatigue per unit sleep deficit
    beta_motion_stress: float = 0.5  # Fatigue per unit motion severity
    gamma_recovery: float = 0.08  # Recovery rate during sleep
    recovery_threshold: float = 0.6  # Sleep quality needed for recovery
    max_fatigue: float = 10.0  # Maximum fatigue index
    min_fatigue: float = 0.0  # Minimum fatigue index
    
    # Nonlinear effects
    fatigue_sensitivity: float = 1.2  # Exponent for fatigue accumulation
    recovery_efficiency: float = 0.8  # Multiplier for deep sleep recovery
    
    def validate(self):
        """Validate parameter bounds."""
        assert 0 <= self.alpha_sleep_debt <= 1, "alpha_sleep_debt must be [0,1]"
        assert 0 <= self.beta_motion_stress <= 1, "beta_motion_stress must be [0,1]"
        assert 0 <= self.gamma_recovery <= 1, "gamma_recovery must be [0,1]"
        assert self.max_fatigue > self.min_fatigue


class FatigueModel:
    """
    Core fatigue accumulation model with bidirectional coupling.
    
    Implements a physiologically-grounded fatigue model that captures:
    1. Cumulative effects of poor sleep
    2. Stress-induced fatigue from motion sickness
    3. Recovery during quality sleep
    4. Nonlinear dynamics and thresholds
    
    The model is designed to be coupled with the event system and
    BioGears physiological responses.
    """
    
    def __init__(self, params: Optional[FatigueParameters] = None):
        """
        Initialize fatigue model with parameters.
        
        Args:
            params: FatigueParameters object (uses defaults if None)
        """
        self.params = params or FatigueParameters()
        self.params.validate()
        
        # State tracking
        self.fatigue_history = []
        self.recovery_events = []
        self.critical_events = []
        
        logger.info(f"Initialized FatigueModel with params: {self.params}")
    
    def compute_fatigue_update(
        self,
        current_fatigue: float,
        sleep_quality: float,
        motion_severity: float,
        time_in_bed_hours: float = 8.0,
        dt_hours: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute fatigue update based on current state.
        
        The core equation with nonlinear effects:
        ΔF = [α*(1 - Q)^p + β*S^q - γ*R*Q_r] * dt
        
        Where:
        - Q: sleep quality [0,1]
        - S: motion severity [0,5]
        - R: recovery factor based on time in bed
        - Q_r: recovery quality threshold
        
        Args:
            current_fatigue: Fatigue at previous timestep
            sleep_quality: Sleep quality [0,1]
            motion_severity: Motion sickness severity [0,5]
            time_in_bed_hours: Hours spent in bed
            dt_hours: Time step duration in hours
            
        Returns:
            Tuple of (new_fatigue, components_dict)
            components_dict contains breakdown of contributions
        """
        # Ensure bounds
        sleep_quality = np.clip(sleep_quality, 0, 1)
        motion_severity = np.clip(motion_severity, 0, 5)
        
        # 1. Sleep debt accumulation (nonlinear)
        sleep_deficit = 1.0 - sleep_quality
        nonlinear_deficit = sleep_deficit ** self.params.fatigue_sensitivity
        fatigue_from_sleep = (
            self.params.alpha_sleep_debt * 
            nonlinear_deficit * 
            dt_hours
        )
        
        # 2. Motion sickness stress contribution
        nonlinear_stress = motion_severity ** 1.5  # Stress grows faster than linear
        fatigue_from_motion = (
            self.params.beta_motion_stress * 
            nonlinear_stress * 
            dt_hours
        )
        
        # 3. Recovery during sleep
        recovery = 0.0
        if sleep_quality > self.params.recovery_threshold:
            # Quality sleep enables recovery
            sleep_quality_factor = (sleep_quality - self.params.recovery_threshold) / (1 - self.params.recovery_threshold)
            time_factor = min(1.0, time_in_bed_hours / 8.0)  # Max recovery at 8 hours
            
            recovery = (
                self.params.gamma_recovery *
                self.params.recovery_efficiency *
                sleep_quality_factor *
                time_factor *
                dt_hours
            )
        
        # 4. Net change
        delta_fatigue = fatigue_from_sleep + fatigue_from_motion - recovery
        
        # 5. Apply bounds
        new_fatigue = current_fatigue + delta_fatigue
        new_fatigue = np.clip(new_fatigue, self.params.min_fatigue, self.params.max_fatigue)
        
        # Track components for analysis
        components = {
            'delta_fatigue': delta_fatigue,
            'fatigue_from_sleep': fatigue_from_sleep,
            'fatigue_from_motion': fatigue_from_motion,
            'recovery': recovery,
            'sleep_deficit': sleep_deficit,
            'effective_recovery': recovery > 0
        }
        
        return new_fatigue, components
    
    def estimate_recovery_time(
        self,
        current_fatigue: float,
        optimal_sleep_quality: float = 0.9,
        time_in_bed_hours: float = 8.0
    ) -> float:
        """
        Estimate hours needed to recover to nominal fatigue levels.
        
        Args:
            current_fatigue: Current fatigue level
            optimal_sleep_quality: Expected sleep quality during recovery
            time_in_bed_hours: Hours in bed per night
            
        Returns:
            Estimated recovery hours
        """
        if current_fatigue <= 1.0:
            return 0.0
        
        # Simulate recovery with optimal conditions
        fatigue = current_fatigue
        hours = 0
        dt = 1.0  # 1-hour steps
        
        while fatigue > 1.0 and hours < 168:  # Max 1 week
            fatigue, _ = self.compute_fatigue_update(
                current_fatigue=fatigue,
                sleep_quality=optimal_sleep_quality,
                motion_severity=0.0,
                time_in_bed_hours=time_in_bed_hours,
                dt_hours=dt
            )
            hours += dt
        
        return hours
    
    def compute_risk_contribution(self, fatigue: float) -> Dict[str, float]:
        """
        Compute risk metrics based on fatigue level.
        
        Args:
            fatigue: Current fatigue level
            
        Returns:
            Dictionary with risk probabilities and severity
        """
        # Normalized fatigue [0,1] for probability calculations
        norm_fatigue = fatigue / self.params.max_fatigue
        
        # Probability of performance decrement
        performance_decrement_prob = 1.0 / (1.0 + np.exp(-10 * (norm_fatigue - 0.5)))
        
        # Probability of error/accident
        error_probability = min(0.8, 0.05 * np.exp(3 * norm_fatigue))
        
        # Sleep disruption probability
        sleep_disruption_prob = min(0.9, 0.1 * np.exp(2 * norm_fatigue))
        
        return {
            'normalized_fatigue': norm_fatigue,
            'performance_decrement_prob': performance_decrement_prob,
            'error_probability': error_probability,
            'sleep_disruption_probability': sleep_disruption_prob,
            'risk_level': 'high' if norm_fatigue > 0.7 else 'medium' if norm_fatigue > 0.4 else 'low'
        }
    
    def get_fatigue_state_description(self, fatigue: float) -> str:
        """
        Get human-readable description of fatigue state.
        
        Args:
            fatigue: Current fatigue level
            
        Returns:
            String description
        """
        if fatigue < 1.0:
            return "Well-rested, nominal performance"
        elif fatigue < 2.5:
            return "Mild fatigue, slight performance decrement"
        elif fatigue < 4.0:
            return "Moderate fatigue, increased reaction time"
        elif fatigue < 6.0:
            return "Significant fatigue, frequent microsleeps risk"
        elif fatigue < 8.0:
            return "Severe fatigue, critical performance degradation"
        else:
            return "Extreme fatigue, imminent safety risk"
    
    def reset(self):
        """Reset model state."""
        self.fatigue_history = []
        self.recovery_events = []
        self.critical_events = []
        logger.info("FatigueModel reset")