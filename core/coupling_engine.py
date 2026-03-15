"""
Coupling Engine for Astronaut Digital Twin
Implements bidirectional coupling between sleep-fatigue and motion sickness systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CouplingDirection(Enum):
    """Direction of coupling effect"""
    SLEEP_TO_MOTION = "sleep_to_motion"
    MOTION_TO_SLEEP = "motion_to_sleep"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class CouplingParameters:
    """
    Parameters for bidirectional coupling between systems.
    
    Motion Sickness → Sleep:
        - Base degradation: Minimum sleep quality reduction
        - Severity multiplier: How much severity affects sleep
        - Duration effect: How long effects persist
        
    Fatigue → Motion Sickness:
        - Base probability increase: Minimum increase in onset probability
        - Threshold: Fatigue level where effects begin
        - Severity multiplier: How fatigue amplifies severity
    """
    
    # Motion sickness → sleep coupling
    ms_to_sleep_base_degradation: float = 0.15  # 15% reduction baseline
    ms_to_sleep_severity_multiplier: float = 0.1  # Additional 10% per severity point
    ms_to_sleep_duration_hours: float = 6.0  # Effect lasts 6 hours
    ms_to_sleep_recovery_rate: float = 0.2  # Recovery per hour after event
    
    # Fatigue → motion sickness coupling
    fatigue_to_ms_base_prob_increase: float = 0.1  # 10% baseline increase
    fatigue_to_ms_threshold: float = 3.0  # Fatigue threshold for effects
    fatigue_to_ms_severity_multiplier: float = 0.15  # Severity increase per fatigue point
    fatigue_to_ms_prob_slope: float = 0.05  # Probability increase per fatigue point
    
    # Shared parameters
    max_coupling_effect: float = 0.8  # Maximum coupling effect (80%)
    min_sleep_quality: float = 0.1  # Minimum possible sleep quality
    max_motion_probability: float = 0.5  # Max motion sickness probability per hour
    
    def validate(self):
        """Validate parameter bounds."""
        assert 0 <= self.ms_to_sleep_base_degradation <= 1
        assert self.ms_to_sleep_duration_hours > 0
        assert self.fatigue_to_ms_threshold > 0
        assert 0 <= self.max_coupling_effect <= 1


@dataclass
class CouplingState:
    """Tracks current coupling effects between systems"""
    active_ms_events: List[Dict[str, Any]] = field(default_factory=list)
    fatigue_effect_multiplier: float = 1.0
    last_sleep_quality: float = 1.0
    coupling_history: List[Dict[str, Any]] = field(default_factory=list)


class CouplingEngine:
    """
    Manages bidirectional coupling between sleep-fatigue and motion sickness.
    
    Implements two-way coupling:
    1. Motion sickness episodes degrade subsequent sleep quality
    2. Accumulated fatigue increases probability and severity of motion sickness
    
    This engine is the core of the digital twin's emergent behavior,
    creating realistic feedback loops between physiological systems.
    """
    
    def __init__(self, params: Optional[CouplingParameters] = None):
        """
        Initialize coupling engine.
        
        Args:
            params: CouplingParameters object (uses defaults if None)
        """
        self.params = params or CouplingParameters()
        self.params.validate()
        
        self.state = CouplingState()
        self.coupling_functions: Dict[str, Callable] = {}
        
        # Register default coupling functions
        self._register_default_couplings()
        
        logger.info(f"Initialized CouplingEngine with params: {self.params}")
    
    def _register_default_couplings(self):
        """Register default bidirectional coupling functions."""
        self.coupling_functions = {
            'apply_motion_sickness_effect': self.apply_motion_sickness_effect,
            'compute_fatigue_effect_on_ms': self.compute_fatigue_effect_on_ms,
            'update_sleep_quality_with_coupling': self.update_sleep_quality_with_coupling,
            'update_motion_probability': self.update_motion_probability
        }
    
    def apply_motion_sickness_effect(
        self,
        base_sleep_quality: float,
        ms_events: List[Dict[str, Any]],
        current_time_hours: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply motion sickness effects on sleep quality.
        
        Motion sickness episodes degrade sleep quality through:
        - Vestibular disturbances during sleep
        - Elevated stress hormones
        - Physical discomfort
        
        Args:
            base_sleep_quality: Intrinsic sleep quality [0,1]
            ms_events: List of active motion sickness events
            current_time_hours: Current simulation time
            
        Returns:
            Tuple of (degraded_sleep_quality, effect_metadata)
        """
        if not ms_events:
            return base_sleep_quality, {'degradation': 0.0, 'active_events': 0}
        
        # Calculate total degradation from all active events
        total_degradation = 0.0
        
        for event in ms_events:
            # Base degradation
            degradation = self.params.ms_to_sleep_base_degradation
            
            # Severity contribution
            severity = event.get('severity', 1.0)
            degradation += severity * self.params.ms_to_sleep_severity_multiplier
            
            # Time decay (recent events have stronger effect)
            event_time = event.get('start_time', current_time_hours)
            time_since_event = current_time_hours - event_time
            if time_since_event > 0:
                # Exponential decay
                decay = np.exp(-self.params.ms_to_sleep_recovery_rate * time_since_event)
                degradation *= decay
            
            total_degradation += degradation
        
        # Apply coupling limit
        total_degradation = min(total_degradation, self.params.max_coupling_effect)
        
        # Calculate degraded sleep quality
        degraded_quality = base_sleep_quality * (1 - total_degradation)
        degraded_quality = max(degraded_quality, self.params.min_sleep_quality)
        
        # Track effect
        effect_metadata = {
            'degradation': total_degradation,
            'active_events': len(ms_events),
            'base_quality': base_sleep_quality,
            'degraded_quality': degraded_quality
        }
        
        return degraded_quality, effect_metadata
    
    def compute_fatigue_effect_on_ms(
        self,
        base_probability: float,
        fatigue_level: float,
        event_type: str = 'onset'
    ) -> Tuple[float, float]:
        """
        Compute how fatigue affects motion sickness probability and severity.
        
        Fatigue increases susceptibility to motion sickness through:
        - Reduced compensatory capacity
        - Impaired adaptation mechanisms
        - Heightened stress response
        
        Args:
            base_probability: Baseline event probability
            fatigue_level: Current fatigue index
            event_type: 'onset' or 'severity'
            
        Returns:
            Tuple of (modified_probability, severity_multiplier)
        """
        if fatigue_level <= self.params.fatigue_to_ms_threshold:
            # Below threshold, no effect
            return base_probability, 1.0
        
        # Calculate fatigue excess above threshold
        excess_fatigue = fatigue_level - self.params.fatigue_to_ms_threshold
        
        # Probability modification
        prob_increase = (
            self.params.fatigue_to_ms_base_prob_increase +
            excess_fatigue * self.params.fatigue_to_ms_prob_slope
        )
        prob_increase = min(prob_increase, self.params.max_coupling_effect)
        
        modified_probability = base_probability * (1 + prob_increase)
        modified_probability = min(modified_probability, self.params.max_motion_probability)
        
        # Severity modification
        severity_multiplier = 1.0 + (
            excess_fatigue * self.params.fatigue_to_ms_severity_multiplier
        )
        severity_multiplier = min(severity_multiplier, 1 + self.params.max_coupling_effect)
        
        logger.debug(
            f"Fatigue effect: fatigue={fatigue_level:.1f}, "
            f"prob={base_probability:.3f}→{modified_probability:.3f}, "
            f"severity_mult={severity_multiplier:.2f}"
        )
        
        return modified_probability, severity_multiplier
    
    def update_sleep_quality_with_coupling(
        self,
        state_manager,
        t: int,
        base_sleep_quality: Optional[float] = None
    ) -> float:
        """
        High-level method to update sleep quality considering all couplings.
        
        Args:
            state_manager: AstronautState instance
            t: Current time index
            base_sleep_quality: Optional override for base quality
            
        Returns:
            Updated sleep quality value
        """
        # Get base sleep quality (from probabilistic model)
        if base_sleep_quality is None:
            base_sleep_quality = state_manager.sleep_quality[t]
        
        # Get active motion sickness events
        current_time_hours = t * state_manager.dt / 60.0
        active_events = self._get_active_events(state_manager, current_time_hours)
        
        # Apply motion sickness effect
        degraded_quality, effect = self.apply_motion_sickness_effect(
            base_sleep_quality=base_sleep_quality,
            ms_events=active_events,
            current_time_hours=current_time_hours
        )
        
        # Log coupling effect
        if effect['degradation'] > 0.01:
            logger.info(
                f"Coupling effect at t={t}: sleep quality "
                f"{base_sleep_quality:.2f}→{degraded_quality:.2f} "
                f"({effect['active_events']} active MS events)"
            )
        
        # Update state
        state_manager.update(t, sleep_quality=degraded_quality)
        
        # Track in history
        self.state.coupling_history.append({
            'time': current_time_hours,
            't': t,
            'base_quality': base_sleep_quality,
            'degraded_quality': degraded_quality,
            'active_events': effect['active_events']
        })
        
        return degraded_quality
    
    def update_motion_probability(
        self,
        state_manager,
        t: int,
        base_probability: float
    ) -> Tuple[float, float]:
        """
        Update motion sickness probability based on fatigue coupling.
        
        Args:
            state_manager: AstronautState instance
            t: Current time index
            base_probability: Baseline probability from Poisson process
            
        Returns:
            Tuple of (modified_probability, severity_multiplier)
        """
        fatigue = state_manager.fatigue[t]
        
        modified_prob, severity_mult = self.compute_fatigue_effect_on_ms(
            base_probability=base_probability,
            fatigue_level=fatigue
        )
        
        # Store effect in state manager metadata
        if not hasattr(state_manager, 'coupling_metadata'):
            state_manager.coupling_metadata = []
        
        state_manager.coupling_metadata.append({
            'time': t,
            'base_prob': base_probability,
            'modified_prob': modified_prob,
            'severity_mult': severity_mult,
            'fatigue': fatigue
        })
        
        return modified_prob, severity_mult
    
    def _get_active_events(
        self,
        state_manager,
        current_time_hours: float
    ) -> List[Dict[str, Any]]:
        """Get motion sickness events active at current time."""
        active = []
        
        for event in state_manager.event_log:
            if event['type'] != 'motion_sickness':
                continue
            
            event_time = event.get('simulation_time', 0)
            duration = event.get('duration', 1.0)
            
            if event_time <= current_time_hours <= event_time + duration:
                active.append({
                    'start_time': event_time,
                    'severity': event.get('severity', 1.0),
                    'duration': duration
                })
        
        return active
    
    def compute_emergent_risk(
        self,
        fatigue: float,
        ms_severity: float,
        sleep_quality: float
    ) -> Dict[str, Any]:
        """
        Compute emergent risk from coupled systems.
        
        The interaction between systems can create risks that aren't
        apparent from individual metrics.
        
        Args:
            fatigue: Current fatigue level
            ms_severity: Current motion sickness severity
            sleep_quality: Current sleep quality
            
        Returns:
            Dictionary with emergent risk metrics
        """
        # Coupled risk index
        coupled_risk = (
            0.3 * (fatigue / 10.0) +
            0.3 * (ms_severity / 5.0) +
            0.2 * (1 - sleep_quality) +
            0.2 * (fatigue * ms_severity / 50.0)  # Interaction term
        )
        
        # Vulnerability window (when both systems are compromised)
        vulnerability_window = (
            fatigue > 5.0 and 
            ms_severity > 2.0 and 
            sleep_quality < 0.4
        )
        
        # Recovery potential (how easily can they recover)
        recovery_potential = max(0, 1.0 - (fatigue * ms_severity / 30.0))
        
        return {
            'coupled_risk_index': coupled_risk,
            'vulnerability_window': vulnerability_window,
            'recovery_potential': recovery_potential,
            'risk_level': 'CRITICAL' if coupled_risk > 0.7 else 'HIGH' if coupled_risk > 0.5 else 'MODERATE' if coupled_risk > 0.3 else 'LOW',
            'interaction_effect': fatigue * ms_severity / 50.0
        }
    
    def reset(self):
        """Reset coupling engine state."""
        self.state = CouplingState()
        logger.info("CouplingEngine reset")
    
    def get_coupling_summary(self) -> Dict[str, Any]:
        """
        Get summary of coupling effects for analysis.
        
        Returns:
            Dictionary with coupling statistics
        """
        if not self.state.coupling_history:
            return {'message': 'No coupling events recorded'}
        
        degradations = [h['base_quality'] - h['degraded_quality']   # positive = worse
                       for h in self.state.coupling_history]
        
        return {
            'total_coupling_events': len(self.state.coupling_history),
            'avg_degradation': np.mean(degradations) if degradations else 0,
            'max_degradation': max(degradations) if degradations else 0,
            'coupling_frequency': len(self.state.coupling_history) / max(1, self.state.coupling_history[-1]['time'])
        }