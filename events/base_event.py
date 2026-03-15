
"""
Base Event Module for Astronaut Digital Twin
Defines abstract interfaces for all discrete events in the simulation.
Enforces consistent event structure and provides common functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for event scheduling"""
    CRITICAL = 1  # Life safety events
    HIGH = 2      # Mission-impacting events
    MEDIUM = 3    # Standard events
    LOW = 4       # Background events
    LOGGING = 5   # Purely informational


class EventStatus(Enum):
    """Lifecycle status of an event"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class EventEffect:
    """
    Represents the effect of an event on the astronaut's state.
    Contains both immediate and delayed effects.
    """
    # Immediate effects (applied at event onset)
    immediate: Dict[str, float] = field(default_factory=dict)
    
    # Duration of effect (hours)
    duration_hours: float = 0.0
    
    # Delayed effects (applied gradually over duration)
    delayed: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # (rate, max)
    
    # Recovery profile after event ends
    recovery_rate: float = 0.1  # per hour
    recovery_delay: float = 0.0  # hours before recovery starts
    
    def validate(self) -> bool:
        """Validate effect parameters."""
        assert self.duration_hours >= 0, "Duration must be non-negative"
        assert 0 <= self.recovery_rate <= 1, "Recovery rate must be [0,1]"
        assert self.recovery_delay >= 0, "Recovery delay must be non-negative"
        return True


class Event(ABC):
    """
    Abstract base class for all discrete events in the digital twin.
    
    Events represent discrete occurrences that perturb the astronaut's state,
    such as motion sickness episodes or sleep disruptions. Each event has:
    - Stochastic onset based on probabilistic models
    - Defined duration and severity
    - Specific effects on physiological state
    - Potential to trigger cascading events
    
    This class enforces the interface that all concrete events must implement.
    """
    
    def __init__(
        self,
        event_id: Optional[str] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        **kwargs
    ):
        """
        Initialize base event.
        
        Args:
            event_id: Unique identifier (auto-generated if None)
            priority: Event priority for scheduling
            **kwargs: Additional event-specific parameters
        """
        self.event_id = event_id or str(uuid.uuid4())[:8]
        self.priority = priority
        self.status = EventStatus.PENDING
        
        # Timing attributes (to be set by scheduler)
        self.onset_time: Optional[float] = None  # Simulation time (hours)
        self.onset_index: Optional[int] = None   # Time step index
        self.duration: Optional[float] = None    # Hours
        self.end_time: Optional[float] = None     # onset_time + duration
        
        # Effect tracking
        self.effect: Optional[EventEffect] = None
        self.effect_history: List[Dict[str, Any]] = []
        
        # Metadata
        self.created_at = datetime.now()
        self.trigger_conditions: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = kwargs
        
        logger.debug(f"Created event {self.event_id} of type {self.__class__.__name__}")
    
    @abstractmethod
    def sample_onset(self, state: Any, t: int, **kwargs) -> Tuple[bool, Optional[float]]:
        """
        Determine if event occurs at current time.
        
        This method implements the stochastic process for event onset,
        typically based on probabilistic models (Poisson, threshold crossing, etc.)
        
        Args:
            state: Current astronaut state
            t: Current time index
            **kwargs: Additional context for onset decision
            
        Returns:
            Tuple of (should_occur, severity)
            - should_occur: True if event should start now
            - severity: Optional severity level (0-1 or custom scale)
        """
        pass
    
    @abstractmethod
    def apply_effect(self, state: Any, t: int, dt_hours: float) -> Dict[str, Any]:
        """
        Apply event effects to astronaut state at current time.
        
        This method is called each timestep while the event is active.
        It modifies the state according to the event's effect profile.
        
        Args:
            state: AstronautState instance to modify
            t: Current time index
            dt_hours: Time step duration in hours
            
        Returns:
            Dictionary with effect metrics for logging/analysis
        """
        pass
    
    @abstractmethod
    def get_duration(self, severity: float, **kwargs) -> float:
        """
        Sample or compute event duration based on severity.
        
        Args:
            severity: Event severity level
            **kwargs: Additional parameters for duration calculation
            
        Returns:
            Duration in hours
        """
        pass
    
    def initialize_event(
        self,
        onset_time: float,
        onset_index: int,
        severity: float,
        **kwargs
    ) -> None:
        """
        Initialize event with onset time and severity.
        
        Args:
            onset_time: Simulation time of onset (hours)
            onset_index: Time step index of onset
            severity: Event severity level
            **kwargs: Additional initialization parameters
        """
        self.onset_time = onset_time
        self.onset_index = onset_index
        self.severity = severity
        self.duration = self.get_duration(severity, **kwargs)
        self.end_time = onset_time + self.duration
        self.status = EventStatus.ACTIVE
        
        # Create effect object
        self.effect = self._create_effect(severity, **kwargs)
        self.effect.validate()
        
        logger.info(
            f"Event {self.event_id} initialized: type={self.__class__.__name__}, "
            f"onset={onset_time:.1f}h, duration={self.duration:.1f}h, "
            f"severity={severity:.2f}"
        )
    
    @abstractmethod
    def _create_effect(self, severity: float, **kwargs) -> EventEffect:
        """
        Create effect object based on severity.
        
        This method defines how severity translates to physiological effects.
        
        Args:
            severity: Event severity level
            **kwargs: Additional effect parameters
            
        Returns:
            Configured EventEffect object
        """
        pass
    
    def is_active(self, current_time: float) -> bool:
        """Check if event is active at given time."""
        if self.onset_time is None or self.end_time is None:
            return False
        return self.onset_time <= current_time <= self.end_time
    
    def complete(self) -> None:
        """Mark event as completed."""
        self.status = EventStatus.COMPLETED
        logger.debug(f"Event {self.event_id} completed")
    
    def cancel(self, reason: str = "") -> None:
        """Cancel the event."""
        self.status = EventStatus.CANCELLED
        logger.info(f"Event {self.event_id} cancelled: {reason}")
    
    def get_remaining_duration(self, current_time: float) -> float:
        """Get remaining duration in hours."""
        if self.end_time is None:
            return 0.0
        return max(0, self.end_time - current_time)
    
    def get_progress(self, current_time: float) -> float:
        """Get event progress as fraction [0,1]."""
        if self.duration is None or self.duration == 0:
            return 1.0
        elapsed = current_time - (self.onset_time or 0)
        return min(1.0, max(0.0, elapsed / self.duration))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'type': self.__class__.__name__,
            'priority': self.priority.value,
            'status': self.status.value,
            'onset_time': self.onset_time,
            'onset_index': self.onset_index,
            'duration': self.duration,
            'end_time': self.end_time,
            'severity': getattr(self, 'severity', None),
            'metadata': self.metadata,
            'effect_history': self.effect_history
        }
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.event_id}, "
                f"status={self.status.value}, priority={self.priority.name})")