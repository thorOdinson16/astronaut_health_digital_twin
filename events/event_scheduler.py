
"""
Event Scheduler Module
Manages the discrete event simulation for the digital twin.
Handles event triggering, queueing, and application.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Type
from collections import defaultdict
import logging
from datetime import datetime
import heapq
from dataclasses import dataclass, field

from events.base_event import Event, EventStatus, EventPriority
from events.motion_sickness_event import MotionSicknessEvent
from events.sleep_disruption_event import SleepDisruptionEvent

logger = logging.getLogger(__name__)


@dataclass
class ScheduledEvent:
    """Wrapper for scheduled events with priority queue support."""
    trigger_time: float  # Simulation time to trigger
    event: Event
    priority: int  # Lower number = higher priority
    
    def __lt__(self, other):
        """For heapq priority queue ordering."""
        if self.trigger_time != other.trigger_time:
            return self.trigger_time < other.trigger_time
        return self.priority < other.priority


class EventScheduler:
    """
    Event scheduler for discrete event simulation.
    
    Manages the entire event lifecycle:
    - Trigger detection (stochastic and threshold-based)
    - Event queueing and prioritization
    - Effect application over time
    - Event logging and history
    
    The scheduler integrates with the main simulation loop and
    coordinates with the coupling engine for bidirectional effects.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize event scheduler.
        
        Args:
            config: Optional configuration dictionary
        """
        # Event queue (priority queue based on trigger time)
        self.event_queue: List[ScheduledEvent] = []
        
        # Active events (events currently in progress)
        self.active_events: Dict[str, Event] = {}
        
        # Event history (completed events)
        self.event_history: List[Dict[str, Any]] = []
        
        # Event statistics
        self.stats: Dict[str, Any] = {
            'total_events_triggered': 0,
            'events_by_type': defaultdict(int),
            'events_by_priority': defaultdict(int),
            'total_active_time': defaultdict(float),
            'max_concurrent_events': 0
        }
        
        # Registered event types
        self.event_types: Dict[str, Type[Event]] = {
            'motion_sickness': MotionSicknessEvent,
            'sleep_disruption': SleepDisruptionEvent
        }
        
        # Configuration
        self.config = config or {}
        self.max_concurrent_events = self.config.get('max_concurrent_events', 10)
        self.enable_logging = self.config.get('enable_event_logging', True)
        
        # Time tracking
        self.current_time: float = 0.0
        self.current_index: int = 0
        
        logger.info("EventScheduler initialized")
    
    def check_triggers(
        self,
        state: Any,
        t: int,
        dt_hours: float,
        coupling_effects: Optional[Dict[str, Any]] = None
    ) -> List[Event]:
        """
        Check for and trigger new events at current timestep.
        
        This method is called each timestep of the main simulation loop.
        It evaluates all registered event types for possible onset.
        
        Args:
            state: Current astronaut state
            t: Current time index
            dt_hours: Time step duration in hours
            coupling_effects: Additional coupling effects from engine
            
        Returns:
            List of newly triggered events
        """
        self.current_time = t * dt_hours
        self.current_index = t
        
        newly_triggered = []
        
        # Check each event type for onset
        for event_name, event_class in self.event_types.items():
            # Create temporary event instance for checking
            temp_event = event_class()
            
            # Get coupling effects for this event type
            event_coupling = {}
            if coupling_effects and event_name in coupling_effects:
                event_coupling = coupling_effects[event_name]
            
            # Sample onset
            should_occur, severity = temp_event.sample_onset(
                state=state,
                t=t,
                **event_coupling
            )
            
            if should_occur and severity is not None:
                # Create actual event instance
                event = event_class()
                
                # Initialize event with onset parameters
                event.initialize_event(
                    onset_time=self.current_time,
                    onset_index=t,
                    severity=severity
                )
                
                # Add to queue (with small delay to ensure ordering)
                trigger_time = self.current_time + 0.001
                self._schedule_event(trigger_time, event)
                
                newly_triggered.append(event)
                
                logger.info(
                    f"Triggered {event_name} at t={t} ({self.current_time:.1f}h)"
                )
        
        # Update statistics
        if newly_triggered:
            self.stats['total_events_triggered'] += len(newly_triggered)
            for event in newly_triggered:
                self.stats['events_by_type'][event.__class__.__name__] += 1
        
        return newly_triggered
    
    def _schedule_event(self, trigger_time: float, event: Event) -> None:
        """Add event to priority queue."""
        scheduled = ScheduledEvent(
            trigger_time=trigger_time,
            event=event,
            priority=event.priority.value
        )
        heapq.heappush(self.event_queue, scheduled)
    
    def process_pending_events(self, state: Any, dt_hours: float) -> List[Dict[str, Any]]:
        """
        Process all events that should be triggered at current time.
        
        Also updates active events and applies their effects.
        
        Args:
            state: Astronaut state to apply effects to
            dt_hours: Time step duration
            
        Returns:
            List of effect metrics from processed events
        """
        applied_effects = []
        
        # Process newly triggered events
        while (self.event_queue and 
               self.event_queue[0].trigger_time <= self.current_time):
            scheduled = heapq.heappop(self.event_queue)
            event = scheduled.event
            
            # Add to active events
            self.active_events[event.event_id] = event
            self.stats['events_by_priority'][event.priority.name] += 1
            
            logger.debug(
                f"Event {event.event_id} activated at {self.current_time:.1f}h"
            )
        
        # Update active events
        if self.active_events:
            # Track concurrency
            self.stats['max_concurrent_events'] = max(
                self.stats['max_concurrent_events'],
                len(self.active_events)
            )
            
            # Apply effects for each active event
            completed_events = []
            for event_id, event in self.active_events.items():
                # Apply effect
                effect = event.apply_effect(state, self.current_index, dt_hours)
                applied_effects.append(effect)
                
                # Track active time
                self.stats['total_active_time'][event.__class__.__name__] += dt_hours
                
                # Check if event is complete
                if not event.is_active(self.current_time + dt_hours):
                    event.complete()
                    completed_events.append(event_id)
                    
                    # Log to history
                    self.event_history.append(event.to_dict())
                    
                    logger.debug(
                        f"Event {event_id} completed at {self.current_time + dt_hours:.1f}h"
                    )
            
            # Remove completed events
            for event_id in completed_events:
                del self.active_events[event_id]
        
        return applied_effects
    
    def process_time_step(
        self,
        state: Any,
        t: int,
        dt_hours: float,
        coupling_effects: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete time step processing: check triggers and apply effects.
        
        This is the main method called by the simulation loop.
        
        Args:
            state: Astronaut state
            t: Current time index
            dt_hours: Time step duration
            coupling_effects: Coupling effects from engine
            
        Returns:
            Summary of events processed this timestep
        """
        # Check for new triggers
        new_events = self.check_triggers(state, t, dt_hours, coupling_effects)
        
        # Process pending events and apply effects
        effects = self.process_pending_events(state, dt_hours)
        
        return {
            'time': t,
            'simulation_time': self.current_time,
            'new_events': [e.to_dict() for e in new_events],
            'active_events': len(self.active_events),
            'effects_applied': len(effects),
            'queue_size': len(self.event_queue)
        }
    
    def get_active_events(self, event_type: Optional[str] = None) -> List[Event]:
        """
        Get currently active events, optionally filtered by type.
        
        Args:
            event_type: Optional event type filter
            
        Returns:
            List of active events
        """
        if event_type:
            return [e for e in self.active_events.values() 
                   if e.__class__.__name__ == event_type]
        return list(self.active_events.values())
    
    def get_upcoming_events(self, time_window: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get events scheduled in the upcoming time window.
        
        Args:
            time_window: Hours to look ahead
            
        Returns:
            List of upcoming event summaries
        """
        upcoming = []
        cutoff_time = self.current_time + time_window
        
        for scheduled in self.event_queue:
            if scheduled.trigger_time <= cutoff_time:
                upcoming.append({
                    'trigger_time': scheduled.trigger_time,
                    'event_type': scheduled.event.__class__.__name__,
                    'event_id': scheduled.event.event_id,
                    'priority': scheduled.priority
                })
            else:
                break
        
        return upcoming
    
    def cancel_event(self, event_id: str, reason: str = "") -> bool:
        """
        Cancel a pending or active event.
        
        Args:
            event_id: ID of event to cancel
            reason: Cancellation reason
            
        Returns:
            True if event was cancelled
        """
        # Check active events
        if event_id in self.active_events:
            self.active_events[event_id].cancel(reason)
            del self.active_events[event_id]
            return True
        
        # Check queue
        for i, scheduled in enumerate(self.event_queue):
            if scheduled.event.event_id == event_id:
                scheduled.event.cancel(reason)
                self.event_queue.pop(i)
                heapq.heapify(self.event_queue)  # Re-heapify
                return True
        
        return False
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive event statistics.
        
        Returns:
            Dictionary with event statistics
        """
        return {
            **self.stats,
            'current_active': len(self.active_events),
            'queue_size': len(self.event_queue),
            'history_size': len(self.event_history),
            'active_event_types': {
                name: len([e for e in self.active_events.values() 
                          if e.__class__.__name__ == name])
                for name in self.event_types.keys()
            }
        }
    
    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.event_queue = []
        self.active_events = {}
        self.event_history = []
        self.stats = {
            'total_events_triggered': 0,
            'events_by_type': defaultdict(int),
            'events_by_priority': defaultdict(int),
            'total_active_time': defaultdict(float),
            'max_concurrent_events': 0
        }
        self.current_time = 0.0
        self.current_index = 0
        logger.info("EventScheduler reset")
    
    def register_event_type(self, name: str, event_class: Type[Event]) -> None:
        """
        Register a new event type.
        
        Args:
            name: Event type name
            event_class: Event class (must inherit from Event)
        """
        if not issubclass(event_class, Event):
            raise ValueError(f"{event_class} must inherit from Event")
        self.event_types[name] = event_class
        logger.info(f"Registered event type: {name}")
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Get complete event timeline for visualization.
        
        Returns:
            List of all events with timing information
        """
        timeline = []
        
        # Add completed events
        for event_dict in self.event_history:
            timeline.append({
                **event_dict,
                'status': 'completed'
            })
        
        # Add active events
        for event in self.active_events.values():
            timeline.append({
                **event.to_dict(),
                'status': 'active'
            })
        
        # Add queued events
        for scheduled in self.event_queue:
            timeline.append({
                **scheduled.event.to_dict(),
                'trigger_time': scheduled.trigger_time,
                'status': 'queued'
            })
        
        # Sort by onset time
        timeline.sort(key=lambda x: x.get('onset_time', float('inf')))
        
        return timeline