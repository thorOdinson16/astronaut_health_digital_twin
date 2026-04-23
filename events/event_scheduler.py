"""
events/event_scheduler.py

Event Scheduler Module — manages the discrete event simulation for the digital twin.

FIX (v1.2):
  1. get_upcoming_events() now sorts the heap copy before iterating.
     heapq guarantees the *minimum* element is at index 0 but does NOT
     guarantee insertion order for remaining elements.  Iterating
     self.event_queue directly (as before) could return events out of
     trigger-time order if any heappop/heappush had occurred.

  2. Added get_active_eva_contributions() helper that collects
     "fatigue_forcing" and "stress_delta" from all currently active
     ExerciseStressEvents.  The simulation loop calls this after
     process_time_step() and before the stress formula so that both
     values can be wired into the correct places:
         fatigue_forcing → engine.step(fatigue_forcing=…)
         stress_delta    → folded into the stress formula
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
from events.exercise_stress_event import ExerciseStressEvent

logger = logging.getLogger(__name__)


@dataclass
class ScheduledEvent:
    """Wrapper for scheduled events with priority queue support."""
    trigger_time: float
    event: Event
    priority: int

    def __lt__(self, other):
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
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize event scheduler.

        Args:
            config: Optional configuration dictionary.  Recognised keys:
                max_concurrent_events  (int, default 10)
                enable_event_logging   (bool, default True)
                disabled_event_types   (list[str])
        """
        self.event_queue:   List[ScheduledEvent]  = []
        self.active_events: Dict[str, Event]      = {}
        self.event_history: List[Dict[str, Any]]  = []

        self.stats: Dict[str, Any] = {
            "total_events_triggered":  0,
            "events_by_type":          defaultdict(int),
            "events_by_priority":      defaultdict(int),
            "total_active_time":       defaultdict(float),
            "max_concurrent_events":   0,
        }

        self.event_types: Dict[str, Type[Event]] = {
            "motion_sickness":  MotionSicknessEvent,
            "sleep_disruption": SleepDisruptionEvent,
            "exercise_stress":  ExerciseStressEvent,
        }

        self.config = config or {}
        self.max_concurrent_events = self.config.get("max_concurrent_events", 10)
        self.enable_logging        = self.config.get("enable_event_logging", True)

        for name in self.config.get("disabled_event_types", []):
            self.event_types.pop(name, None)
            logger.info(f"EventScheduler: disabled event type '{name}' via config")

        self.current_time:  float = 0.0
        self.current_index: int   = 0

        # Track simulation-time of the last onset per event type for refractory checks
        self.last_event_times: Dict[str, float] = {}

        logger.info("EventScheduler initialized")

    # ------------------------------------------------------------------
    # Core trigger / process methods
    # ------------------------------------------------------------------

    def check_triggers(
        self,
        state: Any,
        t: int,
        dt_hours: float,
        coupling_effects: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """
        Check for and trigger new events at current timestep.

        Args:
            state:            Current astronaut state
            t:                Current time index
            dt_hours:         Time step duration in hours
            coupling_effects: Physics coupling for specific event types

        Returns:
            List of newly triggered events
        """
        self.current_time  = t * dt_hours
        self.current_index = t

        newly_triggered = []

        for event_name, event_class in self.event_types.items():
            temp_event = event_class()

            # Build per-event coupling dict.
            # dt_hours and last_event_time are always present so every
            # sample_onset() gets the real step size and refractory info.
            event_coupling: Dict[str, Any] = {
                "dt_hours":       dt_hours,
                "last_event_time": self.last_event_times.get(event_name, -999.0),
            }

            if coupling_effects and event_name in coupling_effects:
                event_coupling.update(coupling_effects[event_name])

            should_occur, severity = temp_event.sample_onset(
                state=state, t=t, **event_coupling
            )

            if should_occur and severity is not None:
                event = event_class()
                event.initialize_event(
                    onset_time=self.current_time,
                    onset_index=t,
                    severity=severity,
                )

                self.last_event_times[event_name] = self.current_time

                trigger_time = self.current_time + 0.001
                self._schedule_event(trigger_time, event)

                newly_triggered.append(event)

                logger.info(
                    f"Triggered {event_name} at t={t} ({self.current_time:.1f}h)"
                )

        if newly_triggered:
            self.stats["total_events_triggered"] += len(newly_triggered)
            for event in newly_triggered:
                self.stats["events_by_type"][event.__class__.__name__] += 1

        return newly_triggered

    def _schedule_event(self, trigger_time: float, event: Event) -> None:
        """Add event to priority queue."""
        scheduled = ScheduledEvent(
            trigger_time=trigger_time,
            event=event,
            priority=event.priority.value,
        )
        heapq.heappush(self.event_queue, scheduled)

    def process_pending_events(
        self, state: Any, dt_hours: float
    ) -> List[Dict[str, Any]]:
        """
        Process all events that should be triggered at current time.
        Activates newly-ready events and applies effects of all active events.

        Returns:
            List of effect dicts from apply_effect() calls.
        """
        applied_effects = []

        # Activate events whose trigger_time has arrived
        while (
            self.event_queue
            and self.event_queue[0].trigger_time <= self.current_time
        ):
            scheduled = heapq.heappop(self.event_queue)
            event = scheduled.event
            self.active_events[event.event_id] = event
            self.stats["events_by_priority"][event.priority.name] += 1
            logger.debug(f"Event {event.event_id} activated at {self.current_time:.1f}h")

        if self.active_events:
            self.stats["max_concurrent_events"] = max(
                self.stats["max_concurrent_events"],
                len(self.active_events),
            )

            completed_events = []
            for event_id, event in self.active_events.items():
                effect = event.apply_effect(state, self.current_index, dt_hours)
                applied_effects.append(effect)

                self.stats["total_active_time"][event.__class__.__name__] += dt_hours

                if not event.is_active(self.current_time + dt_hours):
                    event.complete()
                    completed_events.append(event_id)
                    self.event_history.append(event.to_dict())
                    logger.debug(
                        f"Event {event_id} completed at {self.current_time + dt_hours:.1f}h"
                    )

            for event_id in completed_events:
                del self.active_events[event_id]

        return applied_effects

    def process_time_step(
        self,
        state: Any,
        t: int,
        dt_hours: float,
        coupling_effects: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete time step processing: check triggers and apply effects.

        Returns a summary dict.  Note: apply_effect() for ExerciseStressEvent
        now returns "fatigue_forcing" and "stress_delta" rather than writing
        them to state.  Call get_active_eva_contributions() after this method
        to retrieve those values for the main loop.
        """
        new_events = self.check_triggers(state, t, dt_hours, coupling_effects)
        effects    = self.process_pending_events(state, dt_hours)

        return {
            "time":            t,
            "simulation_time": self.current_time,
            "new_events":      [e.to_dict() for e in new_events],
            "active_events":   len(self.active_events),
            "effects_applied": len(effects),
            "queue_size":      len(self.event_queue),
            "effects":         effects,   # raw effect dicts for callers that need them
        }

    # ------------------------------------------------------------------
    # EVA contribution helper (FIX v1.2)
    # ------------------------------------------------------------------

    def get_active_eva_contributions(self) -> Dict[str, float]:
        """
        Aggregate fatigue_forcing and stress_delta from all currently active
        ExerciseStressEvents.

        Call this AFTER process_time_step() and BEFORE the stress formula and
        the engine.step() call:

            event_summary = scheduler.process_time_step(…)
            eva = scheduler.get_active_eva_contributions()

            # Wire fatigue forcing into the ODE
            phys = engine.step(…, fatigue_forcing=eva["fatigue_forcing"])

            # Fold stress delta into the formula
            total_stress = clip(
                0.12 + circadian + fat_term + ms_term + eva["stress_delta"],
                0.0, 0.95,
            )
            state.update(t, stress=total_stress)

        Returns:
            {
                "fatigue_forcing": float,   # sum of all active EVA forcing rates
                "stress_delta":    float,   # sum of all active EVA stress rates
            }
        """
        total_fatigue_forcing = 0.0
        total_stress_delta    = 0.0

        for event in self.active_events.values():
            if not isinstance(event, ExerciseStressEvent):
                continue
            if not event.effect:
                continue
            imm = event.effect.immediate
            total_fatigue_forcing += float(imm.get("fatigue_acceleration", 0.0))
            total_stress_delta    += float(imm.get("stress_delta", 0.0))

        return {
            "fatigue_forcing": total_fatigue_forcing,
            "stress_delta":    total_stress_delta,
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_active_events(self, event_type: Optional[str] = None) -> List[Event]:
        """Get currently active events, optionally filtered by class name."""
        if event_type:
            return [e for e in self.active_events.values()
                    if e.__class__.__name__ == event_type]
        return list(self.active_events.values())

    def get_upcoming_events(self, time_window: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get events scheduled in the upcoming time window.

        FIX: iterates a sorted copy of the heap rather than the raw heap list.
        heapq only guarantees the minimum is at index 0; iterating the list
        directly can return events out of trigger-time order after heappop/heappush
        operations.

        Args:
            time_window: Hours to look ahead

        Returns:
            List of upcoming event summaries, sorted by trigger_time.
        """
        cutoff_time = self.current_time + time_window
        upcoming    = []

        # Sort a copy — O(n log n) but the queue is typically tiny
        for scheduled in sorted(self.event_queue):
            if scheduled.trigger_time > cutoff_time:
                break
            upcoming.append({
                "trigger_time": scheduled.trigger_time,
                "event_type":   scheduled.event.__class__.__name__,
                "event_id":     scheduled.event.event_id,
                "priority":     scheduled.priority,
            })

        return upcoming

    def cancel_event(self, event_id: str, reason: str = "") -> bool:
        """Cancel a pending or active event."""
        if event_id in self.active_events:
            self.active_events[event_id].cancel(reason)
            del self.active_events[event_id]
            return True

        for i, scheduled in enumerate(self.event_queue):
            if scheduled.event.event_id == event_id:
                scheduled.event.cancel(reason)
                self.event_queue.pop(i)
                heapq.heapify(self.event_queue)
                return True

        return False

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get comprehensive event statistics."""
        return {
            **self.stats,
            "current_active": len(self.active_events),
            "queue_size":     len(self.event_queue),
            "history_size":   len(self.event_history),
            "active_event_types": {
                name: len([e for e in self.active_events.values()
                           if e.__class__.__name__ == name])
                for name in self.event_types
            },
        }

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.event_queue   = []
        self.active_events = {}
        self.event_history = []
        self.stats = {
            "total_events_triggered": 0,
            "events_by_type":         defaultdict(int),
            "events_by_priority":     defaultdict(int),
            "total_active_time":      defaultdict(float),
            "max_concurrent_events":  0,
        }
        self.current_time  = 0.0
        self.current_index = 0
        self.last_event_times = {}
        logger.info("EventScheduler reset")

    def register_event_type(self, name: str, event_class: Type[Event]) -> None:
        """Register a new event type."""
        if not issubclass(event_class, Event):
            raise ValueError(f"{event_class} must inherit from Event")
        self.event_types[name] = event_class
        logger.info(f"Registered event type: {name}")

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get complete event timeline for visualization."""
        timeline = []

        for event_dict in self.event_history:
            timeline.append({**event_dict, "status": "completed"})

        for event in self.active_events.values():
            timeline.append({**event.to_dict(), "status": "active"})

        for scheduled in self.event_queue:
            timeline.append({
                **scheduled.event.to_dict(),
                "trigger_time": scheduled.trigger_time,
                "status":       "queued",
            })

        timeline.sort(key=lambda x: x.get("onset_time", float("inf")))
        return timeline