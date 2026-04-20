"""
Simulation API Routes - Person 1's Core Interface
Handles simulation lifecycle: start, stop, status, and configuration.
These endpoints are called by Person 3's visualization dashboard.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid
import asyncio
import logging
import numpy as np

from core.state_manager import AstronautState
from core.probabilistic_models import ProbabilisticModels
from core.fatigue_model import FatigueModel, FatigueParameters
from core.coupling_engine import CouplingEngine, CouplingParameters
from events.event_scheduler import EventScheduler
from events.motion_sickness_event import MotionSicknessEvent, MotionSicknessParameters
from events.sleep_disruption_event import SleepDisruptionEvent, SleepDisruptionParameters
from biogears.biogears_adapter import BioGearsAdapter
from utils.logger import get_logger
from api.dependencies import get_simulation_manager, SimulationManager
from analytics.risk_engine  import compute_full_risk_report
from analytics.trend_analysis import compute_full_trend_report
from pydantic import BaseModel, Field
from analytics.monte_carlo import run_monte_carlo
from core.probabilistic_models import ProbabilisticModels as PM 

# Configure logging
logger = get_logger(__name__)

router = APIRouter(tags=["simulation"])


# =============================================================================
# PYDANTIC MODELS - These define the API contract with Person 3
# =============================================================================

class SimulationConfig(BaseModel):
    """
    Configuration for a simulation run.
    Person 3 sends this JSON to configure the simulation.
    """
    
    # Mission parameters
    mission_duration_hours: float = Field(
        720.0,
        description="Duration of simulation in hours (default: 30 days)",
        ge=1.0,
        le=8760.0  # 1 year max
    )
    time_step_minutes: float = Field(
        5.0,
        description="Time step resolution in minutes",
        ge=0.1,
        le=60.0
    )
    
    # Astronaut baseline
    astronaut_id: str = Field(
        "default",
        description="Astronaut identifier for baseline profiles"
    )
    baseline_hr: float = Field(
        75.0,
        description="Baseline heart rate (bpm)",
        ge=40.0,
        le=120.0
    )
    baseline_sleep_quality: float = Field(
        0.8,
        description="Baseline sleep quality [0-1]",
        ge=0.0,
        le=1.0
    )
    initial_fatigue: float = Field(
        0.0,
        description="Initial fatigue level [0-10]",
        ge=0.0,
        le=10.0
    )
    num_astronauts: int = Field(
        1,
        description="Number of astronauts to simulate (1-5)",
        ge=1,
        le=5
    )
    # Event enablement
    enable_motion_sickness: bool = Field(
        True,
        description="Enable motion sickness events"
    )
    enable_sleep_disruption: bool = Field(
        True,
        description="Enable sleep disruption events"
    )
    
    # BioGears integration
    use_biogears: bool = Field(
        True,
        description="Use BioGears for physiological responses"
    )
    biogears_scenario_path: Optional[str] = Field(
        None,
        description="Custom BioGears scenario file path"
    )
    
    # Output options
    save_trajectories: bool = Field(
        True,
        description="Save full state trajectories"
    )
    save_events: bool = Field(
        True,
        description="Save event logs"
    )
    
    @validator('mission_duration_hours')
    def validate_duration(cls, v):
        """Ensure duration is reasonable for simulation."""
        if v < 1:
            raise ValueError('Mission duration must be at least 1 hour')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "mission_duration_hours": 720,
                "time_step_minutes": 5,
                "astronaut_id": "astro_001",
                "baseline_hr": 72,
                "baseline_sleep_quality": 0.85,
                "enable_motion_sickness": True,
                "use_biogears": True
            }
        }


class SimulationResponse(BaseModel):
    """
    Response when starting a simulation.
    Person 3 receives this immediately after requesting a simulation.
    """
    run_id: str = Field(..., description="Unique simulation identifier")
    status: str = Field(..., description="Current status (started, queued, failed)")
    message: str = Field(..., description="Human-readable status message")
    data_url: Optional[str] = Field(None, description="URL to fetch results when complete")
    estimated_completion_time: Optional[float] = Field(None, description="Estimated seconds until completion")
    created_at: datetime = Field(default_factory=datetime.now)

# AFTER
class SimulationStatus(BaseModel):
    run_id: str
    status: str
    progress: Optional[float] = 0.0
    current_time_hours: float = 0.0
    events_triggered: int = 0
    active_events: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    time_remaining_seconds: Optional[float] = None

    class Config:
        extra = 'ignore'  # ignore extra fields from to_dict()

class SimulationListResponse(BaseModel):
    """List of available simulations."""
    runs: List[Dict[str, Any]]
    total_count: int


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/run", response_model=SimulationResponse, status_code=202)
async def run_simulation(
    config: SimulationConfig,
    background_tasks: BackgroundTasks,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    Start a new simulation run.
    
    This endpoint accepts simulation configuration and begins execution
    in the background. Returns immediately with a run_id that Person 3
    can use to poll status and fetch results.
    
    The simulation runs asynchronously to prevent blocking the API.
    
    Args:
        config: Simulation configuration from request body
        background_tasks: FastAPI background task manager
        sim_manager: Simulation manager dependency
        
    Returns:
        SimulationResponse with run_id and status URL
    """
    logger.info(f"Received simulation request: {config}")
    
    try:
        run_id = await sim_manager.create_run(config.dict())
        
        # Start simulation in background
        task = asyncio.create_task(execute_simulation(run_id, config, sim_manager))
        run = sim_manager.get_run(run_id)
        if run:
            run.task = task
        
        # Estimate completion time (rough estimate)
        timesteps = int(config.mission_duration_hours * 60 / config.time_step_minutes)
        est_completion = timesteps * 0.01  # ~0.01 seconds per timestep
        
        return SimulationResponse(
            run_id=run_id,
            status="started",
            message=f"Simulation {run_id} started successfully",
            data_url=f"/api/data/results/{run_id}",
            estimated_completion_time=est_completion,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{run_id}", response_model=SimulationStatus)
async def get_simulation_status(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    Get status of a simulation run.
    
    Person 3 can poll this endpoint to check progress and know when
    results are ready.
    
    Args:
        run_id: Simulation identifier
        sim_manager: Simulation manager dependency
        
    Returns:
        Current simulation status
    """
    status = await sim_manager.get_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")
    
    return status


@router.get("/list", response_model=SimulationListResponse)
async def list_simulations(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, pattern="^(pending|running|completed|failed)$"),
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    List all simulation runs with pagination.
    
    Args:
        limit: Maximum number of runs to return
        offset: Pagination offset
        status: Optional filter by status
        sim_manager: Simulation manager dependency
        
    Returns:
        List of simulation runs
    """
    runs = sim_manager.list_runs(limit=limit, offset=offset, status=status)
    total = sim_manager.count_runs(status=status)
    
    return SimulationListResponse(
        runs=runs,
        total_count=total
    )


@router.post("/stop/{run_id}")
async def stop_simulation(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    Stop a running simulation.
    
    Args:
        run_id: Simulation identifier
        sim_manager: Simulation manager dependency
        
    Returns:
        Confirmation message
    """
    success = await sim_manager.stop_run(run_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found or cannot be stopped")
    
    return {"message": f"Simulation {run_id} stopped", "run_id": run_id}


@router.delete("/delete/{run_id}")
async def delete_simulation(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    Delete a simulation run and its data.
    
    Args:
        run_id: Simulation identifier
        sim_manager: Simulation manager dependency
        
    Returns:
        Confirmation message
    """
    success = await sim_manager.delete_run(run_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")
    
    return {"message": f"Simulation {run_id} deleted", "run_id": run_id}


@router.get("/config/{run_id}")
async def get_simulation_config(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    Get the configuration used for a simulation run.
    
    Args:
        run_id: Simulation identifier
        sim_manager: Simulation manager dependency
        
    Returns:
        Simulation configuration
    """
    config = sim_manager.get_config(run_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")
    
    return config

class MonteCarloConfig(BaseModel):
    n_runs:                  int   = Field(50,   ge=5,  le=500)
    mission_duration_hours:  float = Field(720.0, ge=24, le=8760)
    time_step_minutes:       float = Field(30.0,  ge=5,  le=60)
    baseline_hr:             float = Field(75.0,  ge=40, le=120)
    baseline_sleep_quality:  float = Field(0.8,   ge=0,  le=1)
    initial_fatigue:         float = Field(0.0,   ge=0,  le=10)
    ms_lambda:               float = Field(0.03,  ge=0.001, le=0.5)
    risk_fatigue_threshold:  float = Field(5.0,   ge=1,  le=9)
    risk_sleep_threshold:    float = Field(0.4,   ge=0.1, le=0.8)
 
 
@router.post("/monte-carlo")
async def run_monte_carlo_endpoint(config: MonteCarloConfig):
    """
    Run a Monte Carlo batch simulation synchronously and return
    aggregated risk statistics, envelopes, and conclusions.
 
    This is intentionally synchronous (runs in ~1-3 s for n=50, 30-min steps).
    For very large n (>200) consider wrapping in a background task.
    """
    try:
        result = run_monte_carlo(
            n_runs                  = config.n_runs,
            mission_duration_hours  = config.mission_duration_hours,
            time_step_minutes       = config.time_step_minutes,
            baseline_hr             = config.baseline_hr,
            baseline_sleep_quality  = config.baseline_sleep_quality,
            initial_fatigue         = config.initial_fatigue,
            ms_lambda               = config.ms_lambda,
            risk_fatigue_threshold  = config.risk_fatigue_threshold,
            risk_sleep_threshold    = config.risk_sleep_threshold,
        )
        return result
    except Exception as e:
        logger.error(f"Monte Carlo failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/sensitivity/{variable}")
async def get_sensitivity_analysis(
    variable:   str,
    timesteps:  int   = 288,   # 24 hours at 5-min steps
    dt_minutes: float = 5.0,
):
    """
    Return multi-profile sensitivity analysis for a physiological variable.
    Supported variables: heart_rate, sleep_quality
    """
    if variable not in ("heart_rate", "sleep_quality"):
        raise HTTPException(
            status_code=400,
            detail="variable must be 'heart_rate' or 'sleep_quality'"
        )
    try:
        pm     = PM()
        result = pm.generate_sensitivity_profiles(
            variable   = variable,
            timesteps  = timesteps,
            dt_minutes = dt_minutes,
        )
        # Add time axis
        result["time_minutes"] = [i * dt_minutes for i in range(timesteps)]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# BACKGROUND SIMULATION EXECUTION
# =============================================================================

async def execute_simulation(
    run_id:      str,
    config,            # SimulationConfig
    sim_manager,       # SimulationManager
):
    logger.info(f"Starting simulation {run_id} ({config.num_astronauts} astronaut(s))")
 
    try:
        await sim_manager.update_status(run_id, "running", progress=0)
 
        timesteps   = int(config.mission_duration_hours * 60 / config.time_step_minutes)
        dt_hours    = config.time_step_minutes / 60.0
        n_astronauts = getattr(config, "num_astronauts", 1)
 
        # Shared BioGears adapter (one instance, shared across astronauts)
        biogears = BioGearsAdapter() if config.use_biogears else None
 
        all_states     = []
        all_events     = []
        all_statistics = []
 
        for astro_idx in range(n_astronauts):
            # Slight per-astronaut parameter variation (±5% noise on baseline)
            rng_seed   = astro_idx * 7919   # deterministic, distinct per astronaut
            rng        = np.random.default_rng(rng_seed)
            hr_offset  = float(rng.normal(0, 3))
            slp_noise  = float(rng.normal(0, 0.05))
 
            state = AstronautState(
                timesteps             = timesteps,
                dt_minutes            = config.time_step_minutes,
                baseline_hr           = config.baseline_hr + hr_offset,
                baseline_sleep_quality= float(np.clip(config.baseline_sleep_quality + slp_noise, 0.1, 1.0)),
                initial_fatigue       = config.initial_fatigue,
            )
 
            prob_models    = ProbabilisticModels()
            fatigue_model  = FatigueModel()
            coupling_engine= CouplingEngine()
            scheduler      = EventScheduler()
 
            # Baseline trajectories
            time_hours = np.arange(timesteps) * dt_hours
            circadian  = 5.0 * np.sin(2 * np.pi * time_hours / 24.0)
            state.hr[:]            = np.clip(prob_models.sample_heart_rate(size=timesteps) + circadian, 40, 200)
            state.sleep_quality[:] = np.clip(prob_models.sample_sleep_quality(size=timesteps), 0.05, 1.0)
 
            # Main simulation loop
            for t in range(timesteps):
                progress = ((astro_idx * timesteps + t) / (n_astronauts * timesteps)) * 100
                if t % 100 == 0:
                    await sim_manager.update_status(run_id, "running", progress=progress)
 
                coupling_effects = {
                    'motion_sickness': {
                        'fatigue_multiplier': coupling_engine.compute_fatigue_effect_on_ms(
                            base_probability=1.0,
                            fatigue_level=float(state.fatigue[t - 1]) if t > 0 else 0,
                        )[0]
                    }
                }
 
                event_summary = scheduler.process_time_step(
                    state=state, t=t, dt_hours=dt_hours,
                    coupling_effects=coupling_effects
                )
 
                new_events = event_summary.get('new_events', [])
                sms_events = [e for e in new_events if 'motion' in e.get('type', '').lower()]
                sms_severity = max((e.get('severity', 0) for e in sms_events), default=0.0)
 
                current_fat = float(state.fatigue[t - 1]) if t > 0 else config.initial_fatigue
                t_h = float(state.time[t]) / 60.0
                import math
                circadian_stress = 0.08 + 0.06 * math.sin(2 * math.pi * (t_h % 24) / 24 - math.pi / 2)
                fatigue_stress   = min(0.45, (current_fat / 10.0) * 0.60)
                acute_stress     = min(0.50, float(sms_severity) * 0.70)
                total_stress     = min(0.95, 0.12 + circadian_stress + fatigue_stress + acute_stress)
                state.update(t, stress=total_stress)
 
                if t > 0:
                    mot_sev = float(state.motion_severity[t])
                    new_fatigue, _ = fatigue_model.compute_fatigue_update(
                        current_fatigue=state.fatigue[t - 1],
                        sleep_quality=state.sleep_quality[t],
                        motion_severity=mot_sev,
                        dt_hours=dt_hours,
                    )
                    state.update(t, fatigue=new_fatigue)
 
                if config.use_biogears and biogears:
                    for event in event_summary.get('new_events', []):
                        if event['type'] == 'MotionSicknessEvent':
                            perturbation = {
                                'type':             'motion_sickness',
                                'nausea_severity':  event.get('severity', 0.3),
                                'duration_minutes': event.get('duration', 10.0) * 60,
                                'baseline_hr':      config.baseline_hr + hr_offset,
                                'fatigue_level':    float(state.fatigue[t - 1]) if t > 0 else 0.0,
                            }
                            bio_response = await biogears.run_perturbation_async(perturbation)
                            if bio_response:
                                state.update(t, hr=min(float(bio_response.get('hr', state.hr[t])), 160.0))
 
            # ── Per-astronaut analytics ────────────────────────────────────
            state_dict     = state.to_dict()
            event_timeline = scheduler.get_timeline()
            risk_report    = compute_full_risk_report(state=state_dict, events=event_timeline)
            trend_report   = compute_full_trend_report(state=state_dict, events=event_timeline)
            risk_trace     = risk_report.pop("risk_score_trace", [])
 
            state_dict["_astronaut_id"]    = astro_idx
            state_dict["_risk_report"]     = risk_report
            state_dict["_trend_report"]    = trend_report
            state_dict["_risk_trace"]      = risk_trace
 
            all_states.append(state_dict)
            all_events.append(event_timeline)
            all_statistics.append(scheduler.get_event_statistics())
 
        # ── Aggregate across astronauts ────────────────────────────────────
        peak_fatigues    = [s.get("_risk_report", {}).get("threshold_metrics", {}).get("fatigue", {}).get("peak", 0) for s in all_states]
        overall_risk     = "CRITICAL" if any(s.get("_risk_report", {}).get("overall_risk_level") == "CRITICAL" for s in all_states) else \
                           "HIGH"     if any(s.get("_risk_report", {}).get("overall_risk_level") == "HIGH"     for s in all_states) else "MODERATE"
 
        final_status = {
            "run_id":          run_id,
            "status":          "completed",
            "progress":        100,
            "completed_at":    datetime.now(),
            "events_triggered": sum(s.get("total_events_triggered", 0) for s in all_statistics),
            "metrics": {
                "n_astronauts":         n_astronauts,
                "peak_fatigue":         float(max(peak_fatigues)) if peak_fatigues else 0.0,
                "avg_sleep_quality":    float(np.mean([np.mean(s["sleep_quality"]) for s in all_states])),
                "overall_risk_level":   overall_risk,
            }
        }
 
        # Use first astronaut's state as primary for backward-compat
        primary_state = all_states[0] if all_states else {}
 
        await sim_manager.store_results(
            run_id       = run_id,
            state        = primary_state,
            events       = all_events[0] if all_events else [],
            statistics   = all_statistics[0] if all_statistics else {},
            final_status = final_status,
        )
 
        # Store analytics for primary astronaut
        primary_risk  = primary_state.pop("_risk_report",  {})
        primary_trend = primary_state.pop("_trend_report", {})
        primary_trace = primary_state.pop("_risk_trace",   [])
        # Clean up helper keys
        for s in all_states:
            s.pop("_astronaut_id", None)
            s.pop("_risk_report",  None)
            s.pop("_trend_report", None)
            s.pop("_risk_trace",   None)
 
        await sim_manager.store_analytics(
            run_id       = run_id,
            risk_report  = primary_risk,
            trend_report = primary_trend,
            risk_trace   = primary_trace,
        )
 
        logger.info(f"Simulation {run_id} completed — {n_astronauts} astronaut(s), risk={overall_risk}")
 
    except Exception as e:
        logger.error(f"Simulation {run_id} failed: {e}", exc_info=True)
        await sim_manager.update_status(run_id, "failed", error_message=str(e))
        raise