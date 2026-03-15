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
from biogears.biogears_adapter import BioGearsAdapter  # Person 2's code
from utils.logger import get_logger
from api.dependencies import get_simulation_manager, SimulationManager

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
    progress: float
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
        background_tasks.add_task(
            execute_simulation,
            run_id=run_id,
            config=config,
            sim_manager=sim_manager
        )
        
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
    status: Optional[str] = Query(None, regex="^(pending|running|completed|failed)$"),
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
    success = sim_manager.stop_run(run_id)
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
    success = sim_manager.delete_run(run_id)
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


# =============================================================================
# BACKGROUND SIMULATION EXECUTION
# =============================================================================

async def execute_simulation(
    run_id: str,
    config: SimulationConfig,
    sim_manager: SimulationManager
):
    """
    Execute simulation in background.
    
    This is the core simulation loop that Person 1 owns.
    It integrates all the modules we've built.
    
    Args:
        run_id: Unique simulation identifier
        config: Simulation configuration
        sim_manager: Simulation manager for storing results
    """
    logger.info(f"Starting simulation {run_id}")
    
    try:
        # Update status
        sim_manager.update_status(run_id, "running", progress=0)
        
        # =====================================================================
        # 1. INITIALIZE ALL COMPONENTS
        # =====================================================================
        
        # Calculate timesteps
        timesteps = int(config.mission_duration_hours * 60 / config.time_step_minutes)
        dt_hours = config.time_step_minutes / 60.0
        
        # Initialize state manager
        state = AstronautState(
            timesteps=timesteps,
            dt_minutes=config.time_step_minutes,
            baseline_hr=config.baseline_hr,
            baseline_sleep_quality=config.baseline_sleep_quality,
            initial_fatigue=config.initial_fatigue
        )
        
        # Initialize probabilistic models
        prob_models = ProbabilisticModels()
        
        # Initialize fatigue model
        fatigue_params = FatigueParameters()
        fatigue_model = FatigueModel(params=fatigue_params)
        
        # Initialize coupling engine
        coupling_params = CouplingParameters()
        coupling_engine = CouplingEngine(params=coupling_params)
        
        # Initialize event scheduler
        scheduler = EventScheduler()
        
        # Initialize BioGears adapter (Person 2's code)
        biogears = BioGearsAdapter() if config.use_biogears else None
        
        # =====================================================================
        # 2. GENERATE BASELINE TRAJECTORIES
        # =====================================================================
        
        # Generate baseline heart rate with circadian variation
        time_hours = np.arange(timesteps) * dt_hours
        circadian = 5.0 * np.sin(2 * np.pi * time_hours / 24.0)  # 5 bpm variation
        baseline_hr = prob_models.sample_heart_rate(size=timesteps) + circadian
        state.hr = baseline_hr
        
        # Generate baseline sleep quality (will be modified by events)
        baseline_sleep = prob_models.sample_sleep_quality(size=timesteps)
        state.sleep_quality = baseline_sleep
        
        # =====================================================================
        # 3. MAIN SIMULATION LOOP
        # =====================================================================
        
        for t in range(timesteps):
            # Update progress (0-100%)
            progress = (t / timesteps) * 100
            if t % 100 == 0:  # Update every 100 steps to reduce overhead
                await sim_manager.update_status(run_id, "running", progress=progress)
            
            # Get coupling effects for this timestep
            coupling_effects = {
                'motion_sickness': {
                    'fatigue_multiplier': coupling_engine.compute_fatigue_effect_on_ms(
                        base_probability=1.0,
                        fatigue_level=state.fatigue[t-1] if t > 0 else 0
                    )[0]
                }
            }
            
            # Check for and process events
            event_summary = scheduler.process_time_step(
                state=state,
                t=t,
                dt_hours=dt_hours,
                coupling_effects=coupling_effects
            )
            
            # If motion sickness event occurred and BioGears is enabled,
            # call Person 2's code for detailed physiology
            if config.use_biogears and biogears:
                for event in event_summary.get('new_events', []):
                    if event['type'] == 'MotionSicknessEvent':
                        # Get perturbation from event
                        event_obj = scheduler.get_active_events('MotionSicknessEvent')[0]
                        perturbation = event_obj.get_biogears_perturbation()
                        
                        # Call Person 2's BioGears adapter
                        bio_response = await biogears.run_perturbation_async(
                            perturbation=perturbation
                        )
                        
                        # Update state with BioGears response
                        if bio_response:
                            state.update(t, hr=bio_response.get('hr', state.hr[t]))
            
            # Update fatigue using our model
            if t > 0:
                new_fatigue, components = fatigue_model.compute_fatigue_update(
                    current_fatigue=state.fatigue[t-1],
                    sleep_quality=state.sleep_quality[t],
                    motion_severity=state.motion_severity[t],
                    dt_hours=dt_hours
                )
                state.update(t, fatigue=new_fatigue)
        
        # =====================================================================
        # 4. STORE RESULTS
        # =====================================================================
        
        # Get final status
        final_status = {
            "run_id": run_id,
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now(),
            "events_triggered": scheduler.stats['total_events_triggered'],
            "metrics": {
                "peak_fatigue": float(np.max(state.fatigue)),
                "avg_sleep_quality": float(np.mean(state.sleep_quality)),
                "motion_sickness_count": scheduler.stats['events_by_type'].get('MotionSicknessEvent', 0),
                "sleep_disruption_count": scheduler.stats['events_by_type'].get('SleepDisruptionEvent', 0)
            }
        }
        
        # Store results
        await sim_manager.store_results(
            run_id=run_id,
            state=state.to_dict(),
            events=scheduler.get_timeline(),
            statistics=scheduler.get_event_statistics(),
            final_status=final_status
        )
        
        logger.info(f"Simulation {run_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Simulation {run_id} failed: {e}", exc_info=True)
        await sim_manager.update_status(run_id, "failed", error_message=str(e))
        raise