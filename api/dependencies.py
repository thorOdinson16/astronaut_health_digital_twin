"""
API Dependencies and Shared Managers
Provides dependency injection for FastAPI routes and manages simulation state.
This is the backbone of the API layer, handling:
- Simulation run management
- Result storage and retrieval
- Configuration loading
- Shared resources and locks
"""

from fastapi import Request, HTTPException, Depends
from typing import Dict, Optional, List, Any, Union
from datetime import datetime
import asyncio
import threading
import json
import yaml
import os
import shutil
import pickle
from pathlib import Path
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import numpy as np
import uuid
import zipfile
import io

from utils.logger import get_logger
from utils.helpers import ensure_directory, timeit

logger = get_logger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# =============================================================================
# DATA CLASSES FOR SIMULATION MANAGEMENT
# =============================================================================

@dataclass
class SimulationRun:
    """
    Represents a single simulation run with all associated data.
    This is the in-memory representation of a simulation.
    """
    run_id: str
    config: Dict[str, Any]
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    results: Optional[Dict[str, Any]] = None
    events: Optional[List[Dict[str, Any]]] = None
    statistics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    task: Optional[asyncio.Task] = None
    
    def update(self, **kwargs):
        """Update run attributes and timestamp."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'config': self.config,
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'last_updated': self.last_updated.isoformat(),
            'error_message': self.error_message
        }


class SimulationManager:
    """
    Central manager for all simulation runs.
    
    Responsibilities:
    - Store and retrieve simulation configurations
    - Track running, completed, and failed simulations
    - Persist results to disk
    - Clean up old runs
    - Provide thread-safe access to simulation data
    
    This is a singleton that lives for the lifetime of the API server.
    """
    
    def __init__(self, storage_path: str = "./output/simulations"):
        """
        Initialize the simulation manager.
        
        Args:
            storage_path: Path to store simulation results on disk
        """
        self.storage_path = Path(storage_path)
        ensure_directory(self.storage_path)
        
        # In-memory storage for active simulations
        self.runs: Dict[str, SimulationRun] = {}
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        self._file_lock = threading.RLock()
        
        # Load any existing runs from disk
        self._load_existing_runs()
        
        # Start background cleanup task
        self.cleanup_task = None
        
        logger.info(f"SimulationManager initialized with storage path: {storage_path}")
    
    def _load_existing_runs(self):
        """Load previously saved simulation runs from disk."""
        try:
            metadata_file = self.storage_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    runs_data = json.load(f)
                
                for run_id, data in runs_data.items():
                    # Only load completed/failed runs, not pending/running
                    if data['status'] in ['completed', 'failed']:
                        run = SimulationRun(
                            run_id=run_id,
                            config=data['config'],
                            status=data['status'],
                            progress=100.0 if data['status'] == 'completed' else data['progress'],
                            created_at=datetime.fromisoformat(data['created_at']),
                            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
                            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
                            last_updated=datetime.fromisoformat(data['last_updated']),
                            error_message=data.get('error_message')
                        )
                        self.runs[run_id] = run
                        
            logger.info(f"Loaded {len(self.runs)} existing runs from disk")
        except Exception as e:
            logger.error(f"Failed to load existing runs: {e}")
    
    async def _save_metadata(self):
        try:
            runs_data = {run_id: run.to_dict() for run_id, run in self.runs.items()}
            metadata_file = self.storage_path / "metadata.json"
            with self._file_lock:
                with open(metadata_file, 'w') as f:
                    json.dump(runs_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def create_run(self, config: Dict[str, Any]) -> str:
        """
        Create a new simulation run.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Unique run ID
        """
        run_id = f"sim_{uuid.uuid4().hex[:8]}"
        
        run = SimulationRun(
            run_id=run_id,
            config=config,
            status="pending"
        )
        
        async with self._lock:
            self.runs[run_id] = run
        
        await self._save_metadata()
        logger.info(f"Created new simulation run: {run_id}")
        
        return run_id
    
    async def update_run(self, run_id: str, **kwargs):
        async with self._lock:
            if run_id in self.runs:
                self.runs[run_id].update(**kwargs)
            else:
                return False
        await self._save_metadata()
        return True
    
    def get_run(self, run_id: str) -> Optional[SimulationRun]:
        """Get a simulation run by ID."""
        return self.runs.get(run_id)
    
    async def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a simulation run."""
        run = self.get_run(run_id)
        if not run:
            return None
        
        status_dict = run.to_dict()
        
        # Add additional status info
        if run.status == "running" and run.task:
            status_dict['task_done'] = run.task.done()
        
        return status_dict
    
    def store_config(self, run_id: str, config: Dict[str, Any]):
        """Store configuration for a run."""
        if run_id in self.runs:
            self.runs[run_id].config = config
    
    def get_config(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a run."""
        run = self.get_run(run_id)
        return run.config if run else None
    
    async def store_results(
        self,
        run_id: str,
        state: Dict[str, Any],
        events: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        final_status: Dict[str, Any]
    ):
        """
        Store simulation results.
        
        Args:
            run_id: Simulation identifier
            state: State trajectories
            events: Event timeline
            statistics: Event statistics
            final_status: Final status information
        """
        async with self._lock:
            if run_id not in self.runs:
                raise ValueError(f"Run ID {run_id} not found")
            
            run = self.runs[run_id]
            run.results = {
                'state': state,
                'events': events,
                'statistics': statistics
            }
            run.status = final_status.get('status', 'completed')
            run.completed_at = final_status.get('completed_at', datetime.now())
            run.progress = 100
            
            # Save results to disk
            await self._save_results_to_disk(run_id, state, events, statistics)
        
        await self._save_metadata()
        logger.info(f"Stored results for run {run_id}")
    
    async def _save_results_to_disk(
        self,
        run_id: str,
        state: Dict[str, Any],
        events: List[Dict[str, Any]],
        statistics: Dict[str, Any]
    ):
        """Save results to disk files."""
        run_dir = self.storage_path / run_id
        ensure_directory(run_dir)

        with self._file_lock:
            state_file = run_dir / "state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, cls=NumpyEncoder)

            events_file = run_dir / "events.json"
            with open(events_file, 'w') as f:
                json.dump(events, f, indent=2, cls=NumpyEncoder)

            stats_file = run_dir / "statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(statistics, f, indent=2, cls=NumpyEncoder)

            summary = {
                'run_id': run_id,
                'completed_at': datetime.now().isoformat(),
                'metrics': {
                    'peak_fatigue': float(max(state['fatigue'])) if state.get('fatigue') else None,
                    'avg_hr': float(np.mean(state['hr'])) if state.get('hr') else None,
                    'event_count': len(events)
                }
            }
            summary_file = run_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
    async def get_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a simulation run.
        
        Args:
            run_id: Simulation identifier
            
        Returns:
            Complete results dictionary or None if not found
        """
        run = self.get_run(run_id)
        if not run:
            return None
        
        # If results are in memory, return them
        if run.results:
            return run.results
        
        # Otherwise, try to load from disk
        run_dir = self.storage_path / run_id
        if not run_dir.exists():
            return None
        
        try:
            with self._file_lock:
                state_file = run_dir / "state.json"
                events_file = run_dir / "events.json"
                stats_file = run_dir / "statistics.json"
                
                results = {}
                
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        results['state'] = json.load(f)
                
                if events_file.exists():
                    with open(events_file, 'r') as f:
                        results['events'] = json.load(f)
                
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        results['statistics'] = json.load(f)
                
                # Cache in memory
                run.results = results
                
                return results
        except Exception as e:
            logger.error(f"Failed to load results for {run_id}: {e}")
            return None
    
    async def update_status(
        self,
        run_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Update simulation status."""
        await self.update_run(
            run_id,
            status=status,
            progress=progress,
            error_message=error_message,
            last_updated=datetime.now()
        )
    
    def list_runs(
        self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List simulation runs with pagination.
        
        Args:
            limit: Maximum number of runs to return
            offset: Pagination offset
            status: Optional status filter
            
        Returns:
            List of run summaries
        """
        runs_list = []
        
        for run_id, run in self.runs.items():
            if status and run.status != status:
                continue
            
            runs_list.append(run.to_dict())
        
        # Sort by creation date (newest first)
        runs_list.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Apply pagination
        return runs_list[offset:offset + limit]
    
    def count_runs(self, status: Optional[str] = None) -> int:
        """Count simulation runs."""
        if status:
            return sum(1 for run in self.runs.values() if run.status == status)
        return len(self.runs)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics about all runs."""
        active = sum(1 for run in self.runs.values() if run.status == 'running')
        completed = sum(1 for run in self.runs.values() if run.status == 'completed')
        failed = sum(1 for run in self.runs.values() if run.status == 'failed')
        pending = sum(1 for run in self.runs.values() if run.status == 'pending')
        
        return {
            'active_runs': active,
            'completed_runs': completed,
            'failed_runs': failed,
            'pending_runs': pending,
            'total_runs': len(self.runs),
            'queue_size': pending
        }
    
    async def stop_run(self, run_id: str) -> bool:
        """Stop a running simulation."""
        async with self._lock:
            run = self.runs.get(run_id)
            if not run or run.status != 'running':
                return False
            if run.task and not run.task.done():
                run.task.cancel()
            run.status = 'failed'
            run.error_message = 'Manually stopped'
            run.completed_at = datetime.now()
        await self._save_metadata()
        return True
    
    async def delete_run(self, run_id: str) -> bool:
        """Delete a simulation run and its data."""
        async with self._lock:
            if run_id not in self.runs:
                return False
            run = self.runs[run_id]
            if run.status == 'running' and run.task and not run.task.done():
                run.task.cancel()
            del self.runs[run_id]
            run_dir = self.storage_path / run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
        await self._save_metadata()
        return True
        
    async def create_download_package(self, run_id: str) -> str:
        """
        Create a ZIP package with all results for a run.
        
        Args:
            run_id: Simulation identifier
            
        Returns:
            Path to created ZIP file
        """
        run_dir = self.storage_path / run_id
        if not run_dir.exists():
            raise ValueError(f"No data found for run {run_id}")
        
        zip_path = self.storage_path / f"{run_id}_results.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in run_dir.glob("*"):
                zipf.write(file_path, arcname=file_path.name)
        
        return str(zip_path)
    
    async def cleanup_old_runs(self, max_age_days: int = 30):
        """Clean up runs older than max_age_days."""
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        to_delete = []
        async with self._lock:
            for run_id, run in self.runs.items():
                if run.completed_at and run.completed_at.timestamp() < cutoff:
                    to_delete.append(run_id)
        for run_id in to_delete:
            await self.delete_run(run_id)
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old runs")
    
    async def start_cleanup_task(self, interval_hours: int = 24):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval_hours * 3600)
                await self.cleanup_old_runs()
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started cleanup task (interval: {interval_hours}h)")


class ConfigLoader:
    """
    Loads and caches configuration files.
    Provides access to simulation and distribution configurations.
    """
    
    def __init__(self, config_path: str = "./config"):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = Path(config_path)
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        logger.info(f"ConfigLoader initialized with path: {config_path}")
    
    async def load_simulation_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load simulation configuration.
        
        Args:
            reload: Force reload from disk
            
        Returns:
            Simulation configuration dictionary
        """
        cache_key = "simulation_config"
        
        if not reload and cache_key in self._cache:
            return self._cache[cache_key]
        
        async with self._lock:
            config_file = self.config_path / "simulation_config.yaml"
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self._cache[cache_key] = config
            logger.info("Loaded simulation configuration")
            return config
    
    async def load_distributions_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load distributions configuration.
        
        Args:
            reload: Force reload from disk
            
        Returns:
            Distributions configuration dictionary
        """
        cache_key = "distributions_config"
        
        if not reload and cache_key in self._cache:
            return self._cache[cache_key]
        
        async with self._lock:
            config_file = self.config_path / "distributions.yaml"
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self._cache[cache_key] = config
            logger.info("Loaded distributions configuration")
            return config
    
    async def get_distribution_params(self, dist_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific distribution.
        
        Args:
            dist_name: Name of distribution (e.g., 'heart_rate', 'sleep_quality')
            
        Returns:
            Distribution parameters with justification
        """
        config = await self.load_distributions_config()
        
        if dist_name not in config:
            raise KeyError(f"Distribution '{dist_name}' not found in config")
        
        return config[dist_name]
    
    def clear_cache(self):
        """Clear configuration cache."""
        self._cache.clear()
        logger.info("Cleared config cache")


# =============================================================================
# DEPENDENCY FACTORIES
# =============================================================================

# Global instances (singletons)
_simulation_manager: Optional[SimulationManager] = None
_config_loader: Optional[ConfigLoader] = None
_manager_lock = asyncio.Lock()


async def get_simulation_manager() -> SimulationManager:
    """
    Dependency to get the global SimulationManager instance.
    
    This creates the manager on first call and returns the same instance
    for all subsequent calls (singleton pattern).
    
    Returns:
        SimulationManager instance
    """
    global _simulation_manager
    
    if _simulation_manager is None:
        async with _manager_lock:
            if _simulation_manager is None:
                storage_path = os.getenv("SIMULATION_STORAGE_PATH", "./output/simulations")
                _simulation_manager = SimulationManager(storage_path=storage_path)
                await _simulation_manager.start_cleanup_task()
    
    return _simulation_manager


async def get_config_loader() -> ConfigLoader:
    """
    Dependency to get the global ConfigLoader instance.
    
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None:
        async with _manager_lock:
            if _config_loader is None:
                config_path = os.getenv("CONFIG_PATH", "./config")
                _config_loader = ConfigLoader(config_path=config_path)
    
    return _config_loader


# =============================================================================
# REQUEST-SPECIFIC DEPENDENCIES
# =============================================================================

async def get_run_from_path(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
) -> SimulationRun:
    """
    Dependency to get a simulation run from the path parameter.
    
    This validates that the run exists and raises a 404 if not.
    
    Args:
        run_id: Run ID from path
        sim_manager: Simulation manager dependency
        
    Returns:
        SimulationRun object
        
    Raises:
        HTTPException: 404 if run not found
    """
    run = sim_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


async def get_completed_run(
    run: SimulationRun = Depends(get_run_from_path)
) -> SimulationRun:
    """
    Dependency to ensure a run is completed.
    
    Args:
        run: Simulation run from path
        
    Returns:
        SimulationRun if completed
        
    Raises:
        HTTPException: 400 if run not completed
    """
    if run.status != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Run {run.run_id} is {run.status}, not completed"
        )
    return run


async def get_running_run(
    run: SimulationRun = Depends(get_run_from_path)
) -> SimulationRun:
    """
    Dependency to ensure a run is currently running.
    
    Args:
        run: Simulation run from path
        
    Returns:
        SimulationRun if running
        
    Raises:
        HTTPException: 400 if run not running
    """
    if run.status != 'running':
        raise HTTPException(
            status_code=400,
            detail=f"Run {run.run_id} is {run.status}, not running"
        )
    return run


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan context manager.
    
    This handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up API server...")
    
    # Initialize dependencies
    sim_manager = await get_simulation_manager()
    config_loader = await get_config_loader()
    
    logger.info("API server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    
    # Cleanup tasks
    if sim_manager.cleanup_task:
        sim_manager.cleanup_task.cancel()
        try:
            await sim_manager.cleanup_task
        except asyncio.CancelledError:
            pass
    
    logger.info("API server shutdown complete")


# =============================================================================
# UTILITY DEPENDENCIES
# =============================================================================

async def get_request_id(request: Request) -> str:
    """
    Get or generate a request ID for tracing.
    
    Args:
        request: FastAPI request
        
    Returns:
        Request ID string
    """
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = f"req_{uuid.uuid4().hex[:8]}"
    return request_id


async def get_client_info(request: Request) -> Dict[str, str]:
    """
    Get client information from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Dictionary with client info
    """
    return {
        "client_host": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "request_id": await get_request_id(request)
    }


# =============================================================================
# CONFIGURATION DEPENDENCIES
# =============================================================================

async def get_simulation_config(
    config_loader: ConfigLoader = Depends(get_config_loader)
) -> Dict[str, Any]:
    """
    Dependency to get simulation configuration.
    
    Returns:
        Simulation configuration dictionary
    """
    return await config_loader.load_simulation_config()


async def get_distributions_config(
    config_loader: ConfigLoader = Depends(get_config_loader)
) -> Dict[str, Any]:
    """
    Dependency to get distributions configuration.
    
    Returns:
        Distributions configuration dictionary
    """
    return await config_loader.load_distributions_config()


# =============================================================================
# HEALTH CHECK DEPENDENCIES
# =============================================================================

async def verify_biogears_available() -> bool:
    """
    Dependency to verify BioGears is available.
    
    Returns:
        True if available
        
    Raises:
        HTTPException: 503 if BioGears not available
    """
    try:
        from ..biogears.biogears_adapter import BioGearsAdapter
        adapter = BioGearsAdapter()
        if adapter.check_available():
            return True
    except:
        pass
    
    raise HTTPException(
        status_code=503,
        detail="BioGears is not available"
    )


async def verify_disk_space(min_gb: float = 1.0) -> bool:
    """
    Dependency to verify sufficient disk space.
    
    Args:
        min_gb: Minimum required GB
        
    Returns:
        True if sufficient space
        
    Raises:
        HTTPException: 507 if insufficient space
    """
    try:
        import psutil
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < min_gb:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient disk space: {free_gb:.1f} GB free, need {min_gb} GB"
            )
        return True
    except:
        # If we can't check, assume it's ok
        return True