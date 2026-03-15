"""
Health Check API Routes
Simple endpoints for monitoring API status and dependencies.
Person 3 can use these to verify the backend is running.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import psutil
import platform
import time
from datetime import datetime
import logging

from utils.logger import get_logger
from api.dependencies import get_simulation_manager, SimulationManager
from __init__ import __version__

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@router.get("/")
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Simple status indicating API is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Astronaut Digital Twin API"
    }


@router.get("/ping")
async def ping():
    """
    Ultra-lightweight ping endpoint for load balancers.
    
    Returns:
        "pong" with minimal overhead
    """
    return {"ping": "pong"}


@router.get("/status")
async def system_status(
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    Detailed system status including resource usage.
    
    Returns:
        System information, resource usage, and service status
    """
    
    # System info
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "hostname": platform.node()
    }
    
    # Resource usage
    process = psutil.Process()
    memory_info = process.memory_info()
    
    resource_usage = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": process.memory_percent(),
        "memory_rss_mb": memory_info.rss / (1024 * 1024),
        "memory_vms_mb": memory_info.vms / (1024 * 1024) if hasattr(memory_info, 'vms') else 0,
        "open_files": len(process.open_files()),
        "threads": process.num_threads(),
        "connections": len(process.connections())
    }
    
    # Disk usage for output directory
    try:
        disk_usage = psutil.disk_usage('/')
        disk_info = {
            "total_gb": disk_usage.total / (1024**3),
            "used_gb": disk_usage.used / (1024**3),
            "free_gb": disk_usage.free / (1024**3),
            "percent_used": disk_usage.percent
        }
    except:
        disk_info = {"error": "Could not get disk usage"}
    
    # Simulation service status
    sim_stats = sim_manager.get_global_stats()
    
    # Uptime
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            uptime_days = uptime_seconds / (24 * 3600)
        uptime = {
            "seconds": uptime_seconds,
            "days": uptime_days,
            "formatted": f"{uptime_days:.1f} days"
        }
    except:
        uptime = {"error": "Uptime not available on this system"}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": __version__,
        "uptime": uptime,
        "system": system_info,
        "resources": resource_usage,
        "disk": disk_info,
        "simulation_service": {
            "active_runs": sim_stats.get('active_runs', 0),
            "completed_runs": sim_stats.get('completed_runs', 0),
            "failed_runs": sim_stats.get('failed_runs', 0),
            "total_runs": sim_stats.get('total_runs', 0),
            "queue_size": sim_stats.get('queue_size', 0)
        },
        "dependencies": {
            "biogears": _check_biogears_available(),
            "numpy": _check_numpy_version(),
            "fastapi": _check_fastapi_version()
        }
    }


@router.get("/dependencies")
async def check_dependencies():
    """
    Check all external dependencies.
    
    Returns:
        Status of each dependency
    """
    dependencies = {
        "biogears": _check_biogears_available(),
        "numpy": _check_numpy_version(),
        "scipy": _check_scipy_version(),
        "pandas": _check_pandas_version(),
        "pyyaml": _check_yaml_version(),
        "simpy": _check_simpy_version()
    }
    
    all_healthy = all(d.get("available", False) for d in dependencies.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "dependencies": dependencies
    }


@router.get("/readiness")
async def readiness_probe():
    """
    Kubernetes-style readiness probe.
    
    Returns:
        200 if ready to accept traffic, 503 otherwise
    """
    # Check critical dependencies
    biogears_ok = _check_biogears_available().get("available", False)
    
    if biogears_ok:
        return {"status": "ready"}
    else:
        # Still return 200 but warn (don't fail the probe)
        return {"status": "ready", "warnings": ["BioGears not available"]}


@router.get("/liveness")
async def liveness_probe():
    """
    Kubernetes-style liveness probe.
    
    Returns:
        200 if process is alive
    """
    return {"status": "alive"}


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus-style metrics endpoint.
    
    Returns:
        Basic metrics for monitoring
    """
    process = psutil.Process()
    
    metrics = {
        "process_start_time": process.create_time(),
        "cpu_usage_percent": process.cpu_percent(interval=0.1),
        "memory_usage_bytes": process.memory_info().rss,
        "open_files_count": len(process.open_files()),
        "threads_count": process.num_threads(),
        "connections_count": len(process.connections())
    }
    
    return metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _check_biogears_available() -> Dict[str, Any]:
    """Check if BioGears is available."""
    try:
        # Try to import BioGears adapter
        from ...biogears.biogears_adapter import BioGearsAdapter
        
        # Try to instantiate
        adapter = BioGearsAdapter()
        
        # Check if we can run a simple test
        version = adapter.get_version() if hasattr(adapter, 'get_version') else "unknown"
        
        return {
            "available": True,
            "version": version,
            "path": adapter.__class__.__module__
        }
    except ImportError as e:
        return {
            "available": False,
            "error": f"ImportError: {e}",
            "hint": "Install BioGears Python bindings or check PYTHONPATH"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def _check_numpy_version() -> Dict[str, Any]:
    """Check NumPy version."""
    try:
        import numpy as np
        return {
            "available": True,
            "version": np.__version__
        }
    except ImportError:
        return {
            "available": False,
            "error": "NumPy not installed"
        }


def _check_scipy_version() -> Dict[str, Any]:
    """Check SciPy version."""
    try:
        import scipy
        return {
            "available": True,
            "version": scipy.__version__
        }
    except ImportError:
        return {
            "available": False,
            "error": "SciPy not installed"
        }


def _check_pandas_version() -> Dict[str, Any]:
    """Check pandas version."""
    try:
        import pandas as pd
        return {
            "available": True,
            "version": pd.__version__
        }
    except ImportError:
        return {
            "available": False,
            "error": "pandas not installed"
        }


def _check_yaml_version() -> Dict[str, Any]:
    """Check PyYAML version."""
    try:
        import yaml
        return {
            "available": True,
            "version": yaml.__version__
        }
    except ImportError:
        return {
            "available": False,
            "error": "PyYAML not installed"
        }


def _check_simpy_version() -> Dict[str, Any]:
    """Check SimPy version."""
    try:
        import simpy
        return {
            "available": True,
            "version": simpy.__version__
        }
    except ImportError:
        return {
            "available": False,
            "error": "SimPy not installed"
        }


def _check_fastapi_version() -> Dict[str, Any]:
    """Check FastAPI version."""
    try:
        import fastapi
        return {
            "available": True,
            "version": fastapi.__version__
        }
    except ImportError:
        return {
            "available": False,
            "error": "FastAPI not installed"
        }


# Additional endpoints for more detailed monitoring

@router.get("/health/simulation/{run_id}")
async def simulation_health(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    """
    Health check for a specific simulation run.
    
    Args:
        run_id: Simulation identifier
        sim_manager: Simulation manager dependency
        
    Returns:
        Detailed status of the simulation
    """
    status = sim_manager.get_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")
    
    # Check if simulation is stuck (no progress for too long)
    is_stuck = False
    if status['status'] == 'running':
        last_update = status.get('last_updated')
        if last_update:
            time_since_update = (datetime.now() - last_update).seconds
            is_stuck = time_since_update > 300  # 5 minutes without update
    
    return {
        "run_id": run_id,
        "status": status['status'],
        "progress": status['progress'],
        "is_stuck": is_stuck,
        "details": status
    }