"""
Data API Routes - Results Access for Person 3
Provides endpoints for Person 3's visualization dashboard to fetch simulation data.
"""

from fastapi import APIRouter, HTTPException, Query, Response, Depends
from fastapi.responses import FileResponse
from typing import Optional, Dict, List, Any
import pandas as pd
import json
from datetime import datetime
import logging
import numpy as np

from api.dependencies import get_simulation_manager, SimulationManager
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["data"])


# =============================================================================
# NUMPY SERIALIZATION — fixes float32/int32/ndarray JSON errors everywhere
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def numpy_safe(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: numpy_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_safe(i) for i in obj]
    return obj


def json_response(data):
    """Return a Response with numpy-safe JSON encoding."""
    return Response(
        content=json.dumps(data, cls=NumpyEncoder),
        media_type="application/json"
    )


# =============================================================================
# DATA ENDPOINTS - Person 3 uses these for visualization
# =============================================================================

@router.get("/results/{run_id}")
async def get_simulation_results(
    run_id: str,
    format: str = Query("json", pattern="^(json|csv|parquet)$"),
    include_raw_biogears: bool = Query(False),
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    results = await sim_manager.get_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")

    status = await sim_manager.get_status(run_id)
    if status['status'] != 'completed':
        return json_response({
            "run_id": run_id,
            "status": status['status'],
            "progress": status['progress'],
            "message": "Simulation not yet complete"
        })

    if format == "json":
        return json_response(results)

    elif format == "csv":
        df = pd.DataFrame(numpy_safe(results['state']))
        return Response(
            content=df.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=simulation_{run_id}.csv"}
        )

    elif format == "parquet":
        df = pd.DataFrame(numpy_safe(results['state']))
        return Response(
            content=df.to_parquet(index=False),
            media_type="application/parquet",
            headers={"Content-Disposition": f"attachment; filename=simulation_{run_id}.parquet"}
        )


@router.get("/results/{run_id}/trajectory/{variable}")
async def get_variable_trajectory(
    run_id: str,
    variable: str,
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    results = await sim_manager.get_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")

    if variable not in results['state']:
        valid_vars = list(results['state'].keys())
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found. Available: {valid_vars}")

    time   = numpy_safe(results['state']['time'])
    values = numpy_safe(results['state'][variable])

    if start_time is not None or end_time is not None:
        filtered_time, filtered_values = [], []
        for t, v in zip(time, values):
            if start_time is not None and t < start_time:
                continue
            if end_time is not None and t > end_time:
                continue
            filtered_time.append(t)
            filtered_values.append(v)
    else:
        filtered_time, filtered_values = time, values

    return json_response({
        "run_id": run_id,
        "variable": variable,
        "time": filtered_time,
        "values": filtered_values,
        "unit": _get_variable_unit(variable),
        "statistics": {
            "min": float(min(filtered_values)),
            "max": float(max(filtered_values)),
            "mean": float(sum(filtered_values) / len(filtered_values)),
            "current": float(filtered_values[-1]) if filtered_values else None
        }
    })


@router.get("/results/{run_id}/events")
async def get_event_timeline(
    run_id: str,
    event_type: Optional[str] = Query(None, pattern="^(motion_sickness|sleep_disruption)$"),
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    results = await sim_manager.get_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")

    events = numpy_safe(results.get('events', []))

    if event_type:
        events = [e for e in events if e['type'].lower() == event_type.lower()]

    return json_response({
        "run_id": run_id,
        "event_count": len(events),
        "events": events
    })


@router.get("/results/{run_id}/summary")
async def get_simulation_summary(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    results = await sim_manager.get_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")

    state  = numpy_safe(results['state'])
    events = numpy_safe(results.get('events', []))

    _durations = [e.get('duration', 0) for e in events if e.get('duration')]

    summary = {
        "run_id": run_id,
        "duration_hours": float(state['time'][-1]) if state['time'] else 0,
        "metrics": {
            "heart_rate": {
                "mean":        float(np.mean(state['hr'])),
                "min":         float(np.min(state['hr'])),
                "max":         float(np.max(state['hr'])),
                "variability": float(np.std(state['hr']))
            },
            "fatigue": {
                "mean":                float(np.mean(state['fatigue'])),
                "peak":                float(np.max(state['fatigue'])),
                "final":               float(state['fatigue'][-1]) if state['fatigue'] else 0.0,
                "time_above_threshold": float(_time_above_threshold(state['time'], state['fatigue'], 5.0))
            },
            "sleep_quality": {
                "mean":       float(np.mean(state['sleep_quality'])),
                "min":        float(np.min(state['sleep_quality'])),
                "efficiency": float(np.mean(state['sleep_quality'])) * 100
            },
            "motion_sickness": {
                "episodes":       len([e for e in events if 'motion' in e['type'].lower()]),
                "total_severity": float(sum(e.get('severity', 0) for e in events)),
                "avg_duration":   float(np.mean(_durations)) if _durations else 0.0
            }
        },
        "risk_assessment": _compute_risk_summary(state, events)
    }

    return json_response(summary)


@router.get("/results/{run_id}/download")
async def download_results_package(
    run_id: str,
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    results = await sim_manager.get_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")

    zip_path = await sim_manager.create_download_package(run_id)

    return FileResponse(
        path=zip_path,
        filename=f"simulation_{run_id}_results.zip",
        media_type="application/zip"
    )


@router.get("/compare")
async def compare_simulations(
    run_ids: str = Query(..., description="Comma-separated list of run IDs"),
    metrics: str = Query("fatigue,hr,sleep_quality", description="Comma-separated metrics"),
    sim_manager: SimulationManager = Depends(get_simulation_manager)
):
    run_id_list = [rid.strip() for rid in run_ids.split(",")]
    metric_list = [m.strip()  for m  in metrics.split(",")]

    comparison = {"runs": run_id_list, "metrics": metric_list, "data": {}, "statistics": {}}

    for run_id in run_id_list:
        results = await sim_manager.get_results(run_id)
        if not results:
            raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")

        state     = numpy_safe(results['state'])
        run_data  = {}
        run_stats = {}

        for metric in metric_list:
            if metric in state:
                values = state[metric]
                run_data[metric]  = values
                run_stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std":  float(np.std(values)),
                    "min":  float(np.min(values)),
                    "max":  float(np.max(values))
                }

        comparison['data'][run_id]       = run_data
        comparison['statistics'][run_id] = run_stats

    return json_response(comparison)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_variable_unit(variable: str) -> str:
    units = {
        'hr': 'bpm', 'sleep_quality': 'normalized [0-1]',
        'fatigue': 'index [0-10]', 'stress': 'normalized [0-1]',
        'motion_severity': 'severity [0-5]', 'time': 'hours'
    }
    return units.get(variable, 'unknown')


def _time_above_threshold(time: List[float], values: List[float], threshold: float) -> float:
    if not time or not values:
        return 0.0
    total_time = 0.0
    for i, val in enumerate(values):
        if val > threshold and i > 0:
            total_time += time[i] - time[i-1]
    return total_time


def _compute_risk_summary(state: Dict, events: List[Dict]) -> Dict:
    peak_fatigue = float(max(state['fatigue']))
    fatigue_risk = "HIGH" if peak_fatigue > 7 else "MEDIUM" if peak_fatigue > 4 else "LOW"

    ms_events = [e for e in events if 'motion' in e['type'].lower()]
    ms_burden  = sum(e.get('severity', 0) * e.get('duration', 0) for e in ms_events)
    ms_risk    = "HIGH" if ms_burden > 20 else "MEDIUM" if ms_burden > 10 else "LOW"

    avg_sleep  = float(np.mean(state['sleep_quality']))
    sleep_risk = "HIGH" if avg_sleep < 0.4 else "MEDIUM" if avg_sleep < 0.6 else "LOW"

    risk_scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    composite   = (risk_scores[fatigue_risk] + risk_scores[ms_risk] + risk_scores[sleep_risk]) / 3
    composite_risk = "HIGH" if composite > 2.5 else "MEDIUM" if composite > 1.5 else "LOW"

    return {
        "fatigue_risk":        fatigue_risk,
        "motion_sickness_risk": ms_risk,
        "sleep_risk":          sleep_risk,
        "composite_risk":      composite_risk,
        "critical_events":     len([e for e in events if e.get('severity', 0) > 0.8])
    }