import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

#!/usr/bin/env python3
"""
Astronaut Digital Twin API - Main Entry Point
Service-Oriented Hybrid Digital Twin for Coupled Sleep-Fatigue and Space Motion Sickness

This FastAPI application provides the backend for the digital twin simulation,
exposing endpoints for running simulations, retrieving results, and monitoring
system health. It integrates:
- Probabilistic models for physiological variables
- Discrete event simulation for motion sickness and sleep disruption
- BioGears integration for ground-truth physiology
- Monte Carlo analytics for risk assessment

Author: Person 1 - Supervisory Digital Twin Team
Version: 1.0.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime
import warnings

# Third-party imports
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
import uvicorn
import yaml

# Local imports
from api.routes import simulation, data, health
from api.dependencies import lifespan, get_simulation_manager, get_config_loader
from utils.logger import setup_logging, get_logger
from utils.helpers import get_version, get_git_revision

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Version information
__version__ = "1.0.0"
__git_revision__ = get_git_revision()

# Setup logging
setup_logging()
logger = get_logger(__name__)


# =============================================================================
# APPLICATION LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    This handles:
    - Loading configuration on startup
    - Initializing simulation manager
    - Setting up background tasks
    - Graceful shutdown
    """
    logger.info("=" * 60)
    logger.info("ASTRONAUT DIGITAL TWIN API - STARTING UP")
    logger.info("=" * 60)
    logger.info(f"Version: {__version__} (git: {__git_revision__})")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Load configuration
    try:
        config_path = Path("./config")
        if config_path.exists():
            logger.info(f"Loading configuration from: {config_path.absolute()}")
            
            # Load simulation config
            sim_config_path = config_path / "simulation_config.yaml"
            if sim_config_path.exists():
                with open(sim_config_path, 'r') as f:
                    sim_config = yaml.safe_load(f)
                app.state.simulation_config = sim_config
                logger.info(f"✓ Loaded simulation configuration")
            
            # Load distributions config
            dist_config_path = config_path / "distributions.yaml"
            if dist_config_path.exists():
                with open(dist_config_path, 'r') as f:
                    dist_config = yaml.safe_load(f)
                app.state.distributions_config = dist_config
                logger.info(f"✓ Loaded distributions configuration")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        app.state.simulation_config = {}
        app.state.distributions_config = {}
    
    # Initialize simulation manager
    try:
        sim_manager = await get_simulation_manager()
        app.state.simulation_manager = sim_manager
        logger.info(f"✓ Simulation manager initialized")
        logger.info(f"  - Storage path: {sim_manager.storage_path}")
        logger.info(f"  - Active runs: {len([r for r in sim_manager.runs.values() if r.status == 'running'])}")
        logger.info(f"  - Completed runs: {len([r for r in sim_manager.runs.values() if r.status == 'completed'])}")
    except Exception as e:
        logger.error(f"Failed to initialize simulation manager: {e}")
    
    # Check BioGears availability
    try:
        from biogears.biogears_adapter import BioGearsAdapter
        adapter = BioGearsAdapter()
        if not adapter.runner._mock_mode:
            app.state.biogears_available = True
        else:
            app.state.biogears_available = False
    except ImportError:
        logger.warning("⚠ BioGears adapter not found - running in standalone mode")
        app.state.biogears_available = False
    except Exception as e:
        logger.warning(f"⚠ BioGears check failed: {e} - running in standalone mode")
        app.state.biogears_available = False
    
    logger.info("=" * 60)
    logger.info("API startup complete - ready to accept requests")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("ASTRONAUT DIGITAL TWIN API - SHUTTING DOWN")
    logger.info("=" * 60)
    
    # Cleanup any running simulations
    sim_manager = app.state.simulation_manager
    if sim_manager:
        running_runs = [r for r in sim_manager.runs.values() if r.status == 'running']
        if running_runs:
            logger.info(f"Cleaning up {len(running_runs)} running simulations...")
            for run in running_runs:
                await sim_manager.stop_run(run.run_id)
    
    logger.info("Shutdown complete")


# =============================================================================
# FASTAPI APPLICATION CREATION
# =============================================================================

def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Astronaut Digital Twin API",
        description="""
        # 🚀 Astronaut Digital Twin
        
        A service-oriented hybrid digital twin for coupled sleep-fatigue and space motion sickness.
        
        ## Features
        
        * **Probabilistic Modeling** - Stochastic generation of physiological variables
        * **Discrete Event Simulation** - Motion sickness and sleep disruption events
        * **BioGears Integration** - Ground-truth physiology engine
        * **Monte Carlo Analytics** - Risk assessment and uncertainty quantification
        * **Real-time Visualization** - WebSocket support for live updates
        
        ## Architecture
        
        The API implements a three-layer architecture:
        1. **Supervisory Layer** - Event scheduling, coupling logic, state management
        2. **Physiological Core** - BioGears integration for ground-truth responses
        3. **Analytics Layer** - Risk assessment and visualization data
        
        ## Documentation
        
        * Interactive API docs: `/docs`
        * Alternative docs: `/redoc`
        * OpenAPI schema: `/openapi.json`
        
        ## Version
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Person 1 - Supervisory Digital Twin Team",
            "email": "person1@digitaltwin.space",
            "url": "https://github.com/yourusername/astronaut-digital-twin"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        terms_of_service="http://example.com/terms/",
    )
    
    # =========================================================================
    # MIDDLEWARE CONFIGURATION
    # =========================================================================
    
    # CORS middleware - Allow Person 3's React app to connect
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",      # React dev server
            "http://localhost:5173",       # Vite dev server
            "http://localhost:8080",       # Alternative dev server
            "http://localhost:5500",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5500",
            "https://dashboard.astronaut-twins.space",  # Production
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )
    
    # GZip compression for large responses
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,  # Compress responses larger than 1KB
    )
    
    # Trusted host middleware (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "localhost",
            "127.0.0.1",
            "*.astronaut-twins.space",
            "*.digitaltwin.space",
        ],
    )
    
    # =========================================================================
    # CUSTOM MIDDLEWARE
    # =========================================================================
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add X-Process-Time header with request duration."""
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    @app.middleware("http")
    async def add_request_id_header(request: Request, call_next):
        """Add X-Request-ID header for tracing."""
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests."""
        logger.info(f"Request: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response
    
    # =========================================================================
    # EXCEPTION HANDLERS
    # =========================================================================
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with detailed messages."""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": exc.errors(),
                "body": exc.body,
                "message": "Request validation failed"
            },
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle any uncaught exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred",
                "type": exc.__class__.__name__,
                "message": str(exc) if app.debug else "Contact system administrator"
            },
        )
    
    # =========================================================================
    # INCLUDE ROUTERS
    # =========================================================================
    
    # Simulation routes - Person 1's core endpoints
    app.include_router(
        simulation.router,
        prefix="/api/simulation",
        tags=["simulation"]
    )
    
    # Data routes - Person 3's visualization endpoints
    app.include_router(
        data.router,
        prefix="/api/data",
        tags=["data"]
    )
    
    # Health routes - Monitoring and diagnostics
    app.include_router(
        health.router,
        prefix="/api/health",
        tags=["health"]
    )
    
    # =========================================================================
    # ROOT ENDPOINTS
    # =========================================================================
    
    @app.get("/", tags=["root"])
    async def root():
        """
        Root endpoint - API information.
        
        Returns basic information about the API including version,
        available endpoints, and documentation links.
        """
        return {
            "name": "Astronaut Digital Twin API",
            "version": __version__,
            "git_revision": __git_revision__,
            "description": "Service-oriented hybrid digital twin for sleep-fatigue and space motion sickness",
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            },
            "endpoints": {
                "simulation": "/api/simulation",
                "data": "/api/data",
                "health": "/api/health"
            },
            "status": "operational",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/health", tags=["health"])
    async def health_check():
        """
        Simple health check endpoint.
        
        Returns:
            Status indicating API is running
        """
        return {
            "status": "healthy",
            "version": __version__,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/info", tags=["root"])
    async def info():
        """
        Detailed system information.
        
        Returns comprehensive information about the system including
        configuration, dependencies, and runtime statistics.
        """
        sim_manager = app.state.simulation_manager
        
        # Get system info
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        # Get simulation stats
        sim_stats = {}
        if sim_manager:
            sim_stats = sim_manager.get_global_stats()
        
        return {
            "name": "Astronaut Digital Twin API",
            "version": __version__,
            "git_revision": __git_revision__,
            "system": system_info,
            "configuration": {
                "simulation": bool(app.state.simulation_config),
                "distributions": bool(app.state.distributions_config),
                "biogears_available": app.state.biogears_available
            },
            "simulation_statistics": sim_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    # =========================================================================
    # CUSTOM OPENAPI SCHEMA
    # =========================================================================
    
    def custom_openapi():
        """Custom OpenAPI schema with additional information."""
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add server information
        openapi_schema["servers"] = [
            {"url": "http://localhost:8000", "description": "Development server"},
            {"url": "https://api.astronaut-twins.space", "description": "Production server"}
        ]
        
        # Add security schemes if needed
        openapi_schema["components"]["securitySchemes"] = {}
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = create_application()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for running the API server.
    
    This function is called when running `python main.py`
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Astronaut Digital Twin API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", 
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Configure uvicorn
    config = {
        "app": "main:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "workers": args.workers,
        "log_level": args.log_level.lower(),
        "access_log": True,
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
    }
    
    # Print startup banner
    print("\n" + "=" * 60)
    print("  🚀 ASTRONAUT DIGITAL TWIN API SERVER")
    print("=" * 60)
    print(f"  Version: {__version__}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Workers: {args.workers}")
    print(f"  Reload: {args.reload}")
    print(f"  Log level: {args.log_level}")
    print("=" * 60)
    print("  Documentation:")
    print(f"  → http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print(f"  → http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/redoc")
    print("=" * 60 + "\n")
    
    # Run server
    uvicorn.run(**config)


# =============================================================================
# ADDITIONAL CONFIGURATION FOR DEPLOYMENT
# =============================================================================

# For Gunicorn deployment (if using)
gunicorn_config = {
    "bind": "0.0.0.0:8000",
    "workers": 4,
    "worker_class": "uvicorn.workers.UvicornWorker",
    "timeout": 120,
    "keepalive": 5,
    "max_requests": 1000,
    "max_requests_jitter": 100,
}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()