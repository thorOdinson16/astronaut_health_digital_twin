
"""
Logging Configuration for Astronaut Digital Twin
Provides consistent logging across all modules with different log levels,
formatters, and handlers for production and development environments.
"""

import logging
import logging.handlers
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import time
import functools
from typing import Optional, Dict, Any, Union
from pythonjsonlogger import jsonlogger
import traceback
import socket

# Try to import colorlog for development
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Color configuration for development
COLOR_CONFIG = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}


# =============================================================================
# CUSTOM LOGGER CLASS
# =============================================================================

class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds context to log messages.
    Useful for adding request IDs, user IDs, etc.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add extra context to log records
        kwargs['extra'] = kwargs.get('extra', {})
        kwargs['extra'].update(self.extra)
        return msg, kwargs


# =============================================================================
# JSON LOG FORMATTER
# =============================================================================

class JsonLogFormatter(jsonlogger.JsonFormatter):
    """
    JSON formatter for structured logging.
    Useful for production and log aggregation services.
    """
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add hostname
        log_record['hostname'] = socket.gethostname()
        
        # Add process and thread info
        log_record['process'] = record.process
        log_record['thread'] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }


# =============================================================================
# COLORED CONSOLE FORMATTER (Development)
# =============================================================================

class ColoredConsoleFormatter(colorlog.ColoredFormatter):
    """Colored formatter for development console output."""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s'
        
        super().__init__(
            fmt=fmt,
            datefmt=datefmt or DEFAULT_DATE_FORMAT,
            log_colors=COLOR_CONFIG,
            reset=True,
            style='%'
        )


# =============================================================================
# LOG HANDLER FACTORY
# =============================================================================

def create_console_handler(
    level: int = DEFAULT_LOG_LEVEL,
    json_format: bool = False,
    colored: bool = True
) -> logging.Handler:
    """
    Create a console log handler.
    
    Args:
        level: Log level
        json_format: Use JSON format
        colored: Use colored output (development only)
        
    Returns:
        Configured console handler
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if json_format:
        formatter = JsonLogFormatter()
    elif colored and HAS_COLORLOG:
        formatter = ColoredConsoleFormatter()
    else:
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
    
    handler.setFormatter(formatter)
    return handler


def create_file_handler(
    log_file: Union[str, Path],
    level: int = DEFAULT_LOG_LEVEL,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    json_format: bool = True
) -> logging.Handler:
    """
    Create a rotating file log handler.
    
    Args:
        log_file: Path to log file
        level: Log level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        json_format: Use JSON format
        
    Returns:
        Configured file handler
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    handler.setLevel(level)
    
    if json_format:
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
    
    handler.setFormatter(formatter)
    return handler


def create_syslog_handler(
    address: str = '/dev/log',
    facility: str = 'user',
    level: int = logging.WARNING
) -> logging.Handler:
    """
    Create a syslog handler for system logging.
    
    Args:
        address: Syslog address
        facility: Syslog facility
        level: Log level
        
    Returns:
        Configured syslog handler
    """
    import logging.handlers
    
    handler = logging.handlers.SysLogHandler(
        address=address,
        facility=facility
    )
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt='astronaut_digital_twin[%(process)d]: %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    return handler


# =============================================================================
# MAIN LOGGING SETUP FUNCTION
# =============================================================================

def setup_logging(
    log_level: str = 'INFO',
    log_dir: Optional[Union[str, Path]] = None,
    console_json: bool = False,
    file_json: bool = True,
    enable_syslog: bool = False,
    log_to_file: bool = True,
    loggers_to_silence: Optional[list] = None
) -> None:
    """
    Setup logging configuration for the entire application.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console_json: Use JSON format for console output
        file_json: Use JSON format for file output
        enable_syslog: Enable syslog output
        log_to_file: Enable file logging
        loggers_to_silence: List of logger names to set to WARNING
    """
    # Convert string level to int
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add console handler
    console_handler = create_console_handler(
        level=numeric_level,
        json_format=console_json,
        colored=not console_json
    )
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file and log_dir:
        log_file = log_dir / f"astronaut_digital_twin_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = create_file_handler(
            log_file=log_file,
            level=numeric_level,
            json_format=file_json
        )
        root_logger.addHandler(file_handler)
        
        # Also add error file for ERROR and above
        error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = create_file_handler(
            log_file=error_file,
            level=logging.ERROR,
            json_format=file_json
        )
        root_logger.addHandler(error_handler)
    
    # Add syslog handler if requested
    if enable_syslog:
        try:
            syslog_handler = create_syslog_handler(level=logging.WARNING)
            root_logger.addHandler(syslog_handler)
        except Exception as e:
            root_logger.warning(f"Failed to setup syslog: {e}")
    
    # Silence noisy loggers
    loggers_to_silence = loggers_to_silence or [
        'uvicorn.access',
        'uvicorn.error',
        'fastapi',
        'asyncio',
        'matplotlib',
        'PIL',
    ]
    
    for logger_name in loggers_to_silence:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Log startup
    root_logger.info("=" * 60)
    root_logger.info("LOGGING CONFIGURATION INITIALIZED")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Console JSON: {console_json}")
    if log_dir:
        root_logger.info(f"Log directory: {log_dir}")
    root_logger.info("=" * 60)


# =============================================================================
# GET LOGGER FUNCTION
# =============================================================================

def get_logger(
    name: str,
    context: Optional[Dict[str, Any]] = None
) -> Union[logging.Logger, LoggerAdapter]:
    """
    Get a logger instance with optional context.
    
    This is the main function to use throughout the application.
    
    Args:
        name: Logger name (usually __name__)
        context: Optional context dictionary (request ID, user ID, etc.)
        
    Returns:
        Logger instance (adapter if context provided)
    """
    logger = logging.getLogger(name)
    
    if context:
        return LoggerAdapter(logger, context)
    
    return logger


# =============================================================================
# CONTEXT MANAGER FOR TEMPORARY LOG LEVEL
# =============================================================================

class log_level:
    """
    Context manager to temporarily change log level.
    
    Example:
        with log_level('DEBUG'):
            logger.debug("This will be shown")
        # Level returns to previous
    """
    
    def __init__(self, level: str, logger_name: Optional[str] = None):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.logger_name = logger_name
        self.previous_level = None
        self.logger = None
    
    def __enter__(self):
        if self.logger_name:
            self.logger = logging.getLogger(self.logger_name)
        else:
            self.logger = logging.getLogger()
        
        self.previous_level = self.logger.level
        self.logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger and self.previous_level is not None:
            self.logger.setLevel(self.previous_level)


# =============================================================================
# PERFORMANCE LOGGING DECORATOR
# =============================================================================

def log_performance(logger_name: Optional[str] = None):
    """
    Decorator to log function performance.
    
    Example:
        @log_performance()
        def expensive_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                logger.debug(
                    f"Performance: {func.__name__} took {elapsed:.4f} seconds"
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Performance: {func.__name__} failed after {elapsed:.4f} seconds: {e}"
                )
                raise
        
        return wrapper
    return decorator


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_dict(logger: logging.Logger, level: str, msg: str, data: Dict) -> None:
    """Log a dictionary with nice formatting."""
    log_method = getattr(logger, level.lower())
    log_method(f"{msg}:\n{json.dumps(data, indent=2, default=str)}")


def log_exception(logger: logging.Logger, msg: str = "Exception occurred") -> None:
    """Log an exception with full traceback."""
    logger.exception(msg)


def log_section(logger: logging.Logger, title: str, level: str = 'info') -> None:
    """Log a section header."""
    log_method = getattr(logger, level.lower())
    log_method("=" * 60)
    log_method(f"{title.upper()}")
    log_method("=" * 60)


def log_simulation_start(logger: logging.Logger, config: Dict) -> None:
    """Log simulation start with configuration."""
    log_section(logger, "SIMULATION START")
    logger.info(f"Run ID: {config.get('run_id', 'unknown')}")
    logger.info(f"Mission duration: {config.get('mission_duration_hours', 0)} hours")
    logger.info(f"Time step: {config.get('time_step_minutes', 5)} minutes")
    logger.debug(f"Full config: {json.dumps(config, indent=2, default=str)}")


def log_simulation_end(
    logger: logging.Logger,
    run_id: str,
    metrics: Dict,
    duration: float
) -> None:
    """Log simulation end with metrics."""
    log_section(logger, "SIMULATION COMPLETE")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Peak fatigue: {metrics.get('peak_fatigue', 0):.2f}")
    logger.info(f"Events triggered: {metrics.get('event_count', 0)}")
    logger.info(f"Risk level: {metrics.get('risk_level', 'unknown')}")


# =============================================================================
# MAIN GUARD - Test logging configuration
# =============================================================================

if __name__ == "__main__":
    # Test logging setup
    setup_logging(
        log_level='DEBUG',
        log_dir='./logs/test',
        console_json=False,
        file_json=True,
        log_to_file=True
    )
    
    logger = get_logger(__name__, context={'test': True})
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    
    try:
        1 / 0
    except Exception as e:
        logger.exception("Caught an exception")
    
    # Test context manager
    with log_level('DEBUG'):
        logger.debug("This debug message should appear")
    
    logger.info("Logging test complete")