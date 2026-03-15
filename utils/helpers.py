
"""
Helper Utilities for Astronaut Digital Twin
Common utility functions used across the application.
"""

import numpy as np
import json
import yaml
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import os
import re
import subprocess
import functools
import time
import inspect
from contextlib import contextmanager
import pandas as pd


# =============================================================================
# TIME AND DATE UTILITIES
# =============================================================================

def hours_to_timesteps(hours: float, dt_minutes: float = 5.0) -> int:
    """
    Convert hours to number of timesteps.
    
    Args:
        hours: Time in hours
        dt_minutes: Time step duration in minutes
        
    Returns:
        Number of timesteps
    """
    return int(hours * 60 / dt_minutes)


def timesteps_to_hours(timesteps: int, dt_minutes: float = 5.0) -> float:
    """
    Convert timesteps to hours.
    
    Args:
        timesteps: Number of timesteps
        dt_minutes: Time step duration in minutes
        
    Returns:
        Time in hours
    """
    return timesteps * dt_minutes / 60.0


def format_duration_hours(hours: float) -> str:
    """
    Format duration in hours to human-readable string.
    
    Args:
        hours: Duration in hours
        
    Returns:
        Formatted string (e.g., "2d 6h 30m")
    """
    days = int(hours // 24)
    remaining = hours % 24
    hrs = int(remaining)
    mins = int((remaining - hrs) * 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hrs > 0:
        parts.append(f"{hrs}h")
    if mins > 0:
        parts.append(f"{mins}m")
    
    return " ".join(parts) if parts else "0h"


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def get_date_str() -> str:
    """Get current date as string YYYYMMDD."""
    return datetime.now().strftime("%Y%m%d")


def get_time_str() -> str:
    """Get current time as string HHMMSS."""
    return datetime.now().strftime("%H%M%S")


# =============================================================================
# MATH AND STATISTICS UTILITIES
# =============================================================================

def normalize_array(arr: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normalize array to [min_val, max_val] range.
    
    Args:
        arr: Input array
        min_val: Minimum of output range
        max_val: Maximum of output range
        
    Returns:
        Normalized array
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    if arr_max - arr_min == 0:
        return np.full_like(arr, (min_val + max_val) / 2)
    
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized * (max_val - min_val) + min_val


def smooth_series(
    data: np.ndarray,
    window_size: int = 5,
    method: str = 'moving_average'
) -> np.ndarray:
    """
    Smooth a time series.
    
    Args:
        data: Input time series
        window_size: Smoothing window size
        method: 'moving_average' or 'exponential'
        
    Returns:
        Smoothed series
    """
    if method == 'moving_average':
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')
    
    elif method == 'exponential':
        alpha = 2 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def detect_peaks(data: np.ndarray, threshold: float = 0.5) -> List[int]:
    """
    Detect peaks in time series data.
    
    Args:
        data: Input time series
        threshold: Peak threshold (relative to max)
        
    Returns:
        List of peak indices
    """
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(data, height=threshold * np.max(data))
    return peaks.tolist()


def compute_autocorrelation(data: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation of time series.
    
    Args:
        data: Input time series
        max_lag: Maximum lag to compute
        
    Returns:
        Autocorrelation values
    """
    from scipy import signal
    
    if max_lag is None:
        max_lag = len(data) // 2
    
    result = signal.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    result = result[len(result)//2:len(result)//2 + max_lag]
    result /= result[0]
    
    return result


def calculate_risk_percentile(value: float, distribution: np.ndarray) -> float:
    """
    Calculate percentile of a value in a distribution.
    
    Args:
        value: Value to check
        distribution: Reference distribution
        
    Returns:
        Percentile (0-100)
    """
    return np.sum(distribution <= value) / len(distribution) * 100


# =============================================================================
# FILE AND PATH UTILITIES
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """
    Convert string to safe filename.
    
    Args:
        filename: Input filename
        
    Returns:
        Safe filename with only alphanumeric, underscore, hyphen
    """
    # Remove invalid characters
    safe = re.sub(r'[^\w\s-]', '', filename)
    # Replace spaces with underscores
    safe = re.sub(r'[-\s]+', '_', safe)
    return safe


def get_unique_filename(directory: Union[str, Path], base_name: str, extension: str) -> Path:
    """
    Get a unique filename by adding number if file exists.
    
    Args:
        directory: Directory path
        base_name: Base filename without extension
        extension: File extension (with or without dot)
        
    Returns:
        Unique Path object
    """
    directory = Path(directory)
    extension = extension.lstrip('.')
    
    counter = 1
    while True:
        if counter == 1:
            filename = directory / f"{base_name}.{extension}"
        else:
            filename = directory / f"{base_name}_{counter}.{extension}"
        
        if not filename.exists():
            return filename
        counter += 1


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Any, filepath: Union[str, Path], pretty: bool = True) -> None:
    """Save data to JSON file."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    with open(filepath, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2, default=str)
        else:
            json.dump(data, f, default=str)


def load_yaml(filepath: Union[str, Path]) -> Dict:
    """Load YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, filepath: Union[str, Path]) -> None:
    """Save data to YAML file."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_csv_to_dict(filepath: Union[str, Path]) -> List[Dict]:
    """Load CSV file to list of dictionaries."""
    import pandas as pd
    df = pd.read_csv(filepath)
    return df.to_dict('records')


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def validate_range(value: float, min_val: float, max_val: float, name: str = "value") -> bool:
    """
    Validate value is within range.
    
    Args:
        value: Value to check
        min_val: Minimum allowed
        max_val: Maximum allowed
        name: Name for error message
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If out of range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return True


def validate_probability(value: float, name: str = "probability") -> bool:
    """Validate value is a probability [0,1]."""
    return validate_range(value, 0, 1, name)


def validate_positive(value: float, name: str = "value") -> bool:
    """Validate value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return True


def validate_non_negative(value: float, name: str = "value") -> bool:
    """Validate value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return True


# =============================================================================
# ID GENERATION
# =============================================================================

def generate_run_id(prefix: str = "sim") -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{unique}"


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def hash_config(config: Dict) -> str:
    """Generate hash of configuration for caching."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:10]


# =============================================================================
# DECORATORS
# =============================================================================

def timeit(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def memoize(func):
    """Decorator to memoize function results."""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def timer(name: str = "Block"):
    """Context manager to time a block of code."""
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} took {end - start:.4f} seconds")


@contextmanager
def working_directory(path: Union[str, Path]):
    """Context manager to temporarily change working directory."""
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


# =============================================================================
# VERSION CONTROL UTILITIES
# =============================================================================

def get_git_revision() -> str:
    """Get current git revision hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def get_version() -> str:
    """Get application version from VERSION file or git."""
    version_file = Path("VERSION")
    if version_file.exists():
        return version_file.read_text().strip()
    return f"git-{get_git_revision()}"


# =============================================================================
# DATA CONVERSION UTILITIES
# =============================================================================

def dict_to_json_bytes(data: Dict) -> bytes:
    """Convert dictionary to JSON bytes."""
    return json.dumps(data, default=str).encode('utf-8')


def json_bytes_to_dict(data: bytes) -> Dict:
    """Convert JSON bytes to dictionary."""
    return json.loads(data.decode('utf-8'))


def numpy_to_python(obj: Any) -> Any:
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    return obj


def dataframe_to_dict(df: pd.DataFrame) -> Dict:
    """Convert pandas DataFrame to dictionary."""
    return {
        'columns': df.columns.tolist(),
        'data': df.values.tolist(),
        'index': df.index.tolist()
    }


# =============================================================================
# STRING UTILITIES
# =============================================================================

def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length."""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024) if hasattr(memory_info, 'vms') else 0,
        'percent': process.memory_percent()
    }


def get_cpu_usage() -> float:
    """Get current CPU usage percent."""
    import psutil
    return psutil.cpu_percent(interval=0.1)


def get_disk_usage(path: Union[str, Path] = "/") -> Dict[str, float]:
    """Get disk usage in GB."""
    import psutil
    usage = psutil.disk_usage(path)
    
    return {
        'total_gb': usage.total / (1024**3),
        'used_gb': usage.used / (1024**3),
        'free_gb': usage.free / (1024**3),
        'percent': usage.percent
    }


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge two configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_config(config: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested configuration dictionary."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

def bpm_to_seconds(bpm: float) -> float:
    """Convert beats per minute to seconds per beat."""
    return 60.0 / bpm


def seconds_to_bpm(seconds: float) -> float:
    """Convert seconds per beat to beats per minute."""
    return 60.0 / seconds


def hours_to_minutes(hours: float) -> float:
    """Convert hours to minutes."""
    return hours * 60.0


def minutes_to_hours(minutes: float) -> float:
    """Convert minutes to hours."""
    return minutes / 60.0


def hours_to_seconds(hours: float) -> float:
    """Convert hours to seconds."""
    return hours * 3600.0


def seconds_to_hours(seconds: float) -> float:
    """Convert seconds to hours."""
    return seconds / 3600.0


# =============================================================================
# RANDOM UTILITIES
# =============================================================================

def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)


def generate_color_palette(n_colors: int) -> List[str]:
    """Generate a color palette for plotting."""
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab10', n_colors)
    return [f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}' 
            for rgb in cmap(range(n_colors))]


# =============================================================================
# CLASS UTILITIES
# =============================================================================

def singleton(cls):
    """Singleton decorator for classes."""
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def enum_values(enum_class) -> List[Any]:
    """Get all values from an enum class."""
    return [item.value for item in enum_class]


# =============================================================================
# MAIN GUARD
# =============================================================================

if __name__ == "__main__":
    # Test utilities
    print("Testing helpers...")
    print(f"Version: {get_version()}")
    print(f"Git revision: {get_git_revision()}")
    print(f"Git branch: {get_git_branch()}")
    print(f"Format 72.5 hours: {format_duration_hours(72.5)}")
    print(f"Memory usage: {get_memory_usage()}")