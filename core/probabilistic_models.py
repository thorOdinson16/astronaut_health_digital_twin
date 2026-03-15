
"""
Probabilistic Models for Astronaut Digital Twin
Implements statistically justified distributions for physiological variables.
All distributions include academic justification in docstrings.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, Union, List
import logging
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DistributionConfig:
    """Configuration for a probabilistic distribution"""
    type: str  # normal, beta, poisson, gamma, etc.
    params: Dict[str, float]
    justification: str


class ProbabilisticModels:
    """
    Factory class for generating probabilistic physiological signals.
    
    Implements various distributions with proper statistical justification
    for each variable based on physiological principles and aerospace medicine
    literature.
    
    Justifications:
    - Heart Rate: Normal distribution due to homeostatic regulation around setpoint
    - Sleep Quality: Beta distribution (bounded [0,1], can be skewed)
    - Motion Sickness Onset: Poisson process (rare, discrete events)
    - Fatigue: Gamma distribution (positive, right-skewed)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize probabilistic models with optional config file.
        
        Args:
            config_path: Path to YAML distribution configuration
        """
        self.rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        
        # Default distributions with justifications
        self.distributions = {
            'heart_rate': {
                'type': 'normal',
                'params': {'mean': 75.0, 'std': 5.0},
                'justification': 'Heart rate exhibits homeostatic regulation around '
                                  'a setpoint with symmetric random fluctuations due '
                                  'to autonomic nervous system balance.'
            },
            'sleep_quality': {
                'type': 'beta',
                'params': {'alpha': 5.0, 'beta': 2.0},
                'justification': 'Sleep quality is bounded [0,1] and typically '
                                  'left-skewed (more good sleep than bad) due to '
                                  'homeostatic sleep drive.'
            },
            'motion_sickness_onset': {
                'type': 'poisson',
                'params': {'lambda': 0.03},  # events per hour
                'justification': 'Motion sickness episodes occur as rare discrete '
                                  'events, well-modeled as a Poisson process with '
                                  'constant hazard rate during adaptation period.'
            },
            'fatigue_noise': {
                'type': 'gamma',
                'params': {'shape': 2.0, 'scale': 0.1},
                'justification': 'Fatigue accumulation noise is positive and '
                                  'right-skewed, representing random stressors.'
            },
            'stress_response': {
                'type': 'lognormal',
                'params': {'mean': 0.5, 'sigma': 0.3},
                'justification': 'Stress responses are multiplicative and '
                                  'log-normally distributed in physiological systems.'
            }
        }
        
        # Override with config if provided
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """Load distribution configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'distributions' in config:
                    self.distributions.update(config['distributions'])
                    logger.info(f"Loaded distributions from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config {config_path}: {e}")
    
    def sample_heart_rate(self, size: int = 1, **kwargs) -> Union[float, np.ndarray]:
        """
        Sample heart rate from normal distribution.
        
        Args:
            size: Number of samples
            **kwargs: Override distribution parameters
            
        Returns:
            Heart rate samples in bpm
        """
        params = self.distributions['heart_rate']['params'].copy()
        params.update(kwargs)
        
        samples = self.rng.normal(
            loc=params['mean'],
            scale=params['std'],
            size=size
        )
        
        # Clip to physiological bounds (40-200 bpm)
        return np.clip(samples, 40, 200)
    
    def sample_sleep_quality(self, size: int = 1, **kwargs) -> Union[float, np.ndarray]:
        """
        Sample sleep quality from Beta distribution.
        
        Beta distribution is ideal for bounded [0,1] continuous variables.
        Alpha and beta parameters control skewness:
        - alpha > beta: left-skewed (more high values)
        - alpha < beta: right-skewed (more low values)
        - alpha = beta: symmetric
        
        Args:
            size: Number of samples
            **kwargs: Override distribution parameters
            
        Returns:
            Sleep quality samples in [0,1]
        """
        params = self.distributions['sleep_quality']['params'].copy()
        params.update(kwargs)
        
        return self.rng.beta(
            a=params['alpha'],
            b=params['beta'],
            size=size
        )
    
    def sample_motion_sickness_onset(
        self, 
        duration_hours: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample motion sickness event times from Poisson process.
        
        Motion sickness episodes occur as a Poisson process during
        microgravity adaptation. The time between events is exponentially
        distributed.
        
        Args:
            duration_hours: Time horizon for simulation (hours)
            **kwargs: Override distribution parameters
            
        Returns:
            Tuple of (event_times, event_count)
            event_times: Array of event times in hours
            event_count: Number of events
        """
        params = self.distributions['motion_sickness_onset']['params'].copy()
        params.update(kwargs)
        
        lambda_rate = params['lambda']  # events per hour
        
        # Sample number of events from Poisson
        n_events = self.rng.poisson(lambda_rate * duration_hours)
        
        # Sample event times uniformly in [0, duration_hours]
        if n_events > 0:
            event_times = np.sort(self.rng.uniform(0, duration_hours, n_events))
        else:
            event_times = np.array([])
        
        return event_times, n_events
    
    def sample_fatigue_noise(self, size: int = 1, **kwargs) -> Union[float, np.ndarray]:
        """
        Sample fatigue accumulation noise from Gamma distribution.
        
        Gamma distribution is appropriate for positive, right-skewed
        noise representing random stressors and recovery variations.
        
        Args:
            size: Number of samples
            **kwargs: Override distribution parameters
            
        Returns:
            Fatigue noise samples
        """
        params = self.distributions['fatigue_noise']['params'].copy()
        params.update(kwargs)
        
        return self.rng.gamma(
            shape=params['shape'],
            scale=params['scale'],
            size=size
        )
    
    def sample_stress_response(self, size: int = 1, **kwargs) -> Union[float, np.ndarray]:
        """
        Sample stress response from Log-normal distribution.
        
        Stress responses in physiological systems are often multiplicative
        and log-normally distributed due to cascade effects.
        
        Args:
            size: Number of samples
            **kwargs: Override distribution parameters
            
        Returns:
            Stress response multiplier (>=0)
        """
        params = self.distributions['stress_response']['params'].copy()
        params.update(kwargs)
        
        # Convert lognormal parameters
        mu = np.log(params['mean']**2 / np.sqrt(params['sigma']**2 + params['mean']**2))
        sigma = np.sqrt(np.log(1 + params['sigma']**2 / params['mean']**2))
        
        return self.rng.lognormal(
            mean=mu,
            sigma=sigma,
            size=size
        )
    
    def generate_baseline_trajectory(
        self,
        variable: str,
        timesteps: int,
        dt_minutes: float,
        noise_scale: float = 0.1,
        trend_rate: float = 0.0
    ) -> np.ndarray:
        """
        Generate a baseline trajectory with drift and noise.
        
        Combines deterministic drift with stochastic noise for realistic
        temporal evolution.
        
        Args:
            variable: Variable name (heart_rate, sleep_quality, etc.)
            timesteps: Number of time steps
            dt_minutes: Time step duration
            noise_scale: Scale of random noise
            trend_rate: Linear trend per hour
            
        Returns:
            Baseline trajectory array
        """
        # Get baseline samples
        if variable == 'heart_rate':
            baseline = self.sample_heart_rate(size=timesteps)
        elif variable == 'sleep_quality':
            baseline = self.sample_sleep_quality(size=timesteps)
        else:
            raise ValueError(f"Unknown variable: {variable}")
        
        # Add deterministic trend
        time_hours = np.arange(timesteps) * dt_minutes / 60.0
        trend = trend_rate * time_hours
        
        # Add autocorrelated noise (more realistic than white noise)
        noise = self.rng.normal(0, noise_scale, timesteps)
        if timesteps > 1:
            # Simple moving average for autocorrelation
            noise[1:] = 0.7 * noise[1:] + 0.3 * noise[:-1]
        
        trajectory = baseline + trend + noise
        
        # Clip to bounds
        bounds = {
            'heart_rate': (40, 200),
            'sleep_quality': (0, 1)
        }.get(variable, (None, None))
        
        if bounds[0] is not None:
            trajectory = np.clip(trajectory, bounds[0], bounds[1])
        
        return trajectory
    
    def get_distribution_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all distributions with justifications.
        
        Returns:
            Dictionary with distribution details for documentation
        """
        return {
            name: {
                'type': dist['type'],
                'params': dist['params'],
                'justification': dist['justification']
            }
            for name, dist in self.distributions.items()
        }