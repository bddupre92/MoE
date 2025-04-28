"""
Base Generator module for the Enhanced Data Generation Pipeline.

This module provides the abstract base class for all domain-specific data generators,
defining the common interface and functionality.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

class BaseGenerator(ABC):
    """
    Abstract base class for all domain-specific data generators.
    
    This class defines the common interface and functionality for all generators
    in the Enhanced Data Generation Pipeline.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the base generator.
        
        Args:
            config: Configuration dictionary for the generator.
            seed: Optional random seed for reproducibility.
        """
        self.config = config
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.rng = np.random.RandomState(self.seed)
    
    @abstractmethod
    def generate(self, num_samples: int, time_periods: int) -> Dict[str, np.ndarray]:
        """
        Generate synthetic data.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods for time-series data.
            
        Returns:
            Dictionary containing the generated data.
        """
        pass
    
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed value.
        """
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
    
    def _generate_time_series(self, 
                             base_value: float, 
                             std_dev: float, 
                             num_samples: int, 
                             time_periods: int,
                             trend: Optional[float] = None,
                             seasonality: Optional[List[float]] = None,
                             autocorrelation: Optional[float] = None) -> np.ndarray:
        """
        Generate a time series with optional trend, seasonality, and autocorrelation.
        
        Args:
            base_value: Base value for the time series.
            std_dev: Standard deviation of the noise.
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            trend: Optional linear trend coefficient.
            seasonality: Optional list of seasonal coefficients.
            autocorrelation: Optional autocorrelation coefficient.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing the generated time series.
        """
        # Initialize the time series with noise
        time_series = self.rng.normal(0, std_dev, (num_samples, time_periods))
        
        # Add base value
        time_series += base_value
        
        # Add trend if specified
        if trend is not None:
            for t in range(time_periods):
                time_series[:, t] += trend * t
        
        # Add seasonality if specified
        if seasonality is not None:
            season_length = len(seasonality)
            for t in range(time_periods):
                time_series[:, t] += seasonality[t % season_length]
        
        # Add autocorrelation if specified
        if autocorrelation is not None and autocorrelation != 0:
            for t in range(1, time_periods):
                time_series[:, t] += autocorrelation * (time_series[:, t-1] - base_value)
        
        return time_series
    
    def _generate_correlated_series(self, 
                                   base_series: np.ndarray, 
                                   correlation: float, 
                                   base_value: float, 
                                   std_dev: float) -> np.ndarray:
        """
        Generate a time series correlated with a base series.
        
        Args:
            base_series: Base time series to correlate with.
            correlation: Correlation coefficient (-1 to 1).
            base_value: Base value for the new series.
            std_dev: Standard deviation of the noise.
            
        Returns:
            NumPy array with the same shape as base_series containing the correlated time series.
        """
        num_samples, time_periods = base_series.shape
        
        # Generate independent noise
        noise = self.rng.normal(0, std_dev, (num_samples, time_periods))
        
        # Normalize base series
        base_normalized = (base_series - np.mean(base_series)) / (np.std(base_series) + 1e-8)
        
        # Generate correlated series
        correlated_series = base_value + correlation * std_dev * base_normalized + np.sqrt(1 - correlation**2) * noise
        
        return correlated_series
    
    def _apply_circadian_rhythm(self, 
                               time_series: np.ndarray, 
                               amplitude: float, 
                               phase_shift: float = 0.0,
                               periods_per_day: int = 24) -> np.ndarray:
        """
        Apply circadian rhythm to a time series.
        
        Args:
            time_series: Time series to apply circadian rhythm to.
            amplitude: Amplitude of the circadian rhythm.
            phase_shift: Phase shift in radians.
            periods_per_day: Number of time periods per day.
            
        Returns:
            NumPy array with the same shape as time_series with circadian rhythm applied.
        """
        num_samples, time_periods = time_series.shape
        
        # Generate circadian rhythm
        time_indices = np.arange(time_periods)
        circadian = amplitude * np.sin(2 * np.pi * time_indices / periods_per_day + phase_shift)
        
        # Apply to time series
        result = time_series.copy()
        for t in range(time_periods):
            result[:, t] += circadian[t]
        
        return result
    
    def _generate_events(self, 
                        probability: float, 
                        num_samples: int, 
                        time_periods: int,
                        duration_mean: float = 1.0,
                        duration_std: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random events with specified probability and duration.
        
        Args:
            probability: Probability of event occurrence per time period.
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            duration_mean: Mean duration of events.
            duration_std: Standard deviation of event duration.
            
        Returns:
            Tuple of (event_occurrences, event_durations) where:
                - event_occurrences is a boolean array of shape (num_samples, time_periods)
                  indicating whether an event started at each time period.
                - event_durations is an integer array of shape (num_samples, time_periods)
                  indicating the duration of the event that started at each time period (0 if no event).
        """
        # Generate event occurrences
        event_occurrences = self.rng.random((num_samples, time_periods)) < probability
        
        # Generate event durations
        durations = np.maximum(1, self.rng.normal(duration_mean, duration_std, (num_samples, time_periods)))
        durations = np.round(durations).astype(int)
        
        # Set duration to 0 for non-events
        event_durations = np.where(event_occurrences, durations, 0)
        
        return event_occurrences, event_durations
    
    def _apply_events_to_series(self, 
                               time_series: np.ndarray, 
                               event_occurrences: np.ndarray,
                               event_durations: np.ndarray,
                               effect_magnitude: float) -> np.ndarray:
        """
        Apply events to a time series.
        
        Args:
            time_series: Time series to apply events to.
            event_occurrences: Boolean array indicating event occurrences.
            event_durations: Integer array indicating event durations.
            effect_magnitude: Magnitude of the event effect.
            
        Returns:
            NumPy array with the same shape as time_series with events applied.
        """
        num_samples, time_periods = time_series.shape
        result = time_series.copy()
        
        # Apply events
        for i in range(num_samples):
            for t in range(time_periods):
                if event_occurrences[i, t]:
                    duration = min(event_durations[i, t], time_periods - t)
                    for d in range(duration):
                        if t + d < time_periods:
                            result[i, t + d] += effect_magnitude
        
        return result
    
    def _generate_individual_variations(self, 
                                       base_values: np.ndarray, 
                                       variation_std: float) -> np.ndarray:
        """
        Generate individual variations for a set of base values.
        
        Args:
            base_values: Base values to apply variations to.
            variation_std: Standard deviation of the variations.
            
        Returns:
            NumPy array with the same shape as base_values with individual variations applied.
        """
        return base_values + self.rng.normal(0, variation_std, base_values.shape)
