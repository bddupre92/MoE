"""
Sleep Data Generator module for the Enhanced Data Generation Pipeline.

This module provides functionality for generating realistic sleep data patterns
with circadian rhythms, disruptions, and recovery patterns.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from moe_data_pipeline.generators.base_generator import BaseGenerator


class SleepDataGenerator(BaseGenerator):
    """
    Sleep Data Generator for the Enhanced Data Generation Pipeline.
    
    This class generates realistic sleep data patterns with circadian rhythms,
    disruptions, and recovery patterns.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the Sleep Data Generator.
        
        Args:
            config: Configuration dictionary for the generator.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(config, seed)
        
        # Extract sleep-specific configuration
        self.sleep_config = config.get("sleep", {})
        
        # Set default values if not provided in config
        self.duration_mean = self.sleep_config.get("duration_mean", 7.0)
        self.duration_std = self.sleep_config.get("duration_std", 1.5)
        self.quality_mean = self.sleep_config.get("quality_mean", 6.0)
        self.quality_std = self.sleep_config.get("quality_std", 2.0)
        self.deep_sleep_mean = self.sleep_config.get("deep_sleep_mean", 20.0)
        self.deep_sleep_std = self.sleep_config.get("deep_sleep_std", 5.0)
        self.rem_sleep_mean = self.sleep_config.get("rem_sleep_mean", 25.0)
        self.rem_sleep_std = self.sleep_config.get("rem_sleep_std", 5.0)
        self.interruptions_lambda = self.sleep_config.get("interruptions_lambda", 2.0)
        self.time_to_sleep_lambda = self.sleep_config.get("time_to_sleep_lambda", 15.0)
        
        # Feature flags
        self.circadian_rhythm_enabled = self.sleep_config.get("circadian_rhythm_enabled", True)
        self.weekend_variation_enabled = self.sleep_config.get("weekend_variation_enabled", True)
    
    def generate(self, num_samples: int, time_periods: int) -> Dict[str, np.ndarray]:
        """
        Generate synthetic sleep data.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods for time-series data.
            
        Returns:
            Dictionary containing the generated sleep data with the following keys:
                - duration: Sleep duration in hours
                - quality: Sleep quality score (0-10)
                - deep_sleep_percentage: Percentage of deep sleep
                - rem_sleep_percentage: Percentage of REM sleep
                - interruptions: Number of sleep interruptions
                - time_to_sleep: Time to fall asleep in minutes
        """
        # Generate base sleep duration with weekly patterns
        duration = self._generate_sleep_duration(num_samples, time_periods)
        
        # Generate sleep quality
        quality = self._generate_sleep_quality(num_samples, time_periods, duration)
        
        # Generate deep sleep percentage
        deep_sleep_percentage = self._generate_deep_sleep_percentage(num_samples, time_periods, quality)
        
        # Generate REM sleep percentage
        rem_sleep_percentage = self._generate_rem_sleep_percentage(num_samples, time_periods, quality, deep_sleep_percentage)
        
        # Generate sleep interruptions
        interruptions = self._generate_sleep_interruptions(num_samples, time_periods, quality)
        
        # Generate time to fall asleep
        time_to_sleep = self._generate_time_to_sleep(num_samples, time_periods, quality)
        
        return {
            "duration": duration,
            "quality": quality,
            "deep_sleep_percentage": deep_sleep_percentage,
            "rem_sleep_percentage": rem_sleep_percentage,
            "interruptions": interruptions,
            "time_to_sleep": time_to_sleep
        }
    
    def _generate_sleep_duration(self, num_samples: int, time_periods: int) -> np.ndarray:
        """
        Generate sleep duration data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing sleep duration in hours.
        """
        # Generate base sleep duration
        duration = self._generate_time_series(
            base_value=self.duration_mean,
            std_dev=self.duration_std,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.7  # Strong day-to-day correlation
        )
        
        # Apply weekend variation if enabled
        if self.weekend_variation_enabled:
            # Assume time periods represent days, with weekends every 7th and 8th day
            for t in range(time_periods):
                if t % 7 == 5 or t % 7 == 6:  # Weekend (Saturday or Sunday)
                    # People tend to sleep longer on weekends
                    duration[:, t] += self.rng.normal(1.0, 0.5, num_samples)
        
        # Apply individual variations
        individual_variations = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            duration[i, :] += individual_variations[i]
        
        # Ensure sleep duration is within realistic bounds (3-12 hours)
        duration = np.clip(duration, 3.0, 12.0)
        
        return duration
    
    def _generate_sleep_quality(self, num_samples: int, time_periods: int, duration: np.ndarray) -> np.ndarray:
        """
        Generate sleep quality data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            duration: Sleep duration data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing sleep quality scores (0-10).
        """
        # Generate base sleep quality
        quality = self._generate_time_series(
            base_value=self.quality_mean,
            std_dev=self.quality_std,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.6  # Moderate day-to-day correlation
        )
        
        # Apply correlation with sleep duration
        # Sleep quality tends to be better with optimal sleep duration (7-8 hours)
        for t in range(time_periods):
            for i in range(num_samples):
                # Optimal sleep duration effect
                optimal_effect = -0.5 * ((duration[i, t] - 7.5) ** 2)
                quality[i, t] += optimal_effect
        
        # Apply circadian rhythm effects if enabled
        if self.circadian_rhythm_enabled:
            # Simulate disruptions to circadian rhythm
            circadian_disruption = self.rng.normal(0, 1.0, (num_samples, time_periods))
            for t in range(1, time_periods):
                # Propagate disruptions with decay
                circadian_disruption[:, t] += 0.7 * circadian_disruption[:, t-1]
            
            # Apply disruption effect to quality
            quality -= 0.5 * np.abs(circadian_disruption)
        
        # Ensure sleep quality is within bounds (0-10)
        quality = np.clip(quality, 0.0, 10.0)
        
        return quality
    
    def _generate_deep_sleep_percentage(self, num_samples: int, time_periods: int, quality: np.ndarray) -> np.ndarray:
        """
        Generate deep sleep percentage data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            quality: Sleep quality data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing deep sleep percentages.
        """
        # Generate base deep sleep percentage
        deep_sleep = self._generate_time_series(
            base_value=self.deep_sleep_mean,
            std_dev=self.deep_sleep_std,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.5  # Moderate day-to-day correlation
        )
        
        # Apply correlation with sleep quality
        # Deep sleep percentage tends to be higher with better sleep quality
        for t in range(time_periods):
            deep_sleep[:, t] += 0.5 * (quality[:, t] - self.quality_mean)
        
        # Apply age-related variations
        # Simulate different age groups with different deep sleep patterns
        age_factors = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            # Younger people tend to have more deep sleep
            deep_sleep[i, :] -= 5.0 * max(0, age_factors[i])
        
        # Ensure deep sleep percentage is within realistic bounds (5-35%)
        deep_sleep = np.clip(deep_sleep, 5.0, 35.0)
        
        return deep_sleep
    
    def _generate_rem_sleep_percentage(self, num_samples: int, time_periods: int, 
                                      quality: np.ndarray, deep_sleep: np.ndarray) -> np.ndarray:
        """
        Generate REM sleep percentage data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            quality: Sleep quality data.
            deep_sleep: Deep sleep percentage data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing REM sleep percentages.
        """
        # Generate base REM sleep percentage
        rem_sleep = self._generate_time_series(
            base_value=self.rem_sleep_mean,
            std_dev=self.rem_sleep_std,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.4  # Moderate day-to-day correlation
        )
        
        # Apply correlation with sleep quality
        # REM sleep percentage tends to be higher with better sleep quality
        for t in range(time_periods):
            rem_sleep[:, t] += 0.3 * (quality[:, t] - self.quality_mean)
        
        # Ensure total sleep cycle percentages don't exceed 100%
        # Adjust REM sleep to ensure deep_sleep + rem_sleep <= 100
        total_percentage = deep_sleep + rem_sleep
        for t in range(time_periods):
            for i in range(num_samples):
                if total_percentage[i, t] > 100:
                    # Scale down REM sleep to fit within 100% total
                    reduction_factor = (100 - deep_sleep[i, t]) / rem_sleep[i, t]
                    rem_sleep[i, t] *= reduction_factor
        
        # Ensure REM sleep percentage is within realistic bounds (10-40%)
        rem_sleep = np.clip(rem_sleep, 10.0, 40.0)
        
        return rem_sleep
    
    def _generate_sleep_interruptions(self, num_samples: int, time_periods: int, quality: np.ndarray) -> np.ndarray:
        """
        Generate sleep interruption data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            quality: Sleep quality data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing number of sleep interruptions.
        """
        # Generate base interruptions using Poisson distribution
        interruptions = self.rng.poisson(self.interruptions_lambda, (num_samples, time_periods))
        
        # Apply correlation with sleep quality
        # More interruptions lead to lower sleep quality
        for t in range(time_periods):
            # Calculate quality-based adjustment factor
            quality_factor = (self.quality_mean - quality[:, t]) / 5.0
            
            # Apply adjustment (more interruptions for lower quality)
            adjustment = self.rng.poisson(quality_factor * 2, num_samples)
            interruptions[:, t] += adjustment
        
        # Apply individual variations
        sensitivity = self.rng.normal(1.0, 0.3, num_samples)
        for i in range(num_samples):
            interruptions[i, :] = np.round(interruptions[i, :] * sensitivity[i])
        
        # Ensure interruptions are non-negative integers
        interruptions = np.maximum(0, interruptions).astype(int)
        
        return interruptions
    
    def _generate_time_to_sleep(self, num_samples: int, time_periods: int, quality: np.ndarray) -> np.ndarray:
        """
        Generate time to fall asleep data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            quality: Sleep quality data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing time to fall asleep in minutes.
        """
        # Generate base time to sleep using exponential distribution
        time_to_sleep = self.rng.exponential(self.time_to_sleep_lambda, (num_samples, time_periods))
        
        # Apply correlation with sleep quality
        # Longer time to fall asleep correlates with lower sleep quality
        for t in range(time_periods):
            # Calculate quality-based adjustment factor
            quality_factor = (self.quality_mean - quality[:, t]) / 2.0
            
            # Apply adjustment (longer time for lower quality)
            time_to_sleep[:, t] *= (1.0 + quality_factor)
        
        # Apply stress-related variations
        # Simulate different stress levels affecting time to fall asleep
        stress_factors = self.rng.normal(0, 1.0, (num_samples, time_periods))
        time_to_sleep += 5.0 * np.maximum(0, stress_factors)
        
        # Apply individual variations
        insomnia_tendency = self.rng.exponential(1.0, num_samples)
        for i in range(num_samples):
            time_to_sleep[i, :] *= (1.0 + insomnia_tendency[i])
        
        # Ensure time to sleep is within realistic bounds (1-120 minutes)
        time_to_sleep = np.clip(time_to_sleep, 1.0, 120.0)
        
        return time_to_sleep
