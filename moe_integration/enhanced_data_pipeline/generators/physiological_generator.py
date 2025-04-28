"""
Physiological Data Generator module for the Enhanced Data Generation Pipeline.

This module provides functionality for generating realistic physiological data patterns
with circadian rhythms, exercise effects, and interdependencies between parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from moe_data_pipeline.generators.base_generator import BaseGenerator


class PhysiologicalGenerator(BaseGenerator):
    """
    Physiological Data Generator for the Enhanced Data Generation Pipeline.
    
    This class generates realistic physiological data patterns with circadian rhythms,
    exercise effects, and interdependencies between physiological parameters.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the Physiological Data Generator.
        
        Args:
            config: Configuration dictionary for the generator.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(config, seed)
        
        # Extract physiological-specific configuration
        self.physio_config = config.get("physiological", {})
        
        # Set default values if not provided in config
        self.heart_rate_mean = self.physio_config.get("heart_rate_mean", 75.0)
        self.heart_rate_std = self.physio_config.get("heart_rate_std", 10.0)
        self.systolic_mean = self.physio_config.get("systolic_mean", 120.0)
        self.systolic_std = self.physio_config.get("systolic_std", 15.0)
        self.diastolic_mean = self.physio_config.get("diastolic_mean", 80.0)
        self.diastolic_std = self.physio_config.get("diastolic_std", 10.0)
        self.temperature_mean = self.physio_config.get("temperature_mean", 36.8)
        self.temperature_std = self.physio_config.get("temperature_std", 0.5)
        self.respiratory_mean = self.physio_config.get("respiratory_mean", 16.0)
        self.respiratory_std = self.physio_config.get("respiratory_std", 3.0)
        
        # Feature flags
        self.circadian_rhythm_enabled = self.physio_config.get("circadian_rhythm_enabled", True)
        self.exercise_effect_enabled = self.physio_config.get("exercise_effect_enabled", True)
        self.parameter_correlations_enabled = self.physio_config.get("parameter_correlations_enabled", True)
    
    def generate(self, num_samples: int, time_periods: int, 
                stress_level: Optional[np.ndarray] = None,
                exercise_duration: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Generate synthetic physiological data.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods for time-series data.
            stress_level: Optional stress level data for correlation.
            exercise_duration: Optional exercise duration data for correlation.
            
        Returns:
            Dictionary containing the generated physiological data with the following keys:
                - heart_rate: Heart rate in beats per minute
                - systolic: Systolic blood pressure in mmHg
                - diastolic: Diastolic blood pressure in mmHg
                - temperature: Body temperature in degrees Celsius
                - respiratory_rate: Respiratory rate in breaths per minute
                - heart_rate_variability: Heart rate variability (measure of autonomic function)
        """
        # Generate heart rate with circadian rhythm and exercise effects
        heart_rate = self._generate_heart_rate(num_samples, time_periods, stress_level, exercise_duration)
        
        # Generate heart rate variability (derived from heart rate)
        heart_rate_variability = self._generate_heart_rate_variability(num_samples, time_periods, heart_rate, stress_level)
        
        # Generate blood pressure with correlation to heart rate and stress
        systolic, diastolic = self._generate_blood_pressure(num_samples, time_periods, heart_rate, stress_level)
        
        # Generate body temperature with circadian rhythm
        temperature = self._generate_temperature(num_samples, time_periods, exercise_duration)
        
        # Generate respiratory rate with correlation to heart rate and exercise
        respiratory_rate = self._generate_respiratory_rate(num_samples, time_periods, heart_rate, exercise_duration)
        
        return {
            "heart_rate": heart_rate,
            "systolic": systolic,
            "diastolic": diastolic,
            "temperature": temperature,
            "respiratory_rate": respiratory_rate,
            "heart_rate_variability": heart_rate_variability
        }
    
    def _generate_heart_rate(self, num_samples: int, time_periods: int,
                            stress_level: Optional[np.ndarray] = None,
                            exercise_duration: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate heart rate data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            stress_level: Optional stress level data for correlation.
            exercise_duration: Optional exercise duration data for correlation.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing heart rate in beats per minute.
        """
        # Generate base heart rate
        heart_rate = self._generate_time_series(
            base_value=self.heart_rate_mean,
            std_dev=self.heart_rate_std / 2,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.8  # Strong hour-to-hour correlation
        )
        
        # Apply circadian rhythm if enabled
        if self.circadian_rhythm_enabled:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            # Create circadian heart rate pattern
            # Heart rate is lower during sleep and higher during day
            circadian_pattern = np.zeros(hours_per_day)
            # Sleep hours (0-6 AM)
            circadian_pattern[0:6] = -10.0
            # Morning increase (7-9 AM)
            circadian_pattern[7:10] = 5.0
            # Daytime (10 AM-8 PM)
            circadian_pattern[10:20] = 0.0
            # Evening decrease (9-11 PM)
            circadian_pattern[21:24] = -5.0
            
            # Apply pattern to each day
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                
                if end_idx <= time_periods:
                    for h in range(hours_per_day):
                        heart_rate[:, start_idx + h] += circadian_pattern[h]
        
        # Apply exercise effects if enabled and exercise data provided
        if self.exercise_effect_enabled and exercise_duration is not None:
            for t in range(time_periods):
                # Exercise increases heart rate
                # Assume exercise duration is in minutes
                exercise_effect = 0.5 * exercise_duration[:, t]  # 0.5 bpm increase per minute of exercise
                
                # Apply immediate effect
                heart_rate[:, t] += exercise_effect
                
                # Apply lingering effect (gradually decreasing)
                for d in range(1, 4):  # Effect lasts for 3 hours
                    if t + d < time_periods:
                        decay_factor = 1 - (d / 4)  # Linear decay
                        heart_rate[:, t + d] += exercise_effect * decay_factor
        
        # Apply stress effects if stress data provided
        if stress_level is not None:
            for t in range(time_periods):
                # Stress increases heart rate
                stress_effect = 0.5 * (stress_level[:, t] - 5)  # 0.5 bpm per unit of stress above baseline
                heart_rate[:, t] += stress_effect
        
        # Apply individual variations
        # Some people have naturally higher or lower heart rates
        baseline_variation = self.rng.normal(0, 5.0, num_samples)
        for i in range(num_samples):
            heart_rate[i, :] += baseline_variation[i]
        
        # Apply fitness level variations
        # Fitter individuals have lower resting heart rates
        fitness_level = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            # Negative correlation between fitness and heart rate
            heart_rate[i, :] -= 5.0 * fitness_level[i]
        
        # Ensure heart rate is within physiological bounds (40-200 bpm)
        heart_rate = np.clip(heart_rate, 40.0, 200.0)
        
        return heart_rate
    
    def _generate_heart_rate_variability(self, num_samples: int, time_periods: int,
                                        heart_rate: np.ndarray,
                                        stress_level: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate heart rate variability data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            heart_rate: Heart rate data.
            stress_level: Optional stress level data for correlation.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing heart rate variability.
        """
        # Initialize heart rate variability array
        hrv = np.zeros((num_samples, time_periods))
        
        # Generate base HRV (inversely related to heart rate)
        for t in range(time_periods):
            # HRV is generally higher when heart rate is lower
            hrv[:, t] = 100 - 0.5 * (heart_rate[:, t] - 60)
        
        # Apply individual variations
        hrv_baseline = self.rng.normal(0, 10.0, num_samples)
        for i in range(num_samples):
            hrv[i, :] += hrv_baseline[i]
        
        # Apply stress effects if stress data provided
        if stress_level is not None:
            for t in range(time_periods):
                # Stress decreases HRV
                stress_effect = -2.0 * (stress_level[:, t] - 5)  # 2 units per unit of stress above baseline
                hrv[:, t] += stress_effect
        
        # Apply age-related variations
        # HRV tends to decrease with age
        age_factor = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            # Negative correlation between age and HRV
            hrv[i, :] -= 5.0 * max(0, age_factor[i])
        
        # Apply fitness level variations
        # Fitter individuals have higher HRV
        fitness_level = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            # Positive correlation between fitness and HRV
            hrv[i, :] += 10.0 * max(0, fitness_level[i])
        
        # Ensure HRV is within realistic bounds (10-100)
        hrv = np.clip(hrv, 10.0, 100.0)
        
        return hrv
    
    def _generate_blood_pressure(self, num_samples: int, time_periods: int,
                               heart_rate: np.ndarray,
                               stress_level: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate blood pressure data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            heart_rate: Heart rate data.
            stress_level: Optional stress level data for correlation.
            
        Returns:
            Tuple of (systolic, diastolic) arrays of shape (num_samples, time_periods)
            containing blood pressure in mmHg.
        """
        # Generate base systolic blood pressure
        systolic = self._generate_time_series(
            base_value=self.systolic_mean,
            std_dev=self.systolic_std / 2,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.9  # Very strong hour-to-hour correlation
        )
        
        # Generate base diastolic blood pressure
        diastolic = self._generate_time_series(
            base_value=self.diastolic_mean,
            std_dev=self.diastolic_std / 2,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.9  # Very strong hour-to-hour correlation
        )
        
        # Apply correlation with heart rate if parameter correlations enabled
        if self.parameter_correlations_enabled:
            for t in range(time_periods):
                # Blood pressure tends to increase with heart rate
                hr_effect_systolic = 0.3 * (heart_rate[:, t] - self.heart_rate_mean)
                hr_effect_diastolic = 0.15 * (heart_rate[:, t] - self.heart_rate_mean)
                
                systolic[:, t] += hr_effect_systolic
                diastolic[:, t] += hr_effect_diastolic
        
        # Apply stress effects if stress data provided
        if stress_level is not None:
            for t in range(time_periods):
                # Stress increases blood pressure
                stress_effect_systolic = 0.5 * (stress_level[:, t] - 5)  # 0.5 mmHg per unit of stress above baseline
                stress_effect_diastolic = 0.3 * (stress_level[:, t] - 5)  # 0.3 mmHg per unit of stress above baseline
                
                systolic[:, t] += stress_effect_systolic
                diastolic[:, t] += stress_effect_diastolic
        
        # Apply circadian rhythm if enabled
        if self.circadian_rhythm_enabled:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            # Create circadian blood pressure pattern
            # Blood pressure is lower during sleep and higher during day
            circadian_pattern_systolic = np.zeros(hours_per_day)
            circadian_pattern_diastolic = np.zeros(hours_per_day)
            
            # Sleep hours (0-6 AM)
            circadian_pattern_systolic[0:6] = -10.0
            circadian_pattern_diastolic[0:6] = -5.0
            
            # Morning increase (7-9 AM)
            circadian_pattern_systolic[7:10] = 10.0
            circadian_pattern_diastolic[7:10] = 5.0
            
            # Daytime (10 AM-8 PM)
            circadian_pattern_systolic[10:20] = 0.0
            circadian_pattern_diastolic[10:20] = 0.0
            
            # Evening decrease (9-11 PM)
            circadian_pattern_systolic[21:24] = -5.0
            circadian_pattern_diastolic[21:24] = -3.0
            
            # Apply pattern to each day
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                
                if end_idx <= time_periods:
                    for h in range(hours_per_day):
                        systolic[:, start_idx + h] += circadian_pattern_systolic[h]
                        diastolic[:, start_idx + h] += circadian_pattern_diastolic[h]
        
        # Apply individual variations
        # Some people have naturally higher or lower blood pressure
        bp_baseline_variation = self.rng.normal(0, 10.0, num_samples)
        for i in range(num_samples):
            systolic[i, :] += bp_baseline_variation[i]
            diastolic[i, :] += 0.6 * bp_baseline_variation[i]  # Diastolic varies less than systolic
        
        # Ensure systolic is always greater than diastolic
        for t in range(time_periods):
            for i in range(num_samples):
                if systolic[i, t] <= diastolic[i, t]:
                    # Adjust to maintain realistic pulse pressure (systolic - diastolic)
                    mean_bp = (systolic[i, t] + diastolic[i, t]) / 2
                    pulse_pressure = max(30, systolic[i, t] - diastolic[i, t])  # Minimum 30 mmHg pulse pressure
                    
                    systolic[i, t] = mean_bp + pulse_pressure / 2
                    diastolic[i, t] = mean_bp - pulse_pressure / 2
        
        # Ensure blood pressure is within physiological bounds
        systolic = np.clip(systolic, 70.0, 200.0)
        diastolic = np.clip(diastolic, 40.0, 120.0)
        
        return systolic, diastolic
    
    def _generate_temperature(self, num_samples: int, time_periods: int,
                             exercise_duration: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate body temperature data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            exercise_duration: Optional exercise duration data for correlation.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing body temperature in degrees Celsius.
        """
        # Generate base body temperature
        temperature = self._generate_time_series(
            base_value=self.temperature_mean,
            std_dev=self.temperature_std / 3,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.9  # Very strong hour-to-hour correlation
        )
        
        # Apply circadian rhythm if enabled
        if self.circadian_rhythm_enabled:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            # Create circadian temperature pattern
            # Body temperature is lower during sleep and higher during day
            # Typical daily variation is about 0.5°C
            circadian_pattern = 0.25 * np.sin(2 * np.pi * np.arange(hours_per_day) / hours_per_day - np.pi/2)
            
            # Apply pattern to each day
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                
                if end_idx <= time_periods:
                    for h in range(hours_per_day):
                        temperature[:, start_idx + h] += circadian_pattern[h]
        
        # Apply exercise effects if enabled and exercise data provided
        if self.exercise_effect_enabled and exercise_duration is not None:
            for t in range(time_periods):
                # Exercise increases body temperature
                # Assume exercise duration is in minutes
                exercise_effect = 0.01 * exercise_duration[:, t]  # 0.01°C increase per minute of exercise
                
                # Apply immediate effect
                temperature[:, t] += exercise_effect
                
                # Apply lingering effect (gradually decreasing)
                for d in range(1, 3):  # Effect lasts for 2 hours
                    if t + d < time_periods:
                        decay_factor = 1 - (d / 3)  # Linear decay
                        temperature[:, t + d] += exercise_effect * decay_factor
        
        # Apply individual variations
        # Some people have naturally higher or lower body temperature
        temp_baseline_variation = self.rng.normal(0, 0.2, num_samples)
        for i in range(num_samples):
            temperature[i, :] += temp_baseline_variation[i]
        
        # Apply menstrual cycle variations for a subset of individuals
        # Assume approximately half the population experiences menstrual cycles
        has_cycle = self.rng.random(num_samples) < 0.5
        cycle_phase = self.rng.uniform(0, 28, num_samples)  # Random starting point in 28-day cycle
        
        for i in range(num_samples):
            if has_cycle[i]:
                # Assume time periods represent hours
                hours_per_day = 24
                days = time_periods // hours_per_day
                
                for day in range(days):
                    # Update cycle phase
                    current_phase = (cycle_phase[i] + day) % 28
                    
                    # Temperature increases after ovulation (typically days 14-28)
                    if 14 <= current_phase < 28:
                        temp_increase = 0.3  # About 0.3°C increase during luteal phase
                        
                        start_idx = day * hours_per_day
                        end_idx = min(start_idx + hours_per_day, time_periods)
                        
                        temperature[i, start_idx:end_idx] += temp_increase
        
        # Ensure temperature is within physiological bounds (35-42°C)
        temperature = np.clip(temperature, 35.0, 42.0)
        
        return temperature
    
    def _generate_respiratory_rate(self, num_samples: int, time_periods: int,
                                 heart_rate: np.ndarray,
                                 exercise_duration: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate respiratory rate data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            heart_rate: Heart rate data.
            exercise_duration: Optional exercise duration data for correlation.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing respiratory rate in breaths per minute.
        """
        # Generate base respiratory rate
        respiratory_rate = self._generate_time_series(
            base_value=self.respiratory_mean,
            std_dev=self.respiratory_std / 2,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.8  # Strong hour-to-hour correlation
        )
        
        # Apply correlation with heart rate if parameter correlations enabled
        if self.parameter_correlations_enabled:
            for t in range(time_periods):
                # Respiratory rate tends to increase with heart rate
                # Typical ratio is about 1:4 (respiratory rate:heart rate)
                hr_effect = 0.05 * (heart_rate[:, t] - self.heart_rate_mean)
                respiratory_rate[:, t] += hr_effect
        
        # Apply exercise effects if enabled and exercise data provided
        if self.exercise_effect_enabled and exercise_duration is not None:
            for t in range(time_periods):
                # Exercise increases respiratory rate
                # Assume exercise duration is in minutes
                exercise_effect = 0.1 * exercise_duration[:, t]  # 0.1 breaths/min increase per minute of exercise
                
                # Apply immediate effect
                respiratory_rate[:, t] += exercise_effect
                
                # Apply lingering effect (gradually decreasing)
                for d in range(1, 3):  # Effect lasts for 2 hours
                    if t + d < time_periods:
                        decay_factor = 1 - (d / 3)  # Linear decay
                        respiratory_rate[:, t + d] += exercise_effect * decay_factor
        
        # Apply circadian rhythm if enabled
        if self.circadian_rhythm_enabled:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            # Create circadian respiratory pattern
            # Respiratory rate is lower during sleep and higher during day
            circadian_pattern = np.zeros(hours_per_day)
            # Sleep hours (0-6 AM)
            circadian_pattern[0:6] = -2.0
            # Morning increase (7-9 AM)
            circadian_pattern[7:10] = 1.0
            # Daytime (10 AM-8 PM)
            circadian_pattern[10:20] = 0.0
            # Evening decrease (9-11 PM)
            circadian_pattern[21:24] = -1.0
            
            # Apply pattern to each day
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                
                if end_idx <= time_periods:
                    for h in range(hours_per_day):
                        respiratory_rate[:, start_idx + h] += circadian_pattern[h]
        
        # Apply individual variations
        # Some people have naturally higher or lower respiratory rates
        resp_baseline_variation = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            respiratory_rate[i, :] += resp_baseline_variation[i]
        
        # Ensure respiratory rate is within physiological bounds (8-40 breaths/min)
        respiratory_rate = np.clip(respiratory_rate, 8.0, 40.0)
        
        return respiratory_rate
