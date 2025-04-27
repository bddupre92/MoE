"""
Enhanced Synthetic Data Generator for FuseMoE

This module provides an enhanced synthetic data generator for the
FuseMoE system with temporal patterns, concept drift, and complex feature interactions.
"""

import numpy as np
import pandas as pd
import torch
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from scipy.stats import norm, poisson, expon
from sklearn.preprocessing import MinMaxScaler


class EnhancedMigraineDataGenerator:
    """
    Enhanced synthetic data generator for migraine prediction.
    
    This class provides methods for generating synthetic data with temporal patterns,
    concept drift, and complex feature interactions for migraine prediction.
    
    Attributes:
        seed (int): Random seed for reproducibility
        num_patients (int): Number of patients to simulate
        days_per_patient (int): Number of days to simulate per patient
        include_temporal_patterns (bool): Whether to include temporal patterns
        include_concept_drift (bool): Whether to include concept drift
        include_complex_interactions (bool): Whether to include complex feature interactions
        include_patient_heterogeneity (bool): Whether to include patient-specific variations
        include_anomalies (bool): Whether to include anomalies and edge cases
    """
    
    def __init__(self, 
                 seed: int = 42, 
                 num_patients: int = 100, 
                 days_per_patient: int = 90,
                 include_temporal_patterns: bool = True,
                 include_concept_drift: bool = True,
                 include_complex_interactions: bool = True,
                 include_patient_heterogeneity: bool = True,
                 include_anomalies: bool = True):
        """
        Initialize the enhanced migraine data generator.
        
        Args:
            seed: Random seed for reproducibility
            num_patients: Number of patients to simulate
            days_per_patient: Number of days to simulate per patient
            include_temporal_patterns: Whether to include temporal patterns
            include_concept_drift: Whether to include concept drift
            include_complex_interactions: Whether to include complex feature interactions
            include_patient_heterogeneity: Whether to include patient-specific variations
            include_anomalies: Whether to include anomalies and edge cases
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_patients = num_patients
        self.days_per_patient = days_per_patient
        self.total_samples = num_patients * days_per_patient
        
        self.include_temporal_patterns = include_temporal_patterns
        self.include_concept_drift = include_concept_drift
        self.include_complex_interactions = include_complex_interactions
        self.include_patient_heterogeneity = include_patient_heterogeneity
        self.include_anomalies = include_anomalies
        
        # Initialize patient profiles if using patient heterogeneity
        if self.include_patient_heterogeneity:
            self._initialize_patient_profiles()
        
        # Set base parameters for data generation
        self._initialize_base_parameters()
        
        # Initialize seasonal patterns if using concept drift
        if self.include_concept_drift:
            self._initialize_seasonal_patterns()
    
    def _initialize_patient_profiles(self):
        """
        Initialize patient-specific profiles.
        """
        # Generate patient-specific sensitivity to different triggers
        self.patient_profiles = {
            'sleep_sensitivity': np.random.normal(1.0, 0.3, self.num_patients),
            'weather_sensitivity': np.random.normal(1.0, 0.4, self.num_patients),
            'stress_sensitivity': np.random.normal(1.0, 0.5, self.num_patients),
            'physio_sensitivity': np.random.normal(1.0, 0.3, self.num_patients),
            'baseline_frequency': np.random.beta(2, 5, self.num_patients),  # Baseline migraine frequency
            'recovery_time': np.random.poisson(3, self.num_patients) + 1,   # Days to recover from migraine
            'medication_response': np.random.beta(3, 2, self.num_patients)  # Effectiveness of medication
        }
        
        # Ensure sensitivities are positive
        for key in ['sleep_sensitivity', 'weather_sensitivity', 'stress_sensitivity', 'physio_sensitivity']:
            self.patient_profiles[key] = np.maximum(0.1, self.patient_profiles[key])
        
        # Generate patient types (cluster patients into types)
        num_types = 4
        self.patient_types = np.random.randint(0, num_types, self.num_patients)
        
        # Define trigger profiles for each patient type
        self.type_profiles = {
            0: {'name': 'Weather Sensitive', 'primary_trigger': 'weather', 'secondary_trigger': 'stress'},
            1: {'name': 'Sleep Sensitive', 'primary_trigger': 'sleep', 'secondary_trigger': 'physio'},
            2: {'name': 'Stress Sensitive', 'primary_trigger': 'stress', 'secondary_trigger': 'sleep'},
            3: {'name': 'Mixed Triggers', 'primary_trigger': None, 'secondary_trigger': None}
        }
        
        # Adjust sensitivities based on patient type
        for i in range(self.num_patients):
            patient_type = self.patient_types[i]
            if patient_type != 3:  # Not mixed type
                primary = self.type_profiles[patient_type]['primary_trigger'] + '_sensitivity'
                secondary = self.type_profiles[patient_type]['secondary_trigger'] + '_sensitivity'
                
                # Increase primary trigger sensitivity
                self.patient_profiles[primary][i] *= 1.5
                
                # Increase secondary trigger sensitivity
                self.patient_profiles[secondary][i] *= 1.2
    
    def _initialize_base_parameters(self):
        """
        Initialize base parameters for data generation.
        """
        # Sleep parameters
        self.sleep_params = {
            'duration_mean': 7.0,
            'duration_std': 1.5,
            'quality_mean': 6.0,
            'quality_std': 2.0,
            'deep_sleep_mean': 20.0,
            'deep_sleep_std': 5.0,
            'rem_sleep_mean': 25.0,
            'rem_sleep_std': 5.0,
            'interruptions_lambda': 2.0,
            'time_to_sleep_lambda': 15.0,
            'autocorrelation': 0.7,  # Day-to-day correlation
            'weekend_effect': 1.2    # Multiplier for weekend sleep duration
        }
        
        # Weather parameters
        self.weather_params = {
            'temperature_mean': 20.0,
            'temperature_std': 8.0,
            'humidity_mean': 60.0,
            'humidity_std': 15.0,
            'pressure_mean': 1013.0,
            'pressure_std': 10.0,
            'precipitation_lambda': 2.0,
            'wind_speed_lambda': 10.0,
            'cloud_cover_mean': 50.0,
            'cloud_cover_std': 25.0,
            'pressure_change_threshold': 6.0,  # Threshold for significant pressure change
            'temperature_change_threshold': 8.0,  # Threshold for significant temperature change
            'autocorrelation': 0.8  # Day-to-day correlation
        }
        
        # Stress and diet parameters
        self.stress_diet_params = {
            'stress_mean': 5.0,
            'stress_std': 2.0,
            'caffeine_lambda': 2.0,
            'alcohol_lambda': 1.0,
            'meal_regularity_mean': 6.0,
            'meal_regularity_std': 2.0,
            'hydration_mean': 6.0,
            'hydration_std': 2.0,
            'exercise_lambda': 30.0,
            'stress_autocorrelation': 0.6,  # Day-to-day correlation for stress
            'diet_autocorrelation': 0.5,    # Day-to-day correlation for diet
            'weekend_alcohol_multiplier': 2.0,  # Increased alcohol on weekends
            'workweek_stress_multiplier': 1.3   # Increased stress during workweek
        }
        
        # Physiological parameters
        self.physio_params = {
            'heart_rate_mean': 75.0,
            'heart_rate_std': 10.0,
            'systolic_mean': 120.0,
            'systolic_std': 15.0,
            'diastolic_mean': 80.0,
            'diastolic_std': 10.0,
            'body_temperature_mean': 36.8,
            'body_temperature_std': 0.5,
            'respiratory_mean': 16.0,
            'respiratory_std': 3.0,
            'oxygen_saturation_mean': 97.0,
            'oxygen_saturation_std': 2.0,
            'autocorrelation': 0.75,  # Day-to-day correlation
            'exercise_effect': 0.2    # Effect of exercise on heart rate
        }
        
        # Migraine parameters
        self.migraine_params = {
            'base_weights': {
                'sleep': {
                    'sleep_duration': -0.3,
                    'sleep_quality': -0.4,
                    'deep_sleep_percentage': -0.2,
                    'rem_sleep_percentage': 0.1,
                    'sleep_interruptions': 0.3,
                    'time_to_sleep': 0.2
                },
                'weather': {
                    'temperature': 0.1,
                    'humidity': 0.2,
                    'pressure': 0.4,
                    'precipitation': 0.2,
                    'wind_speed': 0.1,
                    'cloud_cover': 0.1,
                    'pressure_change': 0.5,  # New feature for pressure change
                    'temperature_change': 0.3  # New feature for temperature change
                },
                'stress_diet': {
                    'stress_level': 0.4,
                    'caffeine_intake': 0.3,
                    'alcohol_consumption': 0.2,
                    'meal_regularity': -0.3,
                    'hydration_level': -0.3,
                    'exercise_duration': -0.2
                },
                'physio': {
                    'heart_rate': 0.2,
                    'blood_pressure_systolic': 0.3,
                    'blood_pressure_diastolic': 0.3,
                    'body_temperature': 0.1,
                    'respiratory_rate': 0.1,
                    'oxygen_saturation': -0.2
                }
            },
            'threshold': 0.6,
            'noise_std': 0.3,
            'lag_effect_days': 2,  # Number of days for lagged effects
            'recovery_period': 3,  # Days after migraine with reduced probability
            'trigger_accumulation_rate': 0.3,  # Rate at which triggers accumulate
            'trigger_decay_rate': 0.2  # Rate at which triggers decay
        }
        
        # Interaction terms for complex interactions
        if self.include_complex_interactions:
            self.interaction_terms = {
                'stress_weather': 0.4,  # Stress amplifies weather sensitivity
                'sleep_stress': 0.3,    # Poor sleep amplifies stress sensitivity
                'dehydration_alcohol': 0.5,  # Dehydration amplifies alcohol effect
                'pressure_stress': 0.3,  # Pressure changes amplify stress effect
                'sleep_caffeine': 0.4,   # Poor sleep amplifies caffeine effect
                'threshold_effects': {
                    'stress_threshold': 7.0,  # Stress only matters above this level
                    'sleep_threshold': 5.0,   # Sleep only matters below this level
                    'pressure_change_threshold': 5.0  # Pressure change only matters above this
                }
            }
    
    def _initialize_seasonal_patterns(self):
        """
        Initialize seasonal patterns for concept drift.
        """
        # Define seasonal patterns (annual cycle)
        days_in_year = 365
        t = np.linspace(0, 2*np.pi, days_in_year)
        
        # Temperature seasonal pattern
        self.seasonal_patterns = {
            'temperature': 10 * np.sin(t),  # +/- 10 degrees C seasonal variation
            'humidity': 15 * np.sin(t + np.pi/6),  # +/- 15% humidity, slightly offset
            'pressure': 5 * np.sin(t + np.pi/4),  # +/- 5 hPa, offset
            'migraine_base_rate': 0.1 * np.sin(t + np.pi/3) + 0.1  # Seasonal variation in base rate
        }
        
        # Define concept drift patterns (gradual changes over time)
        drift_period = self.days_per_patient
        t_drift = np.linspace(0, 2*np.pi, drift_period)
        
        self.concept_drift = {
            'sleep_importance': 0.2 * np.sin(t_drift) + 1.0,  # Varying importance of sleep
            'weather_importance': 0.3 * np.sin(t_drift + np.pi/2) + 1.0,  # Varying importance of weather
            'stress_importance': 0.25 * np.sin(t_drift + np.pi) + 1.0,  # Varying importance of stress
            'threshold_drift': 0.1 * np.sin(t_drift + np.pi/4) + 0.6  # Drifting threshold
        }
    
    def _get_seasonal_adjustment(self, feature: str, day: int) -> float:
        """
        Get seasonal adjustment for a feature on a specific day.
        
        Args:
            feature: Feature name
            day: Day of the year (0-364)
            
        Returns:
            Seasonal adjustment value
        """
        if not self.include_concept_drift:
            return 0.0
        
        day_of_year = day % 365
        if feature in self.seasonal_patterns:
            return self.seasonal_patterns[feature][day_of_year]
        return 0.0
    
    def _get_concept_drift_factor(self, factor: str, day: int) -> float:
        """
        Get concept drift factor for a specific day.
        
        Args:
            factor: Factor name
            day: Day number
            
        Returns:
            Concept drift factor
        """
        if not self.include_concept_drift:
            return 1.0
        
        day_in_period = day % self.days_per_patient
        if factor in self.concept_drift:
            return self.concept_drift[factor][day_in_period]
        return 1.0
    
    def _apply_autocorrelation(self, previous_value: float, new_value: float, 
                              autocorr: float) -> float:
        """
        Apply autocorrelation between consecutive days.
        
        Args:
            previous_value: Value from previous day
            new_value: Independently generated new value
            autocorr: Autocorrelation coefficient
            
        Returns:
            New value with autocorrelation applied
        """
        if not self.include_temporal_patterns or previous_value is None:
            return new_value
        
        return autocorr * previous_value + (1 - autocorr) * new_value
    
    def _is_weekend(self, day: int) -> bool:
        """
        Check if a day is a weekend.
        
        Args:
            day: Day number
            
        Returns:
            True if weekend, False otherwise
        """
        return (day % 7) >= 5  # Days 5 and 6 are weekend
    
    def generate_sleep_data(self) -> pd.DataFrame:
        """
        Generate synthetic sleep data with temporal patterns.
        
        Returns:
            DataFrame containing synthetic sleep data
        """
        # Initialize arrays
        sleep_duration = np.zeros((self.num_patients, self.days_per_patient))
        sleep_quality = np.zeros((self.num_patients, self.days_per_patient))
        deep_sleep_percentage = np.zeros((self.num_patients, self.days_per_patient))
        rem_sleep_percentage = np.zeros((self.num_patients, self.days_per_patient))
        sleep_interruptions = np.zeros((self.num_patients, self.days_per_patient))
        time_to_sleep = np.zeros((self.num_patients, self.days_per_patient))
        
        # Generate data for each patient
        for patient in range(self.num_patients):
            # Patient-specific base parameters
            if self.include_patient_heterogeneity:
                patient_duration_mean = self.sleep_params['duration_mean'] * (0.8 + 0.4 * np.random.random())
                patient_quality_mean = self.sleep_params['quality_mean'] * (0.8 + 0.4 * np.random.random())
                patient_interruptions_lambda = self.sleep_params['interruptions_lambda'] * (0.7 + 0.6 * np.random.random())
            else:
                patient_duration_mean = self.sleep_params['duration_mean']
                patient_quality_mean = self.sleep_params['quality_mean']
                patient_interruptions_lambda = self.sleep_params['interruptions_lambda']
            
            # Generate data for each day
            for day in range(self.days_per_patient):
                # Apply weekend effect
                weekend_factor = self.sleep_params['weekend_effect'] if self._is_weekend(day) else 1.0
                
                # Generate new values
                new_duration = np.random.normal(
                    patient_duration_mean * weekend_factor,
                    self.sleep_params['duration_std']
                )
                
                new_quality = np.random.normal(
                    patient_quality_mean,
                    self.sleep_params['quality_std']
                )
                
                new_deep_sleep = np.random.normal(
                    self.sleep_params['deep_sleep_mean'],
                    self.sleep_params['deep_sleep_std']
                )
                
                new_rem_sleep = np.random.normal(
                    self.sleep_params['rem_sleep_mean'],
                    self.sleep_params['rem_sleep_std']
                )
                
                new_interruptions = np.random.poisson(patient_interruptions_lambda)
                
                new_time_to_sleep = np.random.poisson(self.sleep_params['time_to_sleep_lambda'])
                
                # Apply autocorrelation if not the first day
                if day > 0 and self.include_temporal_patterns:
                    sleep_duration[patient, day] = self._apply_autocorrelation(
                        sleep_duration[patient, day-1],
                        new_duration,
                        self.sleep_params['autocorrelation']
                    )
                    
                    sleep_quality[patient, day] = self._apply_autocorrelation(
                        sleep_quality[patient, day-1],
                        new_quality,
                        self.sleep_params['autocorrelation']
                    )
                    
                    deep_sleep_percentage[patient, day] = self._apply_autocorrelation(
                        deep_sleep_percentage[patient, day-1],
                        new_deep_sleep,
                        self.sleep_params['autocorrelation']
                    )
                    
                    rem_sleep_percentage[patient, day] = self._apply_autocorrelation(
                        rem_sleep_percentage[patient, day-1],
                        new_rem_sleep,
                        self.sleep_params['autocorrelation']
                    )
                    
                    # Interruptions and time to sleep have lower autocorrelation
                    sleep_interruptions[patient, day] = self._apply_autocorrelation(
                        sleep_interruptions[patient, day-1],
                        new_interruptions,
                        0.4
                    )
                    
                    time_to_sleep[patient, day] = self._apply_autocorrelation(
                        time_to_sleep[patient, day-1],
                        new_time_to_sleep,
                        0.4
                    )
                else:
                    sleep_duration[patient, day] = new_duration
                    sleep_quality[patient, day] = new_quality
                    deep_sleep_percentage[patient, day] = new_deep_sleep
                    rem_sleep_percentage[patient, day] = new_rem_sleep
                    sleep_interruptions[patient, day] = new_interruptions
                    time_to_sleep[patient, day] = new_time_to_sleep
        
        # Apply constraints
        sleep_duration = np.clip(sleep_duration, 0, 12)
        sleep_quality = np.clip(sleep_quality, 0, 10)
        deep_sleep_percentage = np.clip(deep_sleep_percentage, 0, 100)
        rem_sleep_percentage = np.clip(rem_sleep_percentage, 0, 100)
        sleep_interruptions = np.maximum(0, sleep_interruptions)
        time_to_sleep = np.maximum(0, time_to_sleep)
        
        # Reshape to 2D arrays
        sleep_duration_flat = sleep_duration.flatten()
        sleep_quality_flat = sleep_quality.flatten()
        deep_sleep_percentage_flat = deep_sleep_percentage.flatten()
        rem_sleep_percentage_flat = rem_sleep_percentage.flatten()
        sleep_interruptions_flat = sleep_interruptions.flatten()
        time_to_sleep_flat = time_to_sleep.flatten()
        
        # Create DataFrame
        sleep_data = pd.DataFrame({
            'patient_id': np.repeat(np.arange(self.num_patients), self.days_per_patient),
            'day': np.tile(np.arange(self.days_per_patient), self.num_patients),
            'sleep_duration': sleep_duration_flat,
            'sleep_quality': sleep_quality_flat,
            'deep_sleep_percentage': deep_sleep_percentage_flat,
            'rem_sleep_percentage': rem_sleep_percentage_flat,
            'sleep_interruptions': sleep_interruptions_flat,
            'time_to_sleep': time_to_sleep_flat
        })
        
        return sleep_data
    
    def generate_weather_data(self) -> pd.DataFrame:
        """
        Generate synthetic weather data with temporal patterns and seasonal effects.
        
        Returns:
            DataFrame containing synthetic weather data
        """
        # Initialize arrays
        temperature = np.zeros((self.num_patients, self.days_per_patient))
        humidity = np.zeros((self.num_patients, self.days_per_patient))
        pressure = np.zeros((self.num_patients, self.days_per_patient))
        precipitation = np.zeros((self.num_patients, self.days_per_patient))
        wind_speed = np.zeros((self.num_patients, self.days_per_patient))
        cloud_cover = np.zeros((self.num_patients, self.days_per_patient))
        
        # Generate data for each patient's location (patients in same area have similar weather)
        num_locations = min(10, self.num_patients)  # Group patients into locations
        location_assignments = np.random.randint(0, num_locations, self.num_patients)
        
        # Generate base weather patterns for each location
        for location in range(num_locations):
            # Generate data for each day
            for day in range(self.days_per_patient):
                # Apply seasonal adjustments
                temp_seasonal = self._get_seasonal_adjustment('temperature', day)
                humidity_seasonal = self._get_seasonal_adjustment('humidity', day)
                pressure_seasonal = self._get_seasonal_adjustment('pressure', day)
                
                # Generate new values with seasonal adjustments
                new_temp = np.random.normal(
                    self.weather_params['temperature_mean'] + temp_seasonal,
                    self.weather_params['temperature_std']
                )
                
                new_humidity = np.random.normal(
                    self.weather_params['humidity_mean'] + humidity_seasonal,
                    self.weather_params['humidity_std']
                )
                
                new_pressure = np.random.normal(
                    self.weather_params['pressure_mean'] + pressure_seasonal,
                    self.weather_params['pressure_std']
                )
                
                new_precip = np.random.exponential(self.weather_params['precipitation_lambda'])
                
                new_wind = np.random.exponential(self.weather_params['wind_speed_lambda'])
                
                new_cloud = np.random.normal(
                    self.weather_params['cloud_cover_mean'],
                    self.weather_params['cloud_cover_std']
                )
                
                # Apply autocorrelation if not the first day
                if day > 0 and self.include_temporal_patterns:
                    # Get patients in this location
                    location_patients = np.where(location_assignments == location)[0]
                    
                    for patient in location_patients:
                        temperature[patient, day] = self._apply_autocorrelation(
                            temperature[patient, day-1],
                            new_temp,
                            self.weather_params['autocorrelation']
                        )
                        
                        humidity[patient, day] = self._apply_autocorrelation(
                            humidity[patient, day-1],
                            new_humidity,
                            self.weather_params['autocorrelation']
                        )
                        
                        pressure[patient, day] = self._apply_autocorrelation(
                            pressure[patient, day-1],
                            new_pressure,
                            self.weather_params['autocorrelation']
                        )
                        
                        # Precipitation and wind have lower autocorrelation
                        precipitation[patient, day] = self._apply_autocorrelation(
                            precipitation[patient, day-1],
                            new_precip,
                            0.3
                        )
                        
                        wind_speed[patient, day] = self._apply_autocorrelation(
                            wind_speed[patient, day-1],
                            new_wind,
                            0.4
                        )
                        
                        cloud_cover[patient, day] = self._apply_autocorrelation(
                            cloud_cover[patient, day-1],
                            new_cloud,
                            0.6
                        )
                else:
                    # Get patients in this location
                    location_patients = np.where(location_assignments == location)[0]
                    
                    for patient in location_patients:
                        temperature[patient, day] = new_temp
                        humidity[patient, day] = new_humidity
                        pressure[patient, day] = new_pressure
                        precipitation[patient, day] = new_precip
                        wind_speed[patient, day] = new_wind
                        cloud_cover[patient, day] = new_cloud
        
        # Calculate pressure and temperature changes (important migraine triggers)
        pressure_change = np.zeros((self.num_patients, self.days_per_patient))
        temperature_change = np.zeros((self.num_patients, self.days_per_patient))
        
        for patient in range(self.num_patients):
            for day in range(1, self.days_per_patient):
                pressure_change[patient, day] = pressure[patient, day] - pressure[patient, day-1]
                temperature_change[patient, day] = temperature[patient, day] - temperature[patient, day-1]
        
        # Apply constraints
        humidity = np.clip(humidity, 0, 100)
        pressure = np.clip(pressure, 900, 1100)
        precipitation = np.maximum(0, precipitation)
        wind_speed = np.maximum(0, wind_speed)
        cloud_cover = np.clip(cloud_cover, 0, 100)
        
        # Reshape to 2D arrays
        temperature_flat = temperature.flatten()
        humidity_flat = humidity.flatten()
        pressure_flat = pressure.flatten()
        precipitation_flat = precipitation.flatten()
        wind_speed_flat = wind_speed.flatten()
        cloud_cover_flat = cloud_cover.flatten()
        pressure_change_flat = pressure_change.flatten()
        temperature_change_flat = temperature_change.flatten()
        
        # Create DataFrame
        weather_data = pd.DataFrame({
            'patient_id': np.repeat(np.arange(self.num_patients), self.days_per_patient),
            'day': np.tile(np.arange(self.days_per_patient), self.num_patients),
            'temperature': temperature_flat,
            'humidity': humidity_flat,
            'pressure': pressure_flat,
            'precipitation': precipitation_flat,
            'wind_speed': wind_speed_flat,
            'cloud_cover': cloud_cover_flat,
            'pressure_change': pressure_change_flat,
            'temperature_change': temperature_change_flat
        })
        
        return weather_data
    
    def generate_stress_diet_data(self) -> pd.DataFrame:
        """
        Generate synthetic stress and diet data with temporal patterns.
        
        Returns:
            DataFrame containing synthetic stress and diet data
        """
        # Initialize arrays
        stress_level = np.zeros((self.num_patients, self.days_per_patient))
        caffeine_intake = np.zeros((self.num_patients, self.days_per_patient))
        alcohol_consumption = np.zeros((self.num_patients, self.days_per_patient))
        meal_regularity = np.zeros((self.num_patients, self.days_per_patient))
        hydration_level = np.zeros((self.num_patients, self.days_per_patient))
        exercise_duration = np.zeros((self.num_patients, self.days_per_patient))
        
        # Generate data for each patient
        for patient in range(self.num_patients):
            # Patient-specific base parameters
            if self.include_patient_heterogeneity:
                patient_stress_mean = self.stress_diet_params['stress_mean'] * (0.7 + 0.6 * np.random.random())
                patient_caffeine_lambda = self.stress_diet_params['caffeine_lambda'] * (0.5 + 1.0 * np.random.random())
                patient_alcohol_lambda = self.stress_diet_params['alcohol_lambda'] * (0.2 + 1.6 * np.random.random())
                patient_exercise_lambda = self.stress_diet_params['exercise_lambda'] * (0.4 + 1.2 * np.random.random())
            else:
                patient_stress_mean = self.stress_diet_params['stress_mean']
                patient_caffeine_lambda = self.stress_diet_params['caffeine_lambda']
                patient_alcohol_lambda = self.stress_diet_params['alcohol_lambda']
                patient_exercise_lambda = self.stress_diet_params['exercise_lambda']
            
            # Generate data for each day
            for day in range(self.days_per_patient):
                # Apply weekend/workweek effects
                is_weekend = self._is_weekend(day)
                stress_factor = 1.0 if is_weekend else self.stress_diet_params['workweek_stress_multiplier']
                alcohol_factor = self.stress_diet_params['weekend_alcohol_multiplier'] if is_weekend else 1.0
                
                # Generate new values
                new_stress = np.random.normal(
                    patient_stress_mean * stress_factor,
                    self.stress_diet_params['stress_std']
                )
                
                new_caffeine = np.random.poisson(patient_caffeine_lambda)
                
                new_alcohol = np.random.poisson(patient_alcohol_lambda * alcohol_factor)
                
                new_meal_regularity = np.random.normal(
                    self.stress_diet_params['meal_regularity_mean'],
                    self.stress_diet_params['meal_regularity_std']
                )
                
                new_hydration = np.random.normal(
                    self.stress_diet_params['hydration_mean'],
                    self.stress_diet_params['hydration_std']
                )
                
                # Exercise more likely on weekends
                exercise_adj = 1.2 if is_weekend else 0.9
                new_exercise = np.random.exponential(patient_exercise_lambda * exercise_adj)
                
                # Apply autocorrelation if not the first day
                if day > 0 and self.include_temporal_patterns:
                    stress_level[patient, day] = self._apply_autocorrelation(
                        stress_level[patient, day-1],
                        new_stress,
                        self.stress_diet_params['stress_autocorrelation']
                    )
                    
                    caffeine_intake[patient, day] = self._apply_autocorrelation(
                        caffeine_intake[patient, day-1],
                        new_caffeine,
                        self.stress_diet_params['diet_autocorrelation']
                    )
                    
                    alcohol_consumption[patient, day] = self._apply_autocorrelation(
                        alcohol_consumption[patient, day-1],
                        new_alcohol,
                        self.stress_diet_params['diet_autocorrelation']
                    )
                    
                    meal_regularity[patient, day] = self._apply_autocorrelation(
                        meal_regularity[patient, day-1],
                        new_meal_regularity,
                        self.stress_diet_params['diet_autocorrelation']
                    )
                    
                    hydration_level[patient, day] = self._apply_autocorrelation(
                        hydration_level[patient, day-1],
                        new_hydration,
                        self.stress_diet_params['diet_autocorrelation']
                    )
                    
                    exercise_duration[patient, day] = self._apply_autocorrelation(
                        exercise_duration[patient, day-1],
                        new_exercise,
                        0.3  # Lower autocorrelation for exercise
                    )
                else:
                    stress_level[patient, day] = new_stress
                    caffeine_intake[patient, day] = new_caffeine
                    alcohol_consumption[patient, day] = new_alcohol
                    meal_regularity[patient, day] = new_meal_regularity
                    hydration_level[patient, day] = new_hydration
                    exercise_duration[patient, day] = new_exercise
        
        # Apply constraints
        stress_level = np.clip(stress_level, 0, 10)
        caffeine_intake = np.maximum(0, caffeine_intake)
        alcohol_consumption = np.maximum(0, alcohol_consumption)
        meal_regularity = np.clip(meal_regularity, 0, 10)
        hydration_level = np.clip(hydration_level, 0, 10)
        exercise_duration = np.maximum(0, exercise_duration)
        
        # Reshape to 2D arrays
        stress_level_flat = stress_level.flatten()
        caffeine_intake_flat = caffeine_intake.flatten()
        alcohol_consumption_flat = alcohol_consumption.flatten()
        meal_regularity_flat = meal_regularity.flatten()
        hydration_level_flat = hydration_level.flatten()
        exercise_duration_flat = exercise_duration.flatten()
        
        # Create DataFrame
        stress_diet_data = pd.DataFrame({
            'patient_id': np.repeat(np.arange(self.num_patients), self.days_per_patient),
            'day': np.tile(np.arange(self.days_per_patient), self.num_patients),
            'stress_level': stress_level_flat,
            'caffeine_intake': caffeine_intake_flat,
            'alcohol_consumption': alcohol_consumption_flat,
            'meal_regularity': meal_regularity_flat,
            'hydration_level': hydration_level_flat,
            'exercise_duration': exercise_duration_flat
        })
        
        return stress_diet_data
    
    def generate_physiological_data(self, stress_diet_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic physiological data with temporal patterns.
        
        Args:
            stress_diet_data: Stress and diet data to correlate with physiological data
            
        Returns:
            DataFrame containing synthetic physiological data
        """
        # Initialize arrays
        heart_rate = np.zeros((self.num_patients, self.days_per_patient))
        blood_pressure_systolic = np.zeros((self.num_patients, self.days_per_patient))
        blood_pressure_diastolic = np.zeros((self.num_patients, self.days_per_patient))
        body_temperature = np.zeros((self.num_patients, self.days_per_patient))
        respiratory_rate = np.zeros((self.num_patients, self.days_per_patient))
        oxygen_saturation = np.zeros((self.num_patients, self.days_per_patient))
        
        # Reshape stress and exercise data for correlation
        stress_reshaped = stress_diet_data['stress_level'].values.reshape(self.num_patients, self.days_per_patient)
        exercise_reshaped = stress_diet_data['exercise_duration'].values.reshape(self.num_patients, self.days_per_patient)
        
        # Generate data for each patient
        for patient in range(self.num_patients):
            # Patient-specific base parameters
            if self.include_patient_heterogeneity:
                patient_heart_rate_mean = self.physio_params['heart_rate_mean'] * (0.9 + 0.2 * np.random.random())
                patient_systolic_mean = self.physio_params['systolic_mean'] * (0.9 + 0.2 * np.random.random())
                patient_diastolic_mean = self.physio_params['diastolic_mean'] * (0.9 + 0.2 * np.random.random())
            else:
                patient_heart_rate_mean = self.physio_params['heart_rate_mean']
                patient_systolic_mean = self.physio_params['systolic_mean']
                patient_diastolic_mean = self.physio_params['diastolic_mean']
            
            # Generate data for each day
            for day in range(self.days_per_patient):
                # Correlate with stress and exercise
                stress_effect = 0.2 * stress_reshaped[patient, day]  # Stress increases heart rate and BP
                exercise_effect = self.physio_params['exercise_effect'] * exercise_reshaped[patient, day]
                
                # Generate new values
                new_heart_rate = np.random.normal(
                    patient_heart_rate_mean + stress_effect + exercise_effect,
                    self.physio_params['heart_rate_std']
                )
                
                new_systolic = np.random.normal(
                    patient_systolic_mean + 2 * stress_effect,
                    self.physio_params['systolic_std']
                )
                
                new_diastolic = np.random.normal(
                    patient_diastolic_mean + 1.5 * stress_effect,
                    self.physio_params['diastolic_std']
                )
                
                new_temperature = np.random.normal(
                    self.physio_params['body_temperature_mean'],
                    self.physio_params['body_temperature_std']
                )
                
                new_respiratory = np.random.normal(
                    self.physio_params['respiratory_mean'] + 0.1 * stress_effect + 0.2 * exercise_effect,
                    self.physio_params['respiratory_std']
                )
                
                new_oxygen = np.random.normal(
                    self.physio_params['oxygen_saturation_mean'] - 0.1 * stress_effect + 0.1 * exercise_effect,
                    self.physio_params['oxygen_saturation_std']
                )
                
                # Apply autocorrelation if not the first day
                if day > 0 and self.include_temporal_patterns:
                    heart_rate[patient, day] = self._apply_autocorrelation(
                        heart_rate[patient, day-1],
                        new_heart_rate,
                        self.physio_params['autocorrelation']
                    )
                    
                    blood_pressure_systolic[patient, day] = self._apply_autocorrelation(
                        blood_pressure_systolic[patient, day-1],
                        new_systolic,
                        self.physio_params['autocorrelation']
                    )
                    
                    blood_pressure_diastolic[patient, day] = self._apply_autocorrelation(
                        blood_pressure_diastolic[patient, day-1],
                        new_diastolic,
                        self.physio_params['autocorrelation']
                    )
                    
                    body_temperature[patient, day] = self._apply_autocorrelation(
                        body_temperature[patient, day-1],
                        new_temperature,
                        self.physio_params['autocorrelation']
                    )
                    
                    respiratory_rate[patient, day] = self._apply_autocorrelation(
                        respiratory_rate[patient, day-1],
                        new_respiratory,
                        self.physio_params['autocorrelation']
                    )
                    
                    oxygen_saturation[patient, day] = self._apply_autocorrelation(
                        oxygen_saturation[patient, day-1],
                        new_oxygen,
                        self.physio_params['autocorrelation']
                    )
                else:
                    heart_rate[patient, day] = new_heart_rate
                    blood_pressure_systolic[patient, day] = new_systolic
                    blood_pressure_diastolic[patient, day] = new_diastolic
                    body_temperature[patient, day] = new_temperature
                    respiratory_rate[patient, day] = new_respiratory
                    oxygen_saturation[patient, day] = new_oxygen
        
        # Apply constraints
        heart_rate = np.clip(heart_rate, 40, 200)
        blood_pressure_systolic = np.clip(blood_pressure_systolic, 80, 200)
        blood_pressure_diastolic = np.clip(blood_pressure_diastolic, 40, 120)
        body_temperature = np.clip(body_temperature, 35, 42)
        respiratory_rate = np.clip(respiratory_rate, 8, 30)
        oxygen_saturation = np.clip(oxygen_saturation, 80, 100)
        
        # Reshape to 2D arrays
        heart_rate_flat = heart_rate.flatten()
        blood_pressure_systolic_flat = blood_pressure_systolic.flatten()
        blood_pressure_diastolic_flat = blood_pressure_diastolic.flatten()
        body_temperature_flat = body_temperature.flatten()
        respiratory_rate_flat = respiratory_rate.flatten()
        oxygen_saturation_flat = oxygen_saturation.flatten()
        
        # Create DataFrame
        physio_data = pd.DataFrame({
            'patient_id': np.repeat(np.arange(self.num_patients), self.days_per_patient),
            'day': np.tile(np.arange(self.days_per_patient), self.num_patients),
            'heart_rate': heart_rate_flat,
            'blood_pressure_systolic': blood_pressure_systolic_flat,
            'blood_pressure_diastolic': blood_pressure_diastolic_flat,
            'body_temperature': body_temperature_flat,
            'respiratory_rate': respiratory_rate_flat,
            'oxygen_saturation': oxygen_saturation_flat
        })
        
        return physio_data
    
    def _calculate_complex_interactions(self, sleep_data: pd.DataFrame, weather_data: pd.DataFrame,
                                      stress_diet_data: pd.DataFrame, physio_data: pd.DataFrame) -> pd.Series:
        """
        Calculate complex interaction terms between different domains.
        
        Args:
            sleep_data: Sleep data
            weather_data: Weather data
            stress_diet_data: Stress and diet data
            physio_data: Physiological data
            
        Returns:
            Series of interaction scores
        """
        if not self.include_complex_interactions:
            return pd.Series(np.zeros(len(sleep_data)))
        
        # Reshape data for easier access
        sleep_reshaped = {col: sleep_data[col].values.reshape(self.num_patients, self.days_per_patient) 
                         for col in sleep_data.columns if col not in ['patient_id', 'day']}
        
        weather_reshaped = {col: weather_data[col].values.reshape(self.num_patients, self.days_per_patient) 
                           for col in weather_data.columns if col not in ['patient_id', 'day']}
        
        stress_diet_reshaped = {col: stress_diet_data[col].values.reshape(self.num_patients, self.days_per_patient) 
                               for col in stress_diet_data.columns if col not in ['patient_id', 'day']}
        
        physio_reshaped = {col: physio_data[col].values.reshape(self.num_patients, self.days_per_patient) 
                          for col in physio_data.columns if col not in ['patient_id', 'day']}
        
        # Initialize interaction scores
        interaction_scores = np.zeros((self.num_patients, self.days_per_patient))
        
        # Calculate interaction terms
        for patient in range(self.num_patients):
            for day in range(self.days_per_patient):
                # Stress amplifies weather sensitivity
                if stress_diet_reshaped['stress_level'][patient, day] > self.interaction_terms['threshold_effects']['stress_threshold']:
                    # High stress makes pressure changes more impactful
                    if abs(weather_reshaped['pressure_change'][patient, day]) > self.interaction_terms['threshold_effects']['pressure_change_threshold']:
                        interaction_scores[patient, day] += self.interaction_terms['stress_weather'] * \
                                                          abs(weather_reshaped['pressure_change'][patient, day])
                
                # Poor sleep amplifies stress sensitivity
                if sleep_reshaped['sleep_quality'][patient, day] < self.interaction_terms['threshold_effects']['sleep_threshold']:
                    interaction_scores[patient, day] += self.interaction_terms['sleep_stress'] * \
                                                      stress_diet_reshaped['stress_level'][patient, day]
                
                # Dehydration amplifies alcohol effect
                if stress_diet_reshaped['hydration_level'][patient, day] < 4.0:
                    interaction_scores[patient, day] += self.interaction_terms['dehydration_alcohol'] * \
                                                      stress_diet_reshaped['alcohol_consumption'][patient, day]
                
                # Poor sleep amplifies caffeine effect
                if sleep_reshaped['sleep_quality'][patient, day] < self.interaction_terms['threshold_effects']['sleep_threshold']:
                    interaction_scores[patient, day] += self.interaction_terms['sleep_caffeine'] * \
                                                      stress_diet_reshaped['caffeine_intake'][patient, day]
        
        # Flatten and return as Series
        return pd.Series(interaction_scores.flatten())
    
    def _calculate_trigger_accumulation(self, scores: np.ndarray) -> np.ndarray:
        """
        Calculate trigger accumulation over time.
        
        Args:
            scores: Raw trigger scores
            
        Returns:
            Accumulated trigger scores
        """
        if not self.include_temporal_patterns:
            return scores
        
        # Reshape scores
        scores_reshaped = scores.reshape(self.num_patients, self.days_per_patient)
        accumulated_scores = np.zeros_like(scores_reshaped)
        
        # Calculate accumulation for each patient
        for patient in range(self.num_patients):
            accumulated = 0
            for day in range(self.days_per_patient):
                # Add new triggers
                accumulated += scores_reshaped[patient, day] * self.migraine_params['trigger_accumulation_rate']
                
                # Decay existing triggers
                accumulated *= (1 - self.migraine_params['trigger_decay_rate'])
                
                accumulated_scores[patient, day] = accumulated
        
        return accumulated_scores.flatten()
    
    def generate_migraine_labels(self, sleep_data: pd.DataFrame, weather_data: pd.DataFrame,
                               stress_diet_data: pd.DataFrame, physio_data: pd.DataFrame) -> pd.Series:
        """
        Generate synthetic migraine labels with complex patterns.
        
        Args:
            sleep_data: Sleep data
            weather_data: Weather data
            stress_diet_data: Stress and diet data
            physio_data: Physiological data
            
        Returns:
            Series of migraine labels (0 or 1)
        """
        # Calculate domain scores
        sleep_score = np.zeros(len(sleep_data))
        weather_score = np.zeros(len(weather_data))
        stress_diet_score = np.zeros(len(stress_diet_data))
        physio_score = np.zeros(len(physio_data))
        
        # Calculate sleep score
        for col, weight in self.migraine_params['base_weights']['sleep'].items():
            sleep_score += sleep_data[col].values * weight
        
        # Calculate weather score
        for col, weight in self.migraine_params['base_weights']['weather'].items():
            if col in weather_data.columns:
                weather_score += weather_data[col].values * weight
        
        # Calculate stress and diet score
        for col, weight in self.migraine_params['base_weights']['stress_diet'].items():
            stress_diet_score += stress_diet_data[col].values * weight
        
        # Calculate physiological score
        for col, weight in self.migraine_params['base_weights']['physio'].items():
            physio_score += physio_data[col].values * weight
        
        # Apply patient-specific sensitivities if using patient heterogeneity
        if self.include_patient_heterogeneity:
            # Reshape scores
            sleep_score = sleep_score.reshape(self.num_patients, self.days_per_patient)
            weather_score = weather_score.reshape(self.num_patients, self.days_per_patient)
            stress_diet_score = stress_diet_score.reshape(self.num_patients, self.days_per_patient)
            physio_score = physio_score.reshape(self.num_patients, self.days_per_patient)
            
            # Apply sensitivities
            for patient in range(self.num_patients):
                sleep_score[patient, :] *= self.patient_profiles['sleep_sensitivity'][patient]
                weather_score[patient, :] *= self.patient_profiles['weather_sensitivity'][patient]
                stress_diet_score[patient, :] *= self.patient_profiles['stress_sensitivity'][patient]
                physio_score[patient, :] *= self.patient_profiles['physio_sensitivity'][patient]
            
            # Flatten scores
            sleep_score = sleep_score.flatten()
            weather_score = weather_score.flatten()
            stress_diet_score = stress_diet_score.flatten()
            physio_score = physio_score.flatten()
        
        # Calculate complex interactions
        interaction_score = self._calculate_complex_interactions(
            sleep_data, weather_data, stress_diet_data, physio_data
        ).values
        
        # Calculate total score
        total_score = (sleep_score + weather_score + stress_diet_score + physio_score) / 4 + interaction_score
        
        # Apply trigger accumulation
        if self.include_temporal_patterns:
            total_score = self._calculate_trigger_accumulation(total_score)
        
        # Add noise
        total_score += np.random.normal(0, self.migraine_params['noise_std'], len(total_score))
        
        # Apply concept drift to threshold if enabled
        if self.include_concept_drift:
            # Reshape for per-day threshold
            total_score_reshaped = total_score.reshape(self.num_patients, self.days_per_patient)
            labels = np.zeros_like(total_score_reshaped, dtype=int)
            
            for day in range(self.days_per_patient):
                # Get drifting threshold for this day
                threshold = self._get_concept_drift_factor('threshold_drift', day)
                
                # Apply threshold
                labels[:, day] = (total_score_reshaped[:, day] > threshold).astype(int)
            
            # Flatten labels
            labels = labels.flatten()
        else:
            # Use fixed threshold
            labels = (total_score > self.migraine_params['threshold']).astype(int)
        
        # Apply recovery period (reduced probability of migraine after a migraine)
        if self.include_temporal_patterns:
            labels_reshaped = labels.reshape(self.num_patients, self.days_per_patient)
            
            for patient in range(self.num_patients):
                # Get recovery time for this patient
                if self.include_patient_heterogeneity:
                    recovery_time = self.patient_profiles['recovery_time'][patient]
                else:
                    recovery_time = self.migraine_params['recovery_period']
                
                for day in range(1, self.days_per_patient):
                    # Check if previous day was a migraine
                    if labels_reshaped[patient, day-1] == 1:
                        # Reduce probability of migraine during recovery period
                        recovery_end = min(day + recovery_time, self.days_per_patient)
                        for recovery_day in range(day, recovery_end):
                            # 80% chance of changing a migraine to non-migraine during recovery
                            if labels_reshaped[patient, recovery_day] == 1 and np.random.random() < 0.8:
                                labels_reshaped[patient, recovery_day] = 0
            
            # Flatten labels
            labels = labels_reshaped.flatten()
        
        # Add anomalies if enabled
        if self.include_anomalies:
            # Reshape for easier manipulation
            labels_reshaped = labels.reshape(self.num_patients, self.days_per_patient)
            
            for patient in range(self.num_patients):
                # Add rare but significant trigger events (1-2 per patient)
                num_anomalies = np.random.randint(1, 3)
                for _ in range(num_anomalies):
                    # Random day for anomaly
                    anomaly_day = np.random.randint(0, self.days_per_patient)
                    
                    # Force a migraine on this day
                    labels_reshaped[patient, anomaly_day] = 1
                    
                    # Also likely the next day
                    if anomaly_day + 1 < self.days_per_patient:
                        if np.random.random() < 0.7:
                            labels_reshaped[patient, anomaly_day + 1] = 1
            
            # Flatten labels
            labels = labels_reshaped.flatten()
        
        # Ensure we have a mix of 0s and 1s
        if np.all(labels == 1):
            # Find a threshold that gives approximately 70% positive cases
            threshold = np.percentile(total_score, 30)
            labels = (total_score > threshold).astype(int)
        
        if np.all(labels == 0):
            # Find a threshold that gives approximately 30% positive cases
            threshold = np.percentile(total_score, 70)
            labels = (total_score > threshold).astype(int)
        
        # Convert to pandas Series
        return pd.Series(labels)
    
    def generate_dataset(self) -> Dict[str, Any]:
        """
        Generate complete synthetic dataset with temporal patterns and complex interactions.
        
        Returns:
            Dictionary containing synthetic dataset
        """
        # Generate data for each modality
        sleep_data = self.generate_sleep_data()
        weather_data = self.generate_weather_data()
        stress_diet_data = self.generate_stress_diet_data()
        physio_data = self.generate_physiological_data(stress_diet_data)
        
        # Generate labels
        labels = self.generate_migraine_labels(
            sleep_data, weather_data, stress_diet_data, physio_data
        )
        
        # Create dataset
        dataset = {
            'sleep_data': sleep_data,
            'weather_data': weather_data,
            'stress_diet_data': stress_diet_data,
            'physio_data': physio_data,
            'labels': labels
        }
        
        return dataset
    
    def create_torch_datasets(self, dataset: Dict[str, Any], train_ratio: float = 0.7,
                            val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create PyTorch datasets from synthetic dataset.
        
        Args:
            dataset: Synthetic dataset
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Check that ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1")
        
        # Get unique patient IDs
        patient_ids = dataset['sleep_data']['patient_id'].unique()
        
        # Split patients into train, val, and test sets
        np.random.shuffle(patient_ids)
        num_train = int(len(patient_ids) * train_ratio)
        num_val = int(len(patient_ids) * val_ratio)
        
        train_patients = patient_ids[:num_train]
        val_patients = patient_ids[num_train:num_train+num_val]
        test_patients = patient_ids[num_train+num_val:]
        
        # Create masks for each split
        train_mask = dataset['sleep_data']['patient_id'].isin(train_patients)
        val_mask = dataset['sleep_data']['patient_id'].isin(val_patients)
        test_mask = dataset['sleep_data']['patient_id'].isin(test_patients)
        
        # Extract features for each modality
        sleep_features = dataset['sleep_data'].drop(['patient_id', 'day'], axis=1).values
        weather_features = dataset['weather_data'].drop(['patient_id', 'day'], axis=1).values
        stress_diet_features = dataset['stress_diet_data'].drop(['patient_id', 'day'], axis=1).values
        physio_features = dataset['physio_data'].drop(['patient_id', 'day'], axis=1).values
        
        # Create inputs dictionary
        inputs = {
            'sleep_data': sleep_features,
            'weather_data': weather_features,
            'stress_diet_data': stress_diet_features,
            'physio_data': physio_features
        }
        
        # Create targets
        targets = dataset['labels'].values
        
        # Create custom datasets for each split
        train_inputs = {key: torch.tensor(value[train_mask], dtype=torch.float32) for key, value in inputs.items()}
        val_inputs = {key: torch.tensor(value[val_mask], dtype=torch.float32) for key, value in inputs.items()}
        test_inputs = {key: torch.tensor(value[test_mask], dtype=torch.float32) for key, value in inputs.items()}
        
        train_targets = torch.tensor(targets[train_mask], dtype=torch.float32).unsqueeze(1)
        val_targets = torch.tensor(targets[val_mask], dtype=torch.float32).unsqueeze(1)
        test_targets = torch.tensor(targets[test_mask], dtype=torch.float32).unsqueeze(1)
        
        # Create datasets
        class CustomDataset(Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
            
            def __len__(self):
                return len(self.targets)
            
            def __getitem__(self, idx):
                item_inputs = {key: value[idx] for key, value in self.inputs.items()}
                return item_inputs, self.targets[idx]
        
        train_dataset = CustomDataset(train_inputs, train_targets)
        val_dataset = CustomDataset(val_inputs, val_targets)
        test_dataset = CustomDataset(test_inputs, test_targets)
        
        return train_dataset, val_dataset, test_dataset
    
    def save_dataset(self, dataset: Dict[str, Any], file_path: str) -> None:
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset to save
            file_path: Path to save dataset
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save dataset
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Load dataset from file.
        
        Args:
            file_path: Path to load dataset from
            
        Returns:
            Loaded dataset
        """
        # Load dataset
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        return dataset


def generate_enhanced_dataset(num_patients=100, days_per_patient=90, 
                             include_temporal_patterns=True,
                             include_concept_drift=True,
                             include_complex_interactions=True,
                             include_patient_heterogeneity=True,
                             include_anomalies=True,
                             seed=42):
    """
    Generate enhanced synthetic dataset for migraine prediction.
    
    Args:
        num_patients: Number of patients to simulate
        days_per_patient: Number of days to simulate per patient
        include_temporal_patterns: Whether to include temporal patterns
        include_concept_drift: Whether to include concept drift
        include_complex_interactions: Whether to include complex feature interactions
        include_patient_heterogeneity: Whether to include patient-specific variations
        include_anomalies: Whether to include anomalies and edge cases
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing synthetic dataset
    """
    # Create generator
    generator = EnhancedMigraineDataGenerator(
        seed=seed,
        num_patients=num_patients,
        days_per_patient=days_per_patient,
        include_temporal_patterns=include_temporal_patterns,
        include_concept_drift=include_concept_drift,
        include_complex_interactions=include_complex_interactions,
        include_patient_heterogeneity=include_patient_heterogeneity,
        include_anomalies=include_anomalies
    )
    
    # Generate dataset
    return generator.generate_dataset()


def create_torch_datasets_from_enhanced(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create PyTorch datasets from enhanced synthetic dataset.
    
    Args:
        dataset: Enhanced synthetic dataset
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create generator
    generator = EnhancedMigraineDataGenerator()
    
    # Create datasets
    return generator.create_torch_datasets(dataset, train_ratio, val_ratio, test_ratio)
