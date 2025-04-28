"""
Data Validator module for the Enhanced Data Generation Pipeline.

This module provides comprehensive validation mechanisms to ensure data integrity
and physiological plausibility of the generated synthetic data.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable


class DataValidator:
    """
    Data Validator for the Enhanced Data Generation Pipeline.
    
    This class provides comprehensive validation mechanisms to ensure data integrity
    and physiological plausibility of the generated synthetic data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Data Validator.
        
        Args:
            config: Configuration dictionary for validation settings.
        """
        self.config = config
        self.validation_results = {}
        self.validation_errors = []
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validate the generated data.
        
        Args:
            data: Dictionary containing the generated data.
            
        Returns:
            Tuple of (is_valid, validation_results, validation_errors) where:
                - is_valid is a boolean indicating whether the data passed all validations.
                - validation_results is a dictionary containing detailed validation results.
                - validation_errors is a list of validation error messages.
        """
        # Reset validation results and errors
        self.validation_results = {}
        self.validation_errors = []
        
        # Skip validation if disabled
        if not self.config.get("enabled", True):
            return True, {}, []
        
        # Perform validations
        self._validate_input_shapes(data)
        self._validate_nan_values(data)
        self._validate_data_types(data)
        self._validate_value_ranges(data)
        
        if self.config.get("check_temporal_consistency", True):
            self._validate_temporal_consistency(data)
        
        if self.config.get("check_physiological_plausibility", True):
            self._validate_physiological_plausibility(data)
        
        if self.config.get("check_correlations", True):
            self._validate_correlations(data)
        
        if self.config.get("check_class_balance", True):
            self._validate_class_balance(data)
        
        # Check if any validation errors occurred
        is_valid = len(self.validation_errors) == 0
        
        return is_valid, self.validation_results, self.validation_errors
    
    def _validate_input_shapes(self, data: Dict[str, Any]) -> None:
        """
        Validate the shapes of input data.
        
        Args:
            data: Dictionary containing the generated data.
        """
        shape_results = {}
        
        # Check if all required keys are present
        required_keys = ["sleep", "weather", "stress_diet", "physiological", "migraine"]
        for key in required_keys:
            if key not in data:
                self.validation_errors.append(f"Missing required data key: {key}")
                shape_results[key] = "Missing"
        
        # Check if all arrays have consistent first dimension (num_samples)
        num_samples = None
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        if num_samples is None:
                            num_samples = subvalue.shape[0]
                        elif subvalue.shape[0] != num_samples:
                            self.validation_errors.append(
                                f"Inconsistent number of samples in {key}.{subkey}: "
                                f"Expected {num_samples}, got {subvalue.shape[0]}"
                            )
                            shape_results[f"{key}.{subkey}"] = f"Expected {num_samples}, got {subvalue.shape[0]}"
            elif isinstance(value, np.ndarray):
                if num_samples is None:
                    num_samples = value.shape[0]
                elif value.shape[0] != num_samples:
                    self.validation_errors.append(
                        f"Inconsistent number of samples in {key}: "
                        f"Expected {num_samples}, got {value.shape[0]}"
                    )
                    shape_results[key] = f"Expected {num_samples}, got {value.shape[0]}"
        
        # Check if all time series have consistent second dimension (time_periods)
        time_periods = None
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray) and len(subvalue.shape) > 1:
                        if time_periods is None:
                            time_periods = subvalue.shape[1]
                        elif subvalue.shape[1] != time_periods:
                            self.validation_errors.append(
                                f"Inconsistent time periods in {key}.{subkey}: "
                                f"Expected {time_periods}, got {subvalue.shape[1]}"
                            )
                            shape_results[f"{key}.{subkey}"] = f"Expected {time_periods}, got {subvalue.shape[1]}"
            elif isinstance(value, np.ndarray) and len(value.shape) > 1:
                if time_periods is None:
                    time_periods = value.shape[1]
                elif value.shape[1] != time_periods:
                    self.validation_errors.append(
                        f"Inconsistent time periods in {key}: "
                        f"Expected {time_periods}, got {value.shape[1]}"
                    )
                    shape_results[key] = f"Expected {time_periods}, got {value.shape[1]}"
        
        self.validation_results["input_shapes"] = shape_results
    
    def _validate_nan_values(self, data: Dict[str, Any]) -> None:
        """
        Validate that there are no NaN values in the data.
        
        Args:
            data: Dictionary containing the generated data.
        """
        nan_results = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        nan_count = np.isnan(subvalue).sum()
                        if nan_count > 0:
                            self.validation_errors.append(
                                f"NaN values detected in {key}.{subkey}: {nan_count} NaNs"
                            )
                            nan_results[f"{key}.{subkey}"] = nan_count
            elif isinstance(value, np.ndarray):
                nan_count = np.isnan(value).sum()
                if nan_count > 0:
                    self.validation_errors.append(
                        f"NaN values detected in {key}: {nan_count} NaNs"
                    )
                    nan_results[key] = nan_count
        
        self.validation_results["nan_values"] = nan_results
    
    def _validate_data_types(self, data: Dict[str, Any]) -> None:
        """
        Validate the data types of the generated data.
        
        Args:
            data: Dictionary containing the generated data.
        """
        type_results = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if not isinstance(subvalue, (np.ndarray, list, tuple, int, float, bool)):
                        self.validation_errors.append(
                            f"Invalid data type for {key}.{subkey}: {type(subvalue)}"
                        )
                        type_results[f"{key}.{subkey}"] = str(type(subvalue))
            elif not isinstance(value, (np.ndarray, list, tuple, int, float, bool)):
                self.validation_errors.append(
                    f"Invalid data type for {key}: {type(value)}"
                )
                type_results[key] = str(type(value))
        
        self.validation_results["data_types"] = type_results
    
    def _validate_value_ranges(self, data: Dict[str, Any]) -> None:
        """
        Validate that values are within expected physiological ranges.
        
        Args:
            data: Dictionary containing the generated data.
        """
        range_results = {}
        
        # Define expected ranges for different data types
        expected_ranges = {
            "sleep.duration": (0, 24),
            "sleep.quality": (0, 10),
            "sleep.deep_sleep_percentage": (0, 100),
            "sleep.rem_sleep_percentage": (0, 100),
            "weather.temperature": (-50, 50),
            "weather.humidity": (0, 100),
            "weather.pressure": (900, 1100),
            "stress_diet.stress_level": (0, 10),
            "stress_diet.caffeine_intake": (0, 1000),
            "stress_diet.alcohol_consumption": (0, 100),
            "stress_diet.meal_regularity": (0, 10),
            "stress_diet.hydration": (0, 10),
            "physiological.heart_rate": (30, 200),
            "physiological.systolic": (70, 200),
            "physiological.diastolic": (40, 120),
            "physiological.temperature": (35, 42),
            "physiological.respiratory_rate": (8, 40),
            "migraine.intensity": (0, 10)
        }
        
        # Check value ranges
        for path, (min_val, max_val) in expected_ranges.items():
            parts = path.split(".")
            if len(parts) == 1:
                key = parts[0]
                if key in data and isinstance(data[key], np.ndarray):
                    min_actual = np.min(data[key])
                    max_actual = np.max(data[key])
                    if min_actual < min_val or max_actual > max_val:
                        self.validation_errors.append(
                            f"Values out of range for {key}: "
                            f"Expected [{min_val}, {max_val}], got [{min_actual}, {max_actual}]"
                        )
                        range_results[key] = f"Expected [{min_val}, {max_val}], got [{min_actual}, {max_actual}]"
            elif len(parts) == 2:
                key, subkey = parts
                if key in data and isinstance(data[key], dict) and subkey in data[key]:
                    if isinstance(data[key][subkey], np.ndarray):
                        min_actual = np.min(data[key][subkey])
                        max_actual = np.max(data[key][subkey])
                        if min_actual < min_val or max_actual > max_val:
                            self.validation_errors.append(
                                f"Values out of range for {key}.{subkey}: "
                                f"Expected [{min_val}, {max_val}], got [{min_actual}, {max_actual}]"
                            )
                            range_results[f"{key}.{subkey}"] = f"Expected [{min_val}, {max_val}], got [{min_actual}, {max_actual}]"
        
        self.validation_results["value_ranges"] = range_results
    
    def _validate_temporal_consistency(self, data: Dict[str, Any]) -> None:
        """
        Validate temporal consistency of the generated data.
        
        Args:
            data: Dictionary containing the generated data.
        """
        temporal_results = {}
        
        # Check for unrealistic rapid changes in physiological parameters
        if "physiological" in data:
            physio_data = data["physiological"]
            
            # Check heart rate changes
            if "heart_rate" in physio_data and isinstance(physio_data["heart_rate"], np.ndarray):
                heart_rate = physio_data["heart_rate"]
                if heart_rate.ndim > 1 and heart_rate.shape[1] > 1:
                    max_change = np.max(np.abs(np.diff(heart_rate, axis=1)))
                    if max_change > 50:  # Max 50 bpm change between consecutive time periods
                        self.validation_errors.append(
                            f"Unrealistic heart rate change detected: {max_change} bpm"
                        )
                        temporal_results["heart_rate_change"] = max_change
            
            # Check blood pressure changes
            if "systolic" in physio_data and "diastolic" in physio_data:
                systolic = physio_data["systolic"]
                diastolic = physio_data["diastolic"]
                if (systolic.ndim > 1 and systolic.shape[1] > 1 and
                    diastolic.ndim > 1 and diastolic.shape[1] > 1):
                    max_sys_change = np.max(np.abs(np.diff(systolic, axis=1)))
                    max_dia_change = np.max(np.abs(np.diff(diastolic, axis=1)))
                    if max_sys_change > 30:  # Max 30 mmHg change between consecutive time periods
                        self.validation_errors.append(
                            f"Unrealistic systolic pressure change detected: {max_sys_change} mmHg"
                        )
                        temporal_results["systolic_change"] = max_sys_change
                    if max_dia_change > 20:  # Max 20 mmHg change between consecutive time periods
                        self.validation_errors.append(
                            f"Unrealistic diastolic pressure change detected: {max_dia_change} mmHg"
                        )
                        temporal_results["diastolic_change"] = max_dia_change
        
        # Check for unrealistic weather changes
        if "weather" in data:
            weather_data = data["weather"]
            
            # Check temperature changes
            if "temperature" in weather_data and isinstance(weather_data["temperature"], np.ndarray):
                temperature = weather_data["temperature"]
                if temperature.ndim > 1 and temperature.shape[1] > 1:
                    max_change = np.max(np.abs(np.diff(temperature, axis=1)))
                    if max_change > 15:  # Max 15°C change between consecutive time periods
                        self.validation_errors.append(
                            f"Unrealistic temperature change detected: {max_change}°C"
                        )
                        temporal_results["temperature_change"] = max_change
        
        self.validation_results["temporal_consistency"] = temporal_results
    
    def _validate_physiological_plausibility(self, data: Dict[str, Any]) -> None:
        """
        Validate physiological plausibility of the generated data.
        
        Args:
            data: Dictionary containing the generated data.
        """
        physio_results = {}
        
        # Check blood pressure relationship (systolic > diastolic)
        if "physiological" in data:
            physio_data = data["physiological"]
            if "systolic" in physio_data and "diastolic" in physio_data:
                systolic = physio_data["systolic"]
                diastolic = physio_data["diastolic"]
                if systolic.shape == diastolic.shape:
                    invalid_count = np.sum(systolic <= diastolic)
                    if invalid_count > 0:
                        self.validation_errors.append(
                            f"Invalid blood pressure relationship detected: "
                            f"{invalid_count} instances where systolic <= diastolic"
                        )
                        physio_results["blood_pressure_relationship"] = invalid_count
        
        # Check sleep percentages (deep_sleep_percentage + rem_sleep_percentage <= 100)
        if "sleep" in data:
            sleep_data = data["sleep"]
            if "deep_sleep_percentage" in sleep_data and "rem_sleep_percentage" in sleep_data:
                deep_sleep = sleep_data["deep_sleep_percentage"]
                rem_sleep = sleep_data["rem_sleep_percentage"]
                if deep_sleep.shape == rem_sleep.shape:
                    total_percentage = deep_sleep + rem_sleep
                    invalid_count = np.sum(total_percentage > 100)
                    if invalid_count > 0:
                        self.validation_errors.append(
                            f"Invalid sleep percentages detected: "
                            f"{invalid_count} instances where deep_sleep + rem_sleep > 100%"
                        )
                        physio_results["sleep_percentages"] = invalid_count
        
        self.validation_results["physiological_plausibility"] = physio_results
    
    def _validate_correlations(self, data: Dict[str, Any]) -> None:
        """
        Validate expected correlations in the generated data.
        
        Args:
            data: Dictionary containing the generated data.
        """
        correlation_results = {}
        
        # Check correlation between stress and heart rate
        if ("stress_diet" in data and "stress_level" in data["stress_diet"] and
            "physiological" in data and "heart_rate" in data["physiological"]):
            
            stress = data["stress_diet"]["stress_level"]
            heart_rate = data["physiological"]["heart_rate"]
            
            # Calculate correlation for each time period
            if stress.ndim > 1 and heart_rate.ndim > 1 and stress.shape == heart_rate.shape:
                for t in range(stress.shape[1]):
                    corr = np.corrcoef(stress[:, t], heart_rate[:, t])[0, 1]
                    # Stress and heart rate should be positively correlated
                    if corr < 0.1:
                        self.validation_errors.append(
                            f"Weak correlation between stress and heart rate at time {t}: {corr}"
                        )
                        correlation_results[f"stress_heart_rate_t{t}"] = corr
        
        # Check correlation between migraine intensity and stress
        if ("migraine" in data and "intensity" in data["migraine"] and
            "stress_diet" in data and "stress_level" in data["stress_diet"]):
            
            migraine = data["migraine"]["intensity"]
            stress = data["stress_diet"]["stress_level"]
            
            # Calculate correlation for each time period
            if migraine.ndim > 1 and stress.ndim > 1 and migraine.shape == stress.shape:
                for t in range(migraine.shape[1]):
                    corr = np.corrcoef(migraine[:, t], stress[:, t])[0, 1]
                    # Migraine and stress should be positively correlated
                    if corr < 0.1:
                        self.validation_errors.append(
                            f"Weak correlation between migraine and stress at time {t}: {corr}"
                        )
                        correlation_results[f"migraine_stress_t{t}"] = corr
        
        self.validation_results["correlations"] = correlation_results
    
    def _validate_class_balance(self, data: Dict[str, Any]) -> None:
        """
        Validate class balance in the generated data.
        
        Args:
            data: Dictionary containing the generated data.
        """
        balance_results = {}
        
        # Check migraine class balance
        if "migraine" in data and "intensity" in data["migraine"]:
            migraine = data["migraine"]["intensity"]
            
            # Define migraine threshold (intensity >= 5 considered a migraine)
            threshold = 5
            
            if migraine.ndim > 1:
                for t in range(migraine.shape[1]):
                    migraine_count = np.sum(migraine[:, t] >= threshold)
                    migraine_percentage = 100 * migraine_count / migraine.shape[0]
                    
                    # Check if migraine percentage is within expected range (10-30%)
                    if migraine_percentage < 10 or migraine_percentage > 30:
                        self.validation_errors.append(
                            f"Unbalanced migraine class at time {t}: {migraine_percentage:.1f}% "
                            f"(expected 10-30%)"
                        )
                        balance_results[f"migraine_balance_t{t}"] = f"{migraine_percentage:.1f}%"
        
        self.validation_results["class_balance"] = balance_results
