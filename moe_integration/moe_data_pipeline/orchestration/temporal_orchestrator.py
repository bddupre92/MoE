"""
Temporal Pattern Orchestrator module for the Enhanced Data Generation Pipeline.

This module coordinates the temporal relationships between different data domains,
implements lag effects between triggers and migraine onset, and models cumulative
effects from multiple triggers.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

class TemporalOrchestrator:
    """
    Temporal Pattern Orchestrator for the Enhanced Data Generation Pipeline.
    
    This class coordinates the temporal relationships between different data domains,
    implements lag effects between triggers and migraine onset, and models cumulative
    effects from multiple triggers.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the Temporal Pattern Orchestrator.
        
        Args:
            config: Configuration dictionary for the orchestrator.
            seed: Optional random seed for reproducibility.
        """
        self.config = config
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.rng = np.random.RandomState(self.seed)
        
        # Extract migraine-specific configuration
        self.migraine_config = config.get("migraine", {})
        
        # Extract temporal effects configuration
        self.temporal_effects = self.migraine_config.get("temporal_effects", {})
        self.lag_hours = self.temporal_effects.get("lag_hours", [6, 12, 24, 48])
        self.lag_weights = self.temporal_effects.get("lag_weights", [0.2, 0.4, 0.3, 0.1])
        self.cumulative_effect_enabled = self.temporal_effects.get("cumulative_effect_enabled", True)
        self.cumulative_effect_decay = self.temporal_effects.get("cumulative_effect_decay", 0.8)
        
        # Extract individual variability configuration
        self.individual_variability = self.migraine_config.get("individual_variability", {})
        self.variability_enabled = self.individual_variability.get("enabled", True)
        self.sensitivity_std = self.individual_variability.get("sensitivity_std", 0.5)
        self.threshold_mean = self.individual_variability.get("threshold_mean", 0.7)
        self.threshold_std = self.individual_variability.get("threshold_std", 0.1)
        
        # Extract trigger weights
        self.sleep_weights = self.migraine_config.get("sleep_weights", {})
        self.weather_weights = self.migraine_config.get("weather_weights", {})
        self.stress_diet_weights = self.migraine_config.get("stress_diet_weights", {})
        self.physiological_weights = self.migraine_config.get("physiological_weights", {})
    
    def orchestrate(self, 
                   sleep_data: Dict[str, np.ndarray],
                   weather_data: Dict[str, np.ndarray],
                   stress_diet_data: Dict[str, np.ndarray],
                   physiological_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Orchestrate the temporal relationships between different data domains and generate migraine data.
        
        Args:
            sleep_data: Dictionary containing sleep data.
            weather_data: Dictionary containing weather data.
            stress_diet_data: Dictionary containing stress and diet data.
            physiological_data: Dictionary containing physiological data.
            
        Returns:
            Dictionary containing the generated migraine data with the following keys:
                - intensity: Migraine intensity score (0-10)
                - trigger_scores: Individual trigger contribution scores
                - cumulative_trigger: Cumulative trigger effect
        """
        # Extract dimensions
        num_samples, time_periods = next(iter(sleep_data.values())).shape
        
        # Calculate individual trigger contributions
        sleep_trigger = self._calculate_sleep_trigger(sleep_data)
        weather_trigger = self._calculate_weather_trigger(weather_data)
        stress_diet_trigger = self._calculate_stress_diet_trigger(stress_diet_data)
        physiological_trigger = self._calculate_physiological_trigger(physiological_data)
        
        # Combine triggers with temporal effects
        combined_trigger = self._combine_triggers(
            sleep_trigger, weather_trigger, stress_diet_trigger, physiological_trigger
        )
        
        # Apply lag effects
        lagged_trigger = self._apply_lag_effects(combined_trigger)
        
        # Apply cumulative effects if enabled
        if self.cumulative_effect_enabled:
            cumulative_trigger = self._apply_cumulative_effects(combined_trigger)
        else:
            cumulative_trigger = combined_trigger.copy()
        
        # Generate individual sensitivity profiles if enabled
        if self.variability_enabled:
            sensitivity_profiles = self._generate_sensitivity_profiles(num_samples)
            migraine_thresholds = self._generate_migraine_thresholds(num_samples)
        else:
            sensitivity_profiles = np.ones((num_samples, 4))  # Default sensitivity to all trigger types
            migraine_thresholds = np.full(num_samples, self.threshold_mean)
        
        # Generate migraine intensity based on triggers and individual sensitivity
        intensity = self._generate_migraine_intensity(
            lagged_trigger, cumulative_trigger, sensitivity_profiles, migraine_thresholds
        )
        
        # Collect trigger scores for analysis
        trigger_scores = {
            "sleep": sleep_trigger,
            "weather": weather_trigger,
            "stress_diet": stress_diet_trigger,
            "physiological": physiological_trigger,
            "combined": combined_trigger,
            "lagged": lagged_trigger
        }
        
        return {
            "intensity": intensity,
            "trigger_scores": trigger_scores,
            "cumulative_trigger": cumulative_trigger
        }
    
    def _calculate_sleep_trigger(self, sleep_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate sleep trigger contribution.
        
        Args:
            sleep_data: Dictionary containing sleep data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing sleep trigger scores.
        """
        num_samples, time_periods = next(iter(sleep_data.values())).shape
        sleep_trigger = np.zeros((num_samples, time_periods))
        
        # Apply weights to each sleep feature
        for feature, weight in self.sleep_weights.items():
            if feature in sleep_data:
                # For features where lower values are worse (e.g., duration, quality)
                if feature in ["sleep_duration", "sleep_quality", "deep_sleep_percentage"]:
                    # Normalize to 0-1 range where 1 is bad (trigger)
                    if feature == "sleep_duration":
                        # Sleep duration: optimal is 7-8 hours, less or more is worse
                        normalized = np.abs(sleep_data[feature] - 7.5) / 4.5  # Max deviation is 4.5 (from 3 or 12)
                    elif feature == "sleep_quality":
                        # Sleep quality: 0 is worst, 10 is best
                        normalized = (10 - sleep_data[feature]) / 10
                    elif feature == "deep_sleep_percentage":
                        # Deep sleep: optimal is around 20-25%
                        normalized = np.abs(sleep_data[feature] - 22.5) / 17.5  # Max deviation is 17.5 (from 5 or 40)
                    
                    # Clip to 0-1 range
                    normalized = np.clip(normalized, 0, 1)
                    
                    # Apply weight
                    sleep_trigger += weight * normalized
                else:
                    # For features where higher values are worse (e.g., interruptions)
                    if feature == "sleep_interruptions":
                        # Normalize interruptions (0-10 scale)
                        normalized = np.minimum(sleep_data[feature] / 10, 1)
                    elif feature == "time_to_sleep":
                        # Normalize time to sleep (0-120 minutes)
                        normalized = np.minimum(sleep_data[feature] / 120, 1)
                    elif feature == "rem_sleep_percentage":
                        # REM sleep: both too little and too much can be triggers
                        normalized = np.abs(sleep_data[feature] - 25) / 15  # Optimal around 25%
                        normalized = np.clip(normalized, 0, 1)
                    else:
                        # Default normalization
                        normalized = np.clip(sleep_data[feature] / 10, 0, 1)
                    
                    # Apply weight
                    sleep_trigger += weight * normalized
        
        # Normalize to 0-1 range
        total_weight = sum(abs(w) for w in self.sleep_weights.values())
        if total_weight > 0:
            sleep_trigger /= total_weight
        
        return sleep_trigger
    
    def _calculate_weather_trigger(self, weather_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate weather trigger contribution.
        
        Args:
            weather_data: Dictionary containing weather data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing weather trigger scores.
        """
        num_samples, time_periods = next(iter(weather_data.values())).shape
        weather_trigger = np.zeros((num_samples, time_periods))
        
        # Apply weights to each weather feature
        for feature, weight in self.weather_weights.items():
            if feature in weather_data:
                if feature == "temperature":
                    # Temperature extremes are triggers
                    normalized = np.abs(weather_data[feature] - 20) / 30  # Optimal around 20°C
                elif feature == "humidity":
                    # Humidity extremes are triggers
                    normalized = np.abs(weather_data[feature] - 50) / 50  # Optimal around 50%
                elif feature == "pressure":
                    # Pressure extremes are triggers
                    normalized = np.abs(weather_data[feature] - 1013) / 50  # Optimal around 1013 hPa
                elif feature == "pressure_change_rate":
                    # Pressure changes are triggers (absolute value)
                    normalized = np.minimum(np.abs(weather_data[feature]) / 5, 1)  # Normalize to 5 hPa/hour
                elif feature == "precipitation":
                    # Precipitation can be a trigger
                    normalized = np.minimum(weather_data[feature] / 20, 1)  # Normalize to 20mm
                elif feature == "wind_speed":
                    # High wind can be a trigger
                    normalized = np.minimum(weather_data[feature] / 50, 1)  # Normalize to 50 km/h
                else:
                    # Default normalization
                    normalized = np.clip(weather_data[feature] / 10, 0, 1)
                
                # Apply weight
                weather_trigger += weight * normalized
        
        # Normalize to 0-1 range
        total_weight = sum(abs(w) for w in self.weather_weights.values())
        if total_weight > 0:
            weather_trigger /= total_weight
        
        return weather_trigger
    
    def _calculate_stress_diet_trigger(self, stress_diet_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate stress and diet trigger contribution.
        
        Args:
            stress_diet_data: Dictionary containing stress and diet data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing stress and diet trigger scores.
        """
        num_samples, time_periods = next(iter(stress_diet_data.values())).shape
        stress_diet_trigger = np.zeros((num_samples, time_periods))
        
        # Apply weights to each stress/diet feature
        for feature, weight in self.stress_diet_weights.items():
            if feature in stress_diet_data:
                if feature == "stress_level":
                    # Stress is a direct trigger
                    normalized = stress_diet_data[feature] / 10  # 0-10 scale
                elif feature == "caffeine_intake":
                    # Caffeine intake can be a trigger
                    normalized = np.minimum(stress_diet_data[feature] / 300, 1)  # Normalize to 300mg
                elif feature == "alcohol_consumption":
                    # Alcohol can be a trigger
                    normalized = np.minimum(stress_diet_data[feature] / 3, 1)  # Normalize to 3 drinks
                elif feature in ["meal_regularity", "hydration"]:
                    # For features where lower values are worse
                    normalized = (10 - stress_diet_data[feature]) / 10  # 0-10 scale
                else:
                    # Default normalization
                    normalized = np.clip(stress_diet_data[feature] / 10, 0, 1)
                
                # Apply weight
                stress_diet_trigger += weight * normalized
        
        # Normalize to 0-1 range
        total_weight = sum(abs(w) for w in self.stress_diet_weights.values())
        if total_weight > 0:
            stress_diet_trigger /= total_weight
        
        return stress_diet_trigger
    
    def _calculate_physiological_trigger(self, physiological_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate physiological trigger contribution.
        
        Args:
            physiological_data: Dictionary containing physiological data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing physiological trigger scores.
        """
        num_samples, time_periods = next(iter(physiological_data.values())).shape
        physiological_trigger = np.zeros((num_samples, time_periods))
        
        # Apply weights to each physiological feature
        for feature, weight in self.physiological_weights.items():
            if feature == "heart_rate_variability" and "heart_rate_variability" in physiological_data:
                # Lower HRV is associated with migraine
                normalized = (100 - physiological_data["heart_rate_variability"]) / 100
            elif feature == "blood_pressure" and "systolic" in physiological_data and "diastolic" in physiological_data:
                # Blood pressure extremes are triggers
                systolic_normalized = np.abs(physiological_data["systolic"] - 120) / 50
                diastolic_normalized = np.abs(physiological_data["diastolic"] - 80) / 40
                normalized = (systolic_normalized + diastolic_normalized) / 2
            elif feature == "body_temperature" and "temperature" in physiological_data:
                # Temperature extremes are triggers
                normalized = np.abs(physiological_data["temperature"] - 36.8) / 1.2
            else:
                # Skip if feature not found
                continue
            
            # Clip to 0-1 range
            normalized = np.clip(normalized, 0, 1)
            
            # Apply weight
            physiological_trigger += weight * normalized
        
        # Normalize to 0-1 range
        total_weight = sum(abs(w) for w in self.physiological_weights.values())
        if total_weight > 0:
            physiological_trigger /= total_weight
        
        return physiological_trigger
    
    def _combine_triggers(self, 
                         sleep_trigger: np.ndarray,
                         weather_trigger: np.ndarray,
                         stress_diet_trigger: np.ndarray,
                         physiological_trigger: np.ndarray) -> np.ndarray:
        """
        Combine trigger contributions from different domains.
        
        Args:
            sleep_trigger: Sleep trigger scores.
            weather_trigger: Weather trigger scores.
            stress_diet_trigger: Stress and diet trigger scores.
            physiological_trigger: Physiological trigger scores.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing combined trigger scores.
        """
        # Determine domain weights
        sleep_domain_weight = sum(abs(w) for w in self.sleep_weights.values())
        weather_domain_weight = sum(abs(w) for w in self.weather_weights.values())
        stress_diet_domain_weight = sum(abs(w) for w in self.stress_diet_weights.values())
        physiological_domain_weight = sum(abs(w) for w in self.physiological_weights.values())
        
        # Normalize domain weights
        total_weight = sleep_domain_weight + weather_domain_weight + stress_diet_domain_weight + physiological_domain_weight
        if total_weight > 0:
            sleep_domain_weight /= total_weight
            weather_domain_weight /= total_weight
            stress_diet_domain_weight /= total_weight
            physiological_domain_weight /= total_weight
        else:
            # Equal weights if no weights specified
            sleep_domain_weight = weather_domain_weight = stress_diet_domain_weight = physiological_domain_weight = 0.25
        
        # Combine triggers with domain weights
        combined_trigger = (
            sleep_domain_weight * sleep_trigger +
            weather_domain_weight * weather_trigger +
            stress_diet_domain_weight * stress_diet_trigger +
            physiological_domain_weight * physiological_trigger
        )
        
        return combined_trigger
    
    def _apply_lag_effects(self, combined_trigger: np.ndarray) -> np.ndarray:
        """
        Apply lag effects to trigger scores.
        
        Args:
            combined_trigger: Combined trigger scores.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing trigger scores with lag effects.
        """
        num_samples, time_periods = combined_trigger.shape
        lagged_trigger = np.zeros((num_samples, time_periods))
        
        # Convert lag hours to time periods (assuming 1 time period = 1 hour)
        lag_periods = self.lag_hours
        
        # Normalize lag weights
        lag_weights = np.array(self.lag_weights)
        lag_weights = lag_weights / np.sum(lag_weights)
        
        # Apply lag effects
        for i, (lag, weight) in enumerate(zip(lag_periods, lag_weights)):
            for t in range(time_periods):
                if t >= lag:
                    lagged_trigger[:, t] += weight * combined_trigger[:, t - lag]
        
        return lagged_trigger
    
    def _apply_cumulative_effects(self, combined_trigger: np.ndarray) -> np.ndarray:
        """
        Apply cumulative effects to trigger scores.
        
        Args:
            combined_trigger: Combined trigger scores.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing trigger scores with cumulative effects.
        """
        num_samples, time_periods = combined_trigger.shape
        cumulative_trigger = np.zeros((num_samples, time_periods))
        
        # Apply exponential decay to past triggers
        for t in range(time_periods):
            if t == 0:
                cumulative_trigger[:, t] = combined_trigger[:, t]
            else:
                # Current trigger plus decayed cumulative effect from previous time period
                cumulative_trigger[:, t] = combined_trigger[:, t] + self.cumulative_effect_decay * cumulative_trigger[:, t-1]
        
        # Normalize to 0-1 range
        max_possible_value = 1 / (1 - self.cumulative_effect_decay)
        cumulative_trigger /= max_possible_value
        
        return cumulative_trigger
    
    def _generate_sensitivity_profiles(self, num_samples: int) -> np.ndarray:
        """
        Generate individual sensitivity profiles for different trigger types.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            NumPy array of shape (num_samples, 4) containing sensitivity profiles
            for sleep, weather, stress/diet, and physiological triggers.
        """
        # Generate sensitivity profiles with correlation
        # Some people are more sensitive to certain types of triggers
        
        # Base sensitivity (mean 1.0, std 0.5)
        base_sensitivity = self.rng.normal(1.0, self.sensitivity_std, num_samples)
        
        # Individual variations for each trigger type
        sleep_sensitivity = 0.7 * base_sensitivity + 0.3 * self.rng.normal(0, self.sensitivity_std, num_samples)
        weather_sensitivity = 0.7 * base_sensitivity + 0.3 * self.rng.normal(0, self.sensitivity_std, num_samples)
        stress_diet_sensitivity = 0.7 * base_sensitivity + 0.3 * self.rng.normal(0, self.sensitivity_std, num_samples)
        physio_sensitivity = 0.7 * base_sensitivity + 0.3 * self.rng.normal(0, self.sensitivity_std, num_samples)
        
        # Ensure sensitivities are positive
        sleep_sensitivity = np.maximum(0.1, sleep_sensitivity)
        weather_sensitivity = np.maximum(0.1, weather_sensitivity)
        stress_diet_sensitivity = np.maximum(0.1, stress_diet_sensitivity)
        physio_sensitivity = np.maximum(0.1, physio_sensitivity)
        
        # Combine into profiles
        sensitivity_profiles = np.column_stack([
            sleep_sensitivity,
            weather_sensitivity,
            stress_diet_sensitivity,
            physio_sensitivity
        ])
        
        return sensitivity_profiles
    
    def _generate_migraine_thresholds(self, num_samples: int) -> np.ndarray:
        """
        Generate individual migraine thresholds.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            NumPy array of shape (num_samples,) containing migraine thresholds.
        """
        # Generate thresholds (typically 0.5-0.9)
        thresholds = self.rng.normal(self.threshold_mean, self.threshold_std, num_samples)
        
        # Ensure thresholds are within reasonable bounds
        thresholds = np.clip(thresholds, 0.3, 0.95)
        
        return thresholds
    
    def _generate_migraine_intensity(self, 
                                   lagged_trigger: np.ndarray,
                                   cumulative_trigger: np.ndarray,
                                   sensitivity_profiles: np.ndarray,
                                   migraine_thresholds: np.ndarray) -> np.ndarray:
        """
        Generate migraine intensity based on triggers and individual sensitivity.
        
        Args:
            lagged_trigger: Trigger scores with lag effects.
            cumulative_trigger: Trigger scores with cumulative effects.
            sensitivity_profiles: Individual sensitivity profiles.
            migraine_thresholds: Individual migraine thresholds.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing migraine intensity scores (0-10).
        """
        num_samples, time_periods = lagged_trigger.shape
        intensity = np.zeros((num_samples, time_periods))
        
        # Apply individual sensitivity to lagged triggers
        weighted_trigger = np.zeros_like(lagged_trigger)
        
        # Combine lagged and cumulative effects
        effective_trigger = 0.7 * lagged_trigger + 0.3 * cumulative_trigger
        
        for i in range(num_samples):
            # Calculate average sensitivity
            avg_sensitivity = np.mean(sensitivity_profiles[i])
            
            # Apply sensitivity to effective trigger
            weighted_trigger[i] = effective_trigger[i] * avg_sensitivity
            
            # Generate migraine intensity based on threshold
            threshold = migraine_thresholds[i]
            
            for t in range(time_periods):
                if weighted_trigger[i, t] > threshold:
                    # Trigger exceeds threshold, generate migraine
                    # Scale intensity based on how much trigger exceeds threshold
                    excess = (weighted_trigger[i, t] - threshold) / (1 - threshold)
                    
                    # Convert to 0-10 scale with non-linear scaling
                    # Mild: 1-3, Moderate: 4-6, Severe: 7-10
                    if excess < 0.3:
                        # Mild migraine
                        intensity[i, t] = 1 + 2 * (excess / 0.3)
                    elif excess < 0.7:
                        # Moderate migraine
                        intensity[i, t] = 4 + 3 * ((excess - 0.3) / 0.4)
                    else:
                        # Severe migraine
                        intensity[i, t] = 7 + 3 * ((excess - 0.7) / 0.3)
                    
                    # Add random variation
                    intensity[i, t] += self.rng.normal(0, 0.5)
                    
                    # Ensure intensity is within bounds
                    intensity[i, t] = np.clip(intensity[i, t], 0, 10)
                else:
                    # No migraine or very mild symptoms
                    # Some people may have mild symptoms even below threshold
                    if weighted_trigger[i, t] > 0.8 * threshold and self.rng.random() < 0.3:
                        intensity[i, t] = self.rng.uniform(0, 2)
                    else:
                        intensity[i, t] = 0
        
        return intensity
