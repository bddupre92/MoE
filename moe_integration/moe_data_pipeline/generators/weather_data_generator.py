"""
Weather Data Generator module for the Enhanced Data Generation Pipeline.

This module provides functionality for generating realistic weather data patterns
with seasonal variations, daily fluctuations, and extreme weather events.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from moe_data_pipeline.generators.base_generator import BaseGenerator


class WeatherDataGenerator(BaseGenerator):
    """
    Weather Data Generator for the Enhanced Data Generation Pipeline.
    
    This class generates realistic weather data patterns with seasonal variations,
    daily fluctuations, and extreme weather events.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the Weather Data Generator.
        
        Args:
            config: Configuration dictionary for the generator.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(config, seed)
        
        # Extract weather-specific configuration
        self.weather_config = config.get("weather", {})
        
        # Set default values if not provided in config
        self.temperature_mean = self.weather_config.get("temperature_mean", 20.0)
        self.temperature_std = self.weather_config.get("temperature_std", 8.0)
        self.humidity_mean = self.weather_config.get("humidity_mean", 60.0)
        self.humidity_std = self.weather_config.get("humidity_std", 15.0)
        self.pressure_mean = self.weather_config.get("pressure_mean", 1013.0)
        self.pressure_std = self.weather_config.get("pressure_std", 10.0)
        self.precipitation_lambda = self.weather_config.get("precipitation_lambda", 2.0)
        self.wind_speed_lambda = self.weather_config.get("wind_speed_lambda", 10.0)
        
        # Feature flags
        self.seasonal_variation_enabled = self.weather_config.get("seasonal_variation_enabled", True)
        self.daily_fluctuation_enabled = self.weather_config.get("daily_fluctuation_enabled", True)
        self.extreme_events_enabled = self.weather_config.get("extreme_events_enabled", True)
        self.barometric_pressure_changes_enabled = self.weather_config.get("barometric_pressure_changes_enabled", True)
    
    def generate(self, num_samples: int, time_periods: int) -> Dict[str, np.ndarray]:
        """
        Generate synthetic weather data.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods for time-series data.
            
        Returns:
            Dictionary containing the generated weather data with the following keys:
                - temperature: Temperature in degrees Celsius
                - humidity: Relative humidity percentage
                - pressure: Barometric pressure in hPa
                - precipitation: Precipitation amount in mm
                - wind_speed: Wind speed in km/h
                - pressure_change_rate: Rate of pressure change in hPa/hour
        """
        # Generate temperature with seasonal and daily patterns
        temperature = self._generate_temperature(num_samples, time_periods)
        
        # Generate humidity with correlation to temperature
        humidity = self._generate_humidity(num_samples, time_periods, temperature)
        
        # Generate barometric pressure with realistic patterns
        pressure = self._generate_pressure(num_samples, time_periods)
        
        # Calculate pressure change rate
        pressure_change_rate = self._calculate_pressure_change_rate(pressure)
        
        # Generate precipitation with correlation to humidity and pressure
        precipitation = self._generate_precipitation(num_samples, time_periods, humidity, pressure)
        
        # Generate wind speed with correlation to pressure gradient
        wind_speed = self._generate_wind_speed(num_samples, time_periods, pressure)
        
        return {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "pressure_change_rate": pressure_change_rate
        }
    
    def _generate_temperature(self, num_samples: int, time_periods: int) -> np.ndarray:
        """
        Generate temperature data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing temperature in degrees Celsius.
        """
        # Generate base temperature
        temperature = self._generate_time_series(
            base_value=self.temperature_mean,
            std_dev=self.temperature_std / 2,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.8  # Strong day-to-day correlation
        )
        
        # Apply seasonal variation if enabled
        if self.seasonal_variation_enabled:
            # Simulate seasonal cycle with period of 365 days
            # Assume time periods represent days
            seasonal_cycle = np.sin(2 * np.pi * np.arange(time_periods) / 365)
            seasonal_amplitude = self.temperature_std * 1.5  # Seasonal variation amplitude
            
            for t in range(time_periods):
                temperature[:, t] += seasonal_amplitude * seasonal_cycle[t]
        
        # Apply daily fluctuation if enabled
        if self.daily_fluctuation_enabled and time_periods >= 24:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            # Create daily temperature pattern (cooler at night, warmer during day)
            daily_pattern = np.sin(2 * np.pi * np.arange(hours_per_day) / hours_per_day - np.pi/2)
            daily_amplitude = 5.0  # Daily temperature variation amplitude
            
            # Apply pattern to each day
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                if end_idx <= time_periods:
                    for h in range(hours_per_day):
                        temperature[:, start_idx + h] += daily_amplitude * daily_pattern[h]
        
        # Apply extreme weather events if enabled
        if self.extreme_events_enabled:
            # Generate random extreme weather events
            event_probability = 0.02  # 2% chance of extreme event per time period
            event_occurrences = self.rng.random((num_samples, time_periods)) < event_probability
            
            # Apply extreme temperature events
            for i in range(num_samples):
                for t in range(time_periods):
                    if event_occurrences[i, t]:
                        # Determine if it's a heat wave or cold snap
                        if self.rng.random() < 0.5:
                            # Heat wave: increase temperature by 5-15 degrees
                            temperature[i, t] += self.rng.uniform(5, 15)
                        else:
                            # Cold snap: decrease temperature by 5-15 degrees
                            temperature[i, t] -= self.rng.uniform(5, 15)
        
        # Apply geographical variations
        # Simulate different locations with different base temperatures
        location_factors = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            temperature[i, :] += 5.0 * location_factors[i]
        
        # Ensure temperature is within realistic bounds (-40 to 50 degrees Celsius)
        temperature = np.clip(temperature, -40.0, 50.0)
        
        return temperature
    
    def _generate_humidity(self, num_samples: int, time_periods: int, temperature: np.ndarray) -> np.ndarray:
        """
        Generate humidity data with realistic patterns and correlation to temperature.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            temperature: Temperature data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing relative humidity percentage.
        """
        # Generate base humidity
        humidity = self._generate_time_series(
            base_value=self.humidity_mean,
            std_dev=self.humidity_std / 2,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.7  # Strong day-to-day correlation
        )
        
        # Apply negative correlation with temperature
        # Humidity tends to be lower when temperature is higher
        for t in range(time_periods):
            humidity[:, t] -= 0.5 * (temperature[:, t] - self.temperature_mean)
        
        # Apply daily fluctuation if enabled
        if self.daily_fluctuation_enabled and time_periods >= 24:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            # Create daily humidity pattern (higher at night, lower during day)
            daily_pattern = -np.sin(2 * np.pi * np.arange(hours_per_day) / hours_per_day - np.pi/2)
            daily_amplitude = 10.0  # Daily humidity variation amplitude
            
            # Apply pattern to each day
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                if end_idx <= time_periods:
                    for h in range(hours_per_day):
                        humidity[:, start_idx + h] += daily_amplitude * daily_pattern[h]
        
        # Apply geographical variations
        # Simulate different locations with different humidity levels
        location_factors = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            humidity[i, :] += 10.0 * location_factors[i]
        
        # Ensure humidity is within realistic bounds (0-100%)
        humidity = np.clip(humidity, 0.0, 100.0)
        
        return humidity
    
    def _generate_pressure(self, num_samples: int, time_periods: int) -> np.ndarray:
        """
        Generate barometric pressure data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing barometric pressure in hPa.
        """
        # Generate base pressure
        pressure = self._generate_time_series(
            base_value=self.pressure_mean,
            std_dev=self.pressure_std / 3,  # Reduce noise for more realistic patterns
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.9  # Very strong day-to-day correlation
        )
        
        # Apply barometric pressure changes if enabled
        if self.barometric_pressure_changes_enabled:
            # Generate pressure systems (high and low pressure)
            system_duration_mean = 72  # Average duration of pressure systems in hours
            system_probability = 1.0 / system_duration_mean  # Probability of new system per hour
            
            for i in range(num_samples):
                t = 0
                while t < time_periods:
                    if self.rng.random() < system_probability:
                        # New pressure system
                        system_duration = int(self.rng.normal(system_duration_mean, 24))
                        system_duration = max(12, min(system_duration, time_periods - t))
                        
                        # Determine if it's high or low pressure system
                        if self.rng.random() < 0.5:
                            # High pressure system: increase pressure
                            pressure_change = self.rng.uniform(5, 15)
                        else:
                            # Low pressure system: decrease pressure
                            pressure_change = -self.rng.uniform(5, 15)
                        
                        # Apply gradual pressure change
                        for d in range(system_duration):
                            if t + d < time_periods:
                                # Gradual build-up and decay
                                if d < system_duration / 3:
                                    # Build-up phase
                                    factor = d / (system_duration / 3)
                                elif d > 2 * system_duration / 3:
                                    # Decay phase
                                    factor = (system_duration - d) / (system_duration / 3)
                                else:
                                    # Stable phase
                                    factor = 1.0
                                
                                pressure[i, t + d] += pressure_change * factor
                        
                        t += system_duration
                    else:
                        t += 1
        
        # Apply geographical variations
        # Simulate different locations with different base pressure levels
        altitude_factors = self.rng.normal(0, 1.0, num_samples)
        for i in range(num_samples):
            # Pressure decreases with altitude
            pressure[i, :] -= 5.0 * altitude_factors[i]
        
        # Ensure pressure is within realistic bounds (900-1050 hPa)
        pressure = np.clip(pressure, 900.0, 1050.0)
        
        return pressure
    
    def _calculate_pressure_change_rate(self, pressure: np.ndarray) -> np.ndarray:
        """
        Calculate the rate of pressure change, which is a known migraine trigger.
        
        Args:
            pressure: Barometric pressure data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing pressure change rate in hPa/hour.
        """
        num_samples, time_periods = pressure.shape
        
        # Initialize pressure change rate array
        pressure_change_rate = np.zeros((num_samples, time_periods))
        
        # Calculate pressure change rate
        if time_periods > 1:
            # First time period has no previous data, set to 0
            pressure_change_rate[:, 0] = 0
            
            # Calculate change rate for remaining time periods
            for t in range(1, time_periods):
                pressure_change_rate[:, t] = pressure[:, t] - pressure[:, t-1]
        
        return pressure_change_rate
    
    def _generate_precipitation(self, num_samples: int, time_periods: int, 
                               humidity: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        Generate precipitation data with realistic patterns and correlation to humidity and pressure.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            humidity: Humidity data.
            pressure: Barometric pressure data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing precipitation in mm.
        """
        # Initialize precipitation array
        precipitation = np.zeros((num_samples, time_periods))
        
        # Generate precipitation events based on humidity and pressure
        for t in range(time_periods):
            for i in range(num_samples):
                # Calculate precipitation probability based on humidity and pressure
                # Higher humidity and lower pressure increase precipitation probability
                humidity_factor = (humidity[i, t] - 50) / 50  # -1 to 1
                pressure_factor = (1020 - pressure[i, t]) / 50  # Negative for high pressure, positive for low pressure
                
                # Combined probability (0 to 1)
                precip_probability = 0.3 * (humidity_factor + pressure_factor + 1) / 2
                precip_probability = max(0, min(precip_probability, 1))
                
                # Determine if precipitation occurs
                if self.rng.random() < precip_probability:
                    # Generate precipitation amount using exponential distribution
                    precipitation[i, t] = self.rng.exponential(self.precipitation_lambda)
                    
                    # Adjust based on humidity (more humidity = more precipitation)
                    precipitation[i, t] *= (humidity[i, t] / 50)
        
        # Apply extreme weather events if enabled
        if self.extreme_events_enabled:
            # Generate random extreme precipitation events
            event_probability = 0.01  # 1% chance of extreme event per time period
            event_occurrences = self.rng.random((num_samples, time_periods)) < event_probability
            
            # Apply extreme precipitation events
            for i in range(num_samples):
                for t in range(time_periods):
                    if event_occurrences[i, t]:
                        # Heavy rainfall or storm: significant precipitation
                        precipitation[i, t] += self.rng.uniform(20, 50)
        
        # Ensure precipitation is non-negative
        precipitation = np.maximum(0, precipitation)
        
        return precipitation
    
    def _generate_wind_speed(self, num_samples: int, time_periods: int, pressure: np.ndarray) -> np.ndarray:
        """
        Generate wind speed data with realistic patterns and correlation to pressure gradient.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            pressure: Barometric pressure data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing wind speed in km/h.
        """
        # Generate base wind speed
        wind_speed = self._generate_time_series(
            base_value=self.wind_speed_lambda,
            std_dev=self.wind_speed_lambda / 2,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.6  # Moderate day-to-day correlation
        )
        
        # Apply correlation with pressure gradient
        if time_periods > 1:
            for t in range(1, time_periods):
                # Calculate pressure gradient
                pressure_gradient = np.abs(pressure[:, t] - pressure[:, t-1])
                
                # Wind speed increases with pressure gradient
                wind_speed[:, t] += 0.5 * pressure_gradient
        
        # Apply extreme weather events if enabled
        if self.extreme_events_enabled:
            # Generate random extreme wind events
            event_probability = 0.01  # 1% chance of extreme event per time period
            event_occurrences = self.rng.random((num_samples, time_periods)) < event_probability
            
            # Apply extreme wind events
            for i in range(num_samples):
                for t in range(time_periods):
                    if event_occurrences[i, t]:
                        # Storm or gale: significant wind speed
                        wind_speed[i, t] += self.rng.uniform(30, 70)
        
        # Ensure wind speed is non-negative
        wind_speed = np.maximum(0, wind_speed)
        
        return wind_speed
