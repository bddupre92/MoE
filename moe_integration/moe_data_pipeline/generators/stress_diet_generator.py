"""
Stress and Diet Data Generator module for the Enhanced Data Generation Pipeline.

This module provides functionality for generating realistic stress and dietary data patterns
with work/life cycle influences, meal timing, and correlations between stress and diet.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from moe_data_pipeline.generators.base_generator import BaseGenerator


class StressDietGenerator(BaseGenerator):
    """
    Stress and Diet Data Generator for the Enhanced Data Generation Pipeline.
    
    This class generates realistic stress and dietary data patterns with work/life
    cycle influences, meal timing, and correlations between stress and diet.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the Stress and Diet Data Generator.
        
        Args:
            config: Configuration dictionary for the generator.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(config, seed)
        
        # Extract stress/diet-specific configuration
        self.stress_diet_config = config.get("stress_diet", {})
        
        # Set default values if not provided in config
        self.stress_mean = self.stress_diet_config.get("stress_mean", 5.0)
        self.stress_std = self.stress_diet_config.get("stress_std", 2.0)
        self.alcohol_lambda = self.stress_diet_config.get("alcohol_lambda", 1.0)
        self.meal_regularity_mean = self.stress_diet_config.get("meal_regularity_mean", 6.0)
        self.meal_regularity_std = self.stress_diet_config.get("meal_regularity_std", 2.0)
        self.hydration_mean = self.stress_diet_config.get("hydration_mean", 6.0)
        self.hydration_std = self.stress_diet_config.get("hydration_std", 2.0)
        self.exercise_lambda = self.stress_diet_config.get("exercise_lambda", 30.0)
        
        # Feature flags
        self.work_life_cycle_enabled = self.stress_diet_config.get("work_life_cycle_enabled", True)
        self.weekend_variation_enabled = self.stress_diet_config.get("weekend_variation_enabled", True)
        self.stress_diet_correlation_enabled = self.stress_diet_config.get("stress_diet_correlation_enabled", True)
    
    def generate(self, num_samples: int, time_periods: int) -> Dict[str, np.ndarray]:
        """
        Generate synthetic stress and diet data.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods for time-series data.
            
        Returns:
            Dictionary containing the generated stress and diet data with the following keys:
                - stress_level: Stress level score (0-10)
                - caffeine_intake: Caffeine intake in mg
                - alcohol_consumption: Alcohol consumption in standard drinks
                - meal_regularity: Meal regularity score (0-10)
                - hydration: Hydration level score (0-10)
                - exercise_duration: Exercise duration in minutes
        """
        # Generate stress level with work/life cycle patterns
        stress_level = self._generate_stress_level(num_samples, time_periods)
        
        # Generate caffeine intake with correlation to stress
        caffeine_intake = self._generate_caffeine_intake(num_samples, time_periods, stress_level)
        
        # Generate alcohol consumption with weekly patterns
        alcohol_consumption = self._generate_alcohol_consumption(num_samples, time_periods, stress_level)
        
        # Generate meal regularity with correlation to stress
        meal_regularity = self._generate_meal_regularity(num_samples, time_periods, stress_level)
        
        # Generate hydration level
        hydration = self._generate_hydration(num_samples, time_periods, stress_level)
        
        # Generate exercise duration with weekly patterns
        exercise_duration = self._generate_exercise_duration(num_samples, time_periods, stress_level)
        
        return {
            "stress_level": stress_level,
            "caffeine_intake": caffeine_intake,
            "alcohol_consumption": alcohol_consumption,
            "meal_regularity": meal_regularity,
            "hydration": hydration,
            "exercise_duration": exercise_duration
        }
    
    def _generate_stress_level(self, num_samples: int, time_periods: int) -> np.ndarray:
        """
        Generate stress level data with realistic patterns.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing stress level scores (0-10).
        """
        # Generate base stress level
        stress_level = self._generate_time_series(
            base_value=self.stress_mean,
            std_dev=self.stress_std,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.7  # Strong day-to-day correlation
        )
        
        # Apply work/life cycle if enabled
        if self.work_life_cycle_enabled:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            # Create work stress pattern (higher during work hours)
            work_pattern = np.zeros(hours_per_day)
            # Work hours (9 AM to 5 PM, assuming 0 = midnight)
            work_pattern[9:17] = 2.0
            
            # Apply pattern to each weekday
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                
                if end_idx <= time_periods:
                    # Check if it's a weekday (0-4 = Monday-Friday)
                    day_of_week = day % 7
                    if day_of_week < 5:  # Weekday
                        for h in range(hours_per_day):
                            stress_level[:, start_idx + h] += work_pattern[h]
        
        # Apply weekend variation if enabled
        if self.weekend_variation_enabled:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                
                if end_idx <= time_periods:
                    # Check if it's a weekend (5-6 = Saturday-Sunday)
                    day_of_week = day % 7
                    if day_of_week >= 5:  # Weekend
                        # Lower stress on weekends
                        stress_level[:, start_idx:end_idx] -= 1.5
        
        # Apply individual variations
        stress_sensitivity = self.rng.normal(1.0, 0.3, num_samples)
        for i in range(num_samples):
            # Some people are more sensitive to stress
            stress_level[i, :] = self.stress_mean + (stress_level[i, :] - self.stress_mean) * stress_sensitivity[i]
        
        # Apply stressful life events
        # Simulate random stressful events
        event_probability = 0.01  # 1% chance of stressful event per time period
        event_occurrences = self.rng.random((num_samples, time_periods)) < event_probability
        
        # Apply stressful events
        for i in range(num_samples):
            for t in range(time_periods):
                if event_occurrences[i, t]:
                    # Stressful event: significant increase in stress
                    event_duration = self.rng.randint(1, 48)  # Duration in time periods
                    event_intensity = self.rng.uniform(2, 5)  # Intensity of stress increase
                    
                    # Apply stress increase with decay
                    for d in range(event_duration):
                        if t + d < time_periods:
                            decay_factor = 1 - (d / event_duration)
                            stress_level[i, t + d] += event_intensity * decay_factor
        
        # Ensure stress level is within bounds (0-10)
        stress_level = np.clip(stress_level, 0.0, 10.0)
        
        return stress_level
    
    def _generate_caffeine_intake(self, num_samples: int, time_periods: int, stress_level: np.ndarray) -> np.ndarray:
        """
        Generate caffeine intake data with realistic patterns and correlation to stress.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            stress_level: Stress level data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing caffeine intake in mg.
        """
        # Initialize caffeine intake array
        caffeine_intake = np.zeros((num_samples, time_periods))
        
        # Assume time periods represent hours
        hours_per_day = 24
        days = time_periods // hours_per_day
        
        # Define typical caffeine consumption times and amounts
        # Coffee: ~100mg per cup
        # Tea: ~50mg per cup
        # Soda: ~40mg per can
        morning_coffee_time = 8  # 8 AM
        lunch_coffee_time = 12   # 12 PM
        afternoon_coffee_time = 15  # 3 PM
        
        # Individual caffeine consumption patterns
        caffeine_preference = self.rng.normal(1.0, 0.5, num_samples)
        morning_preference = self.rng.random(num_samples) > 0.2  # 80% are morning coffee drinkers
        afternoon_preference = self.rng.random(num_samples) > 0.5  # 50% have afternoon coffee
        
        # Generate caffeine intake patterns
        for day in range(days):
            start_idx = day * hours_per_day
            end_idx = start_idx + hours_per_day
            
            if end_idx <= time_periods:
                # Morning coffee
                morning_idx = start_idx + morning_coffee_time
                for i in range(num_samples):
                    if morning_preference[i] and self.rng.random() > 0.1:  # 90% consistency
                        caffeine_amount = 100 * caffeine_preference[i]
                        # More caffeine when stressed
                        if stress_level[i, morning_idx] > 7:
                            caffeine_amount *= 1.5
                        caffeine_intake[i, morning_idx] += caffeine_amount
                
                # Lunch coffee/tea
                lunch_idx = start_idx + lunch_coffee_time
                for i in range(num_samples):
                    if self.rng.random() > 0.7:  # 30% have coffee/tea with lunch
                        caffeine_amount = 75 * caffeine_preference[i]
                        caffeine_intake[i, lunch_idx] += caffeine_amount
                
                # Afternoon coffee/tea
                afternoon_idx = start_idx + afternoon_coffee_time
                for i in range(num_samples):
                    if afternoon_preference[i] and self.rng.random() > 0.2:  # 80% consistency
                        caffeine_amount = 90 * caffeine_preference[i]
                        # More caffeine when stressed
                        if stress_level[i, afternoon_idx] > 7:
                            caffeine_amount *= 1.3
                        caffeine_intake[i, afternoon_idx] += caffeine_amount
                
                # Random caffeine throughout the day (soda, etc.)
                for h in range(hours_per_day):
                    if h > 7 and h < 20:  # Between 7 AM and 8 PM
                        for i in range(num_samples):
                            if self.rng.random() > 0.95:  # 5% chance per hour
                                caffeine_amount = 40 * caffeine_preference[i]
                                caffeine_intake[i, start_idx + h] += caffeine_amount
        
        # Apply correlation with stress if enabled
        if self.stress_diet_correlation_enabled:
            # Higher stress may lead to more caffeine consumption
            for t in range(time_periods):
                if caffeine_intake[:, t].any():  # Only for time periods with caffeine
                    stress_factor = (stress_level[:, t] - 5) / 5  # -1 to 1
                    caffeine_intake[:, t] *= (1 + 0.2 * stress_factor)
        
        return caffeine_intake
    
    def _generate_alcohol_consumption(self, num_samples: int, time_periods: int, stress_level: np.ndarray) -> np.ndarray:
        """
        Generate alcohol consumption data with realistic patterns and correlation to stress.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            stress_level: Stress level data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing alcohol consumption in standard drinks.
        """
        # Initialize alcohol consumption array
        alcohol_consumption = np.zeros((num_samples, time_periods))
        
        # Assume time periods represent hours
        hours_per_day = 24
        days = time_periods // hours_per_day
        
        # Individual alcohol consumption patterns
        alcohol_preference = self.rng.exponential(1.0, num_samples)
        # Some people don't drink alcohol
        non_drinkers = self.rng.random(num_samples) < 0.2  # 20% non-drinkers
        alcohol_preference[non_drinkers] = 0
        
        # Weekend drinkers vs. daily drinkers
        weekend_drinkers = self.rng.random(num_samples) < 0.7  # 70% primarily drink on weekends
        
        # Generate alcohol consumption patterns
        for day in range(days):
            start_idx = day * hours_per_day
            end_idx = start_idx + hours_per_day
            
            if end_idx <= time_periods:
                # Check if it's a weekend
                day_of_week = day % 7
                is_weekend = day_of_week >= 5  # Weekend (5-6 = Saturday-Sunday)
                
                # Evening drinking hours (typically 5 PM to 11 PM)
                evening_start = start_idx + 17  # 5 PM
                evening_end = min(start_idx + 23, time_periods)  # 11 PM or end of time periods
                
                for i in range(num_samples):
                    if alcohol_preference[i] > 0:  # Not a non-drinker
                        # Determine if person drinks today
                        drinks_today = False
                        
                        if is_weekend and weekend_drinkers[i]:
                            drinks_today = self.rng.random() < 0.8  # 80% chance on weekends for weekend drinkers
                        elif is_weekend and not weekend_drinkers[i]:
                            drinks_today = self.rng.random() < 0.4  # 40% chance on weekends for non-weekend drinkers
                        elif not is_weekend and weekend_drinkers[i]:
                            drinks_today = self.rng.random() < 0.2  # 20% chance on weekdays for weekend drinkers
                        elif not is_weekend and not weekend_drinkers[i]:
                            drinks_today = self.rng.random() < 0.5  # 50% chance on weekdays for non-weekend drinkers
                        
                        # Apply stress-related drinking if enabled
                        if self.stress_diet_correlation_enabled:
                            # Higher stress may increase likelihood of drinking
                            avg_stress = np.mean(stress_level[i, start_idx:end_idx])
                            stress_factor = (avg_stress - 5) / 5  # -1 to 1
                            if stress_factor > 0:  # Only increase for above-average stress
                                drinks_today = drinks_today or (self.rng.random() < 0.3 * stress_factor)
                        
                        if drinks_today:
                            # Determine number of drinks
                            base_drinks = self.rng.poisson(self.alcohol_lambda * alcohol_preference[i])
                            
                            # Distribute drinks over evening hours
                            drink_hours = self.rng.choice(range(evening_start, evening_end), 
                                                         size=min(base_drinks, evening_end - evening_start),
                                                         replace=False)
                            
                            for hour in drink_hours:
                                alcohol_consumption[i, hour] += 1
                            
                            # Sometimes people have multiple drinks in an hour
                            if base_drinks > len(drink_hours) and len(drink_hours) > 0:
                                extra_drinks = base_drinks - len(drink_hours)
                                for _ in range(extra_drinks):
                                    random_hour = self.rng.choice(drink_hours)
                                    alcohol_consumption[i, random_hour] += 1
        
        return alcohol_consumption
    
    def _generate_meal_regularity(self, num_samples: int, time_periods: int, stress_level: np.ndarray) -> np.ndarray:
        """
        Generate meal regularity data with realistic patterns and correlation to stress.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            stress_level: Stress level data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing meal regularity scores (0-10).
        """
        # Generate base meal regularity
        meal_regularity = self._generate_time_series(
            base_value=self.meal_regularity_mean,
            std_dev=self.meal_regularity_std,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.8  # Strong day-to-day correlation
        )
        
        # Apply correlation with stress if enabled
        if self.stress_diet_correlation_enabled:
            # Higher stress tends to decrease meal regularity
            for t in range(time_periods):
                stress_factor = (stress_level[:, t] - 5) / 5  # -1 to 1
                meal_regularity[:, t] -= 1.5 * stress_factor
        
        # Apply weekend variation if enabled
        if self.weekend_variation_enabled:
            # Assume time periods represent hours
            hours_per_day = 24
            days = time_periods // hours_per_day
            
            for day in range(days):
                start_idx = day * hours_per_day
                end_idx = start_idx + hours_per_day
                
                if end_idx <= time_periods:
                    # Check if it's a weekend
                    day_of_week = day % 7
                    if day_of_week >= 5:  # Weekend
                        # Meal times tend to be less regular on weekends
                        meal_regularity[:, start_idx:end_idx] -= 1.0
        
        # Apply individual variations
        regularity_tendency = self.rng.normal(1.0, 0.3, num_samples)
        for i in range(num_samples):
            # Some people are more regular with meals than others
            meal_regularity[i, :] = self.meal_regularity_mean + (meal_regularity[i, :] - self.meal_regularity_mean) * regularity_tendency[i]
        
        # Ensure meal regularity is within bounds (0-10)
        meal_regularity = np.clip(meal_regularity, 0.0, 10.0)
        
        return meal_regularity
    
    def _generate_hydration(self, num_samples: int, time_periods: int, stress_level: np.ndarray) -> np.ndarray:
        """
        Generate hydration level data with realistic patterns and correlation to stress.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            stress_level: Stress level data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing hydration level scores (0-10).
        """
        # Generate base hydration level
        hydration = self._generate_time_series(
            base_value=self.hydration_mean,
            std_dev=self.hydration_std,
            num_samples=num_samples,
            time_periods=time_periods,
            autocorrelation=0.7  # Strong hour-to-hour correlation
        )
        
        # Apply daily pattern
        # Hydration tends to decrease during sleep and increase during waking hours
        # Assume time periods represent hours
        hours_per_day = 24
        days = time_periods // hours_per_day
        
        # Create daily hydration pattern
        daily_pattern = np.zeros(hours_per_day)
        # Hydration decreases during sleep (0-7 AM)
        daily_pattern[0:7] = -1.0
        # Hydration increases during morning (8-11 AM)
        daily_pattern[8:12] = 1.0
        # Stable during day with slight decrease in afternoon (12-5 PM)
        daily_pattern[12:17] = 0.0
        # Slight increase in evening (6-10 PM)
        daily_pattern[18:22] = 0.5
        # Decrease before bed (11 PM-12 AM)
        daily_pattern[22:24] = -0.5
        
        # Apply pattern to each day
        for day in range(days):
            start_idx = day * hours_per_day
            end_idx = start_idx + hours_per_day
            
            if end_idx <= time_periods:
                for h in range(hours_per_day):
                    hydration[:, start_idx + h] += daily_pattern[h]
        
        # Apply correlation with stress if enabled
        if self.stress_diet_correlation_enabled:
            # Higher stress tends to decrease hydration
            for t in range(time_periods):
                stress_factor = (stress_level[:, t] - 5) / 5  # -1 to 1
                hydration[:, t] -= 0.5 * stress_factor
        
        # Apply individual variations
        hydration_habits = self.rng.normal(1.0, 0.3, num_samples)
        for i in range(num_samples):
            # Some people are better at staying hydrated than others
            hydration[i, :] = self.hydration_mean + (hydration[i, :] - self.hydration_mean) * hydration_habits[i]
        
        # Ensure hydration is within bounds (0-10)
        hydration = np.clip(hydration, 0.0, 10.0)
        
        return hydration
    
    def _generate_exercise_duration(self, num_samples: int, time_periods: int, stress_level: np.ndarray) -> np.ndarray:
        """
        Generate exercise duration data with realistic patterns and correlation to stress.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods.
            stress_level: Stress level data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods) containing exercise duration in minutes.
        """
        # Initialize exercise duration array
        exercise_duration = np.zeros((num_samples, time_periods))
        
        # Assume time periods represent hours
        hours_per_day = 24
        days = time_periods // hours_per_day
        
        # Individual exercise patterns
        exercise_frequency = self.rng.exponential(0.5, num_samples)  # Average days per week
        exercise_preference = self.rng.normal(1.0, 0.3, num_samples)  # Duration multiplier
        
        # Morning vs. evening exercisers
        morning_exercisers = self.rng.random(num_samples) < 0.3  # 30% exercise in morning
        
        # Generate exercise patterns
        for day in range(days):
            start_idx = day * hours_per_day
            end_idx = start_idx + hours_per_day
            
            if end_idx <= time_periods:
                # Check if it's a weekend
                day_of_week = day % 7
                is_weekend = day_of_week >= 5  # Weekend (5-6 = Saturday-Sunday)
                
                # Morning exercise time (6-8 AM)
                morning_start = start_idx + 6
                morning_end = start_idx + 8
                
                # Evening exercise time (5-8 PM)
                evening_start = start_idx + 17
                evening_end = start_idx + 20
                
                for i in range(num_samples):
                    # Determine if person exercises today
                    if is_weekend:
                        exercises_today = self.rng.random() < 0.15 * (1 + exercise_frequency[i])
                    else:
                        exercises_today = self.rng.random() < 0.1 * (1 + exercise_frequency[i])
                    
                    # Apply stress effect if enabled
                    if self.stress_diet_correlation_enabled:
                        # Moderate stress may increase exercise, high stress may decrease it
                        avg_stress = np.mean(stress_level[i, start_idx:end_idx])
                        if avg_stress > 8:  # Very high stress
                            exercises_today = exercises_today and (self.rng.random() > 0.5)  # 50% chance of skipping
                        elif 5 < avg_stress <= 8:  # Moderate stress
                            exercises_today = exercises_today or (self.rng.random() < 0.1)  # 10% chance of adding
                    
                    if exercises_today:
                        # Determine exercise duration
                        base_duration = self.rng.exponential(self.exercise_lambda) * exercise_preference[i]
                        
                        # Determine exercise time
                        if morning_exercisers[i]:
                            exercise_hour = self.rng.randint(morning_start, morning_end)
                        else:
                            exercise_hour = self.rng.randint(evening_start, evening_end)
                        
                        # Add exercise duration
                        if exercise_hour < time_periods:
                            exercise_duration[i, exercise_hour] += base_duration
        
        # Ensure exercise duration is non-negative
        exercise_duration = np.maximum(0, exercise_duration)
        
        return exercise_duration
