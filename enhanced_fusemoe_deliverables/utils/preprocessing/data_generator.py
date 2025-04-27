"""
Synthetic Data Generator for Migraine Prediction

This module provides functionality to generate synthetic data for migraine prediction
across multiple domains: patient demographics, sleep patterns, weather conditions,
stress/diet factors, and physiological measurements.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


class SyntheticDataGenerator:
    """
    Generate synthetic data for migraine prediction across multiple domains.
    
    This class generates realistic synthetic data with correlations between
    different factors and migraine occurrence. The data spans multiple domains:
    patient demographics, sleep patterns, weather conditions, stress/diet factors,
    and physiological measurements.
    
    Attributes:
        num_patients (int): Number of patients to generate data for
        days_per_patient (int): Number of days of data per patient
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, num_patients: int = 100, days_per_patient: int = 30, seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            num_patients: Number of patients to generate data for
            days_per_patient: Number of days of data per patient
            seed: Random seed for reproducibility
        """
        self.num_patients = num_patients
        self.days_per_patient = days_per_patient
        self.seed = seed
        np.random.seed(seed)
        
    def generate_patient_data(self) -> pd.DataFrame:
        """
        Generate basic patient demographic data.
        
        Returns:
            DataFrame containing patient demographic data and migraine occurrence
        """
        data = []
        
        for patient_id in range(self.num_patients):
            # Generate patient demographics
            age = np.random.randint(18, 65)
            gender = np.random.choice(['M', 'F'])
            migraine_history = np.random.randint(0, 20)  # Years with migraine
            
            # Generate daily data for this patient
            for day in range(self.days_per_patient):
                # Baseline migraine probability
                base_prob = 0.1 + (migraine_history / 100)  # Higher probability with longer history
                
                # We'll use this later when we combine with other factors
                had_migraine = np.random.random() < base_prob
                
                data.append({
                    'patient_id': patient_id,
                    'day': day,
                    'age': age,
                    'gender': gender,
                    'migraine_history_years': migraine_history,
                    'had_migraine': had_migraine
                })
        
        return pd.DataFrame(data)
    
    def generate_sleep_data(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate sleep-related data that correlates with migraine occurrence.
        
        Args:
            patient_data: DataFrame containing patient demographic data
            
        Returns:
            DataFrame containing sleep-related data
        """
        data = []
        
        for _, row in patient_data.iterrows():
            patient_id = row['patient_id']
            day = row['day']
            had_migraine = row['had_migraine']
            
            # Base sleep parameters
            base_sleep_hours = np.random.normal(7.5, 1.0)  # Mean 7.5 hours, std 1 hour
            
            # If migraine is coming, sleep might be affected the day before
            if had_migraine:
                # Disturbed sleep pattern before migraine
                total_sleep_hours = max(3, base_sleep_hours - np.random.normal(1.5, 0.5))
                deep_sleep_pct = max(10, np.random.normal(15, 5))  # Lower deep sleep
                rem_sleep_pct = max(10, np.random.normal(20, 5))   # Lower REM sleep
                light_sleep_pct = 100 - deep_sleep_pct - rem_sleep_pct
                awake_time_mins = np.random.normal(30, 10)  # More awake time
                sleep_quality = np.random.normal(4, 1)  # Lower sleep quality (1-10)
            else:
                # Normal sleep pattern
                total_sleep_hours = max(5, base_sleep_hours)
                deep_sleep_pct = max(15, np.random.normal(25, 5))
                rem_sleep_pct = max(15, np.random.normal(25, 5))
                light_sleep_pct = 100 - deep_sleep_pct - rem_sleep_pct
                awake_time_mins = np.random.normal(15, 5)
                sleep_quality = np.random.normal(7, 1)  # Higher sleep quality (1-10)
            
            data.append({
                'patient_id': patient_id,
                'day': day,
                'total_sleep_hours': total_sleep_hours,
                'deep_sleep_pct': deep_sleep_pct,
                'rem_sleep_pct': rem_sleep_pct,
                'light_sleep_pct': light_sleep_pct,
                'awake_time_mins': awake_time_mins,
                'sleep_quality': min(10, max(1, sleep_quality))
            })
        
        return pd.DataFrame(data)
    
    def generate_weather_data(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate weather data that correlates with migraine occurrence.
        
        Args:
            patient_data: DataFrame containing patient demographic data
            
        Returns:
            DataFrame containing weather-related data
        """
        data = []
        
        # Generate a base weather pattern for the entire period
        base_temps = np.random.normal(70, 15, self.days_per_patient)  # Base temperatures
        base_pressure = np.random.normal(1013, 10, self.days_per_patient)  # Base barometric pressure
        base_humidity = np.random.normal(60, 20, self.days_per_patient)  # Base humidity
        
        for _, row in patient_data.iterrows():
            patient_id = row['patient_id']
            day = row['day']
            had_migraine = row['had_migraine']
            
            # Get base weather for this day
            base_temp = base_temps[day]
            base_press = base_pressure[day]
            base_humid = base_humidity[day]
            
            # If migraine is occurring, certain weather conditions might be more likely
            if had_migraine:
                # Weather conditions that might trigger migraines
                temperature = base_temp + np.random.normal(5, 2)  # Higher temperature
                pressure_change = np.random.normal(-5, 2)  # Pressure drop
                humidity = min(100, base_humid + np.random.normal(10, 5))  # Higher humidity
                precipitation = max(0, np.random.normal(0.3, 0.2))  # More likely to have precipitation
                wind_speed = np.random.normal(10, 5)  # Wind speed in mph
            else:
                # Normal weather conditions
                temperature = base_temp
                pressure_change = np.random.normal(0, 1)
                humidity = min(100, base_humid)
                precipitation = max(0, np.random.normal(0.1, 0.1))
                wind_speed = np.random.normal(7, 3)
            
            data.append({
                'patient_id': patient_id,
                'day': day,
                'temperature': temperature,
                'barometric_pressure': base_press + pressure_change,
                'humidity': humidity,
                'precipitation': precipitation,
                'wind_speed': max(0, wind_speed)
            })
        
        return pd.DataFrame(data)
    
    def generate_stress_diet_data(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate stress and dietary data that correlates with migraine occurrence.
        
        Args:
            patient_data: DataFrame containing patient demographic data
            
        Returns:
            DataFrame containing stress and dietary data
        """
        data = []
        
        for _, row in patient_data.iterrows():
            patient_id = row['patient_id']
            day = row['day']
            had_migraine = row['had_migraine']
            
            # Base stress and diet parameters
            base_stress = np.random.normal(4, 1.5)  # Base stress level (1-10)
            
            # If migraine is occurring, stress and diet might be factors
            if had_migraine:
                # Stress and diet factors that might trigger migraines
                stress_level = min(10, base_stress + np.random.normal(2, 1))  # Higher stress
                caffeine_intake = np.random.normal(300, 100)  # Higher caffeine (mg)
                alcohol_consumed = np.random.choice([True, False], p=[0.4, 0.6])  # More likely to have consumed alcohol
                processed_food = np.random.choice([True, False], p=[0.6, 0.4])  # More likely to have eaten processed food
                skipped_meals = np.random.choice([True, False], p=[0.4, 0.6])  # More likely to have skipped meals
                water_intake = np.random.normal(4, 1)  # Lower water intake (cups)
            else:
                # Normal stress and diet
                stress_level = min(10, base_stress)
                caffeine_intake = np.random.normal(150, 100)
                alcohol_consumed = np.random.choice([True, False], p=[0.2, 0.8])
                processed_food = np.random.choice([True, False], p=[0.3, 0.7])
                skipped_meals = np.random.choice([True, False], p=[0.1, 0.9])
                water_intake = np.random.normal(6, 1)
            
            data.append({
                'patient_id': patient_id,
                'day': day,
                'stress_level': min(10, max(1, stress_level)),
                'caffeine_intake_mg': max(0, caffeine_intake),
                'alcohol_consumed': alcohol_consumed,
                'processed_food_consumed': processed_food,
                'skipped_meals': skipped_meals,
                'water_intake_cups': max(0, water_intake)
            })
        
        return pd.DataFrame(data)
    
    def generate_physio_data(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate physiological data that correlates with migraine occurrence.
        
        Args:
            patient_data: DataFrame containing patient demographic data
            
        Returns:
            DataFrame containing physiological data
        """
        data = []
        
        for _, row in patient_data.iterrows():
            patient_id = row['patient_id']
            day = row['day']
            had_migraine = row['had_migraine']
            
            # Base physiological parameters
            base_heart_rate = np.random.normal(70, 8)  # Base heart rate
            
            # If migraine is occurring, physiological measures might be affected
            if had_migraine:
                # Physiological changes that might accompany migraines
                heart_rate = base_heart_rate + np.random.normal(10, 5)  # Elevated heart rate
                systolic_bp = np.random.normal(130, 10)  # Higher blood pressure
                diastolic_bp = np.random.normal(85, 8)
                body_temp = np.random.normal(99.0, 0.5)  # Slightly elevated temperature
                cortisol_level = np.random.normal(18, 3)  # Higher cortisol (μg/dL)
            else:
                # Normal physiological measures
                heart_rate = base_heart_rate
                systolic_bp = np.random.normal(120, 10)
                diastolic_bp = np.random.normal(80, 8)
                body_temp = np.random.normal(98.6, 0.3)
                cortisol_level = np.random.normal(14, 3)
            
            data.append({
                'patient_id': patient_id,
                'day': day,
                'heart_rate': max(40, heart_rate),
                'systolic_bp': max(90, systolic_bp),
                'diastolic_bp': max(60, diastolic_bp),
                'body_temp_f': body_temp,
                'cortisol_level': max(0, cortisol_level)
            })
        
        return pd.DataFrame(data)
    
    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all data types and return them.
        
        Returns:
            Tuple of DataFrames containing patient, sleep, weather, stress/diet, and physiological data
        """
        # Generate patient data first
        patient_data = self.generate_patient_data()
        
        # Generate all other data types based on patient data
        sleep_data = self.generate_sleep_data(patient_data)
        weather_data = self.generate_weather_data(patient_data)
        stress_diet_data = self.generate_stress_diet_data(patient_data)
        physio_data = self.generate_physio_data(patient_data)
        
        return patient_data, sleep_data, weather_data, stress_diet_data, physio_data
    
    def combine_data(self, patient_data: pd.DataFrame, sleep_data: pd.DataFrame, 
                    weather_data: pd.DataFrame, stress_diet_data: pd.DataFrame, 
                    physio_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all data types into a single DataFrame.
        
        Args:
            patient_data: DataFrame containing patient demographic data
            sleep_data: DataFrame containing sleep-related data
            weather_data: DataFrame containing weather-related data
            stress_diet_data: DataFrame containing stress and dietary data
            physio_data: DataFrame containing physiological data
            
        Returns:
            Combined DataFrame with all features
        """
        # Merge all DataFrames on patient_id and day
        combined = patient_data.merge(sleep_data, on=['patient_id', 'day'])
        combined = combined.merge(weather_data, on=['patient_id', 'day'])
        combined = combined.merge(stress_diet_data, on=['patient_id', 'day'])
        combined = combined.merge(physio_data, on=['patient_id', 'day'])
        
        return combined
    
    def split_data_by_modality(self, combined_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split combined data into separate modalities for expert models.
        
        Args:
            combined_data: Combined DataFrame with all features
            
        Returns:
            Dictionary of DataFrames for each modality
        """
        # Define columns for each modality
        patient_cols = ['patient_id', 'day', 'age', 'gender', 'migraine_history_years']
        sleep_cols = ['patient_id', 'day', 'total_sleep_hours', 'deep_sleep_pct', 
                     'rem_sleep_pct', 'light_sleep_pct', 'awake_time_mins', 'sleep_quality']
        weather_cols = ['patient_id', 'day', 'temperature', 'barometric_pressure', 
                       'humidity', 'precipitation', 'wind_speed']
        stress_diet_cols = ['patient_id', 'day', 'stress_level', 'caffeine_intake_mg', 
                           'alcohol_consumed', 'processed_food_consumed', 'skipped_meals', 
                           'water_intake_cups']
        physio_cols = ['patient_id', 'day', 'heart_rate', 'systolic_bp', 
                      'diastolic_bp', 'body_temp_f', 'cortisol_level']
        
        # Create DataFrames for each modality
        modalities = {
            'patient': combined_data[patient_cols + ['had_migraine']],
            'sleep': combined_data[sleep_cols + ['had_migraine']],
            'weather': combined_data[weather_cols + ['had_migraine']],
            'stress_diet': combined_data[stress_diet_cols + ['had_migraine']],
            'physio': combined_data[physio_cols + ['had_migraine']]
        }
        
        return modalities
    
    def create_train_test_split(self, combined_data: pd.DataFrame, test_size: float = 0.2, 
                               val_size: float = 0.1, random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            combined_data: Combined DataFrame with all features
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of DataFrames for training, validation, and testing
        """
        if random_state is None:
            random_state = self.seed
            
        # Get unique patient IDs
        patient_ids = combined_data['patient_id'].unique()
        np.random.seed(random_state)
        np.random.shuffle(patient_ids)
        
        # Split patient IDs into train, val, test
        n_patients = len(patient_ids)
        n_test = int(n_patients * test_size)
        n_val = int(n_patients * val_size)
        
        test_ids = patient_ids[:n_test]
        val_ids = patient_ids[n_test:n_test+n_val]
        train_ids = patient_ids[n_test+n_val:]
        
        # Split data based on patient IDs
        test_data = combined_data[combined_data['patient_id'].isin(test_ids)]
        val_data = combined_data[combined_data['patient_id'].isin(val_ids)]
        train_data = combined_data[combined_data['patient_id'].isin(train_ids)]
        
        return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(num_patients=100, days_per_patient=30)
    patient_data, sleep_data, weather_data, stress_diet_data, physio_data = generator.generate_all_data()
    
    # Combine all data
    combined_data = generator.combine_data(patient_data, sleep_data, weather_data, 
                                          stress_diet_data, physio_data)
    
    # Split by modality
    modalities = generator.split_data_by_modality(combined_data)
    
    # Create train/val/test split
    train_data, val_data, test_data = generator.create_train_test_split(combined_data)
    
    print(f"Generated data for {generator.num_patients} patients over {generator.days_per_patient} days")
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Print migraine distribution
    print(f"Overall migraine prevalence: {combined_data['had_migraine'].mean():.2f}")
    print(f"Training migraine prevalence: {train_data['had_migraine'].mean():.2f}")
    print(f"Validation migraine prevalence: {val_data['had_migraine'].mean():.2f}")
    print(f"Test migraine prevalence: {test_data['had_migraine'].mean():.2f}")
