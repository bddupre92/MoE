"""
Test script for the Enhanced Data Generation Pipeline.

This script tests the functionality of the pipeline by generating a small dataset
and validating the output.
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_data_pipeline.pipeline import MigraineDataPipeline


def test_pipeline():
    """
    Test the functionality of the Migraine Data Pipeline.
    """
    print("Testing Migraine Data Pipeline...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create default configuration
    config = {
        "general": {
            "output_format": "numpy",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        },
        "sleep": {
            "sleep_duration_mean": 7.0,
            "sleep_duration_std": 1.0,
            "sleep_quality_mean": 7.0,
            "sleep_quality_std": 1.5,
            "deep_sleep_percentage_mean": 20.0,
            "deep_sleep_percentage_std": 5.0,
            "rem_sleep_percentage_mean": 25.0,
            "rem_sleep_percentage_std": 5.0,
            "sleep_interruptions_lambda": 2.0,
            "time_to_sleep_mean": 20.0,
            "time_to_sleep_std": 15.0,
            "circadian_rhythm_enabled": True,
            "weekend_variation_enabled": True,
            "sleep_debt_enabled": True
        },
        "weather": {
            "temperature_mean": 20.0,
            "temperature_std": 8.0,
            "humidity_mean": 60.0,
            "humidity_std": 15.0,
            "pressure_mean": 1013.0,
            "pressure_std": 10.0,
            "precipitation_lambda": 2.0,
            "wind_speed_lambda": 10.0,
            "seasonal_variation_enabled": True,
            "daily_fluctuation_enabled": True,
            "extreme_events_enabled": True,
            "barometric_pressure_changes_enabled": True
        },
        "stress_diet": {
            "stress_mean": 5.0,
            "stress_std": 2.0,
            "alcohol_lambda": 1.0,
            "meal_regularity_mean": 6.0,
            "meal_regularity_std": 2.0,
            "hydration_mean": 6.0,
            "hydration_std": 2.0,
            "exercise_lambda": 30.0,
            "work_life_cycle_enabled": True,
            "weekend_variation_enabled": True,
            "stress_diet_correlation_enabled": True
        },
        "physiological": {
            "heart_rate_mean": 75.0,
            "heart_rate_std": 10.0,
            "systolic_mean": 120.0,
            "systolic_std": 15.0,
            "diastolic_mean": 80.0,
            "diastolic_std": 10.0,
            "temperature_mean": 36.8,
            "temperature_std": 0.5,
            "respiratory_mean": 16.0,
            "respiratory_std": 3.0,
            "circadian_rhythm_enabled": True,
            "exercise_effect_enabled": True,
            "parameter_correlations_enabled": True
        },
        "migraine": {
            "temporal_effects": {
                "lag_hours": [6, 12, 24, 48],
                "lag_weights": [0.2, 0.4, 0.3, 0.1],
                "cumulative_effect_enabled": True,
                "cumulative_effect_decay": 0.8
            },
            "individual_variability": {
                "enabled": True,
                "sensitivity_std": 0.5,
                "threshold_mean": 0.7,
                "threshold_std": 0.1
            },
            "sleep_weights": {
                "sleep_duration": 0.3,
                "sleep_quality": 0.2,
                "deep_sleep_percentage": 0.15,
                "rem_sleep_percentage": 0.1,
                "sleep_interruptions": 0.15,
                "time_to_sleep": 0.1
            },
            "weather_weights": {
                "temperature": 0.1,
                "humidity": 0.1,
                "pressure": 0.2,
                "pressure_change_rate": 0.4,
                "precipitation": 0.1,
                "wind_speed": 0.1
            },
            "stress_diet_weights": {
                "stress_level": 0.3,
                "caffeine_intake": 0.15,
                "alcohol_consumption": 0.15,
                "meal_regularity": 0.2,
                "hydration": 0.2
            },
            "physiological_weights": {
                "heart_rate_variability": 0.4,
                "blood_pressure": 0.3,
                "body_temperature": 0.3
            }
        }
    }
    
    # Initialize pipeline with configuration
    pipeline = MigraineDataPipeline(config=config, seed=42)
    
    # Generate a small dataset for testing
    num_samples = 10
    time_periods = 72  # 3 days with hourly data
    
    # Generate and save data
    paths = pipeline.generate_and_save_data(num_samples, time_periods, output_dir, "test_data")
    
    # Verify output files exist
    for split in ["train", "val", "test", "metadata"]:
        if split in paths:
            assert os.path.exists(paths[split]), f"Output file for {split} does not exist"
            print(f"✓ {split} file exists: {paths[split]}")
    
    # Load and verify metadata
    with open(paths["metadata"], "r") as f:
        metadata = json.load(f)
    
    assert "sleep_features" in metadata, "Metadata missing sleep features"
    assert "weather_features" in metadata, "Metadata missing weather features"
    assert "stress_diet_features" in metadata, "Metadata missing stress/diet features"
    assert "physiological_features" in metadata, "Metadata missing physiological features"
    assert "target_feature" in metadata, "Metadata missing target feature"
    assert "num_samples" in metadata, "Metadata missing sample counts"
    assert "time_periods" in metadata, "Metadata missing time periods"
    
    print("✓ Metadata validation passed")
    
    # Load and verify data
    test_data = np.load(paths["test"])
    
    # Check if all expected keys are present
    expected_keys = ["sleep", "weather", "stress_diet", "physiological", "target", "X", "y"]
    for key in expected_keys:
        assert key in test_data, f"Test data missing key: {key}"
    
    print("✓ Data structure validation passed")
    
    # Check data shapes
    num_test_samples = metadata["num_samples"]["test"]
    
    assert test_data["sleep"].shape[0] == num_test_samples, "Incorrect number of test samples for sleep data"
    assert test_data["weather"].shape[0] == num_test_samples, "Incorrect number of test samples for weather data"
    assert test_data["stress_diet"].shape[0] == num_test_samples, "Incorrect number of test samples for stress/diet data"
    assert test_data["physiological"].shape[0] == num_test_samples, "Incorrect number of test samples for physiological data"
    assert test_data["target"].shape[0] == num_test_samples, "Incorrect number of test samples for target data"
    
    assert test_data["sleep"].shape[1] == time_periods, "Incorrect time periods for sleep data"
    assert test_data["weather"].shape[1] == time_periods, "Incorrect time periods for weather data"
    assert test_data["stress_diet"].shape[1] == time_periods, "Incorrect time periods for stress/diet data"
    assert test_data["physiological"].shape[1] == time_periods, "Incorrect time periods for physiological data"
    assert test_data["target"].shape[1] == time_periods, "Incorrect time periods for target data"
    
    print("✓ Data shape validation passed")
    
    # Verify test_predictions.npz file exists
    test_predictions_path = os.path.join(output_dir, "test_predictions.npz")
    assert os.path.exists(test_predictions_path), "Test predictions file does not exist"
    
    # Load and verify test predictions
    test_predictions = np.load(test_predictions_path)
    assert "y_test" in test_predictions, "Test predictions missing y_test"
    assert "y_pred_test" in test_predictions, "Test predictions missing y_pred_test"
    
    print("✓ Test predictions validation passed")
    
    # Generate visualizations
    generate_visualizations(test_data, output_dir)
    
    print("✓ All tests passed!")
    return paths


def generate_visualizations(test_data, output_dir):
    """
    Generate visualizations of the test data.
    
    Args:
        test_data: Test data dictionary.
        output_dir: Output directory for visualizations.
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Sample index to visualize
    sample_idx = 0
    
    # Time periods
    time_periods = test_data["target"].shape[1]
    time_axis = np.arange(time_periods)
    
    # 1. Migraine intensity over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, test_data["target"][sample_idx], 'r-', linewidth=2)
    plt.title(f"Migraine Intensity Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Migraine Intensity (0-10)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "migraine_intensity.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Sleep data
    plt.figure(figsize=(12, 6))
    sleep_features = ["Sleep Duration", "Sleep Quality", "Deep Sleep %", "REM Sleep %", "Interruptions", "Time to Sleep"]
    for i in range(test_data["sleep"].shape[2]):
        plt.plot(time_axis, test_data["sleep"][sample_idx, :, i], label=sleep_features[i] if i < len(sleep_features) else f"Feature {i}")
    plt.title(f"Sleep Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "sleep_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Weather data
    plt.figure(figsize=(12, 6))
    weather_features = ["Temperature", "Humidity", "Pressure", "Precipitation", "Wind Speed", "Pressure Change Rate"]
    for i in range(min(6, test_data["weather"].shape[2])):
        plt.plot(time_axis, test_data["weather"][sample_idx, :, i], label=weather_features[i] if i < len(weather_features) else f"Feature {i}")
    plt.title(f"Weather Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "weather_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Stress and Diet data
    plt.figure(figsize=(12, 6))
    stress_diet_features = ["Stress Level", "Caffeine Intake", "Alcohol Consumption", "Meal Regularity", "Hydration", "Exercise Duration"]
    for i in range(min(6, test_data["stress_diet"].shape[2])):
        plt.plot(time_axis, test_data["stress_diet"][sample_idx, :, i], label=stress_diet_features[i] if i < len(stress_diet_features) else f"Feature {i}")
    plt.title(f"Stress and Diet Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "stress_diet_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Physiological data
    plt.figure(figsize=(12, 6))
    physio_features = ["Heart Rate", "Systolic BP", "Diastolic BP", "Temperature", "Respiratory Rate", "HRV"]
    for i in range(min(6, test_data["physiological"].shape[2])):
        plt.plot(time_axis, test_data["physiological"][sample_idx, :, i], label=physio_features[i] if i < len(physio_features) else f"Feature {i}")
    plt.title(f"Physiological Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "physiological_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 6. Correlation between migraine intensity and key features
    plt.figure(figsize=(12, 10))
    
    # Select key features from each domain
    key_features = {
        "Sleep Quality": test_data["sleep"][sample_idx, :, 1],
        "Pressure Change": test_data["weather"][sample_idx, :, 5],
        "Stress Level": test_data["stress_diet"][sample_idx, :, 0],
        "Heart Rate Variability": test_data["physiological"][sample_idx, :, 5]
    }
    
    migraine = test_data["target"][sample_idx]
    
    # Plot correlations
    plt.subplot(2, 2, 1)
    plt.scatter(key_features["Sleep Quality"], migraine, alpha=0.7)
    plt.title("Migraine vs. Sleep Quality")
    plt.xlabel("Sleep Quality")
    plt.ylabel("Migraine Intensity")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(key_features["Pressure Change"], migraine, alpha=0.7)
    plt.title("Migraine vs. Pressure Change")
    plt.xlabel("Pressure Change Rate")
    plt.ylabel("Migraine Intensity")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(key_features["Stress Level"], migraine, alpha=0.7)
    plt.title("Migraine vs. Stress Level")
    plt.xlabel("Stress Level")
    plt.ylabel("Migraine Intensity")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(key_features["Heart Rate Variability"], migraine, alpha=0.7)
    plt.title("Migraine vs. Heart Rate Variability")
    plt.xlabel("Heart Rate Variability")
    plt.ylabel("Migraine Intensity")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_correlations.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    test_pipeline()
