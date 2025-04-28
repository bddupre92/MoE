"""
Demonstration script for the Enhanced Data Generation Pipeline.

This script demonstrates the usage of the pipeline with various configurations
and visualizes the results.
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_data_pipeline.pipeline import MigraineDataPipeline


def run_basic_demo():
    """
    Run a basic demonstration of the pipeline with default configuration.
    """
    print("Running basic demonstration...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basic_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline with default configuration
    pipeline = MigraineDataPipeline(seed=42)
    
    # Generate a small dataset
    num_samples = 20
    time_periods = 168  # 7 days with hourly data
    
    # Generate and save data
    paths = pipeline.generate_and_save_data(num_samples, time_periods, output_dir, "basic_demo")
    
    # Generate visualizations
    generate_basic_visualizations(paths, output_dir)
    
    print(f"Basic demonstration completed. Results saved to {output_dir}")
    return paths


def run_advanced_demo():
    """
    Run an advanced demonstration of the pipeline with custom configuration.
    """
    print("Running advanced demonstration...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advanced_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create custom configuration
    config = {
        "general": {
            "output_format": "pandas",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        },
        "sleep": {
            "sleep_duration_mean": 7.0,
            "sleep_duration_std": 1.5,  # Increased variability
            "sleep_quality_mean": 6.0,  # Lower average quality
            "sleep_quality_std": 2.0,   # Increased variability
            "deep_sleep_percentage_mean": 18.0,  # Lower deep sleep
            "deep_sleep_percentage_std": 6.0,    # Increased variability
            "rem_sleep_percentage_mean": 23.0,   # Lower REM sleep
            "rem_sleep_percentage_std": 6.0,     # Increased variability
            "sleep_interruptions_lambda": 3.0,   # More interruptions
            "time_to_sleep_mean": 30.0,          # Longer time to fall asleep
            "time_to_sleep_std": 20.0,           # Increased variability
            "circadian_rhythm_enabled": True,
            "weekend_variation_enabled": True,
            "sleep_debt_enabled": True
        },
        "weather": {
            "temperature_mean": 22.0,            # Higher average temperature
            "temperature_std": 10.0,             # Increased variability
            "humidity_mean": 65.0,               # Higher humidity
            "humidity_std": 18.0,                # Increased variability
            "pressure_mean": 1013.0,
            "pressure_std": 12.0,                # Increased variability
            "precipitation_lambda": 3.0,         # More precipitation
            "wind_speed_lambda": 12.0,           # Higher wind speed
            "seasonal_variation_enabled": True,
            "daily_fluctuation_enabled": True,
            "extreme_events_enabled": True,
            "barometric_pressure_changes_enabled": True
        },
        "stress_diet": {
            "stress_mean": 6.0,                  # Higher average stress
            "stress_std": 2.5,                   # Increased variability
            "alcohol_lambda": 1.5,               # More alcohol consumption
            "meal_regularity_mean": 5.0,         # Lower meal regularity
            "meal_regularity_std": 2.5,          # Increased variability
            "hydration_mean": 5.0,               # Lower hydration
            "hydration_std": 2.5,                # Increased variability
            "exercise_lambda": 20.0,             # Less exercise
            "work_life_cycle_enabled": True,
            "weekend_variation_enabled": True,
            "stress_diet_correlation_enabled": True
        },
        "physiological": {
            "heart_rate_mean": 78.0,             # Higher heart rate
            "heart_rate_std": 12.0,              # Increased variability
            "systolic_mean": 125.0,              # Higher blood pressure
            "systolic_std": 18.0,                # Increased variability
            "diastolic_mean": 82.0,              # Higher blood pressure
            "diastolic_std": 12.0,               # Increased variability
            "temperature_mean": 36.9,            # Slightly higher temperature
            "temperature_std": 0.6,              # Increased variability
            "respiratory_mean": 17.0,            # Higher respiratory rate
            "respiratory_std": 4.0,              # Increased variability
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
                "sensitivity_std": 0.7,          # Increased variability in sensitivity
                "threshold_mean": 0.65,          # Lower threshold (more migraines)
                "threshold_std": 0.15            # Increased variability in threshold
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
    
    # Initialize pipeline with custom configuration
    pipeline = MigraineDataPipeline(config=config, seed=42)
    
    # Generate a larger dataset
    num_samples = 50
    time_periods = 168  # 7 days with hourly data
    
    # Generate and save data
    paths = pipeline.generate_and_save_data(num_samples, time_periods, output_dir, "advanced_demo")
    
    # Generate visualizations
    generate_advanced_visualizations(paths, output_dir)
    
    print(f"Advanced demonstration completed. Results saved to {output_dir}")
    return paths


def run_comparison_demo():
    """
    Run a comparison demonstration between different configurations.
    """
    print("Running comparison demonstration...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define configurations to compare
    configs = {
        "baseline": {},  # Default configuration
        "high_stress": {
            "stress_diet": {
                "stress_mean": 7.0,  # Higher average stress
                "stress_std": 2.0
            }
        },
        "poor_sleep": {
            "sleep": {
                "sleep_duration_mean": 6.0,  # Lower sleep duration
                "sleep_quality_mean": 5.0,   # Lower sleep quality
                "sleep_interruptions_lambda": 4.0  # More interruptions
            }
        },
        "weather_sensitive": {
            "migraine": {
                "weather_weights": {
                    "temperature": 0.15,
                    "humidity": 0.15,
                    "pressure": 0.3,
                    "pressure_change_rate": 0.6,  # Higher sensitivity to pressure changes
                    "precipitation": 0.15,
                    "wind_speed": 0.15
                }
            }
        }
    }
    
    # Generate data for each configuration
    results = {}
    
    for config_name, config_override in configs.items():
        print(f"Generating data for {config_name} configuration...")
        
        # Create merged configuration
        config = {}
        if config_override:
            # Start with default config by initializing a pipeline
            temp_pipeline = MigraineDataPipeline()
            config = temp_pipeline.config
            
            # Apply overrides
            for section, section_config in config_override.items():
                if section not in config:
                    config[section] = {}
                config[section].update(section_config)
        
        # Initialize pipeline with configuration
        pipeline = MigraineDataPipeline(config=config, seed=42)
        
        # Generate data
        num_samples = 30
        time_periods = 168  # 7 days with hourly data
        
        # Generate data without saving
        generated_data = pipeline.generate_data(num_samples, time_periods)
        
        # Store results
        results[config_name] = generated_data
        
        # Save migraine intensity data for comparison
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        np.save(os.path.join(config_dir, "migraine_intensity.npy"), generated_data["migraine"]["intensity"])
    
    # Generate comparison visualizations
    generate_comparison_visualizations(results, output_dir)
    
    print(f"Comparison demonstration completed. Results saved to {output_dir}")
    return results


def generate_basic_visualizations(paths, output_dir):
    """
    Generate basic visualizations for the demonstration.
    
    Args:
        paths: Dictionary containing the paths to the saved files.
        output_dir: Output directory for visualizations.
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load test data
    test_data = np.load(paths["test"])
    
    # Load metadata
    with open(paths["metadata"], "r") as f:
        metadata = json.load(f)
    
    # Sample indices to visualize
    sample_indices = [0, 1, 2]
    
    # Time periods
    time_periods = metadata["time_periods"]
    time_axis = np.arange(time_periods)
    
    # 1. Migraine intensity over time for multiple samples
    plt.figure(figsize=(14, 8))
    for i, idx in enumerate(sample_indices):
        plt.plot(time_axis, test_data["target"][idx], label=f"Sample {idx}")
    
    # Add day markers
    for day in range(7):
        plt.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)
        plt.text(day*24 + 12, -0.5, f"Day {day+1}", ha='center')
    
    plt.title("Migraine Intensity Over Time")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Migraine Intensity (0-10)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "migraine_intensity_multi.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Heatmap of migraine intensity for all samples
    plt.figure(figsize=(14, 10))
    sns.heatmap(test_data["target"][:10], cmap="YlOrRd", vmin=0, vmax=10)
    plt.title("Migraine Intensity Heatmap (10 Samples)")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Sample")
    plt.savefig(os.path.join(vis_dir, "migraine_intensity_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Sleep data for a single sample
    sample_idx = 0
    plt.figure(figsize=(14, 8))
    
    # Define sleep features
    sleep_features = ["Sleep Duration", "Sleep Quality", "Deep Sleep %", "REM Sleep %", "Interruptions", "Time to Sleep"]
    
    for i in range(min(len(sleep_features), test_data["sleep"].shape[2])):
        plt.plot(time_axis, test_data["sleep"][sample_idx, :, i], label=sleep_features[i])
    
    # Add day markers
    for day in range(7):
        plt.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Sleep Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "sleep_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Weather data for a single sample
    plt.figure(figsize=(14, 8))
    
    # Define weather features
    weather_features = ["Temperature", "Humidity", "Pressure", "Precipitation", "Wind Speed", "Pressure Change Rate"]
    
    for i in range(min(len(weather_features), test_data["weather"].shape[2])):
        plt.plot(time_axis, test_data["weather"][sample_idx, :, i], label=weather_features[i])
    
    # Add day markers
    for day in range(7):
        plt.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Weather Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "weather_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Stress and Diet data for a single sample
    plt.figure(figsize=(14, 8))
    
    # Define stress/diet features
    stress_diet_features = ["Stress Level", "Caffeine Intake", "Alcohol Consumption", "Meal Regularity", "Hydration", "Exercise Duration"]
    
    for i in range(min(len(stress_diet_features), test_data["stress_diet"].shape[2])):
        plt.plot(time_axis, test_data["stress_diet"][sample_idx, :, i], label=stress_diet_features[i])
    
    # Add day markers
    for day in range(7):
        plt.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Stress and Diet Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "stress_diet_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 6. Physiological data for a single sample
    plt.figure(figsize=(14, 8))
    
    # Define physiological features
    physio_features = ["Heart Rate", "Systolic BP", "Diastolic BP", "Temperature", "Respiratory Rate", "HRV"]
    
    for i in range(min(len(physio_features), test_data["physiological"].shape[2])):
        plt.plot(time_axis, test_data["physiological"][sample_idx, :, i], label=physio_features[i])
    
    # Add day markers
    for day in range(7):
        plt.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Physiological Data Over Time (Sample {sample_idx})")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "physiological_data.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 7. Migraine vs. Key Triggers
    plt.figure(figsize=(14, 12))
    
    # Define key triggers
    key_triggers = {
        "Sleep Quality": (test_data["sleep"][sample_idx, :, 1], "Sleep Quality"),
        "Pressure Change": (test_data["weather"][sample_idx, :, 5], "Pressure Change Rate"),
        "Stress Level": (test_data["stress_diet"][sample_idx, :, 0], "Stress Level"),
        "Heart Rate Variability": (test_data["physiological"][sample_idx, :, 5], "Heart Rate Variability")
    }
    
    migraine = test_data["target"][sample_idx]
    
    # Plot migraine and triggers
    plt.subplot(5, 1, 1)
    plt.plot(time_axis, migraine, 'r-', linewidth=2)
    plt.title("Migraine Intensity")
    plt.ylabel("Intensity (0-10)")
    plt.grid(True, alpha=0.3)
    
    for i, (name, (data, label)) in enumerate(key_triggers.items(), 2):
        plt.subplot(5, 1, i)
        plt.plot(time_axis, data, linewidth=2)
        plt.title(name)
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
    
    plt.xlabel("Time Period (hours)")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "migraine_vs_triggers.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 8. Distribution of migraine intensity
    plt.figure(figsize=(10, 6))
    
    # Flatten the target data
    flat_target = test_data["target"].flatten()
    
    # Create histogram
    plt.hist(flat_target, bins=20, alpha=0.7, color='blue')
    plt.title("Distribution of Migraine Intensity")
    plt.xlabel("Migraine Intensity (0-10)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "migraine_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Basic visualizations saved to {vis_dir}")


def generate_advanced_visualizations(paths, output_dir):
    """
    Generate advanced visualizations for the demonstration.
    
    Args:
        paths: Dictionary containing the paths to the saved files.
        output_dir: Output directory for visualizations.
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load data
    # For pandas format, load CSV files
    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])
    
    # Load metadata
    with open(paths["metadata"], "r") as f:
        metadata = json.load(f)
    
    # 1. Migraine intensity by sample
    plt.figure(figsize=(14, 10))
    
    # Get unique sample IDs
    sample_ids = test_df["sample_id"].unique()[:10]  # First 10 samples
    
    # Create pivot table for heatmap
    pivot_df = test_df[test_df["sample_id"].isin(sample_ids)].pivot(
        index="sample_id", columns="time_period", values="migraine_intensity"
    )
    
    # Create heatmap
    sns.heatmap(pivot_df, cmap="YlOrRd", vmin=0, vmax=10)
    plt.title("Migraine Intensity by Sample and Time Period")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Sample ID")
    plt.savefig(os.path.join(vis_dir, "migraine_intensity_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Migraine intensity distribution
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    sns.histplot(test_df["migraine_intensity"], bins=20, kde=True)
    plt.title("Distribution of Migraine Intensity")
    plt.xlabel("Migraine Intensity (0-10)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "migraine_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Correlation matrix
    plt.figure(figsize=(16, 14))
    
    # Select relevant columns for correlation
    cols = [
        "migraine_intensity",
        "sleep_sleep_duration", "sleep_sleep_quality", "sleep_deep_sleep_percentage",
        "weather_pressure_change_rate", "weather_pressure",
        "stress_diet_stress_level", "stress_diet_caffeine_intake", "stress_diet_alcohol_consumption",
        "physiological_heart_rate_variability"
    ]
    
    # Create correlation matrix
    corr_df = test_df[cols].corr()
    
    # Create heatmap
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix of Key Features with Migraine Intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "correlation_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Time series for a single sample
    plt.figure(figsize=(14, 12))
    
    # Select a sample
    sample_id = sample_ids[0]
    sample_df = test_df[test_df["sample_id"] == sample_id]
    
    # Plot migraine intensity
    plt.subplot(5, 1, 1)
    plt.plot(sample_df["time_period"], sample_df["migraine_intensity"], 'r-', linewidth=2)
    plt.title(f"Migraine Intensity (Sample {sample_id})")
    plt.ylabel("Intensity (0-10)")
    plt.grid(True, alpha=0.3)
    
    # Plot sleep quality
    plt.subplot(5, 1, 2)
    plt.plot(sample_df["time_period"], sample_df["sleep_sleep_quality"], 'b-', linewidth=2)
    plt.title("Sleep Quality")
    plt.ylabel("Quality (0-10)")
    plt.grid(True, alpha=0.3)
    
    # Plot pressure change rate
    plt.subplot(5, 1, 3)
    plt.plot(sample_df["time_period"], sample_df["weather_pressure_change_rate"], 'g-', linewidth=2)
    plt.title("Barometric Pressure Change Rate")
    plt.ylabel("Change Rate (hPa/hour)")
    plt.grid(True, alpha=0.3)
    
    # Plot stress level
    plt.subplot(5, 1, 4)
    plt.plot(sample_df["time_period"], sample_df["stress_diet_stress_level"], 'c-', linewidth=2)
    plt.title("Stress Level")
    plt.ylabel("Stress (0-10)")
    plt.grid(True, alpha=0.3)
    
    # Plot heart rate variability
    plt.subplot(5, 1, 5)
    plt.plot(sample_df["time_period"], sample_df["physiological_heart_rate_variability"], 'm-', linewidth=2)
    plt.title("Heart Rate Variability")
    plt.ylabel("HRV")
    plt.grid(True, alpha=0.3)
    
    plt.xlabel("Time Period (hours)")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "time_series_single_sample.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Scatter plots of key features vs. migraine intensity
    plt.figure(figsize=(16, 12))
    
    # Define key features
    key_features = [
        ("sleep_sleep_quality", "Sleep Quality"),
        ("weather_pressure_change_rate", "Pressure Change Rate"),
        ("stress_diet_stress_level", "Stress Level"),
        ("physiological_heart_rate_variability", "Heart Rate Variability")
    ]
    
    # Create scatter plots
    for i, (feature, label) in enumerate(key_features, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x=feature, y="migraine_intensity", data=test_df, alpha=0.5)
        plt.title(f"Migraine Intensity vs. {label}")
        plt.xlabel(label)
        plt.ylabel("Migraine Intensity (0-10)")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "scatter_plots.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 6. Weekly patterns
    plt.figure(figsize=(14, 10))
    
    # Add day of week column
    test_df["day_of_week"] = test_df["time_period"] // 24 % 7
    
    # Create box plots
    plt.subplot(2, 2, 1)
    sns.boxplot(x="day_of_week", y="migraine_intensity", data=test_df)
    plt.title("Migraine Intensity by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
    plt.ylabel("Migraine Intensity (0-10)")
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x="day_of_week", y="sleep_sleep_quality", data=test_df)
    plt.title("Sleep Quality by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
    plt.ylabel("Sleep Quality (0-10)")
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x="day_of_week", y="stress_diet_stress_level", data=test_df)
    plt.title("Stress Level by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
    plt.ylabel("Stress Level (0-10)")
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x="day_of_week", y="stress_diet_alcohol_consumption", data=test_df)
    plt.title("Alcohol Consumption by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
    plt.ylabel("Alcohol Consumption (drinks)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "weekly_patterns.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 7. Daily patterns
    plt.figure(figsize=(14, 10))
    
    # Add hour of day column
    test_df["hour_of_day"] = test_df["time_period"] % 24
    
    # Create line plots
    plt.subplot(2, 2, 1)
    sns.lineplot(x="hour_of_day", y="migraine_intensity", data=test_df, ci=95)
    plt.title("Migraine Intensity by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Migraine Intensity (0-10)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    sns.lineplot(x="hour_of_day", y="physiological_heart_rate", data=test_df, ci=95)
    plt.title("Heart Rate by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Heart Rate (bpm)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    sns.lineplot(x="hour_of_day", y="stress_diet_stress_level", data=test_df, ci=95)
    plt.title("Stress Level by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Stress Level (0-10)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    sns.lineplot(x="hour_of_day", y="stress_diet_caffeine_intake", data=test_df, ci=95)
    plt.title("Caffeine Intake by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Caffeine Intake (mg)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "daily_patterns.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Advanced visualizations saved to {vis_dir}")


def generate_comparison_visualizations(results, output_dir):
    """
    Generate comparison visualizations between different configurations.
    
    Args:
        results: Dictionary containing the generated data for each configuration.
        output_dir: Output directory for visualizations.
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract migraine intensity data
    migraine_data = {}
    for config_name, data in results.items():
        migraine_data[config_name] = data["migraine"]["intensity"]
    
    # 1. Average migraine intensity
    plt.figure(figsize=(12, 8))
    
    # Calculate average intensity for each configuration
    avg_intensity = {config: np.mean(data) for config, data in migraine_data.items()}
    
    # Create bar chart
    plt.bar(avg_intensity.keys(), avg_intensity.values())
    plt.title("Average Migraine Intensity by Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Average Migraine Intensity (0-10)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "avg_intensity_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Migraine frequency
    plt.figure(figsize=(12, 8))
    
    # Calculate migraine frequency (intensity > 3) for each configuration
    migraine_freq = {config: np.mean(data > 3) * 100 for config, data in migraine_data.items()}
    
    # Create bar chart
    plt.bar(migraine_freq.keys(), migraine_freq.values())
    plt.title("Migraine Frequency by Configuration (Intensity > 3)")
    plt.xlabel("Configuration")
    plt.ylabel("Frequency (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "migraine_freq_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Migraine intensity distribution
    plt.figure(figsize=(14, 8))
    
    # Create histograms
    for config, data in migraine_data.items():
        plt.hist(data.flatten(), bins=20, alpha=0.5, label=config)
    
    plt.title("Migraine Intensity Distribution by Configuration")
    plt.xlabel("Migraine Intensity (0-10)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "intensity_dist_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Migraine intensity over time
    plt.figure(figsize=(14, 10))
    
    # Calculate average intensity over time for each configuration
    time_periods = migraine_data[list(migraine_data.keys())[0]].shape[1]
    time_axis = np.arange(time_periods)
    
    for config, data in migraine_data.items():
        avg_over_time = np.mean(data, axis=0)
        plt.plot(time_axis, avg_over_time, label=config)
    
    # Add day markers
    for day in range(7):
        plt.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("Average Migraine Intensity Over Time by Configuration")
    plt.xlabel("Time Period (hours)")
    plt.ylabel("Average Migraine Intensity (0-10)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "intensity_time_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Heatmap comparison
    plt.figure(figsize=(16, 12))
    
    # Create subplots for each configuration
    for i, (config, data) in enumerate(migraine_data.items(), 1):
        plt.subplot(2, 2, i)
        sns.heatmap(data[:10], cmap="YlOrRd", vmin=0, vmax=10)
        plt.title(f"Migraine Intensity Heatmap - {config}")
        plt.xlabel("Time Period (hours)")
        plt.ylabel("Sample")
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "heatmap_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 6. Severe migraine frequency
    plt.figure(figsize=(12, 8))
    
    # Calculate severe migraine frequency (intensity > 7) for each configuration
    severe_freq = {config: np.mean(data > 7) * 100 for config, data in migraine_data.items()}
    
    # Create bar chart
    plt.bar(severe_freq.keys(), severe_freq.values())
    plt.title("Severe Migraine Frequency by Configuration (Intensity > 7)")
    plt.xlabel("Configuration")
    plt.ylabel("Frequency (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "severe_migraine_freq_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Comparison visualizations saved to {vis_dir}")


if __name__ == "__main__":
    # Run all demonstrations
    basic_paths = run_basic_demo()
    advanced_paths = run_advanced_demo()
    comparison_results = run_comparison_demo()
    
    print("All demonstrations completed successfully!")
