# Enhanced Data Generation Pipeline for Migraine Prediction MoE

## Overview

This document provides comprehensive documentation for the Enhanced Data Generation Pipeline developed for the Mixture of Experts (MoE) migraine prediction system. The pipeline generates realistic synthetic data for migraine prediction with proper temporal patterns, correlations between variables, and individual variability.

## Architecture

The pipeline follows a modular architecture with the following components:

1. **Configuration Management**: Centralized configuration system for all pipeline components
2. **Data Generators**: Domain-specific generators for sleep, weather, stress/diet, and physiological data
3. **Temporal Orchestration**: Coordination of temporal relationships between triggers and migraine onset
4. **Data Validation**: Comprehensive validation to ensure data quality and consistency
5. **Output Formatting**: Flexible formatting options for compatibility with the MoE system

### Directory Structure

```
moe_data_pipeline/
├── __init__.py
├── pipeline.py                  # Main pipeline integration
├── config/
│   └── config_manager.py        # Configuration management
├── generators/
│   ├── base_generator.py        # Base class for all generators
│   ├── sleep_data_generator.py  # Sleep data generator
│   ├── weather_data_generator.py # Weather data generator
│   ├── stress_diet_generator.py # Stress and diet data generator
│   └── physiological_generator.py # Physiological data generator
├── orchestration/
│   └── temporal_orchestrator.py # Temporal pattern orchestration
├── validation/
│   └── data_validator.py        # Data validation
├── formatting/
│   └── output_formatter.py      # Output formatting
└── tests/
    └── test_pipeline.py         # Pipeline tests
```

## Components

### Configuration Manager

The `ConfigManager` provides a centralized configuration system for all pipeline components. It supports loading configuration from a file or dictionary, with sensible defaults for all parameters.

Key features:
- Configuration validation
- Default values for all parameters
- Support for JSON and YAML configuration files

### Base Generator

The `BaseGenerator` provides common functionality for all domain-specific generators, including:

- Time series generation with autocorrelation
- Noise generation with configurable parameters
- Random seed management for reproducibility
- Utility methods for data transformation

### Domain-Specific Generators

#### Sleep Data Generator

The `SleepDataGenerator` generates realistic sleep data with the following features:

- Sleep duration with individual variability
- Sleep quality scores
- Deep sleep and REM sleep percentages
- Sleep interruptions
- Time to fall asleep
- Circadian rhythm patterns
- Weekend variation
- Sleep debt accumulation and recovery

#### Weather Data Generator

The `WeatherDataGenerator` generates realistic weather data with the following features:

- Temperature with seasonal and daily patterns
- Humidity with correlation to temperature
- Barometric pressure with realistic patterns
- Pressure change rate (known migraine trigger)
- Precipitation with correlation to humidity and pressure
- Wind speed with correlation to pressure gradient
- Extreme weather events

#### Stress and Diet Generator

The `StressDietGenerator` generates realistic stress and dietary data with the following features:

- Stress level with work/life cycle patterns
- Caffeine intake with correlation to stress
- Alcohol consumption with weekly patterns
- Meal regularity with correlation to stress
- Hydration level
- Exercise duration with weekly patterns
- Weekend variation
- Individual variability in stress sensitivity

#### Physiological Generator

The `PhysiologicalGenerator` generates realistic physiological data with the following features:

- Heart rate with circadian rhythm
- Heart rate variability (measure of autonomic function)
- Blood pressure (systolic and diastolic)
- Body temperature with circadian rhythm
- Respiratory rate
- Exercise effects on physiological parameters
- Correlations between physiological parameters
- Individual variability in baseline values

### Temporal Orchestrator

The `TemporalOrchestrator` coordinates the temporal relationships between different data domains and generates migraine data based on trigger patterns. Key features include:

- Calculation of trigger contributions from each domain
- Lag effects between triggers and migraine onset
- Cumulative effects from multiple triggers
- Individual variability in trigger sensitivity
- Migraine intensity generation based on trigger patterns
- Threshold-based migraine onset

### Data Validator

The `DataValidator` ensures data quality and consistency through comprehensive validation checks:

- Data shape and dimension validation
- Value range validation
- Temporal consistency validation
- Feature-target alignment validation
- Missing value detection
- Statistical property validation

### Output Formatter

The `OutputFormatter` formats the generated data for compatibility with the MoE system:

- Support for multiple output formats (PyTorch, NumPy, Pandas, JSON)
- Data splitting for training/validation/testing
- Metadata generation
- File saving with configurable paths and prefixes
- Dashboard compatibility with test predictions file

### Pipeline Integration

The `MigraineDataPipeline` integrates all components into a cohesive pipeline with a simple interface:

- Single method for generating, validating, formatting, and saving data
- Progress reporting
- Error handling
- Configuration management

## Usage

### Basic Usage

```python
from moe_data_pipeline.pipeline import MigraineDataPipeline

# Initialize pipeline with default configuration
pipeline = MigraineDataPipeline()

# Generate and save data
num_samples = 100
time_periods = 168  # 7 days with hourly data
output_dir = "/path/to/output"

paths = pipeline.generate_and_save_data(num_samples, time_periods, output_dir)
```

### Custom Configuration

```python
from moe_data_pipeline.pipeline import MigraineDataPipeline

# Custom configuration
config = {
    "general": {
        "output_format": "pytorch",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    },
    "sleep": {
        "sleep_duration_mean": 7.0,
        "sleep_duration_std": 1.0,
        # Additional sleep parameters...
    },
    # Additional domain configurations...
}

# Initialize pipeline with custom configuration
pipeline = MigraineDataPipeline(config=config, seed=42)

# Generate and save data
paths = pipeline.generate_and_save_data(100, 168, "/path/to/output")
```

### Advanced Usage

```python
from moe_data_pipeline.pipeline import MigraineDataPipeline

# Initialize pipeline
pipeline = MigraineDataPipeline()

# Generate data without saving
generated_data = pipeline.generate_data(100, 168)

# Access individual domain data
sleep_data = generated_data["sleep"]
weather_data = generated_data["weather"]
stress_diet_data = generated_data["stress_diet"]
physiological_data = generated_data["physiological"]
migraine_data = generated_data["migraine"]

# Format and save data separately
paths = pipeline.format_and_save_data(generated_data, "/path/to/output")
```

## Configuration Options

### General Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_format` | Output format (pytorch, numpy, pandas, json) | `"numpy"` |
| `train_ratio` | Ratio of training data | `0.7` |
| `val_ratio` | Ratio of validation data | `0.15` |
| `test_ratio` | Ratio of test data | `0.15` |

### Sleep Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sleep_duration_mean` | Mean sleep duration in hours | `7.0` |
| `sleep_duration_std` | Standard deviation of sleep duration | `1.0` |
| `sleep_quality_mean` | Mean sleep quality score (0-10) | `7.0` |
| `sleep_quality_std` | Standard deviation of sleep quality | `1.5` |
| `deep_sleep_percentage_mean` | Mean deep sleep percentage | `20.0` |
| `deep_sleep_percentage_std` | Standard deviation of deep sleep percentage | `5.0` |
| `rem_sleep_percentage_mean` | Mean REM sleep percentage | `25.0` |
| `rem_sleep_percentage_std` | Standard deviation of REM sleep percentage | `5.0` |
| `sleep_interruptions_lambda` | Lambda parameter for Poisson distribution of sleep interruptions | `2.0` |
| `time_to_sleep_mean` | Mean time to fall asleep in minutes | `20.0` |
| `time_to_sleep_std` | Standard deviation of time to fall asleep | `15.0` |
| `circadian_rhythm_enabled` | Enable circadian rhythm patterns | `true` |
| `weekend_variation_enabled` | Enable weekend variation | `true` |
| `sleep_debt_enabled` | Enable sleep debt accumulation and recovery | `true` |

### Weather Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `temperature_mean` | Mean temperature in degrees Celsius | `20.0` |
| `temperature_std` | Standard deviation of temperature | `8.0` |
| `humidity_mean` | Mean humidity percentage | `60.0` |
| `humidity_std` | Standard deviation of humidity | `15.0` |
| `pressure_mean` | Mean barometric pressure in hPa | `1013.0` |
| `pressure_std` | Standard deviation of pressure | `10.0` |
| `precipitation_lambda` | Lambda parameter for exponential distribution of precipitation | `2.0` |
| `wind_speed_lambda` | Lambda parameter for exponential distribution of wind speed | `10.0` |
| `seasonal_variation_enabled` | Enable seasonal variation | `true` |
| `daily_fluctuation_enabled` | Enable daily fluctuation | `true` |
| `extreme_events_enabled` | Enable extreme weather events | `true` |
| `barometric_pressure_changes_enabled` | Enable barometric pressure changes | `true` |

### Stress and Diet Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `stress_mean` | Mean stress level (0-10) | `5.0` |
| `stress_std` | Standard deviation of stress level | `2.0` |
| `alcohol_lambda` | Lambda parameter for Poisson distribution of alcohol consumption | `1.0` |
| `meal_regularity_mean` | Mean meal regularity score (0-10) | `6.0` |
| `meal_regularity_std` | Standard deviation of meal regularity | `2.0` |
| `hydration_mean` | Mean hydration level score (0-10) | `6.0` |
| `hydration_std` | Standard deviation of hydration level | `2.0` |
| `exercise_lambda` | Lambda parameter for exponential distribution of exercise duration | `30.0` |
| `work_life_cycle_enabled` | Enable work/life cycle patterns | `true` |
| `weekend_variation_enabled` | Enable weekend variation | `true` |
| `stress_diet_correlation_enabled` | Enable correlation between stress and diet | `true` |

### Physiological Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `heart_rate_mean` | Mean heart rate in beats per minute | `75.0` |
| `heart_rate_std` | Standard deviation of heart rate | `10.0` |
| `systolic_mean` | Mean systolic blood pressure in mmHg | `120.0` |
| `systolic_std` | Standard deviation of systolic blood pressure | `15.0` |
| `diastolic_mean` | Mean diastolic blood pressure in mmHg | `80.0` |
| `diastolic_std` | Standard deviation of diastolic blood pressure | `10.0` |
| `temperature_mean` | Mean body temperature in degrees Celsius | `36.8` |
| `temperature_std` | Standard deviation of body temperature | `0.5` |
| `respiratory_mean` | Mean respiratory rate in breaths per minute | `16.0` |
| `respiratory_std` | Standard deviation of respiratory rate | `3.0` |
| `circadian_rhythm_enabled` | Enable circadian rhythm patterns | `true` |
| `exercise_effect_enabled` | Enable exercise effects | `true` |
| `parameter_correlations_enabled` | Enable correlations between physiological parameters | `true` |

### Migraine Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `temporal_effects.lag_hours` | Lag hours between triggers and migraine onset | `[6, 12, 24, 48]` |
| `temporal_effects.lag_weights` | Weights for each lag hour | `[0.2, 0.4, 0.3, 0.1]` |
| `temporal_effects.cumulative_effect_enabled` | Enable cumulative effects | `true` |
| `temporal_effects.cumulative_effect_decay` | Decay factor for cumulative effects | `0.8` |
| `individual_variability.enabled` | Enable individual variability | `true` |
| `individual_variability.sensitivity_std` | Standard deviation of trigger sensitivity | `0.5` |
| `individual_variability.threshold_mean` | Mean migraine threshold | `0.7` |
| `individual_variability.threshold_std` | Standard deviation of migraine threshold | `0.1` |
| `sleep_weights` | Weights for sleep features | See code |
| `weather_weights` | Weights for weather features | See code |
| `stress_diet_weights` | Weights for stress/diet features | See code |
| `physiological_weights` | Weights for physiological features | See code |

## Output Formats

The pipeline supports the following output formats:

### PyTorch

```python
{
    "train": {
        "sleep": torch.Tensor,  # Shape: (train_samples, time_periods, sleep_features)
        "weather": torch.Tensor,  # Shape: (train_samples, time_periods, weather_features)
        "stress_diet": torch.Tensor,  # Shape: (train_samples, time_periods, stress_diet_features)
        "physiological": torch.Tensor,  # Shape: (train_samples, time_periods, physiological_features)
        "target": torch.Tensor,  # Shape: (train_samples, time_periods)
        "X": [sleep_tensor, weather_tensor, stress_diet_tensor, physiological_tensor],
        "y": target_tensor
    },
    "val": { ... },
    "test": { ... },
    "metadata": { ... }
}
```

### NumPy

```python
{
    "train": {
        "sleep": np.ndarray,  # Shape: (train_samples, time_periods, sleep_features)
        "weather": np.ndarray,  # Shape: (train_samples, time_periods, weather_features)
        "stress_diet": np.ndarray,  # Shape: (train_samples, time_periods, stress_diet_features)
        "physiological": np.ndarray,  # Shape: (train_samples, time_periods, physiological_features)
        "target": np.ndarray,  # Shape: (train_samples, time_periods)
        "X": [sleep_array, weather_array, stress_diet_array, physiological_array],
        "y": target_array
    },
    "val": { ... },
    "test": { ... },
    "test_predictions": {
        "y_test": np.ndarray,  # Shape: (test_samples, time_periods)
        "y_pred_test": np.ndarray  # Shape: (test_samples, time_periods)
    },
    "metadata": { ... }
}
```

### Pandas

```python
{
    "dataframes": {
        "train": pd.DataFrame,  # Columns: sample_id, time_period, features..., migraine_intensity
        "val": pd.DataFrame,
        "test": pd.DataFrame
    },
    "metadata": { ... }
}
```

### JSON

```python
{
    "data": {
        "train": [
            {
                "sample_id": int,
                "time_series": [
                    {
                        "time_period": int,
                        "sleep": { ... },
                        "weather": { ... },
                        "stress_diet": { ... },
                        "physiological": { ... },
                        "migraine_intensity": float
                    },
                    ...
                ]
            },
            ...
        ],
        "val": [ ... ],
        "test": [ ... ]
    },
    "metadata": { ... }
}
```

## Integration with MoE System

The pipeline is designed to integrate seamlessly with the existing MoE system:

1. The output format is compatible with the MoE model input requirements
2. The data splitting ensures proper training, validation, and testing
3. The test predictions file format is compatible with the dashboard
4. The metadata provides information about the generated data

## Improvements Over Previous Approach

The Enhanced Data Generation Pipeline offers several improvements over the previous approach:

1. **Realistic Temporal Patterns**: Proper modeling of circadian rhythms, weekly patterns, and seasonal variations
2. **Correlations Between Variables**: Realistic correlations between related variables (e.g., stress and sleep quality)
3. **Individual Variability**: Modeling of individual differences in baseline values and trigger sensitivity
4. **Comprehensive Validation**: Robust validation to ensure data quality and consistency
5. **Flexible Output Formats**: Support for multiple output formats for different use cases
6. **Modular Architecture**: Easy to extend with new features or modify existing components
7. **Configurable Parameters**: Fine-grained control over data generation parameters
8. **Reproducibility**: Consistent results with the same random seed

## Testing

The pipeline includes comprehensive tests to ensure functionality and data quality:

1. **Unit Tests**: Tests for individual components
2. **Integration Tests**: Tests for component interactions
3. **End-to-End Tests**: Tests for the entire pipeline
4. **Visualization Tests**: Visual inspection of generated data

To run the tests:

```bash
cd moe_data_pipeline
python -m tests.test_pipeline
```

## Future Enhancements

Potential future enhancements to the pipeline include:

1. **Additional Data Domains**: Support for additional data domains (e.g., medication, hormonal cycles)
2. **More Complex Temporal Patterns**: More sophisticated modeling of temporal patterns and interactions
3. **Real Data Integration**: Ability to incorporate real data for calibration or augmentation
4. **Anomaly Generation**: Controlled generation of anomalous data for robustness testing
5. **Parallel Processing**: Optimization for large-scale data generation
6. **Interactive Visualization**: Interactive tools for exploring generated data
7. **API Integration**: REST API for remote data generation

## Conclusion

The Enhanced Data Generation Pipeline provides a robust solution for generating realistic synthetic data for migraine prediction. Its modular architecture, comprehensive validation, and flexible output formats make it a valuable tool for training and evaluating the MoE system.
