"""
Configuration module for the enhanced data pipeline integration.

This module contains configuration parameters for the enhanced data pipeline integration.
"""

# General configuration
GENERAL_CONFIG = {
    'random_seed': 42,
    'sequence_length': 30,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
}

# Temporal pattern configuration
TEMPORAL_CONFIG = {
    'enable_circadian_rhythms': True,
    'enable_weekly_patterns': True,
    'enable_seasonal_variations': True,
    'enable_individual_variability': True,
    'enable_lag_effects': True,
    'circadian_amplitude': 0.5,
    'weekly_amplitude': 0.3,
    'seasonal_amplitude': 0.2,
    'individual_variability_scale': 0.4,
    'lag_effect_hours': [6, 12, 24, 48],
}

# Correlation configuration
CORRELATION_CONFIG = {
    'sleep_weather_correlation': 0.3,
    'sleep_stress_correlation': 0.5,
    'weather_stress_correlation': 0.2,
    'physio_sleep_correlation': 0.6,
    'physio_stress_correlation': 0.4,
    'target_sleep_correlation': 0.7,
    'target_weather_correlation': 0.4,
    'target_stress_correlation': 0.6,
    'target_physio_correlation': 0.5,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'early_stopping_patience': 10,
}

# Model configuration
MODEL_CONFIG = {
    'sleep_expert_hidden_dims': [64, 32],
    'weather_expert_hidden_dims': [32, 16],
    'stress_diet_expert_hidden_dims': [48, 24],
    'physio_expert_hidden_dims': [32, 16],
    'gating_network_hidden_dims': [64, 32],
    'dropout_rate': 0.2,
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'theme': 'light',
    'show_code': False,
    'wide_mode': True,
}
