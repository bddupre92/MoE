# README.md - Enhanced Data Generation Pipeline for Migraine Prediction MoE

This package contains the Enhanced Data Generation Pipeline for the Mixture of Experts (MoE) migraine prediction system. The pipeline generates realistic synthetic data with proper temporal patterns, correlations between variables, and individual variability.

## Contents

- `moe_data_pipeline/`: Main pipeline implementation
  - `__init__.py`: Package initialization
  - `pipeline.py`: Main pipeline integration
  - `config/`: Configuration management
  - `generators/`: Domain-specific data generators
  - `orchestration/`: Temporal pattern orchestration
  - `validation/`: Data validation
  - `formatting/`: Output formatting
  - `tests/`: Pipeline tests
  - `demos/`: Demonstration examples
- `documentation.md`: Comprehensive documentation
- `final_report.md`: Final report and recommendations
- `research/`: Research and analysis
  - `current_limitations.md`: Analysis of current limitations
  - `best_practices_research.md`: Research on best practices
  - `improved_pipeline_design.md`: Design of improved pipeline
  - `implementation_plan.md`: Implementation plan

## Quick Start

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

## Running Demonstrations

```bash
cd moe_data_pipeline
python -m demos.demo_pipeline
```

## Running Tests

```bash
cd moe_data_pipeline
python -m tests.test_pipeline
```

## Integration with MoE System

The pipeline is designed to integrate seamlessly with the existing MoE system:

1. The output format is compatible with the MoE model input requirements
2. The data splitting ensures proper training, validation, and testing
3. The test predictions file format is compatible with the dashboard
4. The metadata provides information about the generated data

For detailed integration instructions, see the documentation.

## Documentation

For comprehensive documentation, see `documentation.md`.

## Final Report

For the final report and recommendations, see `final_report.md`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
