
import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("About the Enhanced Data Pipeline Integration")

st.markdown('''
## Overview

The Enhanced Data Pipeline Integration improves the existing Mixture of Experts (MoE) system
for migraine prediction by generating more realistic synthetic data with proper temporal patterns,
correlations between variables, and individual variability in migraine triggers.

## Key Features

### Realistic Temporal Patterns
- Circadian rhythms (daily patterns)
- Weekly patterns
- Seasonal variations
- Individual variability
- Lag effects between triggers and migraine onset

### Proper Correlations
- Between related variables
- Between different data modalities
- Between triggers and migraine occurrence

### Individual Variability
- Baseline values
- Trigger sensitivity
- Response patterns

### Comprehensive Validation
- Data alignment
- Data types
- Missing values
- Sequence offset correctness
- Data split proportions
- Expert output consistency

### Multiple Output Formats
- PyTorch
- NumPy
- Pandas
- JSON

## Architecture

The integration follows a modular architecture:

```
Enhanced Data Pipeline
    ↓
Adapters (Sleep, Weather, Stress/Diet, Physio)
    ↓
MoE Model (Expert Models, Gating Network, Fusion Mechanism)
    ↓
Performance Evaluation
    ↓
Dashboard Visualization
```

## Benefits

- More realistic training data
- Better model performance
- More interpretable results
- Improved generalization
- Enhanced explainability

## Future Enhancements

- Additional data modalities
- Personalization
- Mobile integration
- Advanced explainability
- Real-time data integration
- Federated learning
''')

st.header("Contact Information")
st.markdown('''
For more information, please contact:
- Email: info@migraineprediction.example.com
- Website: https://migraineprediction.example.com
''')

st.header("References")
st.markdown('''
1. FuseMoE: https://github.com/aaronhan223/FuseMoE
2. Synthea: https://github.com/synthetichealth/synthea
3. PyGMO: https://esa.github.io/pygmo2/
''')
