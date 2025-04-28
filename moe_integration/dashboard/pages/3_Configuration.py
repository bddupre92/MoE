
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Configuration",
    page_icon="⚙️",
    layout="wide"
)

st.title("Configuration")

st.markdown('''
This page allows you to configure the parameters of the enhanced data pipeline.
You can adjust various settings to customize the data generation process and model training.
''')

# General configuration
st.header("General Configuration")

col1, col2 = st.columns(2)
with col1:
    st.number_input("Random Seed", min_value=0, max_value=1000, value=42)
    st.number_input("Sequence Length", min_value=1, max_value=100, value=30)
with col2:
    st.slider("Train Split", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    st.slider("Validation Split", min_value=0.05, max_value=0.3, value=0.15, step=0.05)

# Temporal pattern configuration
st.header("Temporal Pattern Configuration")

col1, col2 = st.columns(2)
with col1:
    st.checkbox("Enable Circadian Rhythms", value=True)
    st.checkbox("Enable Weekly Patterns", value=True)
    st.checkbox("Enable Seasonal Variations", value=True)
with col2:
    st.checkbox("Enable Individual Variability", value=True)
    st.checkbox("Enable Lag Effects", value=True)

st.subheader("Pattern Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    st.slider("Circadian Amplitude", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    st.slider("Weekly Amplitude", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
with col2:
    st.slider("Seasonal Amplitude", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    st.slider("Individual Variability Scale", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
with col3:
    st.multiselect("Lag Effect Hours", options=[6, 12, 24, 48, 72], default=[6, 12, 24, 48])

# Correlation configuration
st.header("Correlation Configuration")

st.subheader("Inter-modality Correlations")
col1, col2 = st.columns(2)
with col1:
    st.slider("Sleep-Weather Correlation", min_value=-1.0, max_value=1.0, value=0.3, step=0.1)
    st.slider("Sleep-Stress Correlation", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)
    st.slider("Weather-Stress Correlation", min_value=-1.0, max_value=1.0, value=0.2, step=0.1)
with col2:
    st.slider("Physio-Sleep Correlation", min_value=-1.0, max_value=1.0, value=0.6, step=0.1)
    st.slider("Physio-Stress Correlation", min_value=-1.0, max_value=1.0, value=0.4, step=0.1)

st.subheader("Target Correlations")
col1, col2 = st.columns(2)
with col1:
    st.slider("Target-Sleep Correlation", min_value=-1.0, max_value=1.0, value=0.7, step=0.1)
    st.slider("Target-Weather Correlation", min_value=-1.0, max_value=1.0, value=0.4, step=0.1)
with col2:
    st.slider("Target-Stress Correlation", min_value=-1.0, max_value=1.0, value=0.6, step=0.1)
    st.slider("Target-Physio Correlation", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)

# Training configuration
st.header("Training Configuration")

col1, col2 = st.columns(2)
with col1:
    st.number_input("Batch Size", min_value=1, max_value=256, value=32)
    st.number_input("Number of Epochs", min_value=1, max_value=200, value=50)
with col2:
    st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    st.number_input("Early Stopping Patience", min_value=1, max_value=50, value=10)

# Save configuration
st.header("Save Configuration")
st.text_input("Configuration Name", value="default_config")
if st.button("Save Configuration"):
    st.success("Configuration saved successfully!")

# Load configuration
st.header("Load Configuration")
st.selectbox("Select Configuration", ["default_config", "optimized_config", "custom_config"])
if st.button("Load Configuration"):
    st.success("Configuration loaded successfully!")
