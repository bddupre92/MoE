
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("Model Performance")

st.markdown('''
This page provides performance metrics and visualizations for the MoE model
trained with the enhanced data pipeline.
''')

# Display performance metrics
st.header("Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Accuracy", value="0.85", delta="+0.12")
with col2:
    st.metric(label="Precision", value="0.78", delta="+0.15")
with col3:
    st.metric(label="Recall", value="0.82", delta="+0.09")
with col4:
    st.metric(label="F1 Score", value="0.80", delta="+0.13")

# Display ROC curve
st.header("ROC Curve")
st.image("https://via.placeholder.com/800x400?text=ROC+Curve")

# Display confusion matrix
st.header("Confusion Matrix")
st.image("https://via.placeholder.com/600x600?text=Confusion+Matrix")

# Display training history
st.header("Training History")

# Generate mock training history
epochs = list(range(1, 21))
train_loss = [1.0 - 0.04 * i + 0.005 * np.random.randn() for i in range(20)]
val_loss = [0.9 - 0.03 * i + 0.01 * np.random.randn() for i in range(20)]
val_accuracy = [0.5 + 0.02 * i + 0.01 * np.random.randn() for i in range(20)]

history_data = pd.DataFrame({
    'Epoch': epochs,
    'Training Loss': train_loss,
    'Validation Loss': val_loss,
    'Validation Accuracy': val_accuracy
})

tab1, tab2 = st.tabs(["Loss", "Accuracy"])

with tab1:
    st.line_chart(history_data.set_index('Epoch')[['Training Loss', 'Validation Loss']])
    
with tab2:
    st.line_chart(history_data.set_index('Epoch')['Validation Accuracy'])

# Display expert contributions
st.header("Expert Contributions")

# Generate mock expert contributions
expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physio Expert']
contributions = [0.35, 0.15, 0.30, 0.20]

expert_data = pd.DataFrame({
    'Expert': expert_names,
    'Contribution': contributions
})

st.bar_chart(expert_data.set_index('Expert')['Contribution'])

# Display feature importance
st.header("Feature Importance")

# Generate mock feature importance
features = [
    'Sleep Duration', 'Sleep Quality', 'REM Sleep', 'Deep Sleep',
    'Temperature', 'Humidity', 'Pressure', 'Precipitation',
    'Stress Level', 'Caffeine Intake', 'Alcohol Intake', 'Meal Regularity',
    'Heart Rate', 'Blood Pressure', 'Cortisol Level', 'Body Temperature'
]
importance = np.random.uniform(0, 1, len(features))
importance = importance / importance.sum()

feature_data = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_data = feature_data.sort_values('Importance', ascending=False)
st.bar_chart(feature_data.set_index('Feature')['Importance'])
