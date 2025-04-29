#!/usr/bin/env python3
"""
Streamlit Dashboard for FuseMoE Migraine Prediction Model

This dashboard visualizes the performance metrics and potentially
other insights from the trained FuseMoE model.
"""

import streamlit as st
import pandas as pd
import json
import os

# --- Configuration ---
RESULTS_FILE = "/home/ubuntu/evaluation_results.json" # Expected output from train_evaluate.py
MODEL_FILE = "/home/ubuntu/fusemoe_migraine_model.pth" # Optional: Path to saved model

# --- Helper Functions ---
def load_results(filepath):
    """Load evaluation results from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            st.error(f"Error loading results file ({filepath}): {e}")
            return None
    else:
        st.warning(f"Evaluation results file not found: {filepath}. Displaying placeholders.")
        # Return placeholder data if file doesn't exist
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0
        }

# --- Dashboard Layout ---
st.set_page_config(layout="wide")
st.title("FuseMoE Migraine Prediction Model Dashboard")

st.header("Model Performance Evaluation")

# Load results
results = load_results(RESULTS_FILE)

if results:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{results.get('accuracy', 0.0):.4f}")
    col2.metric("Precision", f"{results.get('precision', 0.0):.4f}")
    col3.metric("Recall", f"{results.get('recall', 0.0):.4f}")
    col4.metric("F1-Score", f"{results.get('f1_score', 0.0):.4f}")
    col5.metric("ROC AUC", f"{results.get('roc_auc', 0.0):.4f}")
else:
    st.info("Waiting for evaluation results...")

# Placeholder for future visualizations (e.g., confusion matrix, ROC curve, expert contributions)
st.header("Further Analysis (Placeholders)")

with st.expander("Confusion Matrix"):
    st.write("Placeholder for Confusion Matrix visualization.")
    # Example: Load predictions/targets and use st.pyplot(fig) or st.plotly_chart(fig)

with st.expander("ROC Curve"):
    st.write("Placeholder for ROC Curve visualization.")
    # Example: Load probabilities/targets and use st.pyplot(fig) or st.plotly_chart(fig)

with st.expander("Expert Contributions"):
    st.write("Placeholder for visualizing expert contributions (e.g., gating weights). Requires model modifications to save gating outputs.")

st.sidebar.header("About")
st.sidebar.info(
    "This dashboard displays the performance of the FuseMoE model trained for "
    "migraine prediction using synthetic data based on MIMIC-IV structures and expert modalities."
)
if os.path.exists(MODEL_FILE):
    st.sidebar.success(f"Model file found: {MODEL_FILE}")
else:
    st.sidebar.warning(f"Model file not found: {MODEL_FILE}")


