#!/usr/bin/env python3
"""
Train and Evaluate FuseMoE Migraine Prediction Model

This script trains the FuseMoEMigraineModel on the processed synthetic data
and evaluates its performance using standard classification metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json # Add json import
import sys # Import sys for exit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split # Although data_processor does splitting, we might need it

# Import custom modules
from data_processor import process_data
from updated_fusemoe_integration import FuseMoEMigraineModel, FuseMoEAdapter

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 # Keep low for initial testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class MigraineDataset(Dataset):
    """PyTorch Dataset for Migraine Prediction data."""
    def __init__(self, features, targets):
        # Ensure features is a dictionary of tensors and targets is a tensor
        self.features = {key: torch.tensor(val, dtype=torch.float32) if not isinstance(val, torch.Tensor) else val.float()
                         for key, val in features.items()}
        self.targets = torch.tensor(targets, dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets.float()
        # Ensure targets are correctly shaped (N, 1) for BCELoss
        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Return features and target for the given index
        return {key: val[idx] for key, val in self.features.items()}, self.targets[idx]

# --- Training Function ---
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    """Train the model."""
    print(f"Starting training on {DEVICE}...")
    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        train_loss = 0.0

        for i, (inputs_dict, targets) in enumerate(train_loader):
            # Move data to device
            inputs_dict = {key: val.to(DEVICE) for key, val in inputs_dict.items()}
            targets = targets.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(**inputs_dict) # Pass features as keyword arguments

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation phase
        model.eval() # Set model to evaluation mode
        valid_loss = 0.0
        with torch.no_grad():
            for inputs_dict, targets in valid_loader:
                inputs_dict = {key: val.to(DEVICE) for key, val in inputs_dict.items()}
                targets = targets.to(DEVICE)
                outputs = model(**inputs_dict)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Train Loss: {avg_train_loss:.4f}, Avg Valid Loss: {avg_valid_loss:.4f}")

    print("Training finished.")

# --- Evaluation Function ---
def evaluate_model(model, test_loader):
    """Evaluate the model on the test set."""
    print("Evaluating model...")
    model.eval() # Set model to evaluation mode
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs_dict, targets in test_loader:
            inputs_dict = {key: val.to(DEVICE) for key, val in inputs_dict.items()}
            targets = targets.to(DEVICE)

            outputs = model(**inputs_dict)
            # Ensure outputs are squeezed correctly before converting to numpy
            probabilities = outputs.squeeze().cpu().numpy()
            # Handle potential scalar output if batch size is 1
            if probabilities.ndim == 0:
                 probabilities = np.array([probabilities])
                 targets_np = targets.cpu().numpy().flatten()
            else:
                 targets_np = targets.cpu().numpy().flatten()

            predictions = (probabilities > 0.5).astype(int)

            all_targets.extend(targets_np)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    try:
        # Ensure there are probabilities for both classes before calculating ROC AUC
        if len(np.unique(all_targets)) > 1:
             roc_auc = roc_auc_score(all_targets, all_probabilities)
        else:
             print("Warning: Only one class present in test set targets. ROC AUC is not defined.")
             roc_auc = float("nan")
    except ValueError as e:
        print(f"Warning: Could not calculate ROC AUC. Error: {e}")
        roc_auc = float("nan") # Handle cases where only one class is present or predicted

    print("\nEvaluation Results:")
    print(f"- Accuracy:  {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall:    {recall:.4f}")
    print(f"- F1-Score:  {f1:.4f}")
    print(f"- ROC AUC:   {roc_auc:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and Process Data
    print("Loading and processing data...")
    try:
        # Assuming process_data returns dataframes/series
        X_train_df, y_train_series, X_valid_df, y_valid_series, X_test_df, y_test_series = process_data()
    except Exception as e:
        print(f"Error during data processing: {e}")
        sys.exit(1)

    # 2. Create Model and Adapter
    # Use default config for now, can be customized
    model_config = {
        "num_experts": 4,
        "k": 2,
        "model_dim": 64,
        "hidden_dim": 128,
        "sleep_dim": 6,
        "weather_dim": 4,
        "stress_diet_dim": 6,
        "physio_dim": 10 # Must match the placeholder dimension
    }
    model = FuseMoEMigraineModel(config=model_config)
    adapter = FuseMoEAdapter(config=model_config)

    # 3. Adapt Data and Create Datasets/DataLoaders
    print("Adapting data and creating datasets...")
    try:
        # Adapt data returns dict of numpy arrays and numpy array targets
        train_inputs_np, train_targets_np = adapter.adapt_data_for_fusemoe(X_train_df, y_train_series, physio_dim_expected=model.physio_dim)
        valid_inputs_np, valid_targets_np = adapter.adapt_data_for_fusemoe(X_valid_df, y_valid_series, physio_dim_expected=model.physio_dim)
        test_inputs_np, test_targets_np = adapter.adapt_data_for_fusemoe(X_test_df, y_test_series, physio_dim_expected=model.physio_dim)

        if train_inputs_np is None or valid_inputs_np is None or test_inputs_np is None:
            print("Error adapting data. Exiting.")
            sys.exit(1)

        # Convert numpy arrays to tensors within the Dataset class
        train_dataset = MigraineDataset(train_inputs_np, train_targets_np)
        valid_dataset = MigraineDataset(valid_inputs_np, valid_targets_np)
        test_dataset = MigraineDataset(test_inputs_np, test_targets_np)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("DataLoaders created.")

    except Exception as e:
        print(f"Error creating datasets/dataloaders: {e}")
        sys.exit(1)


    # 4. Define Loss and Optimizer
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Train the Model
    try:
        train_model(model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


    # 6. Evaluate the Model
    try:
        evaluation_results = evaluate_model(model, test_loader)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        evaluation_results = {} # Ensure it's defined even on error
        sys.exit(1)


    # 7. Save Evaluation Results
    results_filepath = "/home/ubuntu/evaluation_results.json"
    try:
        # Convert NaN ROC AUC to None for JSON compatibility if necessary
        if "roc_auc" in evaluation_results and np.isnan(evaluation_results["roc_auc"]):
            evaluation_results_json = evaluation_results.copy()
            evaluation_results_json["roc_auc"] = None # Use None for JSON
        else:
            evaluation_results_json = evaluation_results

        with open(results_filepath, 'w') as f:
            json.dump(evaluation_results_json, f, indent=4)
        print(f"Evaluation results saved to {results_filepath}")

    except Exception as e:
        print(f"Error saving evaluation results: {e}")

    # Optionally save the model
    # torch.save(model.state_dict(), "/home/ubuntu/fusemoe_migraine_model.pth")
    # print("Model state dictionary saved.")

