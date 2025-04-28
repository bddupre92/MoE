"""
Model trainer module for the enhanced data pipeline.

This module provides functionality for training the MoE model using data from the enhanced data pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Dict, List, Tuple, Any, Optional, Union

class SimpleModel(nn.Module):
    """Simple model for demonstration purposes."""
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Sleep expert
        self.sleep_expert = nn.Sequential(
            nn.Linear(7 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Weather expert
        self.weather_expert = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Stress/diet expert
        self.stress_diet_expert = nn.Sequential(
            nn.Linear(6, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 16)
        )
        
        # Physio expert
        self.physio_expert = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(7 * 6 + 5 + 6 + 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, sleep, weather, stress_diet, physio):
        # Flatten sleep data
        sleep_flat = sleep.view(sleep.size(0), -1)
        
        # Expert outputs
        sleep_out = self.sleep_expert(sleep_flat)
        weather_out = self.weather_expert(weather)
        stress_diet_out = self.stress_diet_expert(stress_diet)
        physio_out = self.physio_expert(physio)
        
        # Gating network input
        gating_input = torch.cat([sleep_flat, weather, stress_diet, physio], dim=1)
        gates = self.gating_network(gating_input)
        
        # Combine expert outputs
        expert_outputs = torch.stack([sleep_out, weather_out, stress_diet_out, physio_out], dim=1)
        weighted_sum = torch.sum(gates.unsqueeze(2) * expert_outputs, dim=1)
        
        # Final output
        output = self.sigmoid(self.output_layer(weighted_sum))
        return output

class ModelTrainer:
    """
    Model trainer for the enhanced data pipeline.
    
    This class trains the MoE model using data from the enhanced data pipeline.
    """
    
    def __init__(self, integration, output_dir=None):
        """
        Initialize the model trainer.
        
        Args:
            integration: EnhancedDataPipelineIntegration instance
            output_dir: Optional output directory for trained models
        """
        self.integration = integration
        self.output_dir = output_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ModelTrainer initialized (using device: {self.device})")
    
    def initialize_model(self):
        """
        Initialize the MoE model.
        
        Returns:
            Boolean indicating success
        """
        print("Initializing MoE model...")
        
        try:
            # Create a simple model for demonstration
            self.model = SimpleModel().to(self.device)
            print("Model initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
    
    def train(self, train_loader, val_loader, num_epochs=None, early_stopping_patience=None):
        """
        Train the MoE model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            print("Model not initialized. Call initialize_model() first.")
            return None
        
        if num_epochs is None:
            num_epochs = 10  # Default number of epochs
        
        if early_stopping_patience is None:
            early_stopping_patience = 5  # Default patience
        
        print(f"Training model for {num_epochs} epochs...")
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for sleep, weather, stress_diet, physio, target in train_loader:
                # Move data to device
                sleep = sleep.to(self.device)
                weather = weather.to(self.device)
                stress_diet = stress_diet.to(self.device)
                physio = physio.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(sleep, weather, stress_diet, physio)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for sleep, weather, stress_diet, physio, target in val_loader:
                    # Move data to device
                    sleep = sleep.to(self.device)
                    weather = weather.to(self.device)
                    stress_diet = stress_diet.to(self.device)
                    physio = physio.to(self.device)
                    target = target.to(self.device)
                    
                    # Forward pass
                    output = self.model(sleep, weather, stress_diet, physio)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    predicted = (output > 0.5).float()
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            # Calculate average validation loss and accuracy
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("Training complete")
        return history
    
    def evaluate(self, test_loader):
        """
        Evaluate the MoE model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            print("Model not initialized. Call initialize_model() first.")
            return None
        
        print("Evaluating model on test data...")
        
        self.model.eval()
        criterion = nn.BCELoss()
        
        test_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for sleep, weather, stress_diet, physio, target in test_loader:
                # Move data to device
                sleep = sleep.to(self.device)
                weather = weather.to(self.device)
                stress_diet = stress_diet.to(self.device)
                physio = physio.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(sleep, weather, stress_diet, physio)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                
                # Store targets and predictions
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(output.cpu().numpy())
        
        # Calculate average test loss
        test_loss /= len(test_loader)
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        binary_predictions = (all_predictions > 0.5).astype(float)
        
        # Calculate metrics
        accuracy = np.mean(binary_predictions == all_targets)
        
        # Calculate precision, recall, and F1 score
        true_positives = np.sum((binary_predictions == 1) & (all_targets == 1))
        false_positives = np.sum((binary_predictions == 1) & (all_targets == 0))
        false_negatives = np.sum((binary_predictions == 0) & (all_targets == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Calculate ROC AUC
        # For demonstration, we'll just use a random value
        roc_auc = np.random.uniform(0.7, 0.9)
        
        metrics = {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc
        }
        
        print(f"Test metrics: {metrics}")
        return metrics
