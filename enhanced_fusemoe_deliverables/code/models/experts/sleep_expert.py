"""
Sleep Expert Model for Enhanced FuseMoE

This module provides the implementation of the sleep expert model for the
Enhanced FuseMoE system for migraine prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class SleepExpert(nn.Module):
    """
    Expert model for sleep data in the Enhanced FuseMoE system.
    
    This model processes sleep-related features to predict migraine occurrence.
    It uses a combination of convolutional and recurrent layers to capture
    temporal patterns in sleep data.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Size of hidden layers
        output_dim (int): Size of output features
        num_layers (int): Number of LSTM layers
        dropout_rate (float): Dropout probability
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 32,
                num_layers: int = 2, dropout_rate: float = 0.2):
        """
        Initialize the sleep expert model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layers
            output_dim: Size of output features
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability
        """
        super(SleepExpert, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # CNN layers for sequence processing
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sleep expert model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for CNN
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, sequence_length]
        
        # Apply CNN
        x = self.conv(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, hidden_dim]
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, sequence_length, hidden_dim]
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)  # [batch_size, sequence_length]
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)  # [batch_size, hidden_dim]
        
        # Apply output layer
        output = self.output(context)  # [batch_size, output_dim]
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the sleep expert model.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SleepExpert':
        """
        Create a sleep expert model from configuration.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            Initialized sleep expert model
        """
        return cls(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 64),
            output_dim=config.get('output_dim', 32),
            num_layers=config.get('num_layers', 2),
            dropout_rate=config.get('dropout_rate', 0.2)
        )


def create_sleep_expert(input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 32,
                       num_layers: int = 2, dropout_rate: float = 0.2,
                       learning_rate: float = 0.001) -> SleepExpert:
    """
    Create a sleep expert model with optimizer.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Size of hidden layers
        output_dim: Size of output features
        num_layers: Number of LSTM layers
        dropout_rate: Dropout probability
        learning_rate: Learning rate for optimizer
        
    Returns:
        Initialized sleep expert model
    """
    model = SleepExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    # Create optimizer but don't return it as part of a tuple
    # This fixes the type checking issue in the tests
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Store optimizer as an attribute of the model for later use if needed
    model.optimizer = optimizer
    
    return model


class SleepExpertTrainer:
    """
    Trainer for sleep expert models.
    
    This class provides methods for training and evaluating sleep expert models.
    
    Attributes:
        model (SleepExpert): Sleep expert model
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        device (torch.device): Device to use for training
        criterion (nn.Module): Loss function
    """
    
    def __init__(self, model: SleepExpert, optimizer: torch.optim.Optimizer,
                device: Optional[torch.device] = None,
                criterion: Optional[nn.Module] = None):
        """
        Initialize the sleep expert trainer.
        
        Args:
            model: Sleep expert model
            optimizer: Optimizer for model parameters
            device: Device to use for training (default: cuda if available, else cpu)
            criterion: Loss function (default: BCEWithLogitsLoss)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store outputs and targets for ROC AUC calculation
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        # Calculate ROC AUC
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # This would use sklearn.metrics.roc_auc_score in a real implementation
        # For now, we'll just use a placeholder
        auc = 0.5  # Placeholder
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'auc': auc}
    
    def train(self, train_dataloader: torch.utils.data.DataLoader,
             val_dataloader: torch.utils.data.DataLoader,
             num_epochs: int = 10) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            
        Returns:
            Dictionary containing training history
        """
        train_history = []
        val_history = []
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_dataloader)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                 f"Train Loss: {train_metrics['loss']:.4f}, "
                 f"Train Acc: {train_metrics['accuracy']:.4f}, "
                 f"Val Loss: {val_metrics['loss']:.4f}, "
                 f"Val Acc: {val_metrics['accuracy']:.4f}, "
                 f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Store metrics
            train_history.append(train_metrics)
            val_history.append(val_metrics)
        
        return {'train_history': train_history, 'val_history': val_history}
