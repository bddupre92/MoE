"""
Performance metrics module for Enhanced FuseMoE

This module provides functionality for calculating performance metrics for the
Enhanced FuseMoE system for migraine prediction.
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class SimpleMetrics:
    """
    Simple metrics calculator for the Enhanced FuseMoE system.
    
    This class provides methods for calculating performance metrics from
    numpy arrays or saved prediction files.
    """
    
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            Dictionary containing performance metrics
        """
        # Ensure arrays are 1D
        y_true = y_true.flatten()
        y_pred_proba = y_pred.flatten()
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        
        # Handle case where there are no positive samples
        if np.sum(y_true) > 0:
            metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
            metrics['roc_auc'] = 0.5
        
        return metrics
    
    @staticmethod
    def calculate_from_file(file_path: str, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate performance metrics from saved prediction file.
        
        Args:
            file_path: Path to saved prediction file
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            Dictionary containing performance metrics
        """
        # Load predictions
        data = np.load(file_path)
        
        # Extract true labels and predictions
        y_true = data['y_test']
        y_pred = data['y_pred_test']
        
        # Calculate metrics
        return SimpleMetrics.calculate(y_true, y_pred, threshold)


class MigrainePerformanceMetrics:
    """
    Performance metrics calculator for the Enhanced FuseMoE system.
    
    This class provides methods for calculating performance metrics for
    migraine prediction using the Enhanced FuseMoE system.
    
    Attributes:
        model (torch.nn.Module): Model to evaluate
        device (torch.device): Device to use for evaluation
        train_loader (torch.utils.data.DataLoader, optional): DataLoader for training data
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for validation data
        test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                train_loader: Optional[torch.utils.data.DataLoader] = None,
                val_loader: Optional[torch.utils.data.DataLoader] = None,
                test_loader: Optional[torch.utils.data.DataLoader] = None):
        """
        Initialize the migraine performance metrics calculator.
        
        Args:
            model: Model to evaluate
            device: Device to use for evaluation
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def calculate_metrics(self, data_loader: torch.utils.data.DataLoader, 
                         output_dir: Optional[str] = None, 
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            data_loader: DataLoader for data to evaluate
            output_dir: Directory to save predictions and metrics
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            Dictionary containing performance metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize lists for true labels and predictions
        y_true_list = []
        y_pred_list = []
        
        # Disable gradient computation
        with torch.no_grad():
            # Iterate over batches
            for inputs, targets in data_loader:
                # Handle different input formats
                if isinstance(inputs, dict):
                    # Dictionary of inputs for each expert
                    # No need to move to device as CustomDataset should handle this
                    pass
                elif isinstance(inputs, torch.Tensor):
                    # Single tensor input
                    inputs = inputs.to(self.device)
                else:
                    # Unsupported input format
                    raise ValueError(f"Unsupported input format: {type(inputs)}")
                
                # Move targets to device
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    # Model returns (outputs, load_balancing_loss)
                    outputs = outputs[0]
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                
                # Add to lists
                y_true_list.append(targets.cpu().numpy())
                y_pred_list.append(probs.cpu().numpy())
        
        # Concatenate lists
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        
        # Save predictions if output directory is provided
        if output_dir is not None:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save predictions
            np.savez(
                os.path.join(output_dir, 'test_predictions.npz'),
                y_test=y_true,
                y_pred_test=y_pred
            )
            
            # Plot ROC curve
            self._plot_roc_curve(y_true, y_pred, os.path.join(output_dir, 'roc_curve.png'))
            
            # Plot confusion matrix
            self._plot_confusion_matrix(y_true, y_pred, threshold, os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Calculate metrics
        return SimpleMetrics.calculate(y_true, y_pred, threshold)
    
    def calculate_expert_contributions(self, data_loader: torch.utils.data.DataLoader, 
                                      output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Calculate expert contributions.
        
        Args:
            data_loader: DataLoader for data to evaluate
            output_dir: Directory to save expert contributions
            
        Returns:
            Dictionary mapping expert names to contribution arrays
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize lists for expert contributions
        expert_gates_list = []
        
        # Disable gradient computation
        with torch.no_grad():
            # Iterate over batches
            for inputs, _ in data_loader:
                # Handle different input formats
                if isinstance(inputs, dict):
                    # Dictionary of inputs for each expert
                    # No need to move to device as CustomDataset should handle this
                    pass
                elif isinstance(inputs, torch.Tensor):
                    # Single tensor input
                    inputs = inputs.to(self.device)
                else:
                    # Unsupported input format
                    raise ValueError(f"Unsupported input format: {type(inputs)}")
                
                # Forward pass to get gates
                try:
                    # Try to access gating network directly
                    if hasattr(self.model, 'gating') and self.model.gating is not None:
                        # Get list of expert inputs
                        expert_inputs = [inputs[name] for name in self.model.expert_registry.list()]
                        
                        # Forward pass through gating network
                        gates, _, _ = self.model.gating(expert_inputs)
                        
                        # Add to list
                        expert_gates_list.append(gates.cpu().numpy())
                    else:
                        # Model doesn't have gating network
                        return {}
                except Exception as e:
                    # Error accessing gating network
                    print(f"Error calculating expert contributions: {e}")
                    return {}
        
        # Concatenate lists
        expert_gates = np.concatenate(expert_gates_list, axis=0)
        
        # Calculate average contribution for each expert
        expert_contributions = np.mean(expert_gates, axis=0)
        
        # Create dictionary mapping expert names to contributions
        contributions_dict = {}
        try:
            expert_names = self.model.expert_registry.list()
            for i, name in enumerate(expert_names):
                contributions_dict[name] = expert_contributions[i]
        except Exception as e:
            # Error accessing expert names
            print(f"Error mapping expert names to contributions: {e}")
            for i in range(expert_gates.shape[1]):
                contributions_dict[f"Expert {i+1}"] = expert_contributions[i]
        
        # Save expert contributions if output directory is provided
        if output_dir is not None:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save expert contributions
            np.savez(
                os.path.join(output_dir, 'expert_contributions.npz'),
                expert_gates=expert_gates,
                expert_contributions=expert_contributions
            )
            
            # Plot expert contributions
            self._plot_expert_contributions(contributions_dict, os.path.join(output_dir, 'expert_contributions.png'))
        
        return contributions_dict
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            output_path: Path to save plot
        """
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Flatten arrays
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            # Set labels and title
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            
            # Save figure
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            # Error plotting ROC curve
            print(f"Error plotting ROC curve: {e}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              threshold: float, output_path: str) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            threshold: Threshold for converting probabilities to binary predictions
            output_path: Path to save plot
        """
        try:
            # Flatten arrays
            y_true = y_true.flatten()
            y_pred_proba = y_pred.flatten()
            
            # Convert probabilities to binary predictions
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred_binary)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Plot confusion matrix
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            # Set labels
            classes = ['No Migraine', 'Migraine']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            # Set labels and title
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            # Save figure
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            # Error plotting confusion matrix
            print(f"Error plotting confusion matrix: {e}")
    
    def _plot_expert_contributions(self, contributions_dict: Dict[str, float], output_path: str) -> None:
        """
        Plot expert contributions.
        
        Args:
            contributions_dict: Dictionary mapping expert names to contributions
            output_path: Path to save plot
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Sort contributions by value
            sorted_items = sorted(contributions_dict.items(), key=lambda x: x[1], reverse=True)
            expert_names = [item[0] for item in sorted_items]
            contributions = [item[1] for item in sorted_items]
            
            # Plot bar chart
            plt.bar(expert_names, contributions)
            
            # Set labels and title
            plt.xlabel('Expert')
            plt.ylabel('Average Contribution')
            plt.title('Expert Contributions')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add values on top of bars
            for i, v in enumerate(contributions):
                plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            # Error plotting expert contributions
            print(f"Error plotting expert contributions: {e}")


def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], 
                     y_pred: Union[np.ndarray, torch.Tensor], 
                     task_type: str = 'classification',
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate performance metrics for a given task type.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted values or probabilities
        task_type: Type of task ('classification' or 'regression')
        threshold: Threshold for converting probabilities to binary predictions (for classification)
        
    Returns:
        Dictionary containing performance metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Call appropriate metrics calculation function based on task type
    if task_type.lower() == 'classification':
        return calculate_classification_metrics(y_true, y_pred, threshold)
    elif task_type.lower() == 'regression':
        return calculate_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate performance metrics for classification tasks.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary containing performance metrics
    """
    return SimpleMetrics.calculate(y_true, y_pred, threshold)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics for regression tasks.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing performance metrics
    """
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    metrics = {}
    metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    
    # Calculate R-squared
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    metrics['r2'] = 1 - (ss_residual / ss_total)
    
    return metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                  threshold: float = 0.5) -> str:
    """
    Generate a classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Classification report as a string
    """
    from sklearn.metrics import classification_report
    
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred_proba = y_pred.flatten()
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Generate classification report
    return classification_report(y_true, y_pred_binary, target_names=['No Migraine', 'Migraine'])
