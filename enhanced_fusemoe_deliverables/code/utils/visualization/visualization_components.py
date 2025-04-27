"""
Visualization Components for Enhanced FuseMoE

This module provides visualization components for the Enhanced FuseMoE system
for migraine prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import torch
import sklearn.metrics
import sklearn.manifold
from sklearn.decomposition import PCA

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.evaluation.metrics import calculate_metrics


# Standalone visualization functions for compatibility with tests

def plot_training_history(history: Dict[str, List[Dict[str, float]]], 
                         metrics: List[str] = None, 
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot (default: ['loss', 'auc'])
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Set default metrics if not provided
    if metrics is None:
        metrics = ['loss', 'auc']
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    
    # Handle single metric case
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract metric values
        train_values = [epoch[metric] for epoch in history['train_history']]
        val_values = [epoch[metric] for epoch in history['val_history']]
        epochs = range(1, len(train_values) + 1)
        
        # Plot metric
        ax.plot(epochs, train_values, 'b-', label=f'Training {metric}')
        ax.plot(epochs, val_values, 'r-', label=f'Validation {metric}')
        
        # Add target line at 0.95 for performance metrics
        if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            ax.axhline(y=0.95, color='g', linestyle='--', label='Target (0.95)')
        
        # Set labels and legend
        ax.set_ylabel(metric.upper())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set y-limits for specific metrics
        if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            ax.set_ylim([0.0, 1.05])
    
    # Set x-label for bottom plot
    axes[-1].set_xlabel('Epoch')
    
    # Set title
    fig.suptitle('Training History')
    fig.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_confusion_matrix(confusion_matrix: Tuple[int, int, int, int],
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Tuple of (tn, fp, fn, tp)
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract confusion matrix values
    tn, fp, fn, tp = confusion_matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    
    # Add labels
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # Add labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  figsize: Tuple[int, int] = (8, 6),
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Calculate ROC curve
    # Important: Use sklearn.metrics directly to allow mocking in tests
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_proba)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    
    # Add labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                               figsize: Tuple[int, int] = (8, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Calculate precision-recall curve
    # Important: Use sklearn.metrics directly to allow mocking in tests
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred_proba)
    ap = sklearn.metrics.average_precision_score(y_true, y_pred_proba)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, label=f'Precision-Recall curve (AP = {ap:.3f})')
    
    # Add labels and title
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Set y-limits
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_expert_contributions(expert_contributions: Dict[str, float],
                             figsize: Tuple[int, int] = (8, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot expert contributions.
    
    Args:
        expert_contributions: Dictionary mapping expert names to contribution values
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Extract expert names and contributions
    expert_names = list(expert_contributions.keys())
    contributions = list(expert_contributions.values())
    
    # Plot pie chart
    wedges, texts, autotexts = ax.pie(
        contributions, 
        labels=None, 
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Add legend
    ax.legend(wedges, expert_names, title="Experts", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Add title
    ax.set_title('Expert Contributions')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_optimization_progress(generations: List[int], best_fitness: List[float], mean_fitness: List[float],
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot optimization progress.
    
    Args:
        generations: List of generation numbers
        best_fitness: List of best fitness values
        mean_fitness: List of mean fitness values
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot optimization progress
    ax.plot(generations, best_fitness, 'b-', label='Best Fitness')
    ax.plot(generations, mean_fitness, 'r-', label='Mean Fitness')
    
    # Add target line at 0.95
    ax.axhline(y=0.95, color='g', linestyle='--', label='Target (0.95)')
    
    # Add labels and title
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (AUC)')
    ax.set_title('Optimization Progress')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_parameter_importance(parameter_importance: Dict[str, float],
                             figsize: Tuple[int, int] = (10, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot parameter importance.
    
    Args:
        parameter_importance: Dictionary mapping parameter names to importance values
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Sort parameters by importance
    sorted_params = sorted(parameter_importance.items(), key=lambda x: x[1], reverse=True)
    param_names = [p[0] for p in sorted_params]
    importance = [p[1] for p in sorted_params]
    
    # Plot parameter importance
    ax.bar(param_names, importance)
    
    # Add labels and title
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Importance')
    ax.set_title('Parameter Importance')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_performance_comparison(metrics: List[str], original_performance: List[float], 
                               optimized_performance: List[float],
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot performance comparison between original and optimized models.
    
    Args:
        metrics: List of metric names
        original_performance: List of performance values for original model
        optimized_performance: List of performance values for optimized model
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(metrics))
    
    # Plot bars
    ax.bar(index - bar_width/2, original_performance, bar_width, label='Original')
    ax.bar(index + bar_width/2, optimized_performance, bar_width, label='Optimized')
    
    # Add target line at 0.95
    ax.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
    
    # Add labels and title
    ax.set_xlabel('Metric')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Comparison: Original vs. Optimized')
    ax.set_xticks(index)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-limits
    ax.set_ylim([0.0, 1.05])
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_parameter_evolution(parameter_values: Dict[str, List[float]], generations: List[int],
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot parameter evolution over generations.
    
    Args:
        parameter_values: Dictionary mapping parameter names to lists of values over generations
        generations: List of generation numbers
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Determine number of parameters and create subplots
    n_params = len(parameter_values)
    n_cols = min(2, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Plot each parameter
    for i, (param_name, values) in enumerate(parameter_values.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.plot(generations, values, 'b-', marker='o')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Value')
        ax.set_title(f'Parameter: {param_name}')
        ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_prediction_distribution(y_pred_proba: np.ndarray, y_true: np.ndarray = None,
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of prediction probabilities.
    
    Args:
        y_pred_proba: Predicted probabilities
        y_true: True labels (optional, for coloring)
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot histogram
    if y_true is not None:
        # Plot separate histograms for positive and negative classes
        pos_probs = y_pred_proba[y_true == 1]
        neg_probs = y_pred_proba[y_true == 0]
        
        ax.hist(pos_probs, bins=20, alpha=0.5, label='Positive class', color='blue')
        ax.hist(neg_probs, bins=20, alpha=0.5, label='Negative class', color='red')
        ax.legend()
    else:
        # Plot single histogram
        ax.hist(y_pred_proba, bins=20, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Prediction Probabilities')
    ax.grid(True, alpha=0.3)
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


class MigraineVisualization:
    """
    Visualization components for the Enhanced FuseMoE system.
    
    This class provides methods for visualizing various aspects of the
    Enhanced FuseMoE system for migraine prediction.
    
    Attributes:
        output_dir (str): Directory to save visualizations
        figsize (Tuple[int, int]): Default figure size
        dpi (int): Default DPI for saved figures
    """
    
    def __init__(self, output_dir: Optional[str] = None, 
                figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        """
        Initialize the migraine visualization.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Default DPI for saved figures
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_history(self, history: Dict[str, List[Dict[str, float]]],
                             metrics: List[str] = None, 
                             title: str = 'Training History',
                             save_name: Optional[str] = 'training_history.png') -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history: Dictionary containing training history
            metrics: List of metrics to plot (default: ['loss', 'auc'])
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Set default metrics if not provided
        if metrics is None:
            metrics = ['loss', 'auc']
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.figsize, sharex=True)
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract metric values
            train_values = [epoch[metric] for epoch in history['train_history']]
            val_values = [epoch[metric] for epoch in history['val_history']]
            epochs = range(1, len(train_values) + 1)
            
            # Plot metric
            ax.plot(epochs, train_values, 'b-', label=f'Training {metric}')
            ax.plot(epochs, val_values, 'r-', label=f'Validation {metric}')
            
            # Add target line at 0.95 for performance metrics
            if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                ax.axhline(y=0.95, color='g', linestyle='--', label='Target (0.95)')
            
            # Set labels and legend
            ax.set_ylabel(metric.upper())
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Set y-limits for specific metrics
            if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                ax.set_ylim([0.0, 1.05])
        
        # Set x-label for bottom plot
        axes[-1].set_xlabel('Epoch')
        
        # Set title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_expert_contributions(self, model_outputs: Dict[str, np.ndarray],
                                 title: str = 'Expert Contributions',
                                 save_name: Optional[str] = 'expert_contributions.png') -> plt.Figure:
        """
        Plot expert contributions.
        
        Args:
            model_outputs: Dictionary containing model outputs
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Extract expert weights and names
        expert_weights = model_outputs['expert_weights']  # Shape: [n_samples, n_experts]
        expert_names = model_outputs.get('expert_names', [f'Expert {i+1}' for i in range(expert_weights.shape[1])])
        
        # Calculate average contribution of each expert
        avg_contributions = expert_weights.mean(axis=0)
        
        # Plot average contributions as bar chart
        axes[0].bar(expert_names, avg_contributions)
        axes[0].set_title('Average Expert Contributions')
        axes[0].set_ylabel('Contribution')
        axes[0].set_xticklabels(expert_names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot distribution of expert weights as boxplot
        axes[1].boxplot([expert_weights[:, i] for i in range(expert_weights.shape[1])], labels=expert_names)
        axes[1].set_title('Expert Weight Distribution')
        axes[1].set_ylabel('Weight')
        axes[1].set_xticklabels(expert_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, Dict[str, float]],
                               title: str = 'Feature Importance',
                               save_name: Optional[str] = 'feature_importance.png') -> plt.Figure:
        """
        Plot feature importance for each expert.
        
        Args:
            feature_importance: Dictionary mapping expert names to dictionaries of feature importance values
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Get number of experts
        n_experts = len(feature_importance)
        
        # Create figure with subplots for each expert
        fig, axes = plt.subplots(n_experts, 1, figsize=(self.figsize[0], self.figsize[1] * n_experts / 2))
        
        # Handle single expert case
        if n_experts == 1:
            axes = [axes]
        
        # Plot feature importance for each expert
        for i, (expert_name, importances) in enumerate(feature_importance.items()):
            ax = axes[i]
            
            # Convert importances to list of tuples for sorting
            if isinstance(importances, dict):
                # If importances is a dictionary, extract keys and values
                sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                feature_names = [item[0] for item in sorted_items]
                importance_values = [item[1] for item in sorted_items]
            else:
                # If importances is an array or list, use indices as feature names
                importance_values = np.array(importances).flatten()  # Ensure 1D array
                feature_names = [f'Feature {j+1}' for j in range(len(importance_values))]
                # Sort by importance
                sorted_indices = np.argsort(importance_values)[::-1]
                feature_names = [feature_names[j] for j in sorted_indices]
                importance_values = [importance_values[j] for j in sorted_indices]
            
            # Plot horizontal bar chart
            ax.barh(feature_names, importance_values)
            ax.set_xlabel('Importance')
            ax.set_title(f'Expert: {expert_name}')
            ax.grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_embedding_visualization(self, embeddings: np.ndarray, labels: np.ndarray = None,
                                    method: str = 'tsne',
                                    title: str = 'Embedding Visualization',
                                    save_name: Optional[str] = 'embedding_visualization.png') -> plt.Figure:
        """
        Plot embedding visualization using dimensionality reduction.
        
        Args:
            embeddings: Embedding vectors (shape: [n_samples, n_features])
            labels: Labels for coloring (shape: [n_samples])
            method: Dimensionality reduction method ('tsne' or 'pca')
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            # Apply t-SNE - use sklearn.manifold directly to allow mocking in tests
            tsne = sklearn.manifold.TSNE(n_components=2, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)
        elif method == 'pca':
            # Apply PCA
            pca = PCA(n_components=2, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'.")
        
        # Plot scatter plot
        if labels is not None:
            # Color by labels
            scatter = ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.7
            )
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            # No coloring
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=0.7
            )
        
        # Add labels and title
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'{title} ({method.upper()})')
        ax.grid(True, alpha=0.3)
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_optimization_progress(self, optimization_history: Union[List[Dict[str, float]], Dict[str, List[float]]],
                                  title: str = 'Optimization Progress',
                                  save_name: Optional[str] = 'optimization_progress.png') -> plt.Figure:
        """
        Plot optimization progress.
        
        Args:
            optimization_history: List of dictionaries or dictionary of lists containing optimization history
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract optimization history
        if isinstance(optimization_history, list):
            # Handle list of dictionaries
            generations = list(range(len(optimization_history)))
            best_fitness = [gen.get('best_fitness', 0) for gen in optimization_history]
            mean_fitness = [gen.get('mean_fitness', 0) for gen in optimization_history]
        else:
            # Handle dictionary of lists
            generations = optimization_history.get('generations', list(range(len(optimization_history.get('best_fitness', [])))))
            best_fitness = optimization_history.get('best_fitness', [])
            mean_fitness = optimization_history.get('mean_fitness', [])
        
        # Plot optimization progress
        ax.plot(generations, best_fitness, 'b-', label='Best Fitness')
        ax.plot(generations, mean_fitness, 'r-', label='Mean Fitness')
        
        # Add target line at 0.95
        ax.axhline(y=0.95, color='g', linestyle='--', label='Target (0.95)')
        
        # Add labels and title
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (AUC)')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_parameter_evolution(self, parameter_history: Dict[str, List[float]],
                                title: str = 'Parameter Evolution',
                                save_name: Optional[str] = 'parameter_evolution.png') -> plt.Figure:
        """
        Plot parameter evolution over generations.
        
        Args:
            parameter_history: Dictionary mapping parameter names to lists of values over generations
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Get number of parameters
        n_params = len(parameter_history)
        
        # Create figure with subplots for each parameter
        fig, axes = plt.subplots(n_params, 1, figsize=(self.figsize[0], self.figsize[1] * n_params / 2), sharex=True)
        
        # Handle single parameter case
        if n_params == 1:
            axes = [axes]
        
        # Plot each parameter
        for i, (param_name, values) in enumerate(parameter_history.items()):
            ax = axes[i]
            generations = list(range(len(values)))
            
            # Plot parameter evolution
            ax.plot(generations, values, 'b-', marker='o')
            
            # Add labels
            ax.set_ylabel(param_name)
            ax.grid(True, alpha=0.3)
        
        # Set x-label for bottom plot
        axes[-1].set_xlabel('Generation')
        
        # Set overall title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]],
                             metrics: List[str] = None,
                             title: str = 'Model Comparison',
                             save_name: Optional[str] = 'model_comparison.png') -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            model_results: Dictionary mapping model names to dictionaries of metric values
            metrics: List of metrics to plot (default: all metrics in first model)
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set default metrics if not provided
        if metrics is None:
            # Use all metrics from first model
            first_model = next(iter(model_results.values()))
            metrics = list(first_model.keys())
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            model_name: {metric: model_metrics.get(metric, 0) for metric in metrics}
            for model_name, model_metrics in model_results.items()
        })
        
        # Plot bar chart
        df.plot(kind='bar', ax=ax)
        
        # Add target line at 0.95
        ax.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
        
        # Add labels and title
        ax.set_xlabel('Metric')
        ax.set_ylabel('Performance')
        ax.set_title(title)
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        
        # Set y-limits
        ax.set_ylim([0.0, 1.05])
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_prediction_distribution(self, predictions: np.ndarray, targets: np.ndarray = None,
                                    title: str = 'Prediction Distribution',
                                    save_name: Optional[str] = 'prediction_distribution.png') -> plt.Figure:
        """
        Plot distribution of prediction probabilities.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels (optional, for coloring)
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot histogram in first subplot
        if targets is not None:
            # Plot separate histograms for positive and negative classes
            pos_probs = predictions[targets == 1]
            neg_probs = predictions[targets == 0]
            
            axes[0].hist(pos_probs, bins=20, alpha=0.5, label='Positive class', color='blue')
            axes[0].hist(neg_probs, bins=20, alpha=0.5, label='Negative class', color='red')
            axes[0].legend()
        else:
            # Plot single histogram
            axes[0].hist(predictions, bins=20, alpha=0.7)
        
        # Add labels for first subplot
        axes[0].set_xlabel('Prediction Probability')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Histogram')
        axes[0].grid(True, alpha=0.3)
        
        # Plot KDE in second subplot
        if targets is not None:
            # Plot separate KDE for positive and negative classes
            sns.kdeplot(pos_probs, ax=axes[1], label='Positive class', color='blue')
            sns.kdeplot(neg_probs, ax=axes[1], label='Negative class', color='red')
            axes[1].legend()
        else:
            # Plot single KDE
            sns.kdeplot(predictions, ax=axes[1])
        
        # Add labels for second subplot
        axes[1].set_xlabel('Prediction Probability')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Kernel Density Estimate')
        axes[1].grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray,
                               save_prefix: str = 'performance_',
                               include_plots: List[str] = None) -> Dict[str, plt.Figure]:
        """
        Plot various performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            save_prefix: Prefix for saved filenames
            include_plots: List of plots to include (default: all)
            
        Returns:
            Dictionary mapping plot names to figures
        """
        # Set default plots if not provided
        if include_plots is None:
            include_plots = ['confusion_matrix', 'roc_curve', 'precision_recall_curve', 'prediction_distribution']
        
        # Initialize dictionary to store figures
        figures = {}
        
        # Calculate confusion matrix
        if 'confusion_matrix' in include_plots:
            # Calculate confusion matrix values
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
            
            # Plot confusion matrix
            fig_cm = plot_confusion_matrix(
                confusion_matrix=(tn, fp, fn, tp),
                figsize=self.figsize,
                save_path=os.path.join(self.output_dir, f'{save_prefix}confusion_matrix.png') if self.output_dir is not None else None
            )
            
            figures['confusion_matrix'] = fig_cm
        
        # Plot ROC curve
        if 'roc_curve' in include_plots:
            fig_roc = plot_roc_curve(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                figsize=self.figsize,
                save_path=os.path.join(self.output_dir, f'{save_prefix}roc_curve.png') if self.output_dir is not None else None
            )
            
            figures['roc_curve'] = fig_roc
        
        # Plot precision-recall curve
        if 'precision_recall_curve' in include_plots:
            fig_pr = plot_precision_recall_curve(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                figsize=self.figsize,
                save_path=os.path.join(self.output_dir, f'{save_prefix}precision_recall_curve.png') if self.output_dir is not None else None
            )
            
            figures['precision_recall_curve'] = fig_pr
        
        # Plot prediction distribution
        if 'prediction_distribution' in include_plots:
            fig_dist = self.plot_prediction_distribution(
                predictions=y_pred_proba,
                targets=y_true,
                title='Prediction Distribution',
                save_name=f'{save_prefix}prediction_distribution.png'
            )
            
            figures['prediction_distribution'] = fig_dist
        
        return figures
