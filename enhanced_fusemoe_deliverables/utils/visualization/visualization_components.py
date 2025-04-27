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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.evaluation.metrics import calculate_metrics


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
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
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
        # Extract expert weights and predictions
        expert_weights = model_outputs['expert_weights']  # Shape: [n_samples, n_experts]
        expert_names = model_outputs['expert_names']
        
        # Calculate average expert weights
        avg_weights = np.mean(expert_weights, axis=0)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot average expert weights
        ax1.bar(expert_names, avg_weights)
        ax1.set_xlabel('Expert')
        ax1.set_ylabel('Average Weight')
        ax1.set_title('Average Expert Contributions')
        ax1.grid(True, alpha=0.3)
        
        # Plot expert weight distribution
        ax2.boxplot([expert_weights[:, i] for i in range(expert_weights.shape[1])],
                   labels=expert_names)
        ax2.set_xlabel('Expert')
        ax2.set_ylabel('Weight Distribution')
        ax2.set_title('Expert Contribution Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Set title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, np.ndarray],
                               title: str = 'Feature Importance',
                               save_name: Optional[str] = 'feature_importance.png') -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary mapping expert names to feature importance arrays
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Determine number of experts
        n_experts = len(feature_importance)
        
        # Create figure
        fig, axes = plt.subplots(n_experts, 1, figsize=(self.figsize[0], self.figsize[1] * n_experts / 2))
        
        # Handle single expert case
        if n_experts == 1:
            axes = [axes]
        
        # Plot feature importance for each expert
        for i, (expert_name, importance) in enumerate(feature_importance.items()):
            ax = axes[i]
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)
            feature_names = [f'Feature {j}' for j in range(len(importance))]
            sorted_names = [feature_names[j] for j in sorted_idx]
            sorted_importance = importance[sorted_idx]
            
            # Plot feature importance
            ax.barh(sorted_names, sorted_importance)
            ax.set_xlabel('Importance')
            ax.set_title(f'{expert_name} Expert Feature Importance')
            ax.grid(True, alpha=0.3)
        
        # Set title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_embedding_visualization(self, embeddings: np.ndarray, labels: np.ndarray,
                                    method: str = 'tsne',
                                    title: str = 'Embedding Visualization',
                                    save_name: Optional[str] = 'embedding_visualization.png') -> plt.Figure:
        """
        Plot embedding visualization.
        
        Args:
            embeddings: Embedding vectors
            labels: Labels for coloring points
            method: Dimensionality reduction method ('tsne' or 'pca')
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensionality
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot embeddings
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                           c=labels, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Label')
        
        # Set labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'{title} ({method.upper()})')
        ax.grid(True, alpha=0.3)
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_optimization_progress(self, optimization_history: List[Dict[str, float]],
                                  title: str = 'Optimization Progress',
                                  save_name: Optional[str] = 'optimization_progress.png') -> plt.Figure:
        """
        Plot optimization progress.
        
        Args:
            optimization_history: List of dictionaries containing optimization metrics
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics
        iterations = range(1, len(optimization_history) + 1)
        best_fitness = [entry['best_fitness'] for entry in optimization_history]
        avg_fitness = [entry['avg_fitness'] for entry in optimization_history]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot optimization progress
        ax.plot(iterations, best_fitness, 'b-', label='Best Fitness')
        ax.plot(iterations, avg_fitness, 'r-', label='Average Fitness')
        
        # Add target line at 0.95
        ax.axhline(y=0.95, color='g', linestyle='--', label='Target (0.95)')
        
        # Set labels and legend
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness (AUC)')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_parameter_evolution(self, parameter_history: Dict[str, List[float]],
                                title: str = 'Parameter Evolution',
                                save_name: Optional[str] = 'parameter_evolution.png') -> plt.Figure:
        """
        Plot parameter evolution during optimization.
        
        Args:
            parameter_history: Dictionary mapping parameter names to lists of values
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Determine number of parameters
        n_params = len(parameter_history)
        
        # Create figure
        fig, axes = plt.subplots(n_params, 1, figsize=(self.figsize[0], self.figsize[1] * n_params / 2),
                                sharex=True)
        
        # Handle single parameter case
        if n_params == 1:
            axes = [axes]
        
        # Plot parameter evolution for each parameter
        for i, (param_name, values) in enumerate(parameter_history.items()):
            ax = axes[i]
            iterations = range(1, len(values) + 1)
            
            # Plot parameter evolution
            ax.plot(iterations, values)
            ax.set_ylabel(param_name)
            ax.grid(True, alpha=0.3)
        
        # Set x-label for bottom plot
        axes[-1].set_xlabel('Iteration')
        
        # Set title
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, Any]],
                             metrics: List[str] = None,
                             title: str = 'Model Comparison',
                             save_name: Optional[str] = 'model_comparison.png') -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            model_results: Dictionary mapping model names to results dictionaries
            metrics: List of metrics to compare (default: ['accuracy', 'precision', 'recall', 'f1', 'auc'])
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Set default metrics if not provided
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Extract metrics for each model
        model_metrics = {}
        for model_name, results in model_results.items():
            model_metrics[model_name] = {metric: results['metrics'][metric] for metric in metrics}
        
        # Create DataFrame for plotting
        df = pd.DataFrame(model_metrics).T
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot model comparison
        df.plot(kind='bar', ax=ax)
        
        # Add target line at 0.95
        ax.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
        
        # Set labels and legend
        ax.set_xlabel('Model')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_prediction_distribution(self, predictions: np.ndarray, targets: np.ndarray,
                                    title: str = 'Prediction Distribution',
                                    save_name: Optional[str] = 'prediction_distribution.png') -> plt.Figure:
        """
        Plot prediction distribution.
        
        Args:
            predictions: Predicted probabilities
            targets: True binary labels
            title: Plot title
            save_name: Filename to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure
        """
        # Ensure predictions is the right shape
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot prediction distribution
        sns.histplot(predictions[targets == 0], bins=20, alpha=0.5, label='No Migraine', ax=ax)
        sns.histplot(predictions[targets == 1], bins=20, alpha=0.5, label='Migraine', ax=ax)
        
        # Add threshold line at 0.5
        ax.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        
        # Set labels and legend
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot if save_name is provided
        if save_name is not None and self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir, save_name), 
                      bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def create_dashboard_components(self, model_results: Dict[str, Dict[str, Any]],
                                   optimization_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """
        Create dashboard components.
        
        Args:
            model_results: Dictionary mapping model names to results dictionaries
            optimization_results: Dictionary containing optimization results
            
        Returns:
            Dictionary mapping component names to Matplotlib figures
        """
        # Initialize components dictionary
        components = {}
        
        # Training history
        components['training_history'] = self.plot_training_history(
            history=optimization_results['history'],
            metrics=['loss', 'auc', 'f1'],
            title='Training History',
            save_name='dashboard_training_history.png'
        )
        
        # Model comparison
        components['model_comparison'] = self.plot_model_comparison(
            model_results=model_results,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
            title='Model Comparison',
            save_name='dashboard_model_comparison.png'
        )
        
        # Prediction distribution
        components['prediction_distribution'] = self.plot_prediction_distribution(
            predictions=optimization_results['evaluation']['predictions'],
            targets=optimization_results['evaluation']['targets'],
            title='Prediction Distribution',
            save_name='dashboard_prediction_distribution.png'
        )
        
        # Expert contributions (if available)
        if 'expert_weights' in optimization_results['evaluation']:
            components['expert_contributions'] = self.plot_expert_contributions(
                model_outputs=optimization_results['evaluation'],
                title='Expert Contributions',
                save_name='dashboard_expert_contributions.png'
            )
        
        # Feature importance (if available)
        if 'feature_importance' in optimization_results['evaluation']:
            components['feature_importance'] = self.plot_feature_importance(
                feature_importance=optimization_results['evaluation']['feature_importance'],
                title='Feature Importance',
                save_name='dashboard_feature_importance.png'
            )
        
        # Optimization progress (if available)
        if 'optimization_history' in optimization_results:
            components['optimization_progress'] = self.plot_optimization_progress(
                optimization_history=optimization_results['optimization_history'],
                title='Optimization Progress',
                save_name='dashboard_optimization_progress.png'
            )
        
        return components
