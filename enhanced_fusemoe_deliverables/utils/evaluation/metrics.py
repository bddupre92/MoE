"""
Evaluation Metrics for Enhanced FuseMoE

This module provides functionality for evaluating the performance of the Enhanced FuseMoE
system for migraine prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_curve
)
from typing import Dict, List, Tuple, Any, Optional, Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Ensure y_pred is the right shape
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    
    # Precision, Recall, F1 Score
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # ROC AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle the case where there's only one class in y_true
        metrics['auc'] = 0.5
    
    # Average Precision (AP)
    try:
        metrics['ap'] = average_precision_score(y_true, y_pred)
    except ValueError:
        # Handle the case where there's only one class in y_true
        metrics['ap'] = 0.0
    
    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Negative Predictive Value (NPV)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Balanced Accuracy
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray, 
                          metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal threshold for converting probabilities to binary predictions.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        metric: Metric to optimize ('f1', 'balanced_accuracy', or 'youden')
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal_threshold)
    """
    # Ensure y_pred is the right shape
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Get unique thresholds from the data
    thresholds = np.unique(y_pred)
    
    # Add 0 and 1 to thresholds if not already present
    if 0.0 not in thresholds:
        thresholds = np.append(thresholds, 0.0)
    if 1.0 not in thresholds:
        thresholds = np.append(thresholds, 1.0)
    
    # Sort thresholds
    thresholds = np.sort(thresholds)
    
    # Calculate metrics for each threshold
    best_metric_value = -np.inf
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        # Calculate metrics at this threshold
        metrics = calculate_metrics(y_true, y_pred, threshold)
        
        # Determine metric value to optimize
        if metric == 'f1':
            metric_value = metrics['f1']
        elif metric == 'balanced_accuracy':
            metric_value = metrics['balanced_accuracy']
        elif metric == 'youden':
            # Youden's J statistic (sensitivity + specificity - 1)
            metric_value = metrics['recall'] + metrics['specificity'] - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Update best threshold if this is better
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5,
                         figsize: Tuple[int, int] = (8, 6), title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        threshold: Threshold for converting probabilities to binary predictions
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Ensure y_pred is the right shape
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['No Migraine', 'Migraine'],
               yticklabels=['No Migraine', 'Migraine'])
    
    # Set labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray,
                  figsize: Tuple[int, int] = (8, 6), title: str = 'ROC Curve',
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Ensure y_pred is the right shape
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Set labels and limits
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_pred: np.ndarray,
                               figsize: Tuple[int, int] = (8, 6), title: str = 'Precision-Recall Curve',
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Ensure y_pred is the right shape
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate average precision
    ap = average_precision_score(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    
    # Plot baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline = {baseline:.3f}')
    
    # Set labels and limits
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          figsize: Tuple[int, int] = (10, 6), title: str = 'Metrics vs. Threshold',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot metrics as a function of threshold.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Ensure y_pred is the right shape
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Generate thresholds
    thresholds = np.linspace(0, 1, 101)
    
    # Calculate metrics for each threshold
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': []
    }
    
    for threshold in thresholds:
        m = calculate_metrics(y_true, y_pred, threshold)
        for key in metrics:
            metrics[key].append(m[key])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot metrics
    for key, values in metrics.items():
        plt.plot(thresholds, values, label=key.capitalize())
    
    # Set labels and limits
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Find and plot optimal threshold for F1 score
    optimal_threshold, _ = find_optimal_threshold(y_true, y_pred, 'f1')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    # Update legend
    plt.legend(loc='best')
    
    # Save plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5,
                                 output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive classification report.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        threshold: Threshold for converting probabilities to binary predictions
        output_dir: Directory to save plots (if None, plots are not saved)
        
    Returns:
        Dictionary containing report data
    """
    # Ensure y_pred is the right shape
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, threshold)
    
    # Find optimal threshold
    optimal_threshold, optimal_metrics = find_optimal_threshold(y_true, y_pred, 'f1')
    
    # Generate plots
    plots = {}
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png') if output_dir else None
    plots['confusion_matrix'] = plot_confusion_matrix(y_true, y_pred, threshold, 
                                                    save_path=cm_path)
    
    # ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png') if output_dir else None
    plots['roc_curve'] = plot_roc_curve(y_true, y_pred, save_path=roc_path)
    
    # Precision-recall curve
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png') if output_dir else None
    plots['precision_recall_curve'] = plot_precision_recall_curve(y_true, y_pred, 
                                                                save_path=pr_path)
    
    # Threshold metrics
    threshold_path = os.path.join(output_dir, 'threshold_metrics.png') if output_dir else None
    plots['threshold_metrics'] = plot_threshold_metrics(y_true, y_pred, 
                                                      save_path=threshold_path)
    
    # Compile report
    report = {
        'metrics': metrics,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics,
        'plots': plots
    }
    
    return report


def compare_models(model_results: Dict[str, Dict[str, Any]], 
                  figsize: Tuple[int, int] = (12, 8),
                  save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Compare multiple models.
    
    Args:
        model_results: Dictionary mapping model names to results dictionaries
                      Each results dictionary should contain 'y_true' and 'y_pred' arrays
        figsize: Figure size
        save_path: Directory to save plots (if None, plots are not saved)
        
    Returns:
        Dictionary mapping plot names to Matplotlib figures
    """
    # Create output directory if it doesn't exist
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Initialize plots dictionary
    plots = {}
    
    # ROC curves
    fig_roc, ax_roc = plt.subplots(figsize=figsize)
    
    for model_name, results in model_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        # Plot ROC curve
        ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    
    # Plot diagonal line (random classifier)
    ax_roc.plot([0, 1], [0, 1], 'k--')
    
    # Set labels and limits
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    
    # Save plot if save_path is provided
    if save_path is not None:
        fig_roc.savefig(os.path.join(save_path, 'roc_curves_comparison.png'), 
                      bbox_inches='tight', dpi=300)
    
    plots['roc_curves'] = fig_roc
    
    # Precision-recall curves
    fig_pr, ax_pr = plt.subplots(figsize=figsize)
    
    for model_name, results in model_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        
        # Plot precision-recall curve
        ax_pr.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
    
    # Plot baseline (random classifier)
    baseline = np.sum(list(model_results.values())[0]['y_true']) / len(list(model_results.values())[0]['y_true'])
    ax_pr.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline = {baseline:.3f}')
    
    # Set labels and limits
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.legend(loc='best')
    ax_pr.grid(True, alpha=0.3)
    
    # Save plot if save_path is provided
    if save_path is not None:
        fig_pr.savefig(os.path.join(save_path, 'pr_curves_comparison.png'), 
                     bbox_inches='tight', dpi=300)
    
    plots['pr_curves'] = fig_pr
    
    # Metrics comparison
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
    metrics_data = []
    
    for model_name, results in model_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Add to metrics data
        model_metrics = {'Model': model_name}
        for metric in metrics_to_compare:
            model_metrics[metric.upper()] = metrics[metric]
        
        metrics_data.append(model_metrics)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot metrics comparison
    fig_metrics, ax_metrics = plt.subplots(figsize=figsize)
    
    # Melt DataFrame for easier plotting
    metrics_df_melted = pd.melt(metrics_df, id_vars=['Model'], 
                              value_vars=[m.upper() for m in metrics_to_compare],
                              var_name='Metric', value_name='Value')
    
    # Plot metrics comparison
    sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_df_melted, ax=ax_metrics)
    
    # Set labels
    ax_metrics.set_xlabel('Metric')
    ax_metrics.set_ylabel('Value')
    ax_metrics.set_title('Metrics Comparison')
    ax_metrics.set_ylim([0.0, 1.05])
    ax_metrics.legend(loc='best')
    ax_metrics.grid(True, alpha=0.3)
    
    # Save plot if save_path is provided
    if save_path is not None:
        fig_metrics.savefig(os.path.join(save_path, 'metrics_comparison.png'), 
                          bbox_inches='tight', dpi=300)
    
    plots['metrics_comparison'] = fig_metrics
    
    return plots
