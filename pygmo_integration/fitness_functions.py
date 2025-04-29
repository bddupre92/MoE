"""
Fitness Functions for PyGMO integration with MoE

This module provides specialized fitness functions for evaluating
Mixture of Experts (MoE) models during optimization. It includes
functions for accuracy, precision, recall, F1 score, expert balance,
and model complexity.

Author: Manus AI
Date: April 28, 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FitnessFunctions:
    """Collection of fitness functions for MoE optimization."""
    
    @staticmethod
    def accuracy(y_true, y_pred, minimize=True):
        """
        Calculate accuracy fitness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            minimize: Whether to return a value to minimize (1-accuracy)
            
        Returns:
            Accuracy fitness value
        """
        acc = accuracy_score(y_true, y_pred)
        return 1.0 - acc if minimize else acc
    
    @staticmethod
    def precision(y_true, y_pred, minimize=True, average='weighted'):
        """
        Calculate precision fitness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            minimize: Whether to return a value to minimize (1-precision)
            average: Averaging method for multi-class problems
            
        Returns:
            Precision fitness value
        """
        # Handle edge cases to avoid division by zero
        try:
            prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating precision: {e}")
            prec = 0.0
            
        return 1.0 - prec if minimize else prec
    
    @staticmethod
    def recall(y_true, y_pred, minimize=True, average='weighted'):
        """
        Calculate recall fitness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            minimize: Whether to return a value to minimize (1-recall)
            average: Averaging method for multi-class problems
            
        Returns:
            Recall fitness value
        """
        # Handle edge cases to avoid division by zero
        try:
            rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating recall: {e}")
            rec = 0.0
            
        return 1.0 - rec if minimize else rec
    
    @staticmethod
    def f1(y_true, y_pred, minimize=True, average='weighted'):
        """
        Calculate F1 score fitness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            minimize: Whether to return a value to minimize (1-f1)
            average: Averaging method for multi-class problems
            
        Returns:
            F1 score fitness value
        """
        # Handle edge cases to avoid division by zero
        try:
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating F1 score: {e}")
            f1 = 0.0
            
        return 1.0 - f1 if minimize else f1
    
    @staticmethod
    def auc(y_true, y_score, minimize=True, multi_class='ovr'):
        """
        Calculate AUC fitness.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            minimize: Whether to return a value to minimize (1-auc)
            multi_class: Method for multi-class problems
            
        Returns:
            AUC fitness value
        """
        # Handle edge cases
        try:
            auc = roc_auc_score(y_true, y_score, multi_class=multi_class)
        except Exception as e:
            logger.warning(f"Error calculating AUC: {e}")
            auc = 0.5  # Random classifier baseline
            
        return 1.0 - auc if minimize else auc
    
    @staticmethod
    def false_positive_rate(y_true, y_pred, minimize=True):
        """
        Calculate false positive rate fitness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            minimize: Whether to return a value to minimize
            
        Returns:
            False positive rate fitness value
        """
        # Calculate false positive rate
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        # Avoid division by zero
        fpr = fp / (fp + tn + 1e-10)
        
        return fpr if minimize else 1.0 - fpr
    
    @staticmethod
    def false_negative_rate(y_true, y_pred, minimize=True):
        """
        Calculate false negative rate fitness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            minimize: Whether to return a value to minimize
            
        Returns:
            False negative rate fitness value
        """
        # Calculate false negative rate
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Avoid division by zero
        fnr = fn / (fn + tp + 1e-10)
        
        return fnr if minimize else 1.0 - fnr
    
    @staticmethod
    def expert_balance(expert_contributions, minimize=True):
        """
        Calculate expert balance fitness.
        
        Args:
            expert_contributions: Contributions of each expert
            minimize: Whether to return a value to minimize
            
        Returns:
            Expert balance fitness value
        """
        # Calculate variance in expert contributions
        # Lower variance means more balanced contributions
        balance = np.var(expert_contributions)
        
        return balance if minimize else 1.0 - balance
    
    @staticmethod
    def expert_specialization(expert_outputs, expert_domains, minimize=True):
        """
        Calculate expert specialization fitness.
        
        Args:
            expert_outputs: Outputs from each expert
            expert_domains: Domain indicators for each expert
            minimize: Whether to return a value to minimize
            
        Returns:
            Expert specialization fitness value
        """
        # Calculate how well each expert performs in its domain
        # Higher values mean better specialization
        specialization_scores = []
        
        for i, (outputs, domain) in enumerate(zip(expert_outputs, expert_domains)):
            # Calculate accuracy within domain
            domain_mask = domain > 0
            if np.sum(domain_mask) > 0:
                domain_acc = np.mean(outputs[domain_mask])
                specialization_scores.append(domain_acc)
        
        # Average specialization across experts
        if specialization_scores:
            specialization = np.mean(specialization_scores)
        else:
            specialization = 0.0
            
        return 1.0 - specialization if minimize else specialization
    
    @staticmethod
    def model_complexity(model_params, complexity_type='l1', minimize=True, penalty=0.01):
        """
        Calculate model complexity fitness.
        
        Args:
            model_params: Model parameters
            complexity_type: Type of complexity measure ('l1', 'l2', 'count')
            minimize: Whether to return a value to minimize
            penalty: Penalty factor for complexity
            
        Returns:
            Model complexity fitness value
        """
        if complexity_type == 'l1':
            # L1 norm (sum of absolute values)
            complexity = np.sum(np.abs(model_params))
        elif complexity_type == 'l2':
            # L2 norm (sum of squares)
            complexity = np.sum(np.square(model_params))
        elif complexity_type == 'count':
            # Count of non-zero parameters
            complexity = np.sum(np.abs(model_params) > 1e-6)
        else:
            raise ValueError(f"Unsupported complexity type: {complexity_type}")
        
        # Apply penalty factor
        complexity *= penalty
        
        return complexity if minimize else 1.0 - complexity
    
    @staticmethod
    def combined_fitness(metrics_dict, weights=None, minimize=True):
        """
        Calculate combined fitness from multiple metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            weights: Dictionary of weights for each metric
            minimize: Whether to return a value to minimize
            
        Returns:
            Combined fitness value
        """
        if weights is None:
            # Equal weights if not provided
            weights = {metric: 1.0 for metric in metrics_dict}
        
        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, value in metrics_dict.items():
            if metric in weights:
                weight = weights[metric]
                weighted_sum += value * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined = weighted_sum / total_weight
        else:
            combined = 0.0
            
        return combined if minimize else 1.0 - combined


class FitnessFunctionFactory:
    """Factory class for creating fitness functions."""
    
    @staticmethod
    def create_fitness_function(objective, minimize=True, **kwargs):
        """
        Create a fitness function for a specific objective.
        
        Args:
            objective: Objective name
            minimize: Whether to return a value to minimize
            **kwargs: Additional arguments for the fitness function
            
        Returns:
            Fitness function
        """
        if objective == 'accuracy':
            return lambda y_true, y_pred: FitnessFunctions.accuracy(y_true, y_pred, minimize, **kwargs)
        
        elif objective == 'precision':
            return lambda y_true, y_pred: FitnessFunctions.precision(y_true, y_pred, minimize, **kwargs)
        
        elif objective == 'recall':
            return lambda y_true, y_pred: FitnessFunctions.recall(y_true, y_pred, minimize, **kwargs)
        
        elif objective == 'f1':
            return lambda y_true, y_pred: FitnessFunctions.f1(y_true, y_pred, minimize, **kwargs)
        
        elif objective == 'auc':
            return lambda y_true, y_score: FitnessFunctions.auc(y_true, y_score, minimize, **kwargs)
        
        elif objective == 'false_positive_rate':
            return lambda y_true, y_pred: FitnessFunctions.false_positive_rate(y_true, y_pred, minimize)
        
        elif objective == 'false_negative_rate':
            return lambda y_true, y_pred: FitnessFunctions.false_negative_rate(y_true, y_pred, minimize)
        
        elif objective == 'expert_balance':
            return lambda expert_contributions: FitnessFunctions.expert_balance(expert_contributions, minimize)
        
        elif objective == 'expert_specialization':
            return lambda expert_outputs, expert_domains: FitnessFunctions.expert_specialization(
                expert_outputs, expert_domains, minimize)
        
        elif objective == 'model_complexity':
            return lambda model_params: FitnessFunctions.model_complexity(model_params, minimize=minimize, **kwargs)
        
        else:
            raise ValueError(f"Unsupported objective: {objective}")
    
    @staticmethod
    def create_multi_objective_fitness(objectives, minimize=True, weights=None, **kwargs):
        """
        Create a multi-objective fitness function.
        
        Args:
            objectives: List of objective names
            minimize: Whether to return values to minimize
            weights: Dictionary of weights for each objective
            **kwargs: Additional arguments for the fitness functions
            
        Returns:
            Multi-objective fitness function
        """
        fitness_functions = {}
        
        for objective in objectives:
            fitness_functions[objective] = FitnessFunctionFactory.create_fitness_function(
                objective, minimize, **kwargs)
        
        def multi_objective_fitness(*args, **kwargs):
            """Multi-objective fitness function."""
            results = {}
            
            for objective, func in fitness_functions.items():
                try:
                    results[objective] = func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error calculating {objective}: {e}")
                    results[objective] = 1.0 if minimize else 0.0
            
            return results
        
        return multi_objective_fitness


# Example usage
if __name__ == "__main__":
    # Generate mock data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_score = np.random.random(100)
    expert_contributions = np.random.random(5)
    expert_contributions = expert_contributions / np.sum(expert_contributions)
    model_params = np.random.randn(100)
    
    # Calculate individual fitness values
    accuracy = FitnessFunctions.accuracy(y_true, y_pred)
    precision = FitnessFunctions.precision(y_true, y_pred)
    recall = FitnessFunctions.recall(y_true, y_pred)
    f1 = FitnessFunctions.f1(y_true, y_pred)
    fpr = FitnessFunctions.false_positive_rate(y_true, y_pred)
    fnr = FitnessFunctions.false_negative_rate(y_true, y_pred)
    balance = FitnessFunctions.expert_balance(expert_contributions)
    complexity = FitnessFunctions.model_complexity(model_params, complexity_type='l1')
    
    print("Accuracy fitness:", accuracy)
    print("Precision fitness:", precision)
    print("Recall fitness:", recall)
    print("F1 fitness:", f1)
    print("False positive rate:", fpr)
    print("False negative rate:", fnr)
    print("Expert balance:", balance)
    print("Model complexity:", complexity)
    
    # Create fitness function using factory
    accuracy_func = FitnessFunctionFactory.create_fitness_function('accuracy')
    print("Accuracy from factory:", accuracy_func(y_true, y_pred))
    
    # Create multi-objective fitness function
    objectives = ['accuracy', 'f1', 'expert_balance']
    multi_obj_func = FitnessFunctionFactory.create_multi_objective_fitness(objectives)
    
    # Calculate combined fitness
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'expert_balance': balance,
        'model_complexity': complexity
    }
    
    weights = {
        'accuracy': 2.0,
        'f1': 1.5,
        'false_positive_rate': 1.0,
        'false_negative_rate': 1.0,
        'expert_balance': 0.5,
        'model_complexity': 0.2
    }
    
    combined = FitnessFunctions.combined_fitness(metrics_dict, weights)
    print("Combined fitness:", combined)
