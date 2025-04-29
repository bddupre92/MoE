"""
Optimization Visualization for PyGMO integration with MoE

This module provides visualization tools for PyGMO optimization results,
including Pareto fronts, convergence plots, expert contribution analysis,
and performance comparisons.

Author: Manus AI
Date: April 28, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationVisualizer:
    """Visualization tools for PyGMO optimization results."""
    
    def __init__(self, 
                 output_dir: str = './output',
                 style: str = 'darkgrid',
                 palette: str = 'viridis',
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 100):
        """
        Initialize optimization visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Seaborn style
            palette: Color palette
            figsize: Figure size
            dpi: Figure resolution
        """
        self.output_dir = output_dir
        self.style = style
        self.palette = palette
        self.figsize = figsize
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style(style)
        sns.set_palette(palette)
        
        logger.info(f"Initialized OptimizationVisualizer with output directory: {output_dir}")
    
    def plot_pareto_front(self, 
                          fitness_values: np.ndarray,
                          objective_names: List[str],
                          title: str = 'Pareto Front',
                          save_path: str = None):
        """
        Plot Pareto front for multi-objective optimization.
        
        Args:
            fitness_values: Array of fitness values (n_individuals, n_objectives)
            objective_names: Names of objectives
            title: Plot title
            save_path: Path to save plot (default: output_dir/pareto_front.png)
            
        Returns:
            Figure and axes
        """
        n_objectives = fitness_values.shape[1]
        
        if n_objectives < 2 or n_objectives > 3:
            logger.warning(f"Pareto front visualization only supports 2 or 3 objectives, got {n_objectives}")
            return None, None
        
        # Create figure
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        if n_objectives == 2:
            # 2D Pareto front
            ax = fig.add_subplot(111)
            
            # Plot fitness values
            ax.scatter(fitness_values[:, 0], fitness_values[:, 1], alpha=0.7)
            
            # Add labels
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
        else:
            # 3D Pareto front
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot fitness values
            ax.scatter(fitness_values[:, 0], fitness_values[:, 1], fitness_values[:, 2], alpha=0.7)
            
            # Add labels
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
            ax.set_zlabel(objective_names[2])
            ax.set_title(title)
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'pareto_front.png')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved Pareto front plot to {save_path}")
        
        return fig, ax
    
    def plot_convergence(self, 
                         fitness_history: List[np.ndarray],
                         objective_names: List[str],
                         title: str = 'Convergence',
                         save_path: str = None):
        """
        Plot convergence of fitness values over generations.
        
        Args:
            fitness_history: List of fitness arrays for each generation
            objective_names: Names of objectives
            title: Plot title
            save_path: Path to save plot (default: output_dir/convergence.png)
            
        Returns:
            Figure and axes
        """
        n_generations = len(fitness_history)
        n_objectives = fitness_history[0].shape[1]
        
        # Create figure
        fig, axes = plt.subplots(n_objectives, 1, figsize=(self.figsize[0], self.figsize[1] * n_objectives),
                                dpi=self.dpi, sharex=True)
        
        # Handle single objective case
        if n_objectives == 1:
            axes = [axes]
        
        # Extract best fitness for each generation and objective
        best_fitness = np.zeros((n_generations, n_objectives))
        for i, fitness in enumerate(fitness_history):
            best_fitness[i] = np.min(fitness, axis=0)
        
        # Plot convergence for each objective
        for i, ax in enumerate(axes):
            ax.plot(range(n_generations), best_fitness[:, i], marker='o', markersize=4)
            ax.set_ylabel(objective_names[i])
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels
        axes[-1].set_xlabel('Generation')
        fig.suptitle(title)
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'convergence.png')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved convergence plot to {save_path}")
        
        return fig, axes
    
    def plot_expert_contributions(self, 
                                 expert_contributions: np.ndarray,
                                 expert_names: List[str],
                                 title: str = 'Expert Contributions',
                                 save_path: str = None):
        """
        Plot expert contributions.
        
        Args:
            expert_contributions: Array of expert contributions (n_samples, n_experts)
            expert_names: Names of experts
            title: Plot title
            save_path: Path to save plot (default: output_dir/expert_contributions.png)
            
        Returns:
            Figure and axes
        """
        n_experts = expert_contributions.shape[1] if len(expert_contributions.shape) > 1 else len(expert_contributions)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if len(expert_contributions.shape) > 1:
            # Multiple samples
            # Calculate mean and std
            mean_contributions = np.mean(expert_contributions, axis=0)
            std_contributions = np.std(expert_contributions, axis=0)
            
            # Plot bar chart with error bars
            ax.bar(range(n_experts), mean_contributions, yerr=std_contributions, alpha=0.7)
            
        else:
            # Single sample
            ax.bar(range(n_experts), expert_contributions, alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Expert')
        ax.set_ylabel('Contribution')
        ax.set_title(title)
        ax.set_xticks(range(n_experts))
        ax.set_xticklabels(expert_names, rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'expert_contributions.png')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved expert contributions plot to {save_path}")
        
        return fig, ax
    
    def plot_expert_network(self, 
                           expert_weights: List[np.ndarray],
                           gating_weights: np.ndarray,
                           expert_names: List[str],
                           feature_names: List[str],
                           title: str = 'Expert Network',
                           save_path: str = None):
        """
        Plot expert network diagram.
        
        Args:
            expert_weights: List of weight arrays for each expert
            gating_weights: Weight array for gating network
            expert_names: Names of experts
            feature_names: Names of features
            title: Plot title
            save_path: Path to save plot (default: output_dir/expert_network.png)
            
        Returns:
            Figure and axes
        """
        n_experts = len(expert_weights)
        n_features = len(feature_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate positions
        feature_x = np.zeros(n_features)
        feature_y = np.arange(n_features)
        
        expert_x = np.ones(n_experts) * 2
        expert_y = np.arange(n_experts)
        
        output_x = 4
        output_y = n_experts // 2
        
        # Plot nodes
        ax.scatter(feature_x, feature_y, s=100, c='blue', label='Features')
        ax.scatter(expert_x, expert_y, s=100, c='green', label='Experts')
        ax.scatter(output_x, output_y, s=100, c='red', label='Output')
        
        # Plot feature-expert connections
        for i, expert_weight in enumerate(expert_weights):
            for j, weight in enumerate(expert_weight):
                ax.plot([feature_x[j], expert_x[i]], [feature_y[j], expert_y[i]], 
                       alpha=abs(weight) / max(abs(expert_weight)), 
                       linewidth=abs(weight) / max(abs(expert_weight)) * 3,
                       color='blue' if weight > 0 else 'red')
        
        # Plot expert-output connections
        for i, weight in enumerate(gating_weights):
            ax.plot([expert_x[i], output_x], [expert_y[i], output_y], 
                   alpha=abs(weight) / max(abs(gating_weights)), 
                   linewidth=abs(weight) / max(abs(gating_weights)) * 3,
                   color='blue' if weight > 0 else 'red')
        
        # Add labels
        for i, name in enumerate(feature_names):
            ax.text(feature_x[i] - 0.1, feature_y[i], name, ha='right', va='center')
        
        for i, name in enumerate(expert_names):
            ax.text(expert_x[i] + 0.1, expert_y[i], name, ha='left', va='center')
        
        ax.text(output_x + 0.1, output_y, 'Output', ha='left', va='center')
        
        # Set limits
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, max(n_features, n_experts))
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        ax.set_title(title)
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'expert_network.png')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved expert network plot to {save_path}")
        
        return fig, ax
    
    def plot_performance_comparison(self, 
                                   metrics_df: pd.DataFrame,
                                   title: str = 'Performance Comparison',
                                   save_path: str = None):
        """
        Plot performance comparison between baseline and optimized models.
        
        Args:
            metrics_df: DataFrame with performance metrics
            title: Plot title
            save_path: Path to save plot (default: output_dir/performance_comparison.png)
            
        Returns:
            Figure and axes
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot bar chart
        metrics_df[['Baseline', 'Optimized']].plot(kind='bar', ax=ax)
        
        # Add labels
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add improvement labels
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            improvement = row['Improvement (%)']
            ax.text(i, max(row['Baseline'], row['Optimized']), 
                   f"{improvement:+.1f}%", 
                   ha='center', va='bottom')
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'performance_comparison.png')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved performance comparison plot to {save_path}")
        
        return fig, ax
    
    def plot_optimization_landscape(self, 
                                   param_ranges: Dict[str, Tuple[float, float]],
                                   fitness_function: callable,
                                   title: str = 'Optimization Landscape',
                                   save_path: str = None):
        """
        Plot optimization landscape for 1 or 2 parameters.
        
        Args:
            param_ranges: Dictionary of parameter ranges (name: (min, max))
            fitness_function: Function to evaluate fitness
            title: Plot title
            save_path: Path to save plot (default: output_dir/optimization_landscape.png)
            
        Returns:
            Figure and axes
        """
        n_params = len(param_ranges)
        
        if n_params < 1 or n_params > 2:
            logger.warning(f"Optimization landscape visualization only supports 1 or 2 parameters, got {n_params}")
            return None, None
        
        # Create figure
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        if n_params == 1:
            # 1D landscape
            ax = fig.add_subplot(111)
            
            # Get parameter name and range
            param_name = list(param_ranges.keys())[0]
            param_min, param_max = param_ranges[param_name]
            
            # Create parameter values
            param_values = np.linspace(param_min, param_max, 100)
            
            # Evaluate fitness
            fitness_values = np.array([fitness_function({param_name: val}) for val in param_values])
            
            # Plot landscape
            ax.plot(param_values, fitness_values)
            
            # Add labels
            ax.set_xlabel(param_name)
            ax.set_ylabel('Fitness')
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
        else:
            # 2D landscape
            ax = fig.add_subplot(111, projection='3d')
            
            # Get parameter names and ranges
            param_names = list(param_ranges.keys())
            param1_min, param1_max = param_ranges[param_names[0]]
            param2_min, param2_max = param_ranges[param_names[1]]
            
            # Create parameter grids
            param1_values = np.linspace(param1_min, param1_max, 20)
            param2_values = np.linspace(param2_min, param2_max, 20)
            param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
            
            # Evaluate fitness
            fitness_grid = np.zeros_like(param1_grid)
            for i in range(param1_grid.shape[0]):
                for j in range(param1_grid.shape[1]):
                    fitness_grid[i, j] = fitness_function({
                        param_names[0]: param1_grid[i, j],
                        param_names[1]: param2_grid[i, j]
                    })
            
            # Plot landscape
            surf = ax.plot_surface(param1_grid, param2_grid, fitness_grid, cmap='viridis', alpha=0.8)
            
            # Add labels
            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            ax.set_zlabel('Fitness')
            ax.set_title(title)
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'optimization_landscape.png')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved optimization landscape plot to {save_path}")
        
        return fig, ax
    
    def create_dashboard_figures(self, 
                                optimization_results: Dict[str, Any],
                                metrics_df: pd.DataFrame,
                                expert_contributions: np.ndarray,
                                expert_names: List[str],
                                feature_names: List[str],
                                output_dir: str = None):
        """
        Create figures for dashboard.
        
        Args:
            optimization_results: Dictionary of optimization results
            metrics_df: DataFrame with performance metrics
            expert_contributions: Array of expert contributions
            expert_names: Names of experts
            feature_names: Names of features
            output_dir: Directory to save figures (default: self.output_dir)
            
        Returns:
            Dictionary of figure paths
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        figure_paths = {}
        
        # Plot Pareto front
        if 'fitness_history' in optimization_results and len(optimization_results['fitness_history']) > 0:
            fitness_values = optimization_results['fitness_history'][-1]
            objective_names = optimization_results.get('objective_names', [f'Objective {i+1}' for i in range(fitness_values.shape[1])])
            
            fig, ax = self.plot_pareto_front(
                fitness_values=fitness_values,
                objective_names=objective_names,
                title='Pareto Front',
                save_path=os.path.join(output_dir, 'pareto_front.png')
            )
            
            figure_paths['pareto_front'] = os.path.join(output_dir, 'pareto_front.png')
        
        # Plot convergence
        if 'fitness_history' in optimization_results and len(optimization_results['fitness_history']) > 0:
            fitness_history = optimization_results['fitness_history']
            objective_names = optimization_results.get('objective_names', [f'Objective {i+1}' for i in range(fitness_history[0].shape[1])])
            
            fig, ax = self.plot_convergence(
                fitness_history=fitness_history,
                objective_names=objective_names,
                title='Convergence',
                save_path=os.path.join(output_dir, 'convergence.png')
            )
            
            figure_paths['convergence'] = os.path.join(output_dir, 'convergence.png')
        
        # Plot expert contributions
        if expert_contributions is not None and expert_names is not None:
            fig, ax = self.plot_expert_contributions(
                expert_contributions=expert_contributions,
                expert_names=expert_names,
                title='Expert Contributions',
                save_path=os.path.join(output_dir, 'expert_contributions.png')
            )
            
            figure_paths['expert_contributions'] = os.path.join(output_dir, 'expert_contributions.png')
        
        # Plot expert network
        if 'expert_weights' in optimization_results and 'gating_weights' in optimization_results:
            expert_weights = optimization_results['expert_weights']
            gating_weights = optimization_results['gating_weights']
            
            fig, ax = self.plot_expert_network(
                expert_weights=expert_weights,
                gating_weights=gating_weights,
                expert_names=expert_names,
                feature_names=feature_names,
                title='Expert Network',
                save_path=os.path.join(output_dir, 'expert_network.png')
            )
            
            figure_paths['expert_network'] = os.path.join(output_dir, 'expert_network.png')
        
        # Plot performance comparison
        if metrics_df is not None:
            fig, ax = self.plot_performance_comparison(
                metrics_df=metrics_df,
                title='Performance Comparison',
                save_path=os.path.join(output_dir, 'performance_comparison.png')
            )
            
            figure_paths['performance_comparison'] = os.path.join(output_dir, 'performance_comparison.png')
        
        logger.info(f"Created dashboard figures: {list(figure_paths.keys())}")
        return figure_paths


# Example usage
if __name__ == "__main__":
    # Create visualizer
    visualizer = OptimizationVisualizer(output_dir='./output')
    
    # Generate mock data
    n_generations = 20
    n_individuals = 50
    n_objectives = 2
    n_experts = 3
    n_features = 5
    
    # Fitness history
    fitness_history = [np.random.random((n_individuals, n_objectives)) for _ in range(n_generations)]
    
    # Expert contributions
    expert_contributions = np.random.random(n_experts)
    expert_contributions = expert_contributions / np.sum(expert_contributions)
    
    # Expert weights
    expert_weights = [np.random.randn(n_features) for _ in range(n_experts)]
    gating_weights = np.random.randn(n_experts)
    
    # Performance metrics
    metrics = {
        'accuracy': [0.75, 0.85],
        'precision': [0.70, 0.82],
        'recall': [0.68, 0.79],
        'f1': [0.69, 0.80],
        'false_positive_rate': [0.32, 0.22],
        'false_negative_rate': [0.28, 0.18]
    }
    
    metrics_df = pd.DataFrame(metrics, index=['Baseline', 'Optimized']).T
    metrics_df['Improvement'] = metrics_df['Optimized'] - metrics_df['Baseline']
    metrics_df['Improvement (%)'] = (metrics_df['Improvement'] / metrics_df['Baseline'] * 100).round(2)
    
    # Plot Pareto front
    visualizer.plot_pareto_front(
        fitness_values=fitness_history[-1],
        objective_names=['Objective 1', 'Objective 2'],
        title='Pareto Front'
    )
    
    # Plot convergence
    visualizer.plot_convergence(
        fitness_history=fitness_history,
        objective_names=['Objective 1', 'Objective 2'],
        title='Convergence'
    )
    
    # Plot expert contributions
    visualizer.plot_expert_contributions(
        expert_contributions=expert_contributions,
        expert_names=['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert'],
        title='Expert Contributions'
    )
    
    # Plot expert network
    visualizer.plot_expert_network(
        expert_weights=expert_weights,
        gating_weights=gating_weights,
        expert_names=['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert'],
        feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        title='Expert Network'
    )
    
    # Plot performance comparison
    visualizer.plot_performance_comparison(
        metrics_df=metrics_df,
        title='Performance Comparison'
    )
    
    # Create dashboard figures
    optimization_results = {
        'fitness_history': fitness_history,
        'objective_names': ['Objective 1', 'Objective 2'],
        'expert_weights': expert_weights,
        'gating_weights': gating_weights
    }
    
    figure_paths = visualizer.create_dashboard_figures(
        optimization_results=optimization_results,
        metrics_df=metrics_df,
        expert_contributions=expert_contributions,
        expert_names=['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert'],
        feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
    )
    
    print("Visualization completed successfully!")
    print(f"Figure paths: {figure_paths}")
