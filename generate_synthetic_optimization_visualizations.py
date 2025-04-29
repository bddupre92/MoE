"""
Generate synthetic PyGMO optimization visualizations for the publication.

This script creates synthetic visualizations for the PyGMO optimization section
(Figures 15-19) since the actual PyGMO optimization encountered integration issues.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Verdana', 'Helvetica', 'sans-serif']

# Create output directory
output_dir = '/home/ubuntu/full_pipeline_output/pygmo_optimization'
os.makedirs(output_dir, exist_ok=True)

def generate_pygmo_optimization_visualizations():
    """Generate synthetic visualizations for PyGMO optimization (Figures 15-19)"""
    print("Generating PyGMO optimization visualizations (synthetic data)...")
    
    # Figure 15: Convergence plot
    generate_convergence_plot()
    
    # Figure 16: Pareto front visualization
    generate_pareto_front()
    
    # Figure 17: Population diversity
    generate_population_diversity()
    
    # Figure 18: Island model migration topology
    generate_island_model_topology()
    
    # Figure 19: Hyperparameter importance
    generate_hyperparameter_importance()
    
    print("PyGMO optimization visualizations completed (synthetic data)")

def generate_convergence_plot():
    """Generate Figure 15: Convergence plot"""
    # Generate synthetic data
    generations = np.arange(1, 21)
    best_fitness = 0.8 * np.exp(-0.15 * generations) + 0.2
    avg_fitness = 0.9 * np.exp(-0.1 * generations) + 0.25
    
    plt.figure(figsize=(12, 8))
    plt.plot(generations, best_fitness, 'b-', linewidth=2, marker='o', label='Best Fitness')
    plt.plot(generations, avg_fitness, 'r-', linewidth=2, marker='s', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (lower is better)')
    plt.title('Figure 15: PyGMO Optimization Convergence (Synthetic Data)')
    plt.legend()
    plt.grid(True)
    plt.annotate('Convergence point', xy=(15, best_fitness[14]), 
                 xytext=(12, best_fitness[14] + 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    plt.text(0.5, 0.01, 'Note: This visualization uses synthetic data as a placeholder',
             horizontalalignment='center', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=10, style='italic', 
             bbox=dict(facecolor='yellow', alpha=0.2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_15_convergence_plot.png'))
    plt.close()

def generate_pareto_front():
    """Generate Figure 16: Pareto front visualization"""
    # Generate synthetic data for a Pareto front
    np.random.seed(42)
    n_points = 100
    
    # Generate dominated solutions
    x_dominated = np.random.uniform(0.3, 1.0, size=n_points)
    y_dominated = np.random.uniform(0.3, 1.0, size=n_points)
    
    # Generate Pareto front
    x_pareto = np.linspace(0.05, 0.5, 20)
    y_pareto = 0.05 + 0.45 * (1 - x_pareto/0.5)**2
    
    plt.figure(figsize=(12, 8))
    plt.scatter(x_dominated, y_dominated, c='lightgray', s=50, alpha=0.7, label='Dominated Solutions')
    plt.plot(x_pareto, y_pareto, 'r-', linewidth=3, label='Pareto Front')
    plt.scatter(x_pareto, y_pareto, c='red', s=80, zorder=3)
    
    # Highlight selected solutions
    selected_indices = [0, 5, 10, 15, 19]
    plt.scatter(x_pareto[selected_indices], y_pareto[selected_indices], 
                c='blue', s=120, zorder=4, label='Selected Solutions')
    
    plt.xlabel('Objective 1: False Positive Rate')
    plt.ylabel('Objective 2: 1 - Accuracy')
    plt.title('Figure 16: Multi-objective Optimization Pareto Front (Synthetic Data)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.text(0.5, 0.01, 'Note: This visualization uses synthetic data as a placeholder',
             horizontalalignment='center', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=10, style='italic', 
             bbox=dict(facecolor='yellow', alpha=0.2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_16_pareto_front.png'))
    plt.close()

def generate_population_diversity():
    """Generate Figure 17: Population diversity"""
    # Generate synthetic data
    generations = np.arange(1, 21)
    diversity = 0.9 * np.exp(-0.08 * generations) + 0.1
    
    # Create a colormap for the scatter points
    colors = np.linspace(0, 1, len(generations))
    cmap = plt.cm.viridis
    
    # Generate synthetic population data for selected generations
    selected_gens = [1, 5, 10, 20]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, gen in enumerate(selected_gens):
        np.random.seed(gen * 10)  # For reproducibility
        
        # Generate 2D points with decreasing spread
        spread = 0.9 * np.exp(-0.08 * gen) + 0.1
        n_points = 50
        x = np.random.normal(0.5, spread, n_points)
        y = np.random.normal(0.5, spread, n_points)
        
        # Plot
        axs[i].scatter(x, y, c=np.random.uniform(0, 1, n_points), cmap=cmap, s=80, alpha=0.7)
        axs[i].set_title(f'Generation {gen}')
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)
        
        # Add diversity measure
        axs[i].text(0.05, 0.05, f'Diversity: {spread:.2f}', 
                   transform=axs[i].transAxes, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.7))
    
    plt.suptitle('Figure 17: Population Diversity Across Generations (Synthetic Data)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add note about synthetic data
    fig.text(0.5, 0.01, 'Note: This visualization uses synthetic data as a placeholder',
             horizontalalignment='center', verticalalignment='bottom',
             fontsize=10, style='italic', 
             bbox=dict(facecolor='yellow', alpha=0.2))
    
    plt.savefig(os.path.join(output_dir, 'figure_17_population_diversity.png'))
    plt.close()

def generate_island_model_topology():
    """Generate Figure 18: Island model migration topology"""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (islands)
    islands = ['Island 1', 'Island 2', 'Island 3', 'Island 4', 'Island 5']
    for i, island in enumerate(islands):
        G.add_node(island, algorithm=f'Algorithm {i+1}')
    
    # Add edges (migrations)
    G.add_edge('Island 1', 'Island 2', weight=5)
    G.add_edge('Island 2', 'Island 3', weight=3)
    G.add_edge('Island 3', 'Island 4', weight=4)
    G.add_edge('Island 4', 'Island 5', weight=2)
    G.add_edge('Island 5', 'Island 1', weight=6)
    G.add_edge('Island 1', 'Island 4', weight=1)
    G.add_edge('Island 2', 'Island 5', weight=2)
    G.add_edge('Island 3', 'Island 1', weight=3)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Define node positions
    pos = nx.circular_layout(G)
    
    # Define edge weights for line thickness
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edges with varying thickness based on weight
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, 
                          edge_color='gray', arrows=True, arrowsize=20)
    
    # Add edge labels (migration rates)
    edge_labels = {(u, v): f"{G[u][v]['weight']} ind." for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Add algorithm labels
    for node, (x, y) in pos.items():
        plt.text(x, y-0.15, G.nodes[node]['algorithm'], 
                horizontalalignment='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('Figure 18: Island Model Migration Topology (Synthetic Data)', fontsize=16)
    plt.axis('off')
    
    # Add note about synthetic data
    plt.text(0.5, 0.01, 'Note: This visualization uses synthetic data as a placeholder',
             horizontalalignment='center', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=10, style='italic', 
             bbox=dict(facecolor='yellow', alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_18_island_model_topology.png'))
    plt.close()

def generate_hyperparameter_importance():
    """Generate Figure 19: Hyperparameter importance"""
    # Define hyperparameters and their importance scores
    hyperparams = [
        'Learning Rate', 
        'Batch Size', 
        'Expert Count', 
        'Hidden Layers', 
        'Dropout Rate',
        'Weight Decay',
        'Activation Function',
        'Optimizer'
    ]
    
    importance = [0.85, 0.72, 0.65, 0.58, 0.45, 0.38, 0.25, 0.18]
    
    # Sort by importance
    sorted_indices = np.argsort(importance)[::-1]
    hyperparams = [hyperparams[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(hyperparams, importance, color='skyblue', height=0.6)
    
    # Add a color gradient
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(importance[i]))
    
    # Add value labels
    for i, v in enumerate(importance):
        plt.text(v + 0.02, i, f'{v:.2f}', va='center', fontweight='bold')
    
    plt.xlabel('Relative Importance')
    plt.title('Figure 19: Hyperparameter Importance for Model Performance (Synthetic Data)')
    plt.xlim(0, 1.0)
    plt.grid(axis='x')
    
    # Add note about synthetic data
    plt.text(0.5, 0.01, 'Note: This visualization uses synthetic data as a placeholder',
             horizontalalignment='center', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=10, style='italic', 
             bbox=dict(facecolor='yellow', alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_19_hyperparameter_importance.png'))
    plt.close()

if __name__ == "__main__":
    generate_pygmo_optimization_visualizations()
    print("All synthetic PyGMO optimization visualizations generated successfully.")
