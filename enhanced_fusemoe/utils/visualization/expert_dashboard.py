"""
Expert dashboard component for the Enhanced FuseMoE system.

This module provides visualization components for the expert dashboard in the
Enhanced FuseMoE system for migraine prediction.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import base64

# For Streamlit integration
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class ExpertDashboard:
    """
    Dashboard for visualizing expert contributions and performance in the Enhanced FuseMoE system.
    
    This class provides methods for visualizing the gating network, expert weights,
    and performance comparisons between optimized and baseline models.
    
    Attributes:
        model: Enhanced FuseMoE model
        expert_names: List of expert names
        test_data: Test data for evaluation
        optimized_metrics: Performance metrics for optimized model
        baseline_metrics: Performance metrics for baseline model
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        expert_names: Optional[List[str]] = None,
        test_data: Optional[Any] = None,
        optimized_metrics: Optional[Dict[str, Any]] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the expert dashboard.
        
        Args:
            model: Enhanced FuseMoE model (optional)
            expert_names: List of expert names (optional)
            test_data: Test data for evaluation (optional)
            optimized_metrics: Performance metrics for optimized model (optional)
            baseline_metrics: Performance metrics for baseline model (optional)
        """
        self.model = model
        self.expert_names = expert_names or ["Sleep", "Weather", "Stress/Diet", "Physiological"]
        self.test_data = test_data
        self.optimized_metrics = optimized_metrics
        self.baseline_metrics = baseline_metrics
    
    def plot_network_diagram(
        self,
        expert_weights: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot an interactive network diagram of the gating network and experts.
        
        Args:
            expert_weights: Expert weights of shape [num_experts] (optional)
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure containing the network diagram
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node("Input", pos=(0, 0))
        G.add_node("Gating Network", pos=(1, 0))
        
        # Add expert nodes
        num_experts = len(self.expert_names)
        expert_positions = [(2, i - (num_experts - 1) / 2) for i in range(num_experts)]
        for i, name in enumerate(self.expert_names):
            G.add_node(name, pos=expert_positions[i])
        
        # Add output node
        G.add_node("Output", pos=(3, 0))
        
        # Add edges
        G.add_edge("Input", "Gating Network")
        
        # Add edges from gating network to experts
        for name in self.expert_names:
            G.add_edge("Gating Network", name)
            G.add_edge(name, "Output")
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_size=3000, 
                              node_color='lightblue',
                              node_shape='o',
                              alpha=0.8,
                              ax=ax)
        
        # Draw edges
        if expert_weights is not None:
            # Normalize weights for edge width
            max_weight = max(expert_weights)
            normalized_weights = [w / max_weight * 5 for w in expert_weights]
            
            # Draw edges with varying width based on weights
            for i, name in enumerate(self.expert_names):
                nx.draw_networkx_edges(G, pos, 
                                      edgelist=[("Gating Network", name)],
                                      width=normalized_weights[i],
                                      alpha=0.7,
                                      ax=ax)
                
                nx.draw_networkx_edges(G, pos, 
                                      edgelist=[(name, "Output")],
                                      width=normalized_weights[i],
                                      alpha=0.7,
                                      ax=ax)
        else:
            # Draw all edges with same width
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax)
        
        # Draw edge from input to gating network
        nx.draw_networkx_edges(G, pos, 
                              edgelist=[("Input", "Gating Network")],
                              width=2,
                              alpha=0.7,
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', ax=ax)
        
        # Remove axis
        ax.axis('off')
        
        # Set title
        ax.set_title("Mixture of Experts Network Diagram", fontsize=16)
        
        plt.tight_layout()
        
        return fig
    
    def plot_expert_weights(
        self,
        expert_weights: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot a bar chart of expert weights.
        
        Args:
            expert_weights: Expert weights of shape [num_experts] (optional)
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure containing the bar chart
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate sample weights if not provided
        if expert_weights is None:
            expert_weights = np.random.dirichlet(np.ones(len(self.expert_names)))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Expert': self.expert_names,
            'Weight': expert_weights
        })
        
        # Sort by weight
        df = df.sort_values('Weight', ascending=False)
        
        # Plot bar chart
        sns.barplot(x='Expert', y='Weight', data=df, ax=ax)
        
        # Add value labels on top of bars
        for i, v in enumerate(df['Weight']):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('Expert', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title('Expert Weight Distribution', fontsize=16)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig
    
    def plot_decision_tree_visualization(
        self,
        X: Optional[np.ndarray] = None,
        expert_gates: Optional[np.ndarray] = None,
        max_depth: int = 3,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot a decision tree visualization showing how inputs affect expert selection.
        
        Args:
            X: Input features of shape [num_samples, num_features] (optional)
            expert_gates: Expert gate values of shape [num_samples, num_experts] (optional)
            max_depth: Maximum depth of the decision tree
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure containing the decision tree visualization
        """
        # Generate sample data if not provided
        if X is None or expert_gates is None:
            # Generate sample data
            num_samples = 100
            num_features = 5
            X = np.random.randn(num_samples, num_features)
            
            # Generate sample expert gates
            num_experts = len(self.expert_names)
            expert_gates = np.random.rand(num_samples, num_experts)
            expert_gates = expert_gates / expert_gates.sum(axis=1, keepdims=True)
        
        # Get expert with highest gate value for each sample
        expert_indices = np.argmax(expert_gates, axis=1)
        
        # Create feature names if not available
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Train decision tree to predict expert selection
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X, expert_indices)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Export decision tree to graphviz
        dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=feature_names,
            class_names=self.expert_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        
        # Convert to image
        graph = graphviz.Source(dot_data)
        
        # Save to temporary file and read back
        temp_file = "temp_tree.png"
        graph.render(filename="temp_tree", format="png", cleanup=True)
        
        # Read image and display
        img = plt.imread(temp_file)
        ax.imshow(img)
        ax.axis('off')
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        plt.tight_layout()
        
        return fig
    
    def plot_performance_comparison(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot a comparison of performance metrics between optimized and baseline models.
        
        Args:
            metrics: List of metrics to compare (optional)
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure containing the performance comparison
        """
        # Default metrics to compare
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate sample metrics if not provided
        if self.optimized_metrics is None:
            self.optimized_metrics = {
                'accuracy': 0.95,
                'precision': 0.92,
                'recall': 0.94,
                'f1': 0.93,
                'auc': 0.97
            }
        
        if self.baseline_metrics is None:
            self.baseline_metrics = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.84,
                'f1': 0.83,
                'auc': 0.87
            }
        
        # Create DataFrame for plotting
        data = []
        for metric in metrics:
            if metric in self.optimized_metrics and metric in self.baseline_metrics:
                data.append({
                    'Metric': metric.capitalize(),
                    'Optimized': self.optimized_metrics[metric],
                    'Baseline': self.baseline_metrics[metric]
                })
        
        df = pd.DataFrame(data)
        
        # Melt DataFrame for grouped bar chart
        df_melted = pd.melt(df, id_vars=['Metric'], var_name='Model', value_name='Value')
        
        # Plot grouped bar chart
        sns.barplot(x='Metric', y='Value', hue='Model', data=df_melted, ax=ax)
        
        # Add value labels on top of bars
        for i, v in enumerate(df_melted['Value']):
            ax.text(i % len(metrics) - 0.2 + (i // len(metrics)) * 0.4, 
                   v + 0.01, 
                   f"{v:.3f}", 
                   ha='center', 
                   va='bottom', 
                   fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Performance Comparison: Optimized vs. Baseline', fontsize=16)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Add legend
        ax.legend(title='Model')
        
        plt.tight_layout()
        
        return fig
    
    def render_streamlit_dashboard(self) -> None:
        """
        Render the expert dashboard in Streamlit.
        
        This method creates a Streamlit dashboard with multiple tabs for different
        visualizations of the expert model.
        """
        if not STREAMLIT_AVAILABLE:
            print("Streamlit is not available. Please install it with 'pip install streamlit'.")
            return
        
        st.title("Enhanced FuseMoE Expert Dashboard")
        
        # Create tabs
        tabs = st.tabs(["Network Diagram", "Expert Weights", "Decision Tree", "Performance Comparison"])
        
        # Network Diagram tab
        with tabs[0]:
            st.header("Mixture of Experts Network Diagram")
            
            # Get expert weights
            expert_weights = self._get_expert_weights()
            
            # Plot network diagram
            fig = self.plot_network_diagram(expert_weights)
            st.pyplot(fig)
        
        # Expert Weights tab
        with tabs[1]:
            st.header("Expert Weight Distribution")
            
            # Get expert weights
            expert_weights = self._get_expert_weights()
            
            # Plot expert weights
            fig = self.plot_expert_weights(expert_weights)
            st.pyplot(fig)
        
        # Decision Tree tab
        with tabs[2]:
            st.header("Decision Tree Visualization")
            
            # Get input features and expert gates
            X, expert_gates = self._get_features_and_gates()
            
            # Plot decision tree visualization
            fig = self.plot_decision_tree_visualization(X, expert_gates)
            st.pyplot(fig)
        
        # Performance Comparison tab
        with tabs[3]:
            st.header("Performance Comparison: Optimized vs. Baseline")
            
            # Plot performance comparison
            fig = self.plot_performance_comparison()
            st.pyplot(fig)
    
    def _get_expert_weights(self) -> np.ndarray:
        """
        Get expert weights from the model or generate sample weights.
        
        Returns:
            Expert weights of shape [num_experts]
        """
        if self.model is not None and hasattr(self.model, 'get_expert_weights'):
            # Get expert weights from model
            return self.model.get_expert_weights()
        else:
            # Generate sample weights
            return np.random.dirichlet(np.ones(len(self.expert_names)))
    
    def _get_features_and_gates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get input features and expert gates from the model or generate sample data.
        
        Returns:
            Tuple containing:
                - Input features of shape [num_samples, num_features]
                - Expert gate values of shape [num_samples, num_experts]
        """
        if self.model is not None and self.test_data is not None:
            # Get features and gates from model and test data
            if hasattr(self.model, 'get_expert_gates'):
                X = self.test_data[0]
                expert_gates = self.model.get_expert_gates(X)
                return X, expert_gates
        
        # Generate sample data
        num_samples = 100
        num_features = 5
        X = np.random.randn(num_samples, num_features)
        
        # Generate sample expert gates
        num_experts = len(self.expert_names)
        expert_gates = np.random.rand(num_samples, num_experts)
        expert_gates = expert_gates / expert_gates.sum(axis=1, keepdims=True)
        
        return X, expert_gates
