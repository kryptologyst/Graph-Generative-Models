"""Evaluation utilities for graph generative models."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from torch_geometric.utils import to_networkx
import networkx as nx

from ..utils import get_device


class Evaluator:
    """Evaluator class for graph generative models.
    
    Args:
        model: Trained model
        data: Test data
        device: Device to use for evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.data = data
        self.device = device or get_device()
        
        # Move model and data to device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_link_prediction(
        self, 
        metrics: List[str] = ["auc", "ap", "f1"]
    ) -> Dict[str, float]:
        """Evaluate link prediction performance.
        
        Args:
            metrics: List of metrics to compute
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        results = {}
        
        with torch.no_grad():
            # Get node embeddings
            if isinstance(self.model, VGAE):
                z = self.model.encode(self.data.x, self.data.train_pos_edge_index)
            else:
                z = self.model.encode(self.data.x, self.data.train_pos_edge_index)
            
            # Test positive edges
            pos_edge_index = self.data.test_pos_edge_index
            pos_prob = self.model.decode(z, pos_edge_index)
            pos_labels = torch.ones_like(pos_prob)
            
            # Test negative edges
            neg_edge_index = self.data.test_neg_edge_index
            neg_prob = self.model.decode(z, neg_edge_index)
            neg_labels = torch.zeros_like(neg_prob)
            
            # Combine predictions and labels
            all_prob = torch.cat([pos_prob, neg_prob]).cpu().numpy()
            all_labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
            
            # Compute metrics
            if "auc" in metrics:
                results["auc"] = roc_auc_score(all_labels, all_prob)
            
            if "ap" in metrics:
                results["ap"] = average_precision_score(all_labels, all_prob)
            
            if "f1" in metrics:
                # Convert probabilities to binary predictions
                pred_labels = (all_prob > 0.5).astype(int)
                results["f1"] = f1_score(all_labels, pred_labels)
            
            if "accuracy" in metrics:
                pred_labels = (all_prob > 0.5).astype(int)
                results["accuracy"] = accuracy_score(all_labels, pred_labels)
        
        return results
    
    def evaluate_graph_generation(
        self,
        num_samples: int = 100,
        metrics: List[str] = ["density", "clustering", "degree_dist"]
    ) -> Dict[str, float]:
        """Evaluate graph generation quality.
        
        Args:
            num_samples: Number of graphs to generate
            metrics: List of metrics to compute
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        results = {}
        
        with torch.no_grad():
            if isinstance(self.model, GraphVAE):
                # Generate graphs
                generated_graphs = []
                
                for _ in range(num_samples):
                    # Sample from prior
                    z = torch.randn(1, self.model.latent_dim, device=self.device)
                    
                    # Generate graph
                    node_features, adj_matrix, node_probs = self.model.decode(z)
                    
                    # Convert to networkx graph
                    G = self._adj_matrix_to_graph(adj_matrix, node_probs)
                    generated_graphs.append(G)
                
                # Compute metrics
                if "density" in metrics:
                    densities = [nx.density(G) for G in generated_graphs]
                    results["density_mean"] = np.mean(densities)
                    results["density_std"] = np.std(densities)
                
                if "clustering" in metrics:
                    clusterings = [nx.average_clustering(G) for G in generated_graphs]
                    results["clustering_mean"] = np.mean(clusterings)
                    results["clustering_std"] = np.std(clusterings)
                
                if "degree_dist" in metrics:
                    degree_means = [np.mean([d for n, d in G.degree()]) for G in generated_graphs]
                    results["degree_mean"] = np.mean(degree_means)
                    results["degree_std"] = np.std(degree_means)
            
            else:
                raise NotImplementedError(f"Graph generation evaluation not implemented for {type(self.model)}")
        
        return results
    
    def _adj_matrix_to_graph(
        self, 
        adj_matrix: torch.Tensor, 
        node_probs: torch.Tensor,
        threshold: float = 0.5
    ) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph.
        
        Args:
            adj_matrix: Adjacency matrix
            node_probs: Node existence probabilities
            threshold: Threshold for edge existence
            
        Returns:
            nx.Graph: NetworkX graph
        """
        # Filter nodes based on existence probability
        node_mask = node_probs > threshold
        num_nodes = node_mask.sum().item()
        
        if num_nodes == 0:
            return nx.Graph()
        
        # Filter adjacency matrix
        adj_np = adj_matrix[node_mask][:, node_mask].cpu().numpy()
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_np[i, j] > threshold:
                    G.add_edge(i, j)
        
        return G
    
    def compute_graph_statistics(self, G: nx.Graph) -> Dict[str, float]:
        """Compute basic graph statistics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dict[str, float]: Graph statistics
        """
        if G.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "clustering_coefficient": 0.0,
                "avg_path_length": 0.0,
            }
        
        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": np.mean([d for n, d in G.degree()]),
            "clustering_coefficient": nx.average_clustering(G),
        }
        
        # Average path length (only for connected graphs)
        if nx.is_connected(G):
            stats["avg_path_length"] = nx.average_shortest_path_length(G)
        else:
            stats["avg_path_length"] = float("inf")
        
        return stats
    
    def compare_graphs(
        self, 
        G1: nx.Graph, 
        G2: nx.Graph
    ) -> Dict[str, float]:
        """Compare two graphs.
        
        Args:
            G1: First graph
            G2: Second graph
            
        Returns:
            Dict[str, float]: Comparison metrics
        """
        stats1 = self.compute_graph_statistics(G1)
        stats2 = self.compute_graph_statistics(G2)
        
        comparison = {}
        for key in stats1:
            if key in stats2:
                comparison[f"{key}_diff"] = abs(stats1[key] - stats2[key])
                comparison[f"{key}_ratio"] = stats1[key] / stats2[key] if stats2[key] != 0 else float("inf")
        
        return comparison
    
    def generate_and_evaluate(
        self,
        num_samples: int = 100,
        save_path: Optional[str] = None
    ) -> Dict[str, Union[float, List]]:
        """Generate graphs and evaluate them.
        
        Args:
            num_samples: Number of graphs to generate
            save_path: Path to save generated graphs
            
        Returns:
            Dict[str, Union[float, List]]: Evaluation results
        """
        results = {
            "generated_graphs": [],
            "statistics": [],
            "metrics": {},
        }
        
        with torch.no_grad():
            if isinstance(self.model, GraphVAE):
                for i in range(num_samples):
                    # Sample from prior
                    z = torch.randn(1, self.model.latent_dim, device=self.device)
                    
                    # Generate graph
                    node_features, adj_matrix, node_probs = self.model.decode(z)
                    
                    # Convert to NetworkX graph
                    G = self._adj_matrix_to_graph(adj_matrix, node_probs)
                    
                    # Compute statistics
                    stats = self.compute_graph_statistics(G)
                    
                    results["generated_graphs"].append(G)
                    results["statistics"].append(stats)
                
                # Compute aggregate metrics
                densities = [s["density"] for s in results["statistics"]]
                clusterings = [s["clustering_coefficient"] for s in results["statistics"]]
                degrees = [s["avg_degree"] for s in results["statistics"]]
                
                results["metrics"] = {
                    "density_mean": np.mean(densities),
                    "density_std": np.std(densities),
                    "clustering_mean": np.mean(clusterings),
                    "clustering_std": np.std(clusterings),
                    "degree_mean": np.mean(degrees),
                    "degree_std": np.std(degrees),
                }
                
                # Save if requested
                if save_path:
                    self._save_generated_graphs(results["generated_graphs"], save_path)
            
            else:
                raise NotImplementedError(f"Graph generation not implemented for {type(self.model)}")
        
        return results
    
    def _save_generated_graphs(
        self, 
        graphs: List[nx.Graph], 
        save_path: str
    ) -> None:
        """Save generated graphs to file.
        
        Args:
            graphs: List of generated graphs
            save_path: Path to save graphs
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as GraphML files
        for i, G in enumerate(graphs):
            nx.write_graphml(G, f"{save_path}_graph_{i}.graphml")
        
        # Save statistics
        stats = [self.compute_graph_statistics(G) for G in graphs]
        np.save(f"{save_path}_statistics.npy", stats)
    
    def create_evaluation_report(
        self,
        num_samples: int = 100,
        save_path: Optional[str] = None
    ) -> str:
        """Create a comprehensive evaluation report.
        
        Args:
            num_samples: Number of samples for evaluation
            save_path: Path to save the report
            
        Returns:
            str: Evaluation report
        """
        report = []
        report.append("=" * 50)
        report.append("GRAPH GENERATIVE MODEL EVALUATION REPORT")
        report.append("=" * 50)
        
        # Link prediction evaluation
        if hasattr(self.data, 'test_pos_edge_index'):
            lp_results = self.evaluate_link_prediction()
            report.append("\nLINK PREDICTION EVALUATION:")
            report.append("-" * 30)
            for metric, value in lp_results.items():
                report.append(f"{metric.upper()}: {value:.4f}")
        
        # Graph generation evaluation
        gen_results = self.generate_and_evaluate(num_samples)
        report.append("\nGRAPH GENERATION EVALUATION:")
        report.append("-" * 30)
        
        for metric, value in gen_results["metrics"].items():
            report.append(f"{metric.upper()}: {value:.4f}")
        
        # Sample statistics
        report.append("\nSAMPLE STATISTICS:")
        report.append("-" * 20)
        sample_stats = gen_results["statistics"][:5]  # Show first 5 samples
        for i, stats in enumerate(sample_stats):
            report.append(f"\nSample {i + 1}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
        
        report_text = "\n".join(report)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_text)
        
        return report_text
