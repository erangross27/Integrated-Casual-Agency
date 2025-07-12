"""
Metrics and evaluation utilities for ICA Framework
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from scipy.stats import entropy
import networkx as nx


class Metrics:
    """Metrics calculation and tracking for ICA Framework"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for concept coherence"""
        if len(np.unique(labels)) < 2:
            return 0.0
        return silhouette_score(embeddings, labels)
    
    def calculate_motif_discovery_metrics(self, 
                                        discovered_motifs: List[Any], 
                                        ground_truth_motifs: List[Any]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for motif discovery"""
        # Convert motifs to comparable format (simplified)
        discovered_set = set(str(motif) for motif in discovered_motifs)
        truth_set = set(str(motif) for motif in ground_truth_motifs)
        
        if len(discovered_set) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        intersection = discovered_set.intersection(truth_set)
        
        precision = len(intersection) / len(discovered_set) if discovered_set else 0.0
        recall = len(intersection) / len(truth_set) if truth_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def calculate_prediction_error(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate prediction error for world model"""
        return torch.nn.functional.mse_loss(predictions, targets).item()
    
    def calculate_uncertainty(self, predictions: torch.Tensor, variance: torch.Tensor) -> float:
        """Calculate model uncertainty"""
        return torch.mean(variance).item()
    
    def calculate_global_confidence(self, graph: nx.Graph, edge_confidences: Dict) -> float:
        """Calculate global confidence metric for the knowledge graph"""
        if not edge_confidences:
            return 0.0
        
        # Calculate centrality-weighted confidence
        centrality = nx.betweenness_centrality(graph)
        total_confidence = 0.0
        total_weight = 0.0
        
        for edge, confidence in edge_confidences.items():
            if edge in graph.edges:
                # Weight by centrality of connected nodes
                node1, node2 = edge
                weight = (centrality.get(node1, 0) + centrality.get(node2, 0)) / 2
                total_confidence += confidence * weight
                total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def calculate_entropy(self, distribution: np.ndarray) -> float:
        """Calculate entropy of a probability distribution"""
        return entropy(distribution)
    
    def calculate_complexity_cost(self, model_before: int, model_after: int) -> float:
        """Calculate complexity cost based on model size change"""
        return abs(model_after - model_before) / max(model_before, 1)
    
    def calculate_utility_score_convergence(self, utility_scores: List[float], window_size: int = 10) -> float:
        """Calculate convergence metric for utility scores"""
        if len(utility_scores) < window_size:
            return 0.0
        
        recent_scores = utility_scores[-window_size:]
        variance = np.var(recent_scores)
        return 1.0 / (1.0 + variance)  # Higher value means better convergence
    
    def update_metrics(self, metric_name: str, value: float, step: int):
        """Update metrics history"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append({
            "step": step,
            "value": value
        })
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            values = [item["value"] for item in history]
            summary[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "latest": values[-1] if values else 0.0
            }
        
        return summary
    
    def calculate_performance_improvement(self, baseline_scores: List[float], 
                                        enhanced_scores: List[float]) -> Dict[str, float]:
        """Calculate performance improvement over baseline"""
        if not baseline_scores or not enhanced_scores:
            return {"improvement": 0.0, "relative_improvement": 0.0}
        
        baseline_mean = np.mean(baseline_scores)
        enhanced_mean = np.mean(enhanced_scores)
        
        improvement = enhanced_mean - baseline_mean
        relative_improvement = improvement / baseline_mean if baseline_mean != 0 else 0.0
        
        return {
            "improvement": improvement,
            "relative_improvement": relative_improvement
        }
