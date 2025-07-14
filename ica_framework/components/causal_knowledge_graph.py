"""
Causal Knowledge Graph implementation for ICA Framework
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import beta
import json
import pickle
from dataclasses import dataclass
from ..utils.config import GraphConfig
from ..utils.logger import ica_logger


@dataclass
class Node:
    """Node in the Causal Knowledge Graph"""
    id: str
    label: str
    properties_static: Dict[str, Any]
    properties_dynamic: Dict[str, Any]
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "properties_static": self.properties_static,
            "properties_dynamic": self.properties_dynamic,
            "confidence": self.confidence
        }


@dataclass
class Edge:
    """Edge in the Causal Knowledge Graph"""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]
    confidence: float
    weight: float = 1.0
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "properties": self.properties,
            "confidence": self.confidence,
            "weight": self.weight,
            "conditions": self.conditions
        }


class CausalKnowledgeGraph:
    """
    Causal Knowledge Graph implementation as a directed, labeled, property multi-graph
    G = (V, E) where V are nodes representing entities and E are edges representing
    causal relationships with confidence modeled using Beta distributions.
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.nodes_dict = {}  # id -> Node
        self.edges_dict = {}  # (source, target, key) -> Edge
        self.confidence_distributions = {}  # edge_id -> Beta distribution parameters
        self.logger = ica_logger
        
        # Initialize beta priors
        self.beta_alpha = config.beta_prior_alpha
        self.beta_beta = config.beta_prior_beta
        
    
    def add_node(self, node: Node) -> bool:
        """Add a node to the graph"""
        try:
            if node.id in self.nodes_dict:
                self.logger.debug(f"Node {node.id} already exists, updating...")
                return self.update_node(node)
            
            self.nodes_dict[node.id] = node
            self.graph.add_node(node.id, **node.to_dict())
            
            self.logger.debug(f"Added node {node.id} with label {node.label} (total nodes: {self.graph.number_of_nodes()})")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding node {node.id}: {str(e)}")
            return False
    
    def add_edge(self, edge: Edge) -> bool:
        """Add an edge to the graph"""
        try:
            if edge.source not in self.nodes_dict:
                self.logger.warning(f"Source node {edge.source} not found (available nodes: {list(self.nodes_dict.keys())[:10]}...)")
                return False
            
            if edge.target not in self.nodes_dict:
                self.logger.warning(f"Target node {edge.target} not found (available nodes: {list(self.nodes_dict.keys())[:10]}...)")
                return False
            
            # Add edge to networkx graph
            key = self.graph.add_edge(edge.source, edge.target, **edge.to_dict())
            
            # Store edge object
            edge_id = (edge.source, edge.target, key)
            self.edges_dict[edge_id] = edge
            
            # Initialize confidence distribution
            self.confidence_distributions[edge_id] = {
                'alpha': self.beta_alpha,
                'beta': self.beta_beta
            }
            
            self.logger.debug(f"Added edge {edge.source} -> {edge.target} with relationship {edge.relationship} (total edges: {self.graph.number_of_edges()})")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding edge {edge.source} -> {edge.target}: {str(e)}")
            return False
    
    def update_node(self, node: Node) -> bool:
        """Update an existing node"""
        try:
            if node.id not in self.nodes_dict:
                self.logger.warning(f"Node {node.id} not found for update")
                return False
            
            self.nodes_dict[node.id] = node
            self.graph.nodes[node.id].update(node.to_dict())
            
            self.logger.debug(f"Updated node {node.id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating node {node.id}: {str(e)}")
            return False
    
    def update_edge_confidence(self, edge_id: Tuple[str, str, int], 
                              observation: bool, weight: float = 1.0) -> bool:
        """Update edge confidence using Bayesian inference"""
        try:
            if edge_id not in self.confidence_distributions:
                self.logger.warning(f"Edge {edge_id} not found for confidence update")
                return False
            
            # Update Beta distribution parameters
            if observation:
                self.confidence_distributions[edge_id]['alpha'] += weight
            else:
                self.confidence_distributions[edge_id]['beta'] += weight
            
            # Update edge confidence (Beta distribution mean)
            alpha = self.confidence_distributions[edge_id]['alpha']
            beta_param = self.confidence_distributions[edge_id]['beta']
            new_confidence = alpha / (alpha + beta_param)
            
            # Update the edge object
            if edge_id in self.edges_dict:
                self.edges_dict[edge_id].confidence = new_confidence
                self.graph.edges[edge_id]['confidence'] = new_confidence
            
            self.logger.debug(f"Updated edge {edge_id} confidence to {new_confidence:.3f}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating edge confidence {edge_id}: {str(e)}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        return self.nodes_dict.get(node_id)
    
    def get_edge(self, edge_id: Tuple[str, str, int]) -> Optional[Edge]:
        """Get an edge by ID"""
        return self.edges_dict.get(edge_id)
    
    def get_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[str]:
        """Get neighbors of a node, optionally filtered by relationship"""
        try:
            neighbors = []
            for neighbor in self.graph.neighbors(node_id):
                if relationship is None:
                    neighbors.append(neighbor)
                else:
                    # Check if any edge has the specified relationship
                    for edge_data in self.graph[node_id][neighbor].values():
                        if edge_data.get('relationship') == relationship:
                            neighbors.append(neighbor)
                            break
            return neighbors
        
        except Exception as e:
            self.logger.error(f"Error getting neighbors for {node_id}: {str(e)}")
            return []
    
    def get_subgraph(self, nodes: List[str]) -> nx.MultiDiGraph:
        """Get a subgraph containing specified nodes"""
        try:
            return self.graph.subgraph(nodes).copy()
        except Exception as e:
            self.logger.error(f"Error creating subgraph: {str(e)}")
            return nx.MultiDiGraph()
    
    def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """Find all paths between two nodes"""
        try:
            paths = []
            for path in nx.all_simple_paths(self.graph, source, target, cutoff=max_length):
                paths.append(path)
            return paths
        
        except Exception as e:
            self.logger.error(f"Error finding paths from {source} to {target}: {str(e)}")
            return []
    
    def calculate_centrality(self, centrality_type: str = "betweenness") -> Dict[str, float]:
        """Calculate centrality measures for nodes"""
        try:
            if centrality_type == "betweenness":
                return nx.betweenness_centrality(self.graph)
            elif centrality_type == "closeness":
                return nx.closeness_centrality(self.graph)
            elif centrality_type == "degree":
                return nx.degree_centrality(self.graph)
            elif centrality_type == "pagerank":
                return nx.pagerank(self.graph)
            else:
                self.logger.warning(f"Unknown centrality type: {centrality_type}")
                return {}
        
        except Exception as e:
            self.logger.error(f"Error calculating centrality: {str(e)}")
            return {}
    
    def get_confidence_stats(self) -> Dict[str, float]:
        """Get statistics about edge confidences"""
        try:
            confidences = [edge.confidence for edge in self.edges_dict.values()]
            
            if not confidences:
                return {}
            
            return {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "median": np.median(confidences)
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating confidence stats: {str(e)}")
            return {}
    
    def prune_low_confidence_edges(self, threshold: Optional[float] = None) -> int:
        """Remove edges with confidence below threshold"""
        try:
            if threshold is None:
                threshold = self.config.confidence_threshold
            
            edges_to_remove = []
            for edge_id, edge in self.edges_dict.items():
                if edge.confidence < threshold:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                source, target, key = edge_id
                self.graph.remove_edge(source, target, key)
                del self.edges_dict[edge_id]
                del self.confidence_distributions[edge_id]
            
            self.logger.info(f"Pruned {len(edges_to_remove)} edges with confidence < {threshold}")
            return len(edges_to_remove)
        
        except Exception as e:
            self.logger.error(f"Error pruning edges: {str(e)}")
            return 0
    
    def save(self, filepath: str) -> bool:
        """Save the graph to file"""
        try:
            data = {
                "graph": nx.node_link_data(self.graph),
                "nodes_dict": {k: v.to_dict() for k, v in self.nodes_dict.items()},
                "edges_dict": {str(k): v.to_dict() for k, v in self.edges_dict.items()},
                "confidence_distributions": {str(k): v for k, v in self.confidence_distributions.items()},
                "config": self.config.dict()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved graph to {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving graph: {str(e)}")
            return False
    
    def load(self, filepath: str) -> bool:
        """Load the graph from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore graph
            self.graph = nx.node_link_graph(data["graph"], directed=True, multigraph=True)
            
            # Restore nodes
            self.nodes_dict = {}
            for node_id, node_data in data["nodes_dict"].items():
                self.nodes_dict[node_id] = Node(**node_data)
            
            # Restore edges
            self.edges_dict = {}
            for edge_id_str, edge_data in data["edges_dict"].items():
                edge_id = eval(edge_id_str)  # Convert string back to tuple
                self.edges_dict[edge_id] = Edge(**edge_data)
            
            # Restore confidence distributions
            self.confidence_distributions = {}
            for edge_id_str, dist_data in data["confidence_distributions"].items():
                edge_id = eval(edge_id_str)
                self.confidence_distributions[edge_id] = dist_data
            
            self.logger.info(f"Loaded graph from {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading graph: {str(e)}")
            return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            stats = {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "density": self._safe_density(),
                "is_connected": self._safe_connectivity(),
                "confidence_stats": self.get_confidence_stats()
            }
            
            # Add centrality statistics
            try:
                centrality = self.calculate_centrality("betweenness")
                if centrality:
                    centrality_values = list(centrality.values())
                    stats["centrality_stats"] = {
                        "mean": np.mean(centrality_values),
                        "std": np.std(centrality_values),
                        "max": np.max(centrality_values)
                    }
            except:
                pass
            
            return stats
        
        except Exception as e:
            self.logger.error(f"Error calculating graph stats: {str(e)}")
            return {}
    
    def _safe_density(self):
        """Calculate density safely for MultiDiGraph"""
        try:
            return nx.density(self.graph)
        except (nx.NetworkXError, NotImplementedError):
            # Fallback calculation for MultiDiGraph
            n = self.graph.number_of_nodes()
            if n <= 1:
                return 0
            m = self.graph.number_of_edges()
            max_edges = n * (n - 1)  # Directed graph
            return m / max_edges if max_edges > 0 else 0
    
    def _safe_connectivity(self):
        """Check connectivity safely for MultiDiGraph"""
        try:
            return nx.is_weakly_connected(self.graph)
        except (nx.NetworkXError, NotImplementedError):
            # Fallback for MultiDiGraph
            return self.graph.number_of_nodes() > 0
