"""
Hierarchical Abstraction module for ICA Framework
Implements motif discovery, embedding generation, clustering, and utility adjustment
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import itertools
from ..utils.config import AbstractionConfig
from ..utils.logger import ica_logger


class MotifDiscovery:
    """Motif discovery using frequent subgraph mining"""
    
    def __init__(self, config: AbstractionConfig):
        self.config = config
        self.logger = ica_logger
        self.discovered_motifs = []
        self.motif_frequencies = {}
    
    def discover_motifs(self, graph: nx.Graph, min_support: int = 2) -> List[Dict]:
        """
        Discover frequent motifs in the graph using a simplified GSA-like algorithm
        
        Args:
            graph: Input graph
            min_support: Minimum frequency threshold
            
        Returns:
            List of discovered motifs with their properties
        """
        
        motifs = []
        
        # Start with single nodes
        node_candidates = list(graph.nodes())
        
        # Iteratively grow motifs
        for size in range(1, self.config.motif_max_size + 1):
            if size == 1:
                # Single node motifs
                for node in node_candidates:
                    motif = {
                        'nodes': [node],
                        'edges': [],
                        'size': 1,
                        'frequency': 1,
                        'graph': graph.subgraph([node]).copy()
                    }
                    motifs.append(motif)
            else:
                # Multi-node motifs
                new_motifs = self._grow_motifs(graph, motifs, size, min_support)
                motifs.extend(new_motifs)
        
        # Filter by minimum size
        filtered_motifs = [m for m in motifs if m['size'] >= self.config.motif_min_size]
        
        # Calculate frequencies
        self._calculate_frequencies(graph, filtered_motifs)
        
        # Sort by frequency and size
        filtered_motifs.sort(key=lambda x: (x['frequency'], x['size']), reverse=True)
        
        self.discovered_motifs = filtered_motifs
        self.logger.info(f"Discovered {len(filtered_motifs)} motifs")
        
        return filtered_motifs
    
    def _grow_motifs(self, graph: nx.Graph, current_motifs: List[Dict], 
                    target_size: int, min_support: int) -> List[Dict]:
        """Grow motifs by adding connected nodes"""
        
        new_motifs = []
        
        # Group motifs by size
        size_motifs = [m for m in current_motifs if m['size'] == target_size - 1]
        
        for motif in size_motifs:
            # Find candidate nodes to extend the motif
            motif_nodes = set(motif['nodes'])
            candidates = set()
            
            for node in motif_nodes:
                neighbors = set(graph.neighbors(node))
                candidates.update(neighbors - motif_nodes)
            
            # Try extending with each candidate
            for candidate in candidates:
                extended_nodes = motif['nodes'] + [candidate]
                extended_subgraph = graph.subgraph(extended_nodes)
                
                # Check if the extended motif is connected (handle both directed and undirected)
                if graph.is_directed():
                    # For directed graphs, check if weakly connected
                    is_connected = nx.is_weakly_connected(extended_subgraph)
                else:
                    # For undirected graphs, use regular connectivity
                    is_connected = nx.is_connected(extended_subgraph)
                
                if is_connected:
                    extended_motif = {
                        'nodes': extended_nodes,
                        'edges': list(extended_subgraph.edges()),
                        'size': len(extended_nodes),
                        'frequency': 1,  # Will be calculated later
                        'graph': extended_subgraph.copy()
                    }
                    new_motifs.append(extended_motif)
        
        # Remove duplicates based on graph isomorphism (simplified)
        unique_motifs = self._remove_duplicate_motifs(new_motifs)
        
        return unique_motifs
    
    def _remove_duplicate_motifs(self, motifs: List[Dict]) -> List[Dict]:
        """Remove duplicate motifs based on structural similarity"""
        
        unique_motifs = []
        
        for motif in motifs:
            is_duplicate = False
            motif_graph = motif['graph']
            
            for unique_motif in unique_motifs:
                unique_graph = unique_motif['graph']
                
                # Simple structural comparison
                if (motif_graph.number_of_nodes() == unique_graph.number_of_nodes() and
                    motif_graph.number_of_edges() == unique_graph.number_of_edges()):
                    
                    # Check degree sequence
                    motif_degrees = sorted([d for n, d in motif_graph.degree()])
                    unique_degrees = sorted([d for n, d in unique_graph.degree()])
                    
                    if motif_degrees == unique_degrees:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_motifs.append(motif)
        
        return unique_motifs
    
    def _calculate_frequencies(self, graph: nx.Graph, motifs: List[Dict]):
        """Calculate frequency of each motif in the graph"""
        
        for motif in motifs:
            motif_graph = motif['graph']
            frequency = self._count_subgraph_occurrences(graph, motif_graph)
            motif['frequency'] = frequency
            self.motif_frequencies[str(motif['nodes'])] = frequency
    
    def _count_subgraph_occurrences(self, graph: nx.Graph, motif_graph: nx.Graph) -> int:
        """Count occurrences of motif in the main graph"""
        
        count = 0
        motif_nodes = list(motif_graph.nodes())
        
        # Generate all possible node combinations
        for node_combination in itertools.combinations(graph.nodes(), len(motif_nodes)):
            subgraph = graph.subgraph(node_combination)
            
            # Simple structural comparison with proper connectivity check
            # Check connectivity based on graph type
            if graph.is_directed():
                subgraph_connected = nx.is_weakly_connected(subgraph)
                motif_connected = nx.is_weakly_connected(motif_graph)
            else:
                subgraph_connected = nx.is_connected(subgraph)
                motif_connected = nx.is_connected(motif_graph)
            
            if (subgraph.number_of_edges() == motif_graph.number_of_edges() and
                subgraph_connected == motif_connected):
                count += 1
        
        return count


class GraphEmbedding:
    """Graph embedding generation for motifs"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.logger = ica_logger
        self.embeddings = {}
    
    def generate_embeddings(self, motifs: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate embeddings for motifs using graph neural networks"""
        
        embeddings = {}
        
        for i, motif in enumerate(motifs):
            # Create a simple feature representation
            motif_features = self._extract_motif_features(motif)
            
            # Generate embedding (simplified - in practice use pre-trained GNN)
            embedding = self._simple_embedding(motif_features)
            
            motif_id = str(motif['nodes'])
            embeddings[motif_id] = embedding
            
            self.logger.debug(f"Generated embedding for motif {motif_id}")
        
        self.embeddings = embeddings
        return embeddings
    
    def _extract_motif_features(self, motif: Dict) -> np.ndarray:
        """Extract structural features from motif"""
        
        graph = motif['graph']
        
        features = [
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(motif['nodes']),
            motif['frequency'],
            nx.density(graph) if graph.number_of_nodes() > 1 else 0,
            np.mean([d for n, d in graph.degree()]) if graph.number_of_nodes() > 0 else 0,
            nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0
        ]
        
        # Pad or truncate to fixed size
        target_size = 20
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _simple_embedding(self, features: np.ndarray) -> np.ndarray:
        """Generate simple embedding (replace with actual GNN in production)"""
        
        # Simple linear transformation with random weights
        np.random.seed(42)  # For reproducibility
        W = np.random.normal(0, 0.1, (len(features), self.embedding_dim))
        
        embedding = np.dot(features, W)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding


class ConceptInduction:
    """Concept induction through clustering of motif embeddings"""
    
    def __init__(self, config: AbstractionConfig):
        self.config = config
        self.logger = ica_logger
        self.concepts = {}
        self.cluster_labels = {}
        self.silhouette_scores = {}
    
    def induce_concepts(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Induce concepts by clustering motif embeddings"""
        
        if not embeddings:
            return {}
        
        # Prepare data for clustering
        motif_ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[mid] for mid in motif_ids])
        
        # Perform clustering
        if self.config.clustering_algorithm == "kmeans":
            clusterer = KMeans(n_clusters=min(self.config.num_clusters, len(motif_ids)), 
                              random_state=42, n_init=10)
        elif self.config.clustering_algorithm == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.config.clustering_algorithm}")
        
        labels = clusterer.fit_predict(embedding_matrix)
        
        # Calculate silhouette score
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(embedding_matrix, labels)
            self.silhouette_scores['overall'] = silhouette_avg
        else:
            self.silhouette_scores['overall'] = 0.0
        
        # Group motifs by cluster
        concepts = defaultdict(list)
        for motif_id, label in zip(motif_ids, labels):
            concepts[f"concept_{label}"].append(motif_id)
        
        # Create concept objects
        concept_objects = {}
        for concept_name, motif_list in concepts.items():
            concept_objects[concept_name] = {
                'motifs': motif_list,
                'size': len(motif_list),
                'centroid': self._calculate_centroid(motif_list, embeddings),
                'utility_score': 1.0,  # Initial utility score
                'creation_time': 0,
                'last_update': 0
            }
        
        self.concepts = concept_objects
        self.cluster_labels = dict(zip(motif_ids, labels))
        
        self.logger.info(f"Induced {len(concept_objects)} concepts with avg silhouette score: {self.silhouette_scores['overall']:.3f}")
        
        return concept_objects
    
    def _calculate_centroid(self, motif_list: List[str], 
                          embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate centroid of motif embeddings"""
        
        if not motif_list:
            return np.zeros(128)  # Default embedding size
        
        motif_embeddings = [embeddings[mid] for mid in motif_list]
        centroid = np.mean(motif_embeddings, axis=0)
        
        return centroid


class UtilityAdjustment:
    """Utility score adjustment based on predictive performance"""
    
    def __init__(self, config: AbstractionConfig):
        self.config = config
        self.logger = ica_logger
        self.utility_history = defaultdict(list)
    
    def adjust_utility_scores(self, concepts: Dict[str, Dict], 
                            performance_feedback: Dict[str, float]) -> Dict[str, Dict]:
        """Adjust utility scores based on predictive model performance"""
        
        updated_concepts = concepts.copy()
        
        for concept_name, concept_data in updated_concepts.items():
            # Get performance feedback for this concept
            performance_change = performance_feedback.get(concept_name, 0.0)
            
            # Calculate utility adjustment
            current_utility = concept_data['utility_score']
            
            # Decay factor
            decay = self.config.utility_decay * (current_utility - 0.5)  # Decay towards 0.5
            
            # Performance-based adjustment
            performance_adjustment = performance_change * 0.1  # Scale adjustment
            
            # New utility score
            new_utility = current_utility - decay + performance_adjustment
            
            # Clamp to [0, 1] range
            new_utility = max(0.0, min(1.0, new_utility))
            
            # Update concept
            updated_concepts[concept_name]['utility_score'] = new_utility
            updated_concepts[concept_name]['last_update'] = updated_concepts[concept_name].get('last_update', 0) + 1
            
            # Track history
            self.utility_history[concept_name].append(new_utility)
            
            self.logger.debug(f"Updated utility for {concept_name}: {current_utility:.3f} -> {new_utility:.3f}")
        
        return updated_concepts
    
    def get_utility_convergence(self, concept_name: str, window_size: int = 10) -> float:
        """Calculate utility score convergence for a concept"""
        
        history = self.utility_history.get(concept_name, [])
        
        if len(history) < window_size:
            return 0.0
        
        recent_scores = history[-window_size:]
        variance = np.var(recent_scores)
        
        # Convergence score (lower variance = better convergence)
        convergence = 1.0 / (1.0 + variance)
        
        return convergence


class HierarchicalAbstraction:
    """
    Main Hierarchical Abstraction module combining all components
    """
    
    def __init__(self, config: AbstractionConfig):
        self.config = config
        self.logger = ica_logger
        
        # Initialize components
        self.motif_discovery = MotifDiscovery(config)
        self.graph_embedding = GraphEmbedding(config.embedding_dim if hasattr(config, 'embedding_dim') else 128)
        self.concept_induction = ConceptInduction(config)
        self.utility_adjustment = UtilityAdjustment(config)
        
        # State
        self.current_motifs = []
        self.current_embeddings = {}
        self.current_concepts = {}
        
        self.logger.info("Initialized Hierarchical Abstraction module")
    
    def process_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """Process graph through full abstraction pipeline"""
        
        # Step 1: Motif Discovery
        motifs = self.motif_discovery.discover_motifs(graph)
        self.current_motifs = motifs
        
        # Step 2: Generate Embeddings
        embeddings = self.graph_embedding.generate_embeddings(motifs)
        self.current_embeddings = embeddings
        
        # Step 3: Concept Induction
        concepts = self.concept_induction.induce_concepts(embeddings)
        self.current_concepts = concepts
        
        # Return comprehensive results
        results = {
            'motifs': motifs,
            'embeddings': embeddings,
            'concepts': concepts,
            'num_motifs': len(motifs),
            'num_concepts': len(concepts),
            'silhouette_score': self.concept_induction.silhouette_scores.get('overall', 0.0)
        }
        
        self.logger.info(f"Processed graph: {results['num_motifs']} motifs, {results['num_concepts']} concepts")
        
        return results
    
    def update_with_feedback(self, performance_feedback: Dict[str, float]) -> Dict[str, Dict]:
        """Update concepts based on performance feedback"""
        
        updated_concepts = self.utility_adjustment.adjust_utility_scores(
            self.current_concepts, performance_feedback
        )
        
        self.current_concepts = updated_concepts
        
        return updated_concepts
    
    def get_abstraction_metrics(self) -> Dict[str, float]:
        """Get comprehensive abstraction metrics"""
        
        metrics = {
            'num_motifs': len(self.current_motifs),
            'num_concepts': len(self.current_concepts),
            'avg_motif_size': np.mean([m['size'] for m in self.current_motifs]) if self.current_motifs else 0,
            'avg_concept_size': np.mean([c['size'] for c in self.current_concepts.values()]) if self.current_concepts else 0,
            'silhouette_score': self.concept_induction.silhouette_scores.get('overall', 0.0)
        }
        
        # Utility score statistics
        if self.current_concepts:
            utility_scores = [c['utility_score'] for c in self.current_concepts.values()]
            metrics.update({
                'avg_utility_score': np.mean(utility_scores),
                'utility_score_std': np.std(utility_scores),
                'min_utility_score': np.min(utility_scores),
                'max_utility_score': np.max(utility_scores)
            })
        
        # Convergence metrics
        convergence_scores = []
        for concept_name in self.current_concepts.keys():
            convergence = self.utility_adjustment.get_utility_convergence(concept_name)
            convergence_scores.append(convergence)
        
        if convergence_scores:
            metrics['avg_utility_convergence'] = np.mean(convergence_scores)
        
        return metrics
    
    def save_state(self, filepath: str):
        """Save abstraction state"""
        
        state = {
            'config': self.config.dict(),
            'current_motifs': self.current_motifs,
            'current_embeddings': {k: v.tolist() for k, v in self.current_embeddings.items()},
            'current_concepts': self.current_concepts,
            'utility_history': dict(self.utility_adjustment.utility_history),
            'silhouette_scores': self.concept_induction.silhouette_scores
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Saved abstraction state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load abstraction state"""
        
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_motifs = state['current_motifs']
        self.current_embeddings = {k: np.array(v) for k, v in state['current_embeddings'].items()}
        self.current_concepts = state['current_concepts']
        
        # Restore utility history
        for concept_name, history in state['utility_history'].items():
            self.utility_adjustment.utility_history[concept_name] = history
        
        self.concept_induction.silhouette_scores = state['silhouette_scores']
        
        self.logger.info(f"Loaded abstraction state from {filepath}")
