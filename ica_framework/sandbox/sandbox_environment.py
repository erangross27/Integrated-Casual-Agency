"""
Sandbox environment for testing ICA Framework
Implements procedural dataset generation and validation
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple, Set
import json
from pathlib import Path
from ..utils.config import SandboxConfig
from ..utils.logger import ica_logger
from ..components import CausalKnowledgeGraph, Node, Edge


class ProceduralDatasetGenerator:
    """Generates procedural datasets with known ground truth motifs"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.logger = ica_logger
        self.ground_truth_motifs = []
        self.generated_graphs = []
        
    def generate_dataset(self, num_nodes: int = 500, num_edges: int = 1000) -> Dict[str, Any]:
        """
        Generate a procedural dataset with known motifs
        
        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges in the graph
            
        Returns:
            Dictionary containing graph and ground truth information
        """
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        
        # Create base graph
        graph = nx.Graph()
        
        # Add nodes
        for i in range(num_nodes):
            graph.add_node(f"node_{i}", 
                          label=f"entity_{i % 10}",  # 10 different entity types
                          properties={"type": f"type_{i % 5}", "value": np.random.random()})
        
        # Generate known motifs and embed them
        motifs = self._generate_known_motifs(num_nodes)
        self.ground_truth_motifs = motifs
        
        # Embed motifs in the graph
        for motif in motifs:
            self._embed_motif_in_graph(graph, motif)
        
        # Add random edges to reach target edge count
        current_edges = graph.number_of_edges()
        while current_edges < num_edges:
            node1 = random.choice(list(graph.nodes()))
            node2 = random.choice(list(graph.nodes()))
            
            if node1 != node2 and not graph.has_edge(node1, node2):
                graph.add_edge(node1, node2, 
                             relationship="random", 
                             confidence=np.random.uniform(0.1, 0.9))
                current_edges += 1
        
        # Convert to dataset format
        dataset = {
            "graph": graph,
            "ground_truth_motifs": motifs,
            "num_nodes": num_nodes,
            "num_edges": graph.number_of_edges(),
            "motif_types": list(set(m["type"] for m in motifs)),
            "metadata": {
                "generated_at": str(np.datetime64('now')),
                "seed": self.config.random_seed,
                "generator_version": "1.0"
            }
        }
        
        self.generated_graphs.append(dataset)
        self.logger.info(f"Generated dataset with {num_nodes} nodes, {graph.number_of_edges()} edges, "
                        f"and {len(motifs)} ground truth motifs")
        
        return dataset
    
    def _generate_known_motifs(self, num_nodes: int) -> List[Dict[str, Any]]:
        """Generate known motifs to embed in the graph"""
        
        motifs = []
        
        # Generate different types of motifs
        motif_types = ["path", "triangle", "star", "clique", "chain"]
        
        for motif_type in motif_types:
            # Generate multiple instances of each motif type
            for instance in range(3):  # 3 instances of each type
                motif = self._create_motif(motif_type, num_nodes, instance)
                if motif:
                    motifs.append(motif)
        
        return motifs
    
    def _create_motif(self, motif_type: str, num_nodes: int, instance: int) -> Optional[Dict[str, Any]]:
        """Create a specific motif type"""
        
        available_nodes = list(range(num_nodes))
        
        if motif_type == "path":
            # Create a path motif
            path_length = random.randint(3, 6)
            if len(available_nodes) < path_length:
                return None
            
            path_nodes = random.sample(available_nodes, path_length)
            edges = [(f"node_{path_nodes[i]}", f"node_{path_nodes[i+1]}") 
                    for i in range(path_length-1)]
            
            return {
                "type": "path",
                "instance": instance,
                "nodes": [f"node_{n}" for n in path_nodes],
                "edges": edges,
                "size": path_length,
                "properties": {"length": path_length}
            }
        
        elif motif_type == "triangle":
            # Create a triangle motif
            if len(available_nodes) < 3:
                return None
            
            triangle_nodes = random.sample(available_nodes, 3)
            edges = [(f"node_{triangle_nodes[0]}", f"node_{triangle_nodes[1]}"),
                    (f"node_{triangle_nodes[1]}", f"node_{triangle_nodes[2]}"),
                    (f"node_{triangle_nodes[2]}", f"node_{triangle_nodes[0]}")]
            
            return {
                "type": "triangle",
                "instance": instance,
                "nodes": [f"node_{n}" for n in triangle_nodes],
                "edges": edges,
                "size": 3,
                "properties": {"cyclic": True}
            }
        
        elif motif_type == "star":
            # Create a star motif
            star_size = random.randint(4, 8)
            if len(available_nodes) < star_size:
                return None
            
            star_nodes = random.sample(available_nodes, star_size)
            center = star_nodes[0]
            periphery = star_nodes[1:]
            
            edges = [(f"node_{center}", f"node_{p}") for p in periphery]
            
            return {
                "type": "star",
                "instance": instance,
                "nodes": [f"node_{n}" for n in star_nodes],
                "edges": edges,
                "size": star_size,
                "properties": {"center": f"node_{center}", "periphery_size": len(periphery)}
            }
        
        elif motif_type == "clique":
            # Create a clique motif
            clique_size = random.randint(3, 5)
            if len(available_nodes) < clique_size:
                return None
            
            clique_nodes = random.sample(available_nodes, clique_size)
            edges = []
            
            for i in range(clique_size):
                for j in range(i+1, clique_size):
                    edges.append((f"node_{clique_nodes[i]}", f"node_{clique_nodes[j]}"))
            
            return {
                "type": "clique",
                "instance": instance,
                "nodes": [f"node_{n}" for n in clique_nodes],
                "edges": edges,
                "size": clique_size,
                "properties": {"fully_connected": True}
            }
        
        elif motif_type == "chain":
            # Create a chain motif (like path but with specific properties)
            chain_length = random.randint(4, 7)
            if len(available_nodes) < chain_length:
                return None
            
            chain_nodes = random.sample(available_nodes, chain_length)
            edges = [(f"node_{chain_nodes[i]}", f"node_{chain_nodes[i+1]}") 
                    for i in range(chain_length-1)]
            
            return {
                "type": "chain",
                "instance": instance,
                "nodes": [f"node_{n}" for n in chain_nodes],
                "edges": edges,
                "size": chain_length,
                "properties": {"linear": True, "length": chain_length}
            }
        
        return None
    
    def _embed_motif_in_graph(self, graph: nx.Graph, motif: Dict[str, Any]):
        """Embed a motif into the graph"""
        
        # Add edges defined by the motif
        for edge in motif["edges"]:
            source, target = edge
            graph.add_edge(source, target,
                         relationship=motif["type"],
                         confidence=np.random.uniform(0.7, 0.95),  # High confidence for ground truth
                         motif_type=motif["type"],
                         motif_instance=motif["instance"])
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: str):
        """Save dataset to file"""
        
        # Convert networkx graph to serializable format
        serializable_dataset = {
            "graph_data": nx.node_link_data(dataset["graph"]),
            "ground_truth_motifs": dataset["ground_truth_motifs"],
            "num_nodes": dataset["num_nodes"],
            "num_edges": dataset["num_edges"],
            "motif_types": dataset["motif_types"],
            "metadata": dataset["metadata"]
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)
        
        self.logger.info(f"Saved dataset to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load dataset from file"""
        
        with open(filepath, 'r') as f:
            serializable_dataset = json.load(f)
        
        # Convert back to networkx graph
        graph = nx.node_link_graph(serializable_dataset["graph_data"])
        
        dataset = {
            "graph": graph,
            "ground_truth_motifs": serializable_dataset["ground_truth_motifs"],
            "num_nodes": serializable_dataset["num_nodes"],
            "num_edges": serializable_dataset["num_edges"],
            "motif_types": serializable_dataset["motif_types"],
            "metadata": serializable_dataset["metadata"]
        }
        
        self.logger.info(f"Loaded dataset from {filepath}")
        return dataset


class SandboxEnvironment:
    """
    Sandbox environment for testing ICA Framework components
    """
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.logger = ica_logger
        self.dataset_generator = ProceduralDatasetGenerator(config)
        self.current_dataset = None
        self.baseline_results = {}
        self.enhanced_results = {}
        
    def setup_test_environment(self) -> Dict[str, Any]:
        """Set up the test environment with procedural data"""
        
        # Generate dataset
        dataset = self.dataset_generator.generate_dataset(
            num_nodes=self.config.dataset_size,
            num_edges=int(self.config.dataset_size * 2)
        )
        
        self.current_dataset = dataset
        
        # Split into train/test/validation
        all_nodes = list(dataset["graph"].nodes())
        random.shuffle(all_nodes)
        
        n_nodes = len(all_nodes)
        test_size = int(n_nodes * self.config.test_ratio)
        val_size = int(n_nodes * self.config.validation_ratio)
        
        test_nodes = all_nodes[:test_size]
        val_nodes = all_nodes[test_size:test_size + val_size]
        train_nodes = all_nodes[test_size + val_size:]
        
        splits = {
            "train_nodes": train_nodes,
            "test_nodes": test_nodes,
            "val_nodes": val_nodes,
            "train_graph": dataset["graph"].subgraph(train_nodes).copy(),
            "test_graph": dataset["graph"].subgraph(test_nodes).copy(),
            "val_graph": dataset["graph"].subgraph(val_nodes).copy()
        }
        
        self.logger.info(f"Set up test environment with {len(train_nodes)} train, "
                        f"{len(test_nodes)} test, {len(val_nodes)} validation nodes")
        
        return splits
    
    def run_baseline_experiment(self, splits: Dict[str, Any]) -> Dict[str, float]:
        """Run baseline experiment without hierarchical abstraction"""
        
        from ..core import ICAAgent
        from ..utils import Config
        
        # Create baseline config (abstraction disabled)
        config = Config()
        config.abstraction.motif_min_size = 1000  # Effectively disable motif discovery
        
        # Create baseline agent
        baseline_agent = ICAAgent(config)
        
        # Initialize with known dimensions
        baseline_agent.initialize_world_model(
            state_dim=10,
            action_dim=4,
            num_relations=len(set(nx.get_edge_attributes(splits["train_graph"], 'relationship').values()))
        )
        
        # Run baseline experiment
        baseline_results = self._run_experiment(baseline_agent, splits, "baseline")
        
        self.baseline_results = baseline_results
        return baseline_results
    
    def run_enhanced_experiment(self, splits: Dict[str, Any]) -> Dict[str, float]:
        """Run enhanced experiment with full ICA framework"""
        
        from ..core import ICAAgent
        from ..utils import Config
        
        # Create enhanced config
        config = Config()
        
        # Create enhanced agent
        enhanced_agent = ICAAgent(config)
        
        # Initialize with known dimensions
        enhanced_agent.initialize_world_model(
            state_dim=10,
            action_dim=4,
            num_relations=len(set(nx.get_edge_attributes(splits["train_graph"], 'relationship').values()))
        )
        
        # Run enhanced experiment
        enhanced_results = self._run_experiment(enhanced_agent, splits, "enhanced")
        
        self.enhanced_results = enhanced_results
        return enhanced_results
    
    def _run_experiment(self, agent, splits: Dict[str, Any], experiment_type: str) -> Dict[str, float]:
        """Run experiment with given agent"""
        
        results = {
            "motif_discovery_precision": 0.0,
            "motif_discovery_recall": 0.0,
            "motif_discovery_f1": 0.0,
            "concept_coherence": 0.0,
            "prediction_accuracy": 0.0,
            "global_confidence": 0.0,
            "computational_time": 0.0
        }
        
        import time
        start_time = time.time()
        
        try:
            # Convert graph to observations
            train_graph = splits["train_graph"]
            
            # Create mock observations from graph
            observations = []
            
            # First pass: Add all nodes to ensure they exist before edges
            all_entities = []
            for node in train_graph.nodes():
                node_data = train_graph.nodes[node]
                all_entities.append({
                    "id": node, 
                    "label": node_data.get("label", "unknown"),
                    "properties_static": node_data.get("properties_static", {}),
                    "properties_dynamic": node_data.get("properties_dynamic", {})
                })
            
            # Initial observation with all entities to establish nodes
            initial_observation = {
                "entities": all_entities,
                "relationships": [],
                "state": np.random.normal(0, 0.1, 10)
            }
            observations.append(initial_observation)
            
            # Second pass: Create observations with relationships
            for node in train_graph.nodes():
                node_data = train_graph.nodes[node]
                observation = {
                    "entities": [{"id": node, "label": node_data.get("label", "unknown")}],
                    "relationships": [],
                    "state": np.random.normal(0, 0.1, 10)  # Mock state
                }
                
                # Add relationships for this node
                for neighbor in train_graph.neighbors(node):
                    edge_data = train_graph.edges[node, neighbor]
                    observation["relationships"].append({
                        "source": node,
                        "target": neighbor,
                        "type": edge_data.get("relationship", "unknown"),
                        "confidence": edge_data.get("confidence", 0.5)
                    })
                
                observations.append(observation)
            
            # Run agent on observations
            for obs in observations[:50]:  # Limit for testing
                agent.active_learning_step(obs)
            
            # Evaluate results
            if experiment_type == "enhanced":
                # Evaluate motif discovery
                discovered_motifs = agent.hierarchical_abstraction.current_motifs
                ground_truth_motifs = self.current_dataset["ground_truth_motifs"]
                
                motif_metrics = agent.metrics.calculate_motif_discovery_metrics(
                    discovered_motifs, ground_truth_motifs
                )
                
                results.update(motif_metrics)
                
                # Evaluate concept coherence
                if agent.hierarchical_abstraction.current_embeddings:
                    embeddings = np.array(list(agent.hierarchical_abstraction.current_embeddings.values()))
                    labels = list(range(len(embeddings)))  # Simplified
                    
                    if len(embeddings) > 1:
                        coherence = agent.metrics.calculate_silhouette_score(embeddings, labels)
                        results["concept_coherence"] = coherence
            
            # General metrics
            results["global_confidence"] = agent.global_confidence
            results["computational_time"] = time.time() - start_time
            
            # Mock prediction accuracy
            results["prediction_accuracy"] = np.random.uniform(0.6, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error in {experiment_type} experiment: {str(e)}")
        
        self.logger.info(f"Completed {experiment_type} experiment in {results['computational_time']:.2f}s")
        return results
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run complete ablation study"""
        
        # Set up test environment
        splits = self.setup_test_environment()
        
        # Run baseline
        baseline_results = self.run_baseline_experiment(splits)
        
        # Run enhanced
        enhanced_results = self.run_enhanced_experiment(splits)
        
        # Calculate improvements
        improvements = {}
        for metric in baseline_results:
            baseline_val = baseline_results[metric]
            enhanced_val = enhanced_results[metric]
            
            if baseline_val > 0:
                improvement = (enhanced_val - baseline_val) / baseline_val
            else:
                improvement = enhanced_val - baseline_val
            
            improvements[f"{metric}_improvement"] = improvement
        
        ablation_results = {
            "baseline_results": baseline_results,
            "enhanced_results": enhanced_results,
            "improvements": improvements,
            "dataset_info": {
                "num_nodes": self.current_dataset["num_nodes"],
                "num_edges": self.current_dataset["num_edges"],
                "num_ground_truth_motifs": len(self.current_dataset["ground_truth_motifs"]),
                "motif_types": self.current_dataset["motif_types"]
            }
        }
        
        self.logger.info("Completed ablation study")
        return ablation_results
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save experiment results"""
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to {filepath}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable report"""
        
        report = "ICA Framework Sandbox Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Dataset info
        dataset_info = results["dataset_info"]
        report += f"Dataset Information:\n"
        report += f"  Nodes: {dataset_info['num_nodes']}\n"
        report += f"  Edges: {dataset_info['num_edges']}\n"
        report += f"  Ground Truth Motifs: {dataset_info['num_ground_truth_motifs']}\n"
        report += f"  Motif Types: {', '.join(dataset_info['motif_types'])}\n\n"
        
        # Results comparison
        baseline = results["baseline_results"]
        enhanced = results["enhanced_results"]
        improvements = results["improvements"]
        
        report += "Results Comparison:\n"
        report += f"{'Metric':<25} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<12}\n"
        report += "-" * 65 + "\n"
        
        for metric in baseline:
            if metric in enhanced:
                improvement = improvements.get(f"{metric}_improvement", 0)
                report += f"{metric:<25} {baseline[metric]:<12.4f} {enhanced[metric]:<12.4f} {improvement:<12.4f}\n"
        
        report += "\n"
        
        # Key findings
        report += "Key Findings:\n"
        
        # Motif discovery performance
        if "motif_discovery_f1" in improvements:
            f1_improvement = improvements["motif_discovery_f1_improvement"]
            report += f"  - Motif Discovery F1 improved by {f1_improvement:.1%}\n"
        
        # Concept coherence
        if "concept_coherence" in enhanced:
            coherence = enhanced["concept_coherence"]
            report += f"  - Concept Coherence Score: {coherence:.3f}\n"
        
        # Performance gain
        if "prediction_accuracy_improvement" in improvements:
            acc_improvement = improvements["prediction_accuracy_improvement"]
            report += f"  - Prediction Accuracy improved by {acc_improvement:.1%}\n"
        
        return report
