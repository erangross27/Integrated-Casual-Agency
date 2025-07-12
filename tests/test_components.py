"""
Test script for ICA Framework components
"""

import pytest
import numpy as np
import networkx as nx
from pathlib import Path
import sys

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ica_framework.components import CausalKnowledgeGraph, Node, Edge
from ica_framework.utils import Config
from ica_framework.sandbox import ProceduralDatasetGenerator


def test_causal_knowledge_graph():
    """Test CausalKnowledgeGraph functionality"""
    
    print("Testing CausalKnowledgeGraph...")
    
    # Create config
    config = Config()
    
    # Create knowledge graph
    kg = CausalKnowledgeGraph(config.graph)
    
    # Test node addition
    node1 = Node(
        id="test_node_1",
        label="test_entity",
        properties_static={"type": "test"},
        properties_dynamic={"value": 1.0}
    )
    
    node2 = Node(
        id="test_node_2",
        label="test_entity",
        properties_static={"type": "test"},
        properties_dynamic={"value": 2.0}
    )
    
    assert kg.add_node(node1), "Failed to add node 1"
    assert kg.add_node(node2), "Failed to add node 2"
    
    # Test edge addition
    edge = Edge(
        source="test_node_1",
        target="test_node_2",
        relationship="test_relationship",
        properties={"strength": 0.8},
        confidence=0.7
    )
    
    assert kg.add_edge(edge), "Failed to add edge"
    
    # Test graph statistics
    stats = kg.get_graph_stats()
    assert stats["num_nodes"] == 2, f"Expected 2 nodes, got {stats['num_nodes']}"
    assert stats["num_edges"] == 1, f"Expected 1 edge, got {stats['num_edges']}"
    
    # Test confidence update
    edge_id = ("test_node_1", "test_node_2", 0)
    assert kg.update_edge_confidence(edge_id, True), "Failed to update edge confidence"
    
    print("✓ CausalKnowledgeGraph tests passed")


def test_procedural_dataset_generator():
    """Test ProceduralDatasetGenerator"""
    
    print("Testing ProceduralDatasetGenerator...")
    
    # Create config
    config = Config()
    
    # Create generator
    generator = ProceduralDatasetGenerator(config.sandbox)
    
    # Generate dataset
    dataset = generator.generate_dataset(num_nodes=50, num_edges=100)
    
    # Test dataset properties
    assert "graph" in dataset, "Dataset missing graph"
    assert "ground_truth_motifs" in dataset, "Dataset missing ground truth motifs"
    assert dataset["num_nodes"] == 50, f"Expected 50 nodes, got {dataset['num_nodes']}"
    assert len(dataset["ground_truth_motifs"]) > 0, "No ground truth motifs generated"
    
    # Test graph properties
    graph = dataset["graph"]
    assert isinstance(graph, nx.Graph), "Graph is not a NetworkX graph"
    assert graph.number_of_nodes() == 50, f"Graph has {graph.number_of_nodes()} nodes, expected 50"
    
    # Test motif types
    motif_types = set(m["type"] for m in dataset["ground_truth_motifs"])
    expected_types = {"path", "triangle", "star", "clique", "chain"}
    assert motif_types.issubset(expected_types), f"Unexpected motif types: {motif_types - expected_types}"
    
    print("✓ ProceduralDatasetGenerator tests passed")


def test_config():
    """Test configuration management"""
    
    print("Testing Config...")
    
    # Create default config
    config = Config()
    
    # Test default values
    assert config.graph.initial_nodes == 500, "Default graph nodes incorrect"
    assert config.world_model.embedding_dim == 128, "Default embedding dim incorrect"
    assert config.curiosity.complexity_weight == 0.1, "Default complexity weight incorrect"
    assert config.planner.lr_actor == 3e-4, "Default actor LR incorrect"
    assert config.abstraction.motif_min_size == 3, "Default motif min size incorrect"
    
    # Test config modification
    config.graph.initial_nodes = 1000
    assert config.graph.initial_nodes == 1000, "Config modification failed"
    
    # Test device detection
    device = config.get_device()
    assert device in ["cpu", "cuda"], f"Invalid device: {device}"
    
    print("✓ Config tests passed")


def run_all_tests():
    """Run all tests"""
    
    print("Running ICA Framework Tests")
    print("=" * 30)
    
    try:
        test_config()
        test_causal_knowledge_graph()
        test_procedural_dataset_generator()
        
        print("\n" + "=" * 30)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
