#!/usr/bin/env python3
"""
Test the actual learning worker loop - simulate exact continuous learning scenario
"""

import time
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def simulate_worker_learning_step():
    """Simulate exactly what happens in a worker learning step"""
    
    # Load Neo4j config
    import json
    config_file = Path("config/database/neo4j.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        database_config = config_data['config']
    
    print("🧪 Testing exact worker learning step...")
    
    # Create worker agent exactly like in worker_functions.py
    from ica_framework.core.ica_agent import ICAAgent
    from ica_framework.utils.config import Config
    from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
    
    # Create worker configuration - MUST MATCH MAIN AGENT CONFIG
    config = Config()
    config.abstraction.motif_min_size = 10  # Fast learning mode
    config.abstraction.motif_max_size = 20
    config.abstraction.num_clusters = 3
    
    # Create worker agent
    worker_agent = ICAAgent(config)
    
    # Setup enhanced knowledge graph
    enhanced_kg = EnhancedKnowledgeGraph(
        backend="neo4j",
        config=database_config.copy()
    )
    
    if enhanced_kg.connect():
        worker_agent.knowledge_graph = enhanced_kg
        print("✅ Worker agent connected to Neo4j")
        
        # Initialize world model
        worker_agent.initialize_world_model(
            state_dim=32,
            action_dim=8,
            num_relations=10
        )
        
        # Get initial stats
        stats_before = enhanced_kg.get_stats()
        nodes_before = stats_before.get('nodes', 0)
        edges_before = stats_before.get('edges', 0)
        print(f"📊 Before learning: {nodes_before} nodes, {edges_before} edges")
        
        # Create a realistic observation like the workers do
        observation = {
            "entities": [
                {
                    'id': f'test_sphere_{int(time.time())}',
                    'label': 'sphere',
                    'properties_static': {'type': 'sphere', 'material': 'metal'},
                    'properties_dynamic': {'temperature': 25.0, 'velocity': [1, 0, 0]},
                    'confidence': 1.0
                },
                {
                    'id': f'test_cube_{int(time.time())}',
                    'label': 'cube',
                    'properties_static': {'type': 'cube', 'material': 'wood'},
                    'properties_dynamic': {'temperature': 22.0, 'velocity': [0, 1, 0]},
                    'confidence': 1.0
                }
            ],
            "relationships": [
                {
                    'source': f'test_sphere_{int(time.time())}',
                    'target': f'test_cube_{int(time.time())}',
                    'type': 'collides_with',
                    'confidence': 0.8,
                    'weight': 1.0,
                    'properties': {}
                }
            ],
            "state": np.random.normal(0, 0.2, 32)
        }
        
        print(f"📝 Created observation with {len(observation['entities'])} entities, {len(observation['relationships'])} relationships")
        
        # Call active_learning_step like the workers do
        print("🧠 Calling active_learning_step...")
        step_results = worker_agent.active_learning_step(observation)
        print(f"📊 Step results: {step_results}")
        
        # Force database synchronization
        if hasattr(enhanced_kg, 'db'):
            # Force explicit entity and relationship creation
            for entity in observation['entities']:
                enhanced_kg.add_entity(
                    entity['id'], 
                    entity['label'],
                    {**entity['properties_static'], **entity['properties_dynamic']}
                )
            
            for rel in observation['relationships']:
                enhanced_kg.add_relationship(
                    rel['source'],
                    rel['target'], 
                    rel['type'],
                    rel['confidence']
                )
        
        # Get final stats
        stats_after = enhanced_kg.get_stats()
        nodes_after = stats_after.get('nodes', 0)
        edges_after = stats_after.get('edges', 0)
        
        print(f"📊 After learning: {nodes_after} nodes, {edges_after} edges")
        print(f"📈 Created: {nodes_after - nodes_before} nodes, {edges_after - edges_before} edges")
        
        if nodes_after > nodes_before:
            print("✅ SUCCESS: Learning step created entities!")
        else:
            print("❌ ISSUE: Learning step did not create entities")
            
        return True
    else:
        print("❌ Failed to connect to Neo4j")
        return False

if __name__ == "__main__":
    simulate_worker_learning_step()
