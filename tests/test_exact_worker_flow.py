#!/usr/bin/env python3
"""
Test the EXACT worker flow step by step
"""

import time
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_exact_worker_flow():
    """Test the exact flow that happens in a worker"""
    
    # Load Neo4j config
    import json
    config_file = Path("config/database/neo4j.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        database_config = config_data['config']
    
    print("🧪 Testing EXACT worker flow...")
    
    # Import everything exactly like worker_functions.py does
    from ica_framework.core.ica_agent import ICAAgent
    from ica_framework.utils.config import Config
    from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
    from ica_framework.learning.scenario_generators import PhysicsSimulation
    from ica_framework.learning.worker_functions import _generate_enhanced_scenario, _create_observation
    
    # Step 1: Create worker agent exactly like in worker_functions.py
    config = Config()
    config.abstraction.motif_min_size = 10  
    config.abstraction.motif_max_size = 20
    config.abstraction.num_clusters = 3
    
    worker_agent = ICAAgent(config)
    
    # Step 2: Setup enhanced knowledge graph like workers do
    enhanced_kg = EnhancedKnowledgeGraph(
        backend="neo4j",
        config=database_config.copy()
    )
    
    if enhanced_kg.connect():
        worker_agent.knowledge_graph = enhanced_kg
        print("✅ Worker agent connected to Neo4j")
        
        # Step 3: Initialize world model
        worker_agent.initialize_world_model(
            state_dim=32,
            action_dim=8,
            num_relations=10
        )
        
        # Step 4: Get initial stats
        stats_before = enhanced_kg.get_stats()
        nodes_before = stats_before.get('nodes', 0)
        edges_before = stats_before.get('edges', 0)
        print(f"📊 Before processing: {nodes_before} nodes, {edges_before} edges")
        
        # Step 5: Generate scenario like workers do
        physics_sim = PhysicsSimulation()
        
        # Test a fallback scenario (type 2) that was previously empty
        scenario_type = 2  # This was broken before our fix
        scenario_count = 200000  # Simulate high scenario count
        current_round = 800  # Current round from continuous learning
        base_scenario = {}
        
        print(f"🎯 Testing scenario type {scenario_type} (should now have entities)...")
        scenario = _generate_enhanced_scenario(
            scenario_type, scenario_count, current_round,
            physics_sim, None, None, base_scenario
        )
        
        print(f"📝 Generated scenario: {len(scenario.get('entities', []))} entities, {len(scenario.get('relationships', []))} relationships")
        print(f"   Name: {scenario.get('name', 'Unknown')}")
        
        # Step 6: Create observation like workers do
        observation = _create_observation(scenario, current_round)
        print(f"📝 Created observation: {len(observation.get('entities', []))} entities, {len(observation.get('relationships', []))} relationships")
        
        # Step 7: Call active_learning_step like workers do
        print("🧠 Calling worker_agent.active_learning_step...")
        step_results = worker_agent.active_learning_step(observation)
        
        # Step 8: Force entity creation like workers do
        entities_created = 0
        relationships_created = 0
        
        for entity in observation.get('entities', []):
            entity_id = entity['id']
            all_properties = {}
            all_properties.update(entity.get('properties_static', {}))
            all_properties.update(entity.get('properties_dynamic', {}))
            all_properties['confidence'] = entity.get('confidence', 1.0)
            
            success = enhanced_kg.add_entity(
                entity_id, 
                entity.get('label', 'entity'),
                all_properties
            )
            if success:
                entities_created += 1
        
        for rel in observation.get('relationships', []):
            success = enhanced_kg.add_relationship(
                rel['source'],
                rel['target'], 
                rel['type'],
                rel.get('confidence', 0.5)
            )
            if success:
                relationships_created += 1
        
        print(f"📈 Forced creation: {entities_created} entities, {relationships_created} relationships")
        
        # Step 9: Get final stats
        stats_after = enhanced_kg.get_stats()
        nodes_after = stats_after.get('nodes', 0)
        edges_after = stats_after.get('edges', 0)
        
        print(f"📊 After processing: {nodes_after} nodes, {edges_after} edges")
        print(f"📈 Net change: {nodes_after - nodes_before} nodes, {edges_after - edges_before} edges")
        
        if nodes_after > nodes_before:
            print("✅ SUCCESS: Worker flow creates entities!")
        else:
            print("❌ ISSUE: Worker flow still not creating entities")
            
        return True
    else:
        print("❌ Failed to connect to Neo4j")
        return False

if __name__ == "__main__":
    test_exact_worker_flow()
