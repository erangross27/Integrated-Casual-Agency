#!/usr/bin/env python3
"""
Debug Worker Issues - Simple test without m       print(f"   Entities: {len(observation['entities'])}")
    print(f"   Relationships: {len(observation['relationships'])}")
    print(f"   First entity ID: {observation['entities'][0]['id'] if observation['entities'] else 'None'}")
    if observation['relationships']:
        first_rel = observation['relationships'][0]
        print(f"   First relationship: {first_rel['source']} -> {first_rel['target']}")
    else:
        print("   First relationship: None")nt(f"   First entity ID: {observation['entities'][0]['id'] if observation['entities'] else 'None'}")
    if observation['relationships']:
        first_rel = observation['relationships'][0]
        print(f"   First relationship: {first_rel['source']} -> {first_rel['target']}")
    else:
        print("   First relationship: None")tiprocessing complexity
"""

import sys
from pathlib import Path
import time
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.learning.worker_functions import _create_observation
from ica_framework.learning.scenario_generators import PhysicsSimulation
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
from ica_framework.core.ica_agent import ICAAgent
from ica_framework.utils.config import Config

def load_database_config():
    """Load Neo4j configuration"""
    config_file = Path("config/database/neo4j.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            return config_data['config']
    return {
        'uri': 'neo4j://127.0.0.1:7687',
        'username': 'neo4j',
        'password': 'password',
        'database': 'neo4j'
    }

def test_worker_logic_step_by_step():
    """Test worker logic step by step to find the issue"""
    print("🔍 Debug: Testing worker logic step by step...")
    
    database_config = load_database_config()
    
    # Step 1: Create agent exactly like workers
    print("\n📍 Step 1: Creating agent like workers...")
    config = Config()
    config.abstraction.motif_min_size = 10
    config.abstraction.motif_max_size = 20
    config.abstraction.num_clusters = 3
    
    worker_agent = ICAAgent(config)
    
    # Step 2: Setup enhanced knowledge graph
    print("📍 Step 2: Setting up enhanced knowledge graph...")
    enhanced_kg = EnhancedKnowledgeGraph(backend="neo4j", config=database_config)
    if enhanced_kg.connect():
        worker_agent.knowledge_graph = enhanced_kg
        print("✅ Enhanced KG connected")
    else:
        print("❌ Enhanced KG connection failed")
        return
    
    # Step 3: Initialize world model
    print("📍 Step 3: Initializing world model...")
    worker_agent.initialize_world_model(state_dim=32, action_dim=8, num_relations=10)
    print("✅ World model initialized")
    
    # Step 4: Generate scenario and observation
    print("📍 Step 4: Generating scenario and observation...")
    physics_sim = PhysicsSimulation()
    scenario = physics_sim.generate_physics_scenario(complexity_level=0.5)
    observation = _create_observation(scenario, current_round=1)
    
    print(f"   Entities: {len(observation['entities'])}")
    print(f"   Relationships: {len(observation['relationships'])}")
    print(f"   First entity ID: {observation['entities'][0]['id'] if observation['entities'] else 'None'}")
    if observation['relationships']:
        first_rel = observation['relationships'][0]
        print(f"   First relationship: {first_rel['source']} -> {first_rel['target']}")
    else:
        print("   First relationship: None")
    
    # Step 5: Get initial stats
    print("📍 Step 5: Getting initial stats...")
    initial_stats = enhanced_kg.get_stats()
    print(f"   Initial: {initial_stats['nodes']} nodes, {initial_stats['edges']} edges")
    
    # Step 6: Test active_learning_step
    print("📍 Step 6: Testing active_learning_step...")
    try:
        step_results = worker_agent.active_learning_step(observation)
        print(f"   ✅ Active learning completed: {step_results.get('knowledge_updates', 0)} updates")
    except Exception as e:
        print(f"   ❌ Error in active_learning_step: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 7: Check stats after active_learning_step
    print("📍 Step 7: Checking stats after active_learning_step...")
    after_step_stats = enhanced_kg.get_stats()
    step_nodes_added = after_step_stats['nodes'] - initial_stats['nodes']
    step_edges_added = after_step_stats['edges'] - initial_stats['edges']
    print(f"   After step: {after_step_stats['nodes']} nodes, {after_step_stats['edges']} edges")
    print(f"   Step added: {step_nodes_added} nodes, {step_edges_added} edges")
    
    # Step 8: Force additional entity creation (like workers do)
    print("📍 Step 8: Force additional entity creation...")
    entities_forced = 0
    relationships_forced = 0
    
    for entity in observation.get('entities', []):
        entity_id = entity['id']
        all_properties = {}
        all_properties.update(entity.get('properties_static', {}))
        all_properties.update(entity.get('properties_dynamic', {}))
        all_properties['confidence'] = entity.get('confidence', 1.0)
        
        success = enhanced_kg.add_entity(entity_id, entity.get('label', 'entity'), all_properties)
        if success:
            entities_forced += 1
            print(f"   Force added entity: {entity_id}")
    
    for rel in observation.get('relationships', []):
        success = enhanced_kg.add_relationship(
            rel['source'], rel['target'], rel['type'], rel.get('confidence', 0.5)
        )
        if success:
            relationships_forced += 1
            print(f"   Force added relationship: {rel['source']} -> {rel['target']}")
    
    print(f"   Forced: {entities_forced} entities, {relationships_forced} relationships")
    
    # Step 9: Final stats
    print("📍 Step 9: Final stats...")
    final_stats = enhanced_kg.get_stats()
    total_nodes_added = final_stats['nodes'] - initial_stats['nodes']
    total_edges_added = final_stats['edges'] - initial_stats['edges']
    print(f"   Final: {final_stats['nodes']} nodes, {final_stats['edges']} edges")
    print(f"   Total added: {total_nodes_added} nodes, {total_edges_added} edges")
    
    if total_nodes_added > 0:
        print("✅ SUCCESS: Entities were created!")
        print(f"   Via active_learning_step: {step_nodes_added} nodes")
        print(f"   Via forced addition: {entities_forced} entities")
    else:
        print("❌ PROBLEM: No entities created at all!")
    
    enhanced_kg.disconnect()

if __name__ == "__main__":
    test_worker_logic_step_by_step()
