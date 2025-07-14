#!/usr/bin/env python3
"""
Test exact worker scenario processing
"""

import sys
from pathlib import Path
import time
import json
import numpy as np

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

def test_exact_worker_scenario():
    """Test exact worker scenario step by step"""
    print("🔧 Testing exact worker scenario processing...")
    
    database_config = load_database_config()
    
    # Create agent exactly like workers do
    config = Config()
    config.abstraction.motif_min_size = 10
    config.abstraction.motif_max_size = 20
    config.abstraction.num_clusters = 3
    
    agent = ICAAgent(config)
    
    # Setup enhanced knowledge graph
    enhanced_kg = EnhancedKnowledgeGraph(backend="neo4j", config=database_config)
    if enhanced_kg.connect():
        agent.knowledge_graph = enhanced_kg
        print("✅ Connected to Neo4j")
    else:
        print("❌ Failed to connect to Neo4j")
        return
    
    # Initialize world model
    agent.initialize_world_model(state_dim=32, action_dim=8, num_relations=10)
    
    # Get initial stats
    initial_stats = enhanced_kg.get_stats()
    print(f"📊 Initial: {initial_stats['nodes']} nodes, {initial_stats['edges']} edges")
    
    # Generate exact same scenario as workers do
    physics_sim = PhysicsSimulation()
    scenario = physics_sim.generate_physics_scenario(complexity_level=0.5)
    observation = _create_observation(scenario, current_round=1)
    
    print(f"🔍 Worker observation: {len(observation['entities'])} entities, {len(observation['relationships'])} relationships")
    print(f"   First entity: {observation['entities'][0] if observation['entities'] else 'None'}")
    
    # Step 1: Test observe_environment directly first
    print("\n📍 Step 1: Testing observe_environment directly...")
    try:
        success = agent.observe_environment(observation)
        print(f"   observe_environment returned: {success}")
        
        # Check stats immediately after observe_environment
        immediate_stats = enhanced_kg.get_stats()
        print(f"   Immediate stats: {immediate_stats['nodes']} nodes, {immediate_stats['edges']} edges")
        
        nodes_added_step1 = immediate_stats['nodes'] - initial_stats['nodes']
        edges_added_step1 = immediate_stats['edges'] - initial_stats['edges']
        print(f"   Step 1 added: {nodes_added_step1} nodes, {edges_added_step1} edges")
        
    except Exception as e:
        print(f"   ❌ Error in observe_environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Test full active_learning_step
    print("\n📍 Step 2: Testing full active_learning_step...")
    
    # Reset to initial state for fair comparison
    if nodes_added_step1 > 0:
        print("   Note: observe_environment already added entities, continuing with active_learning_step...")
    
    step1_stats = enhanced_kg.get_stats()
    
    try:
        step_results = agent.active_learning_step(observation)
        print(f"   active_learning_step returned: {step_results}")
        
        # Check final stats
        final_stats = enhanced_kg.get_stats()
        print(f"   Final stats: {final_stats['nodes']} nodes, {final_stats['edges']} edges")
        
        nodes_added_step2 = final_stats['nodes'] - step1_stats['nodes']
        edges_added_step2 = final_stats['edges'] - step1_stats['edges']
        print(f"   Step 2 added: {nodes_added_step2} nodes, {edges_added_step2} edges")
        
        total_nodes_added = final_stats['nodes'] - initial_stats['nodes']
        total_edges_added = final_stats['edges'] - initial_stats['edges']
        print(f"   Total added: {total_nodes_added} nodes, {total_edges_added} edges")
        
    except Exception as e:
        print(f"   ❌ Error in active_learning_step: {e}")
        import traceback
        traceback.print_exc()
        return
    
    enhanced_kg.disconnect()

if __name__ == "__main__":
    test_exact_worker_scenario()
