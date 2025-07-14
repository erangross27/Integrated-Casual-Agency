#!/usr/bin/env python3
"""
Test Worker Direct Processing
Test the worker logic without multiprocessing to see any errors
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

def test_worker_logic_direct():
    """Test the exact worker logic without multiprocessing"""
    print("🔧 Testing worker logic directly (no multiprocessing)...")
    
    database_config = load_database_config()
    
    # Create worker configuration exactly like in worker_functions.py
    config = Config()
    config.abstraction.motif_min_size = 10
    config.abstraction.motif_max_size = 20
    config.abstraction.num_clusters = 3
    
    # Create worker agent
    worker_agent = ICAAgent(config)
    
    # Setup enhanced knowledge graph
    enhanced_kg = EnhancedKnowledgeGraph(
        backend="neo4j",
        config=database_config
    )
    
    if enhanced_kg.connect():
        worker_agent.knowledge_graph = enhanced_kg
        print("✅ Worker agent connected to Neo4j")
    else:
        print("❌ Failed to connect worker agent to Neo4j")
        return
    
    # Initialize world model like workers do
    worker_agent.initialize_world_model(
        state_dim=32,
        action_dim=8,
        num_relations=10
    )
    
    # Get initial stats
    initial_stats = enhanced_kg.get_stats()
    print(f"📊 Initial: {initial_stats['nodes']} nodes, {initial_stats['edges']} edges")
    
    # Generate scenario like workers do
    physics_sim = PhysicsSimulation()
    scenario = physics_sim.generate_physics_scenario(complexity_level=0.5)
    observation = _create_observation(scenario, current_round=1)
    
    print(f"🔍 Processing observation with {len(observation['entities'])} entities...")
    
    # Process exactly like workers do - but without output suppression
    try:
        print("🚀 Calling active_learning_step...")
        step_results = worker_agent.active_learning_step(observation)
        print(f"✅ Active learning step completed: {step_results}")
        
        # Force any pending database operations
        if hasattr(enhanced_kg, 'db') and hasattr(enhanced_kg.db, 'session'):
            print("💾 Forcing database commit...")
            enhanced_kg.db.session.close()
            enhanced_kg.db.session = enhanced_kg.db.driver.session(database=enhanced_kg.db.database)
        
    except Exception as e:
        print(f"❌ Error in active_learning_step: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get final stats
    final_stats = enhanced_kg.get_stats()
    print(f"📊 Final: {final_stats['nodes']} nodes, {final_stats['edges']} edges")
    
    nodes_added = final_stats['nodes'] - initial_stats['nodes']
    edges_added = final_stats['edges'] - initial_stats['edges']
    
    print(f"➕ Added: {nodes_added} nodes, {edges_added} edges")
    
    if nodes_added > 0:
        print("✅ SUCCESS: Worker logic creates entities!")
    else:
        print("❌ PROBLEM: Worker logic not creating entities!")
        
        # Debug: Check if entities were processed but not committed
        print("🔍 Debug: Checking local graph state...")
        if hasattr(worker_agent.knowledge_graph, 'graph'):
            local_nodes = worker_agent.knowledge_graph.graph.number_of_nodes()
            local_edges = worker_agent.knowledge_graph.graph.number_of_edges()
            print(f"   Local NetworkX graph: {local_nodes} nodes, {local_edges} edges")
    
    enhanced_kg.disconnect()

if __name__ == "__main__":
    test_worker_logic_direct()
