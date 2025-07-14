#!/usr/bin/env python3
"""
Test Worker Learning - Check if workers are actually creating entities
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.core.ica_agent import ICAAgent
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
from ica_framework.utils.config import Config
import json
import time

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

def test_worker_learning():
    """Test if a single worker can create entities"""
    print("🧪 Testing worker learning capabilities...")
    
    # Create agent like workers do
    config = Config()
    config.abstraction.motif_min_size = 10
    config.abstraction.motif_max_size = 20
    config.abstraction.num_clusters = 3
    
    agent = ICAAgent(config)
    
    # Setup enhanced knowledge graph
    database_config = load_database_config()
    enhanced_kg = EnhancedKnowledgeGraph(
        backend="neo4j",
        config=database_config
    )
    
    if enhanced_kg.connect():
        agent.knowledge_graph = enhanced_kg
        print("✅ Connected to Neo4j")
    else:
        print("❌ Failed to connect to Neo4j")
        return
    
    # Initialize world model
    agent.initialize_world_model(
        state_dim=32,
        action_dim=8,
        num_relations=10
    )
    
    # Get initial stats
    initial_stats = enhanced_kg.get_stats()
    print(f"📊 Initial: {initial_stats['nodes']} nodes, {initial_stats['edges']} edges")
    
    # Create test observation (similar to what workers generate)
    import numpy as np
    observation = {
        'state': np.array([0.1, 0.2, 0.3] + [0.0] * 29),  # 32-dimensional state to match world model
        'entities': [
            {
                'id': f'test_entity_{int(time.time())}',
                'label': 'test_sensor',
                'properties_static': {'type': 'temperature'},
                'properties_dynamic': {'value': 23.5},
                'confidence': 1.0
            },
            {
                'id': f'test_entity_{int(time.time())}_2',
                'label': 'test_controller',
                'properties_static': {'type': 'hvac'},
                'properties_dynamic': {'status': 'active'},
                'confidence': 1.0
            }
        ],
        'relationships': [
            {
                'source': f'test_entity_{int(time.time())}',
                'target': f'test_entity_{int(time.time())}_2',
                'type': 'controls',
                'confidence': 0.8,
                'weight': 1.0,
                'properties': {'delay': 0.1}
            }
        ]
    }
    
    print("🔄 Processing test observation...")
    
    # Process observation using active_learning_step (like workers do)
    result = agent.active_learning_step(observation)
    
    # Force commit - try different approaches
    if hasattr(enhanced_kg, 'db'):
        try:
            print("💾 Forcing Neo4j commit...")
            if hasattr(enhanced_kg.db, 'session') and enhanced_kg.db.session:
                enhanced_kg.db.session.close()
                enhanced_kg.db.session = enhanced_kg.db.driver.session(database=enhanced_kg.db.database)
            elif hasattr(enhanced_kg.db, 'commit'):
                enhanced_kg.db.commit()
        except Exception as e:
            print(f"⚠️ Commit error (might be OK): {e}")
    
    # Get final stats
    final_stats = enhanced_kg.get_stats()
    print(f"📊 Final: {final_stats['nodes']} nodes, {final_stats['edges']} edges")
    
    nodes_added = final_stats['nodes'] - initial_stats['nodes']
    edges_added = final_stats['edges'] - initial_stats['edges']
    
    print(f"➕ Added: {nodes_added} nodes, {edges_added} edges")
    
    if nodes_added > 0:
        print("✅ SUCCESS: Worker can create entities!")
    else:
        print("❌ PROBLEM: Worker not creating entities!")
        
        # Debug: Try direct entity creation
        print("🔍 Testing direct entity creation...")
        test_id = f"direct_test_{int(time.time())}"
        success = enhanced_kg.add_entity(test_id, "direct_test", {"timestamp": time.time()})
        print(f"Direct add_entity success: {success}")
        
        # Check stats again
        debug_stats = enhanced_kg.get_stats()
        print(f"📊 After direct add: {debug_stats['nodes']} nodes, {debug_stats['edges']} edges")
    
    enhanced_kg.disconnect()

if __name__ == "__main__":
    test_worker_learning()
