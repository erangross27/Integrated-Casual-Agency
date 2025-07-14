#!/usr/bin/env python3
"""
Test observe_environment method directly
"""

import sys
from pathlib import Path
import time
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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

def test_observe_environment_directly():
    """Test if observe_environment works with enhanced knowledge graph"""
    print("🔍 Testing observe_environment method directly...")
    
    database_config = load_database_config()
    
    # Create agent and enhanced KG
    config = Config()
    agent = ICAAgent(config)
    
    enhanced_kg = EnhancedKnowledgeGraph(backend="neo4j", config=database_config)
    if enhanced_kg.connect():
        agent.knowledge_graph = enhanced_kg
        print("✅ Connected to Neo4j")
    else:
        print("❌ Failed to connect to Neo4j")
        return
    
    # Get initial stats
    initial_stats = enhanced_kg.get_stats()
    print(f"📊 Initial: {initial_stats['nodes']} nodes, {initial_stats['edges']} edges")
    
    # Create test observation exactly like our working test
    observation = {
        'state': np.array([0.1, 0.2, 0.3] + [0.0] * 29),
        'entities': [
            {
                'id': f'test_observe_{int(time.time() * 1000) % 10000}',
                'label': 'test_sensor',
                'properties_static': {'type': 'temperature'},
                'properties_dynamic': {'value': 23.5},
                'confidence': 1.0
            },
            {
                'id': f'test_observe_{int(time.time() * 1000) % 10000}_2',
                'label': 'test_controller',
                'properties_static': {'type': 'hvac'},
                'properties_dynamic': {'status': 'active'},
                'confidence': 1.0
            }
        ],
        'relationships': [
            {
                'source': f'test_observe_{int(time.time() * 1000) % 10000}',
                'target': f'test_observe_{int(time.time() * 1000) % 10000}_2',
                'type': 'controls',
                'confidence': 0.8,
                'weight': 1.0,
                'properties': {'delay': 0.1}
            }
        ]
    }
    
    # Fix relationship source/target IDs to match entities
    if observation['entities'] and observation['relationships']:
        observation['relationships'][0]['source'] = observation['entities'][0]['id']
        observation['relationships'][0]['target'] = observation['entities'][1]['id']
    
    print(f"🔍 Testing observe_environment with {len(observation['entities'])} entities...")
    
    # Test observe_environment directly
    try:
        success = agent.observe_environment(observation)
        print(f"✅ observe_environment returned: {success}")
        
        # Force commit
        if hasattr(enhanced_kg.db, 'session') and enhanced_kg.db.session:
            enhanced_kg.db.session.close()
            enhanced_kg.db.session = enhanced_kg.db.driver.session(database=enhanced_kg.db.database)
        
    except Exception as e:
        print(f"❌ Error in observe_environment: {e}")
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
        print("✅ SUCCESS: observe_environment creates entities!")
    else:
        print("❌ PROBLEM: observe_environment not creating entities!")
        
        # Check if the add_node/add_edge methods are working
        print("\n🔧 Testing add_node/add_edge directly...")
        try:
            test_id = f"direct_test_{int(time.time() * 1000) % 10000}"
            success = enhanced_kg.add_entity(test_id, "direct_test", {"timestamp": time.time()})
            print(f"   Direct add_entity: {success}")
            
            # Check stats again
            debug_stats = enhanced_kg.get_stats()
            print(f"   Stats after direct add: {debug_stats['nodes']} nodes")
            
        except Exception as e:
            print(f"   Error in direct add_entity: {e}")
    
    enhanced_kg.disconnect()

if __name__ == "__main__":
    test_observe_environment_directly()
