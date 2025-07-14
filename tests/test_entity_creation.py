#!/usr/bin/env python3
"""
Simple test to verify that observe_environment creates Entity nodes in Neo4j
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ica_framework.core.ica_agent import ICAAgent
from ica_framework.utils.config import Config
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def test_entity_creation():
    """Test that observe_environment creates Entity nodes properly"""
    
    # Test Neo4j configuration
    neo4j_config = {
        'uri': 'neo4j://127.0.0.1:7687',
        'username': 'neo4j',
        'password': 'Mk!p93Mk!p93',
        'database': 'neo4j'
    }
    
    print("🧪 Testing Entity Creation in Neo4j...")
    
    # Create ICA Agent
    config = Config()
    agent = ICAAgent(config)
    
    # Replace with Enhanced Knowledge Graph
    enhanced_kg = EnhancedKnowledgeGraph(
        backend='neo4j',
        config=neo4j_config
    )
    
    if not enhanced_kg.connect():
        print("❌ Failed to connect to Neo4j")
        return False
    
    print("✅ Connected to Neo4j")
    
    # Replace agent's knowledge graph
    agent.knowledge_graph = enhanced_kg
    
    # Get initial stats
    initial_stats = enhanced_kg.get_stats()
    print(f"📊 Initial stats: {initial_stats}")
    
    # Create test observation with entities and relationships
    test_observation = {
        "entities": [
            {"id": "test_sensor_1", "label": "sensor", "confidence": 0.9},
            {"id": "test_controller_1", "label": "controller", "confidence": 0.95},
            {"id": "test_environment_1", "label": "environment", "confidence": 0.8}
        ],
        "relationships": [
            {"source": "test_sensor_1", "target": "test_controller_1", "type": "reports_to", "confidence": 0.85},
            {"source": "test_controller_1", "target": "test_environment_1", "type": "controls", "confidence": 0.9}
        ],
        "state": [1.0, 2.0, 3.0] * 10 + [0.0, 0.0]  # 32-element state
    }
    
    print("🔍 Calling observe_environment...")
    
    # Call observe_environment
    success = agent.observe_environment(test_observation)
    
    if success:
        print("✅ observe_environment returned True")
    else:
        print("❌ observe_environment returned False")
        return False
    
    # Get stats after observation
    after_stats = enhanced_kg.get_stats()
    print(f"📊 Stats after observation: {after_stats}")
    
    # Check if entities were created
    nodes_added = after_stats['nodes'] - initial_stats['nodes']
    edges_added = after_stats['edges'] - initial_stats['edges']
    
    print(f"➕ Nodes added: {nodes_added}")
    print(f"➕ Edges added: {edges_added}")
    
    if nodes_added >= 3 and edges_added >= 2:
        print("✅ Entity creation successful!")
        
        # Verify specific entities exist
        for entity in test_observation['entities']:
            entity_data = enhanced_kg.get_entity(entity['id'])
            if entity_data:
                print(f"✅ Entity {entity['id']} exists in Neo4j")
            else:
                print(f"❌ Entity {entity['id']} NOT found in Neo4j")
        
        return True
    else:
        print("❌ Expected entities were not created")
        
        # Direct database query to see what's actually there
        try:
            if hasattr(enhanced_kg.db, '_execute_query'):
                result = enhanced_kg.db._execute_query("MATCH (n) RETURN n.id, labels(n), n.label LIMIT 10")
                print(f"🔍 Direct Neo4j query result: {result}")
        except Exception as e:
            print(f"🔍 Query error: {e}")
        
        return False

if __name__ == "__main__":
    test_entity_creation()
