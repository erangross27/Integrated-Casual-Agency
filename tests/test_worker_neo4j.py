#!/usr/bin/env python3
"""
Test worker Neo4j connection and entity creation
"""

import time
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_worker_neo4j_write():
    """Test if a single worker can write to Neo4j"""
    
    # Load Neo4j config
    import json
    config_file = Path("config/database/neo4j.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        database_config = config_data['config']
    
    # Create a fresh enhanced knowledge graph
    from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
    
    enhanced_kg = EnhancedKnowledgeGraph(
        backend="neo4j",
        config=database_config.copy()
    )
    
    # Connect and test
    if enhanced_kg.connect():
        print("✅ Successfully connected to Neo4j")
        
        # Get stats before
        stats_before = enhanced_kg.get_stats()
        nodes_before = stats_before.get('nodes', 0)
        print(f"📊 Nodes before: {nodes_before}")
        
        # Create test entities
        test_id = f"test_worker_entity_{int(time.time())}"
        success = enhanced_kg.add_entity(
            test_id, 
            "test_entity",
            {"test": True, "timestamp": time.time(), "process_id": os.getpid()}
        )
        
        if success:
            print(f"✅ Created test entity: {test_id}")
        else:
            print(f"❌ Failed to create test entity: {test_id}")
        
        # Get stats after
        stats_after = enhanced_kg.get_stats()
        nodes_after = stats_after.get('nodes', 0)
        print(f"📊 Nodes after: {nodes_after}")
        print(f"📈 Nodes added: {nodes_after - nodes_before}")
        
        # Test relationship
        test_id2 = f"test_worker_entity2_{int(time.time())}"
        enhanced_kg.add_entity(test_id2, "test_entity", {"test": True})
        
        rel_success = enhanced_kg.add_relationship(
            test_id, test_id2, "connects_to", 0.8
        )
        
        if rel_success:
            print(f"✅ Created test relationship: {test_id} -> {test_id2}")
        else:
            print(f"❌ Failed to create test relationship")
            
        # Final stats
        final_stats = enhanced_kg.get_stats()
        print(f"📊 Final stats: {final_stats.get('nodes', 0)} nodes, {final_stats.get('edges', 0)} edges")
        
        return True
    else:
        print("❌ Failed to connect to Neo4j")
        return False

if __name__ == "__main__":
    print("🧪 Testing worker Neo4j connection and entity creation...")
    test_worker_neo4j_write()
