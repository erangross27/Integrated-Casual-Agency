#!/usr/bin/env python3
"""
Reset Neo4j session data to start fresh learning
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def reset_session():
    """Reset the session to start fresh learning"""
    
    # Load Neo4j configuration
    config_file = Path("config/database/neo4j.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            neo4j_config = config_data['config']
    else:
        print("❌ Config file not found")
        return False
    
    print("🧹 Resetting Neo4j session data...")
    
    # Connect to Neo4j
    enhanced_kg = EnhancedKnowledgeGraph(
        backend='neo4j',
        config=neo4j_config
    )
    
    if not enhanced_kg.connect():
        print("❌ Failed to connect to Neo4j")
        return False
    
    print("✅ Connected to Neo4j")
    
    # Get current stats
    initial_stats = enhanced_kg.get_stats()
    print(f"📊 Current: {initial_stats}")
    
    # Clear only SessionMeta nodes (keep Entity nodes for learning continuity)
    try:
        if hasattr(enhanced_kg.db, '_execute_query'):
            # Delete SessionMeta nodes
            result = enhanced_kg.db._execute_query("MATCH (n:SessionMeta) DETACH DELETE n RETURN count(n) as deleted")
            deleted_count = result[0]['deleted'] if result else 0
            print(f"🗑️ Deleted {deleted_count} SessionMeta nodes")
            
            # Check what's left
            entity_result = enhanced_kg.db._execute_query("MATCH (n:Entity) RETURN count(n) as count")
            entity_count = entity_result[0]['count'] if entity_result else 0
            
            edge_result = enhanced_kg.db._execute_query("MATCH ()-[r]->() RETURN count(r) as count")
            edge_count = edge_result[0]['count'] if edge_result else 0
            
            print(f"📊 Remaining: {entity_count} Entity nodes, {edge_count} relationships")
            print("✅ Session reset complete - Entity nodes preserved for continued learning")
            
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    reset_session()
