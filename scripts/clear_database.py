#!/usr/bin/env python3
"""
Clear Neo4j Database for Fresh Start
"""
import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def clear_neo4j_database():
    """Clear the Neo4j database for a fresh start"""
    print("🔄 Clearing Neo4j Database...")
    
    # Load configuration
    config_file = PROJECT_ROOT / "config/database/neo4j.json"
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        db_config = config_data['config']
    
    # Initialize knowledge graph
    kg = EnhancedKnowledgeGraph(
        backend='neo4j',
        config=db_config
    )
    
    # Connect and clear
    if kg.connect():
        print("✅ Connected to Neo4j")
        
        # Clear all data
        try:
            result = kg.db._execute_query("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared successfully!")
            
            # Verify it's empty
            stats = kg.get_statistics()
            print(f"📊 Database statistics after clearing: {stats}")
            
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
    else:
        print("❌ Failed to connect to Neo4j")
    
    kg.disconnect()

if __name__ == "__main__":
    clear_neo4j_database()
