#!/usr/bin/env python3
"""
Test multiprocessing Neo4j connection - simulate exact worker scenario
"""

import time
import os
import sys
import multiprocessing as mp
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def worker_process(worker_id, database_config):
    """Simulate exactly what the learning worker does"""
    try:
        # Import inside worker to avoid pickling issues
        from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
        
        # Create FRESH connection for this worker process
        enhanced_kg = EnhancedKnowledgeGraph(
            backend="neo4j",
            config=database_config.copy()
        )
        
        # Force disconnect any inherited connections
        if hasattr(enhanced_kg, 'disconnect'):
            enhanced_kg.disconnect()
        
        # Create new connection for this worker process
        if enhanced_kg.connect():
            print(f"Worker {worker_id}: Connected to Neo4j")
            
            # Get stats before
            stats_before = enhanced_kg.get_stats()
            nodes_before = stats_before.get('nodes', 0)
            
            # Create multiple test entities like learning workers do
            entities_created = 0
            for i in range(3):
                test_id = f"worker_{worker_id}_entity_{i}_{int(time.time())}"
                success = enhanced_kg.add_entity(
                    test_id, 
                    "test_entity",
                    {
                        "worker_id": worker_id, 
                        "entity_index": i,
                        "timestamp": time.time(), 
                        "process_id": os.getpid()
                    }
                )
                if success:
                    entities_created += 1
            
            # Get stats after
            stats_after = enhanced_kg.get_stats()
            nodes_after = stats_after.get('nodes', 0)
            
            print(f"Worker {worker_id}: Created {entities_created} entities, "
                  f"DB shows {nodes_after - nodes_before} new nodes")
            
            return {
                'worker_id': worker_id,
                'entities_created': entities_created,
                'nodes_before': nodes_before,
                'nodes_after': nodes_after,
                'success': True
            }
        else:
            print(f"Worker {worker_id}: Failed to connect to Neo4j")
            return {'worker_id': worker_id, 'success': False}
            
    except Exception as e:
        print(f"Worker {worker_id}: Error - {e}")
        return {'worker_id': worker_id, 'error': str(e), 'success': False}

def test_multiprocessing_neo4j():
    """Test Neo4j with multiprocessing workers"""
    
    # Load Neo4j config
    import json
    config_file = Path("config/database/neo4j.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        database_config = config_data['config']
    
    print("🧪 Testing multiprocessing Neo4j connection...")
    print("🚀 Starting 3 worker processes...")
    
    # Get initial stats
    from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
    main_kg = EnhancedKnowledgeGraph(backend="neo4j", config=database_config)
    main_kg.connect()
    initial_stats = main_kg.get_stats()
    print(f"📊 Initial: {initial_stats.get('nodes', 0)} nodes, {initial_stats.get('edges', 0)} edges")
    
    # Start workers
    with mp.Pool(3) as pool:
        results = pool.starmap(worker_process, [
            (0, database_config),
            (1, database_config),
            (2, database_config)
        ])
    
    # Check results
    print("\n📊 Worker Results:")
    total_created = 0
    for result in results:
        if result['success']:
            print(f"  Worker {result['worker_id']}: {result['entities_created']} entities created")
            total_created += result['entities_created']
        else:
            print(f"  Worker {result['worker_id']}: Failed")
    
    # Get final stats
    final_stats = main_kg.get_stats()
    actual_new_nodes = final_stats.get('nodes', 0) - initial_stats.get('nodes', 0)
    
    print(f"\n📈 Results Summary:")
    print(f"  Workers reported creating: {total_created} entities")
    print(f"  Database actually has: {actual_new_nodes} new nodes")
    print(f"  Final total: {final_stats.get('nodes', 0)} nodes, {final_stats.get('edges', 0)} edges")
    
    if actual_new_nodes == total_created:
        print("✅ SUCCESS: Multiprocessing works correctly!")
    else:
        print("❌ ISSUE: Multiprocessing database sync problem detected")

if __name__ == "__main__":
    test_multiprocessing_neo4j()
