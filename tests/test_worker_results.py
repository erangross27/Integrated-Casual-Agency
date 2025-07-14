#!/usr/bin/env python3
"""
Test multiprocessing worker results to see what's being returned
"""

import time
import os
import sys
import multiprocessing as mp
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def debug_worker_process(worker_id, scenario_queue, results_queue, database_config, stop_event):
    """Debug version of worker to see what results are generated"""
    
    try:
        # Import inside worker
        from ica_framework.learning.worker_functions import continuous_parallel_worker
        from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
        
        print(f"DEBUG Worker {worker_id}: Starting...")
        
        # Create fresh database connection
        enhanced_kg = EnhancedKnowledgeGraph(
            backend="neo4j",
            config=database_config.copy()
        )
        
        if enhanced_kg.connect():
            print(f"DEBUG Worker {worker_id}: Connected to Neo4j")
            
            # Process a few scenarios and report results
            for i in range(3):
                # Get stats before
                stats_before = enhanced_kg.get_stats()
                nodes_before = stats_before.get('nodes', 0)
                edges_before = stats_before.get('edges', 0)
                
                # Create test entities
                test_id = f"debug_worker_{worker_id}_entity_{i}_{int(time.time())}"
                success = enhanced_kg.add_entity(
                    test_id, 
                    "debug_entity",
                    {"worker_id": worker_id, "test_index": i, "timestamp": time.time()}
                )
                
                # Get stats after
                stats_after = enhanced_kg.get_stats()
                nodes_after = stats_after.get('nodes', 0)
                edges_after = stats_after.get('edges', 0)
                
                result = {
                    'worker_id': worker_id,
                    'test_index': i,
                    'nodes_before': nodes_before,
                    'nodes_after': nodes_after,
                    'nodes_added': nodes_after - nodes_before,
                    'edges_before': edges_before,
                    'edges_after': edges_after,
                    'edges_added': edges_after - edges_before,
                    'entity_created': success,
                    'timestamp': time.time()
                }
                
                print(f"DEBUG Worker {worker_id}: Test {i} - Created entity: {success}, "
                      f"Nodes: {nodes_before} -> {nodes_after} (added: {nodes_after - nodes_before})")
                
                results_queue.put(result)
                time.sleep(0.1)  # Small delay
        else:
            print(f"DEBUG Worker {worker_id}: Failed to connect")
            
    except Exception as e:
        print(f"DEBUG Worker {worker_id}: Error - {e}")
        import traceback
        traceback.print_exc()

def test_multiprocessing_worker_results():
    """Test what results multiprocessing workers actually return"""
    
    # Load Neo4j config
    import json
    config_file = Path("config/database/neo4j.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        database_config = config_data['config']
    
    print("🧪 Testing multiprocessing worker results...")
    
    # Create queues
    scenario_queue = mp.Queue()
    results_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Get initial database state
    from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
    main_kg = EnhancedKnowledgeGraph(backend="neo4j", config=database_config)
    main_kg.connect()
    initial_stats = main_kg.get_stats()
    print(f"📊 Initial DB state: {initial_stats.get('nodes', 0)} nodes, {initial_stats.get('edges', 0)} edges")
    
    # Start debug workers
    workers = []
    for worker_id in range(2):
        worker = mp.Process(
            target=debug_worker_process,
            args=(worker_id, scenario_queue, results_queue, database_config, stop_event)
        )
        worker.start()
        workers.append(worker)
    
    # Collect results
    print("\n📊 Collecting worker results...")
    results = []
    timeout = 10  # 10 seconds timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = results_queue.get(timeout=1.0)
            results.append(result)
            print(f"📈 Result from Worker {result['worker_id']}: "
                  f"nodes_added={result['nodes_added']}, edges_added={result['edges_added']}, "
                  f"entity_created={result['entity_created']}")
        except:
            break
    
    # Stop workers
    stop_event.set()
    for worker in workers:
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
    
    # Get final database state
    final_stats = main_kg.get_stats()
    print(f"\n📊 Final DB state: {final_stats.get('nodes', 0)} nodes, {final_stats.get('edges', 0)} edges")
    print(f"📈 Total change: {final_stats.get('nodes', 0) - initial_stats.get('nodes', 0)} nodes")
    
    # Analyze results
    total_reported_nodes = sum(r['nodes_added'] for r in results)
    actual_new_nodes = final_stats.get('nodes', 0) - initial_stats.get('nodes', 0)
    
    print(f"\n📊 Analysis:")
    print(f"  Workers reported adding: {total_reported_nodes} nodes")
    print(f"  Database actually gained: {actual_new_nodes} nodes")
    print(f"  Results collected: {len(results)}")
    
    if total_reported_nodes == 0:
        print("❌ ISSUE: Workers are reporting 0 nodes_added!")
    elif total_reported_nodes != actual_new_nodes:
        print("⚠️ WARNING: Mismatch between reported and actual nodes")
    else:
        print("✅ SUCCESS: Worker results match database changes")

if __name__ == "__main__":
    test_multiprocessing_worker_results()
