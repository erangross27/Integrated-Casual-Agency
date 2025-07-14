#!/usr/bin/env python3
"""
Simple test of the worker without early stop_event
"""

import time
import multiprocessing as mp
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_simple_worker():
    """Test the worker with no early stop"""
    
    # Load Neo4j config
    config_file = Path("config/database/neo4j.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        database_config = config_data['config']
    
    print("🧪 Testing worker with no early stop...")
    
    # Create queues
    scenario_queue = mp.Queue(maxsize=150)
    results_queue = mp.Queue()
    stop_event = mp.Event()  # DON'T set this early
    
    # Add ONE test scenario
    print("📝 Adding 1 test scenario...")
    scenario = {
        'name': 'Simple Test Scenario',
        'entities': ['entity_1', 'entity_2', 'entity_3'],
        'relationships': [
            {'source': 'entity_1', 'target': 'entity_2', 'type': 'connects_to'},
            {'source': 'entity_2', 'target': 'entity_3', 'type': 'relates_to'}
        ]
    }
    scenario_queue.put((1000, scenario, 1))
    print("  Scenario added!")
    
    # Import and start worker
    from ica_framework.learning.worker_functions import continuous_parallel_worker
    
    print("🚀 Starting worker process...")
    worker = mp.Process(
        target=continuous_parallel_worker,
        args=(0, scenario_queue, results_queue, database_config, "neo4j", stop_event)
    )
    worker.start()
    
    # Wait longer for initialization and processing
    print("📊 Waiting for worker to process scenario...")
    results = []
    timeout = 30  # 30 seconds - much longer
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = results_queue.get(timeout=3.0)  # Longer timeout per result
            results.append(result)
            print(f"📈 Got result: {result}")
            
            # If we got one result, that's enough for this test
            if len(results) >= 1 and 'worker_id' in results[0]:
                print("✅ Got actual result from worker!")
                break
                
        except Exception as e:
            print(f"Waiting... ({e})")
            # Don't break on timeout, keep waiting
            continue
    
    # Now set stop event and wait for worker to finish
    print("🛑 Setting stop event...")
    stop_event.set()
    worker.join(timeout=10.0)
    if worker.is_alive():
        print("Terminating worker...")
        worker.terminate()
        worker.join()
    
    print(f"\n📊 Results Summary:")
    print(f"  Scenarios added: 1")
    print(f"  Results collected: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"  Result {i}: {result}")
    
    if len(results) > 0 and 'worker_id' in results[0]:
        result = results[0]
        nodes_added = result.get('nodes_added', 0)
        edges_added = result.get('edges_added', 0)
        print(f"  Nodes added: {nodes_added}")
        print(f"  Edges added: {edges_added}")
        print("✅ SUCCESS: Worker processed scenario and returned results!")
        return True
    else:
        print("❌ FAILURE: No valid results from worker")
        return False

if __name__ == "__main__":
    success = test_simple_worker()
    if success:
        print("\n🎉 WORKER IS WORKING! The continuous learning issue is likely with queue timing or stop events.")
    else:
        print("\n💥 WORKER IS STILL BROKEN - need more debugging")
