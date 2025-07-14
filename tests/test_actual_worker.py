#!/usr/bin/env python3
"""
Test the actual continuous_parallel_worker function
"""

import time
import multiprocessing as mp
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_actual_worker():
    """Test the real continuous_parallel_worker function"""
    
    # Load Neo4j config
    config_file = Path("config/database/neo4j.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        database_config = config_data['config']
    
    print("🧪 Testing actual continuous_parallel_worker...")
    
    # Create queues like continuous learning does
    scenario_queue = mp.Queue(maxsize=150)  # Same as continuous learning
    results_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Add some test scenarios to the queue
    print("📝 Adding test scenarios to queue...")
    for i in range(3):
        scenario = {
            'name': f'Test Scenario {i}',
            'entities': [f'entity_{i}_1', f'entity_{i}_2', f'entity_{i}_3'],
            'relationships': [
                {'source': f'entity_{i}_1', 'target': f'entity_{i}_2', 'type': 'connects_to'},
                {'source': f'entity_{i}_2', 'target': f'entity_{i}_3', 'type': 'relates_to'}
            ]
        }
        scenario_queue.put((1000 + i, scenario, 1))
        print(f"  Added scenario {i}")
    
    # Import and start the actual worker function
    from ica_framework.learning.worker_functions import continuous_parallel_worker
    
    print("🚀 Starting actual worker process...")
    worker = mp.Process(
        target=continuous_parallel_worker,
        args=(0, scenario_queue, results_queue, database_config, "neo4j", stop_event)
    )
    worker.start()
    
    # Collect results for a short time
    print("📊 Collecting results from actual worker...")
    results = []
    timeout = 10  # 10 seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = results_queue.get(timeout=2.0)
            results.append(result)
            print(f"📈 Got result: worker_id={result.get('worker_id')}, "
                  f"nodes_added={result.get('nodes_added', 'N/A')}, "
                  f"edges_added={result.get('edges_added', 'N/A')}")
            
            # If we got results from all 3 scenarios, we can stop
            if len(results) >= 3:
                break
                
        except Exception as e:
            print(f"Timeout waiting for results: {e}")
            break
    
    # Stop the worker
    print("🛑 Stopping worker...")
    stop_event.set()
    worker.join(timeout=5.0)
    if worker.is_alive():
        print("Terminating worker...")
        worker.terminate()
    
    print(f"\n📊 Results Summary:")
    print(f"  Scenarios added to queue: 3")
    print(f"  Results collected: {len(results)}")
    
    if len(results) > 0:
        total_nodes = sum(r.get('nodes_added', 0) for r in results)
        total_edges = sum(r.get('edges_added', 0) for r in results)
        print(f"  Total nodes added: {total_nodes}")
        print(f"  Total edges added: {total_edges}")
        print("✅ SUCCESS: Worker produced results!")
    else:
        print("❌ ISSUE: No results from worker!")
        print("This means the worker is not processing scenarios or encountering errors")

if __name__ == "__main__":
    test_actual_worker()
