#!/usr/bin/env python3
"""
Simple test to see if multiprocessing workers can even start
"""

import time
import multiprocessing as mp
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def simple_worker(worker_id, results_queue):
    """Simplest possible worker test"""
    try:
        print(f"Worker {worker_id}: Starting!", flush=True)
        
        # Just put a simple message
        result = {
            'worker_id': worker_id,
            'message': f'Hello from worker {worker_id}',
            'timestamp': time.time()
        }
        
        results_queue.put(result)
        print(f"Worker {worker_id}: Sent result!", flush=True)
        
    except Exception as e:
        print(f"Worker {worker_id}: ERROR - {e}", flush=True)
        import traceback
        traceback.print_exc()

def test_simple_multiprocessing():
    """Test if multiprocessing works at all"""
    
    print("🧪 Testing simple multiprocessing...")
    
    # Create queue
    results_queue = mp.Queue()
    
    # Start simple workers
    workers = []
    for worker_id in range(2):
        print(f"Starting worker {worker_id}...")
        worker = mp.Process(
            target=simple_worker,
            args=(worker_id, results_queue)
        )
        worker.start()
        workers.append(worker)
    
    # Collect results
    print("📊 Collecting results...")
    results = []
    timeout = 5  # 5 seconds timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout and len(results) < 2:
        try:
            result = results_queue.get(timeout=1.0)
            results.append(result)
            print(f"📈 Got result: {result}")
        except:
            print("No result received in timeout")
            break
    
    # Wait for workers
    for worker in workers:
        worker.join(timeout=2.0)
        if worker.is_alive():
            print(f"Terminating worker {worker.pid}")
            worker.terminate()
    
    print(f"\n📊 Summary:")
    print(f"  Results collected: {len(results)}")
    
    if len(results) == 2:
        print("✅ SUCCESS: Multiprocessing works!")
    else:
        print("❌ ISSUE: Multiprocessing failed!")

if __name__ == "__main__":
    test_simple_multiprocessing()
