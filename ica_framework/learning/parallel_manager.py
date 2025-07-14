"""
Parallel Processing Manager for ICA Framework Learning
Handles both batch and continuous parallel learning modes
"""

import multiprocessing as mp
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

from .scenario_generators import PhysicsSimulation
from .worker_functions import continuous_parallel_worker, parallel_learning_worker


class ContinuousParallelManager:
    """Manager for coordinating continuous parallel learning processes"""
    
    def __init__(self, num_workers: Optional[int] = None):
        if num_workers is None:
            # Use 90% of available cores for maximum utilization
            self.num_workers = max(1, int(mp.cpu_count() * 0.9))
        else:
            self.num_workers = min(num_workers, mp.cpu_count())
        
        self.scenario_queue = mp.Queue(maxsize=self.num_workers * 10)  # Larger buffer
        self.results_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.workers = []
        self.worker_stats = {}
        self.total_scenarios_processed = 0
        self.total_nodes_added = 0
        self.total_edges_added = 0
        self.queue_maxsize = self.num_workers * 10  # Store maxsize for reference
        
    def start_workers(self, database_config: Dict[str, Any], database_backend: str):
        """Start all worker processes"""
        print(f"Starting {self.num_workers} continuous parallel workers...")
        
        for worker_id in range(self.num_workers):
            worker = mp.Process(
                target=continuous_parallel_worker,
                args=(worker_id, self.scenario_queue, self.results_queue, 
                     database_config, database_backend, self.stop_event)
            )
            worker.start()
            self.workers.append(worker)
            
        print(f"{len(self.workers)} workers started and ready for continuous processing")
    
    def add_scenario(self, scenario_count: int, scenario: Dict[str, Any], current_round: int) -> bool:
        """Add a scenario to the processing queue"""
        try:
            self.scenario_queue.put((scenario_count, scenario, current_round), timeout=0.1)
            return True
        except:
            return False  # Queue is full
    
    def get_results(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Get available results from workers"""
        results = []
        
        while True:
            try:
                result = self.results_queue.get(timeout=timeout)
                results.append(result)
            except:
                break
                
        return results
    
    def stop_workers(self):
        """Stop all worker processes gracefully"""
        print("\nStopping continuous parallel workers...")
        
        # Signal stop
        self.stop_event.set()
        
        # Send poison pills to ensure workers exit
        for _ in range(self.num_workers):
            try:
                self.scenario_queue.put(None, timeout=1.0)
            except:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
        
        print("All workers stopped")
    
    def get_queue_size(self) -> int:
        """Get current scenario queue size"""
        return self.scenario_queue.qsize()
    
    def is_queue_full(self) -> bool:
        """Check if scenario queue is full"""
        return self.scenario_queue.full()


class ParallelLearningManager:
    """Manager for coordinating batch parallel learning processes"""
    
    def __init__(self, num_workers: Optional[int] = None):
        if num_workers is None:
            # Use 75% of available cores, leaving some for system processes
            self.num_workers = max(1, int(mp.cpu_count() * 0.75))
        else:
            self.num_workers = min(num_workers, mp.cpu_count())
        
        self.results_queue = mp.Queue()
        self.total_scenarios_processed = 0
        self.total_nodes_added = 0
        self.total_edges_added = 0
        self.worker_stats = {}
        
    def process_batch_parallel(self, 
                             base_scenarios: List[Dict[str, Any]], 
                             start_scenario: int, 
                             database_config: Dict[str, Any], 
                             database_backend: str, 
                             batch_size: int = 10) -> Dict[str, Any]:
        """Process a batch of scenarios in parallel"""
        # Create scenario batches
        batches, next_scenario = self._create_scenario_batches(
            base_scenarios, start_scenario, self.num_workers, batch_size
        )
        
        # Start parallel workers
        batch_start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit worker tasks
            futures = []
            for worker_id, batch in enumerate(batches):
                if batch:  # Only submit if batch has scenarios
                    future = executor.submit(
                        parallel_learning_worker, 
                        worker_id, batch, database_config, database_backend, self.results_queue
                    )
                    futures.append(future)
            
            # Collect results as they complete
            batch_results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per worker
                    batch_results.append(result)
                except Exception as e:
                    batch_results.append({
                        'worker_id': -1,
                        'error': str(e),
                        'scenarios_processed': 0,
                        'nodes_added': 0,
                        'edges_added': 0,
                        'processing_time': 0,
                        'learning_results': []
                    })
        
        batch_time = time.time() - batch_start_time
        
        # Aggregate results
        batch_summary = {
            'total_scenarios': sum(r['scenarios_processed'] for r in batch_results),
            'total_nodes_added': sum(r['nodes_added'] for r in batch_results),
            'total_edges_added': sum(r['edges_added'] for r in batch_results),
            'batch_time': batch_time,
            'workers_used': len([r for r in batch_results if r['scenarios_processed'] > 0]),
            'next_scenario': next_scenario,
            'worker_results': batch_results
        }
        
        return batch_summary
    
    def _create_scenario_batches(self, 
                               base_scenarios: List[Dict[str, Any]], 
                               start_scenario: int, 
                               num_workers: int, 
                               batch_size: int = 10) -> Tuple[List[List], int]:
        """Create scenario batches for parallel processing"""
        batches = [[] for _ in range(num_workers)]
        
        scenario_count = start_scenario
        worker_idx = 0
        
        # Create batches with round-robin distribution
        for _ in range(batch_size * num_workers):
            current_round = (scenario_count // len(base_scenarios)) + 1
            scenario_in_round = scenario_count % len(base_scenarios)
            scenario = base_scenarios[scenario_in_round].copy()
            
            batches[worker_idx].append((scenario_count, scenario, current_round))
            
            worker_idx = (worker_idx + 1) % num_workers
            scenario_count += 1
        
        return batches, scenario_count
