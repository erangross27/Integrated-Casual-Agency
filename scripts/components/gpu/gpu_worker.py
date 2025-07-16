#!/usr/bin/env python3
"""
GPU Worker Module
Handles dedicated GPU worker thread for continuous parallel processing
"""

import time
import threading
import torch


class GPUWorker:
    """Dedicated GPU worker for continuous parallel processing"""
    
    def __init__(self, gpu_processor, agi_agent, database_manager):
        self.gpu_processor = gpu_processor
        self.agi_agent = agi_agent
        self.database_manager = database_manager
        
        self.running = False
        self.worker_thread = None
        self.worker_cycle = 0
    
    def start_worker(self):
        """Start the GPU worker thread"""
        if not self.gpu_processor.use_gpu:
            return False
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        print("[GPU] 🔥 Started dedicated GPU worker thread for parallel processing")
        return True
    
    def stop_worker(self):
        """Stop the GPU worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _worker_loop(self):
        """Main GPU worker loop"""
        print("[GPU] 🔥 GPU worker thread started - continuous parallel processing")
        
        while self.running:
            try:
                self.worker_cycle += 1
                
                # Continuous GPU processing
                observation_data = {
                    'worker_cycle': self.worker_cycle,
                    'timestamp': time.time(),
                    'type': 'gpu_worker_observation',
                    'continuous_processing': True
                }
                
                # Process multiple batches in parallel
                batch_results = []
                
                # Process 20 batches simultaneously for better 6GB GPU utilization
                for batch_num in range(20):  # Increased from 10 to 20 for better GPU utilization
                    batch_observation = {
                        **observation_data,
                        'batch_number': batch_num
                    }
                    
                    results = self.gpu_processor.process_agi_learning(batch_observation)
                    if results:
                        batch_results.append(results)
                        
                        # Store learning data to database
                        self.database_manager.store_agi_learning(self.agi_agent)
                
                # Adaptive sleep based on GPU utilization
                sleep_time = self._calculate_sleep_time()
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"[GPU] ⚠️ Error in GPU worker: {e}")
                time.sleep(5)
    
    def _calculate_sleep_time(self):
        """Calculate adaptive sleep time based on GPU utilization"""
        if self.worker_cycle % 50 == 0:
            # Check GPU memory usage
            if torch.cuda.is_available():
                gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                if gpu_memory_percent > 80:  # High usage
                    return 2  # Brief pause to prevent overflow
                elif gpu_memory_percent > 60:
                    return 1  # Moderate processing
                else:
                    return 0.5  # Normal processing
            else:
                return 1
        else:
            return 0.5  # Default for 6GB utilization
