#!/usr/bin/env python3
"""
Main Loop Controller Module
Handles the main execution loop and periodic operations
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ..system import SystemUtils

# Setup utilities
flush_print = SystemUtils.flush_print


class MainLoopController:
    """Controls the main execution loop and periodic operations"""
    
    def __init__(self, world_simulator, agi_agent, database_manager, gpu_processor):
        self.world_simulator = world_simulator
        self.agi_agent = agi_agent
        self.database_manager = database_manager
        self.gpu_processor = gpu_processor
        self.running = True
        
        # Periodic save configuration
        self.save_counter = 0
        self.save_interval = 8  # Save every 8 iterations (4 minutes)
    
    def set_running(self, running):
        """Set the running state"""
        self.running = running
    
    def perform_system_health_check(self):
        """Perform system health check"""
        if self.world_simulator and self.agi_agent:
            try:
                world_stats = self.world_simulator.get_learning_statistics()
                sim_stats = world_stats.get('simulation', {})
                
                if sim_stats.get('steps', 0) > 0:
                    # Simple health check without confusing step counts
                    return True
                else:
                    flush_print("[STATUS] ⚠️ TRUE AGI Learning appears inactive")
                    return False
            except Exception as e:
                flush_print(f"[STATUS] ⚠️ Health check failed: {e}")
                return False
        else:
            flush_print("[STATUS] ⚠️ Missing components for health check")
            return False
    
    def perform_periodic_save(self):
        """Perform periodic save of learning state"""
        if self.database_manager and self.agi_agent:
            try:
                flush_print("[PERIODIC] 💾 Performing periodic save...")
                flush_print("[PERIODIC] 🧠 Saving neural network weights and biases...")
                
                # Save complete learning state (neural networks + AGI state)
                save_success = self.database_manager.store_learning_state(self.agi_agent, self.gpu_processor)
                
                if save_success:
                    flush_print("[PERIODIC] ✅ Learning state saved successfully")
                    flush_print("[PERIODIC] ✅ Neural network weights and biases saved!")
                else:
                    flush_print("[PERIODIC] ⚠️ Some components failed to save")
                
                # Get current GPU stats if available
                gpu_stats = None
                if self.gpu_processor:
                    gpu_stats = self.gpu_processor.get_gpu_stats()
                
                # Save session metadata
                self.database_manager.store_learning_state(self.agi_agent, gpu_stats)
                
                # Log learning metrics to W&B
                try:
                    if hasattr(self.database_manager, 'log_learning_metrics'):
                        # Get world simulation stats
                        world_stats = {}
                        if self.world_simulator:
                            world_stats = self.world_simulator.get_learning_statistics()
                        
                        # Prepare metrics for W&B
                        learning_metrics = {
                            'save_success': save_success,
                            'periodic_save_completed': True,
                            'simulation_steps': world_stats.get('simulation', {}).get('steps', 0),
                            'learning_iterations': world_stats.get('learning', {}).get('total_iterations', 0),
                            'timestamp': time.time()
                        }
                        
                        # Add GPU metrics if available
                        if gpu_stats:
                            learning_metrics.update({
                                'gpu_utilization': gpu_stats.get('utilization', 0),
                                'gpu_memory_used': gpu_stats.get('memory_used', 0),
                                'gpu_memory_total': gpu_stats.get('memory_total', 0)
                            })
                        
                        self.database_manager.log_learning_metrics(learning_metrics)
                        flush_print("[PERIODIC] 📊 Metrics logged to W&B dashboard")
                        
                except Exception as metrics_e:
                    flush_print(f"[PERIODIC] ⚠️ W&B metrics logging failed: {metrics_e}")
                
                return save_success
                
            except Exception as e:
                flush_print(f"[PERIODIC] ⚠️ Save failed: {e}")
                # Try to save neural networks even if other components fail
                try:
                    if self.database_manager and self.gpu_processor:
                        flush_print("[PERIODIC] 🔄 Attempting emergency neural network save...")
                        neural_save_success = self.database_manager.neural_persistence.save_gpu_models(self.gpu_processor)
                        if neural_save_success:
                            flush_print("[PERIODIC] ✅ Emergency neural network save successful!")
                        else:
                            flush_print("[PERIODIC] ⚠️ Emergency neural network save failed!")
                except Exception as emergency_e:
                    flush_print(f"[PERIODIC] ⚠️ Emergency save also failed: {emergency_e}")
                return False
        else:
            flush_print("[PERIODIC] ⚠️ Cannot save - missing database manager or AGI agent")
            return False
    
    def run_main_loop(self):
        """Run the main system loop"""
        flush_print("[LOOP] 🔄 Starting main system loop...")
        
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                self.save_counter += 1
                
                # System health check
                self.perform_system_health_check()
                
                # Periodic save every 2 minutes (4 x 30 seconds)
                if self.save_counter >= self.save_interval:
                    self.perform_periodic_save()
                    self.save_counter = 0
                
        except KeyboardInterrupt:
            flush_print("\n[STOP] 🛑 Interrupted by user (Ctrl+C)")
            self.running = False
        except Exception as e:
            flush_print(f"[ERROR] ❌ Unexpected error in main loop: {e}")
            self.running = False
    
    def get_loop_stats(self):
        """Get main loop statistics"""
        return {
            'running': self.running,
            'save_counter': self.save_counter,
            'save_interval': self.save_interval,
            'next_save_in': self.save_interval - self.save_counter
        }
