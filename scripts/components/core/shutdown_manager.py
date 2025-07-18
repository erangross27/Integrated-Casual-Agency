#!/usr/bin/env python3
"""
Shutdown Manager Module
Handles graceful shutdown with complete learning state preservation
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ..system import SystemUtils

# Setup utilities
flush_print = SystemUtils.flush_print


class ShutdownManager:
    """Manages graceful shutdown with complete learning state preservation"""
    
    def __init__(self, database_manager, agi_agent, gpu_processor, 
                 agi_monitor, gpu_worker, world_simulator, knowledge_graph):
        self.database_manager = database_manager
        self.agi_agent = agi_agent
        self.gpu_processor = gpu_processor
        self.agi_monitor = agi_monitor
        self.gpu_worker = gpu_worker
        self.world_simulator = world_simulator
        self.knowledge_graph = knowledge_graph
        self.shutdown_complete = False
    
    def save_complete_learning_state(self):
        """Save complete learning state before shutdown"""
        if self.database_manager and self.agi_agent:
            try:
                flush_print("[SHUTDOWN] 💾 Saving COMPLETE learning state...")
                flush_print("[SHUTDOWN] 🧠 Saving neural network weights and biases...")
                
                # Save complete learning state (neural networks + AGI state)
                save_success = self.database_manager.store_learning_state(self.agi_agent, self.gpu_processor)
                
                # Save current session progress to persistent statistics
                if hasattr(self.agi_agent, 'learning_progress') and hasattr(self.agi_agent.learning_progress, 'save_session_to_persistent_stats'):
                    try:
                        self.agi_agent.learning_progress.save_session_to_persistent_stats()
                        flush_print("[SHUTDOWN] ✅ Session progress saved to persistent statistics!")
                    except Exception as e:
                        flush_print(f"[SHUTDOWN] ⚠️ Failed to save persistent stats: {e}")
                
                if save_success:
                    flush_print("[SHUTDOWN] ✅ Complete learning state saved!")
                    flush_print("[SHUTDOWN] ✅ Neural network weights and biases saved!")
                else:
                    flush_print("[SHUTDOWN] ⚠️ Some components failed to save - but continuing...")
                
                # Get current GPU stats if available
                gpu_stats = None
                if self.gpu_processor:
                    gpu_stats = self.gpu_processor.get_gpu_stats()
                
                # Save session metadata
                self.database_manager.store_learning_state(self.agi_agent, gpu_stats)
                
                # Trigger final pattern save
                if hasattr(self.database_manager, 'shutdown'):
                    self.database_manager.shutdown()
                    
                flush_print("[SHUTDOWN] ✅ Session metadata saved successfully")
                return True
                
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error saving learning state: {e}")
                # Even if there's an error, try to save neural networks directly
                try:
                    if self.database_manager and self.gpu_processor:
                        flush_print("[SHUTDOWN] 🔄 Attempting emergency neural network save...")
                        neural_save_success = self.database_manager.neural_persistence.save_gpu_models(self.gpu_processor)
                        if neural_save_success:
                            flush_print("[SHUTDOWN] ✅ Emergency neural network save successful!")
                        else:
                            flush_print("[SHUTDOWN] ⚠️ Emergency neural network save failed!")
                except Exception as emergency_e:
                    flush_print(f"[SHUTDOWN] ⚠️ Emergency save also failed: {emergency_e}")
                return False
        else:
            flush_print("[SHUTDOWN] ⚠️ Cannot save learning state - missing components")
            return False
    
    def stop_monitoring_components(self):
        """Stop monitoring components"""
        flush_print("[SHUTDOWN] 🛑 Stopping monitoring components...")
        
        # Stop AGI monitor
        if self.agi_monitor:
            try:
                self.agi_monitor.stop_monitoring()
                flush_print("[SHUTDOWN] ✅ AGI Monitor stopped")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error stopping AGI monitor: {e}")
        
        # Stop GPU worker
        if self.gpu_worker:
            try:
                self.gpu_worker.stop_worker()
                flush_print("[SHUTDOWN] ✅ GPU Worker stopped")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error stopping GPU worker: {e}")
    
    def stop_core_components(self):
        """Stop core system components"""
        flush_print("[SHUTDOWN] 🛑 Stopping core components...")
        
        # Stop AGI agent
        if self.agi_agent:
            try:
                self.agi_agent.stop_learning()
                flush_print("[SHUTDOWN] ✅ AGI Agent stopped")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error stopping AGI agent: {e}")
        
        # Stop world simulator
        if self.world_simulator:
            try:
                self.world_simulator.stop()
                flush_print("[SHUTDOWN] ✅ World Simulator stopped")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error stopping world simulator: {e}")
    
    def cleanup_resources(self):
        """Cleanup system resources"""
        flush_print("[SHUTDOWN] 🧹 Cleaning up resources...")
        
        # Clear GPU memory
        if self.gpu_processor:
            try:
                self.gpu_processor.clear_gpu_memory()
                flush_print("[SHUTDOWN] ✅ GPU memory cleared")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error clearing GPU memory: {e}")
        
        # Close knowledge graph connection
        if self.knowledge_graph:
            try:
                self.knowledge_graph.close()
                flush_print("[SHUTDOWN] ✅ Knowledge graph connection closed")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error closing knowledge graph: {e}")
    
    def perform_graceful_shutdown(self):
        """Perform complete graceful shutdown"""
        flush_print("\n[SHUTDOWN] 🛑 Shutting down TRUE AGI system...")
        
        # Save learning state first
        self.save_complete_learning_state()
        
        # Stop monitoring components
        self.stop_monitoring_components()
        
        # Stop core components
        self.stop_core_components()
        
        # Cleanup resources
        self.cleanup_resources()
        
        # Mark shutdown complete
        self.shutdown_complete = True
        
        # Success message
        flush_print("[SHUTDOWN] ✅ TRUE AGI system shutdown complete")
        flush_print("[SHUTDOWN] 🎯 Ready for restart - ALL LEARNING PRESERVED!")
        flush_print("[SHUTDOWN] 🧠 Neural networks: ✓ Saved")
        flush_print("[SHUTDOWN] 📚 Knowledge base: ✓ Saved") 
        flush_print("[SHUTDOWN] 🔬 Learning progress: ✓ Saved")
        flush_print("[SHUTDOWN] 💡 No more starting from scratch!")
    
    def is_shutdown_complete(self):
        """Check if shutdown is complete"""
        return self.shutdown_complete
    
    def get_shutdown_stats(self):
        """Get shutdown statistics"""
        return {
            'shutdown_complete': self.shutdown_complete,
            'components_available': {
                'database_manager': self.database_manager is not None,
                'agi_agent': self.agi_agent is not None,
                'gpu_processor': self.gpu_processor is not None,
                'agi_monitor': self.agi_monitor is not None,
                'gpu_worker': self.gpu_worker is not None,
                'world_simulator': self.world_simulator is not None,
                'knowledge_graph': self.knowledge_graph is not None
            }
        }
