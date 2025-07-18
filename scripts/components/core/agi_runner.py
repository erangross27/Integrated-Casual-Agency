#!/usr/bin/env python3
"""
AGI Runner Module
Main coordinator for the TRUE AGI continuous learning system
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ..system import SystemUtils, SignalHandler
from .session_manager import SessionManager
from .component_initializer import ComponentInitializer
from .learning_coordinator import LearningCoordinator
from .main_loop_controller import MainLoopController
from .shutdown_manager import ShutdownManager

# Setup utilities
SystemUtils.setup_windows_encoding()
flush_print = SystemUtils.flush_print

# Disable verbose logging
logging.getLogger('ica_framework').setLevel(logging.WARNING)


class AGIRunner:
    """Main TRUE AGI continuous learning system coordinator"""
    
    def __init__(self):
        self.running = True
        # No database config needed - using W&B analytics and file storage
        
        # Component managers
        self.session_manager = None
        self.component_initializer = None
        self.learning_coordinator = None
        self.main_loop_controller = None
        self.shutdown_manager = None
        
        # Setup signal handling
        self.signal_handler = SignalHandler(self._shutdown_gracefully)
        
        flush_print("[INIT] ✅ TRUE AGI Runner initialized")
    
    def initialize_system(self):
        """Initialize the complete system"""
        flush_print("🚀 STARTING TRUE AGI CONTINUOUS LEARNING SYSTEM")
        flush_print("=" * 60)
        
        # Initialize component initializer (no database config needed)
        self.component_initializer = ComponentInitializer()
        
        # Initialize all components
        if not self.component_initializer.initialize_all_components():
            flush_print("[ERROR] ❌ Failed to initialize components")
            return False
        
        # Get initialized components
        components = self.component_initializer.get_components()
        
        # Initialize session manager
        self.session_manager = SessionManager(
            components['knowledge_graph'],
            components['database_manager']
        )
        
        # Try to restore previous session
        self.session_manager.attempt_session_restoration()
        
        # Apply session restoration to components
        self.session_manager.apply_session_restoration(
            components['agi_agent'],
            components['gpu_processor'],
            components['world_simulator']
        )
        
        # Initialize learning coordinator
        self.learning_coordinator = LearningCoordinator(
            components['agi_agent'],
            components['agi_monitor'],
            components['gpu_worker']
        )
        
        # Initialize main loop controller
        self.main_loop_controller = MainLoopController(
            components['world_simulator'],
            components['agi_agent'],
            components['database_manager'],
            components['gpu_processor']
        )
        
        # Initialize shutdown manager
        self.shutdown_manager = ShutdownManager(
            components['database_manager'],
            components['agi_agent'],
            components['gpu_processor'],
            components['agi_monitor'],
            components['gpu_worker'],
            components['world_simulator'],
            components['knowledge_graph']
        )
        
        # Show GPU status
        if components['gpu_processor'] and components['gpu_processor'].use_gpu:
            flush_print("✅ [GPU] GPU Acceleration: ENABLED - utilization optimized")
        else:
            flush_print("⚠️ [GPU] GPU Acceleration: DISABLED (using CPU)")
        
        flush_print("[INIT] ✅ System initialization complete")
        return True
    
    def start_learning(self):
        """Start the learning process"""
        flush_print("=" * 60)
        
        if not self.learning_coordinator:
            flush_print("[ERROR] ❌ Cannot start learning - learning coordinator not initialized")
            return False
        
        if not self.learning_coordinator.start_learning_process():
            flush_print("[ERROR] ❌ Failed to start learning")
            return False
        
        flush_print("[SUCCESS] ✅ TRUE AGI Continuous Learning System running!")
        flush_print("[INFO] 🎯 System learning autonomously through environmental interaction")
        flush_print("[INFO] 🛑 Press Ctrl+C to stop gracefully")
        flush_print("=" * 60)
        
        return True
    
    def run_main_loop(self):
        """Run the main system loop"""
        if not self.main_loop_controller:
            flush_print("[ERROR] ❌ Cannot run main loop - controller not initialized")
            return
        
        self.main_loop_controller.run_main_loop()
    
    def _shutdown_gracefully(self):
        """Gracefully shutdown the system"""
        self.running = False
        
        # Stop main loop
        if self.main_loop_controller:
            self.main_loop_controller.set_running(False)
        
        # Perform graceful shutdown
        if self.shutdown_manager:
            self.shutdown_manager.perform_graceful_shutdown()
        else:
            flush_print("[SHUTDOWN] ⚠️ No shutdown manager - performing basic shutdown")
    
    def run(self):
        """Main run method"""
        try:
            # Initialize system
            if not self.initialize_system():
                return
            
            # Start learning
            if not self.start_learning():
                return
            
            # Run main loop
            self.run_main_loop()
            
        except Exception as e:
            flush_print(f"[ERROR] ❌ Critical error: {e}")
        finally:
            self._shutdown_gracefully()
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        stats = {
            'running': self.running,
            'session_manager': self.session_manager.get_stats() if self.session_manager else None,
            'learning_coordinator': self.learning_coordinator.get_learning_stats() if self.learning_coordinator else None,
            'main_loop_controller': self.main_loop_controller.get_loop_stats() if self.main_loop_controller else None,
            'shutdown_manager': self.shutdown_manager.get_shutdown_stats() if self.shutdown_manager else None
        }
        return stats


def main():
    """Main entry point"""
    runner = AGIRunner()
    runner.run()


if __name__ == "__main__":
    main()
