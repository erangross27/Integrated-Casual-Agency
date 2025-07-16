#!/usr/bin/env python3
"""
Main TRUE AGI Runner Module
Coordinates all components for the TRUE AGI continuous learning system
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import component modules
from .system import SystemUtils, SignalHandler, ProcessManager
from .gpu import GPUProcessor
from .database import DatabaseManager
from .monitoring import AGIMonitor
from .gpu import GPUWorker

# Import TRUE AGI Framework
from ica_framework.sandbox import WorldSimulator, AGIAgent
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

# Setup utilities
SystemUtils.setup_windows_encoding()
flush_print = SystemUtils.flush_print

# Disable verbose logging
logging.getLogger('ica_framework').setLevel(logging.WARNING)


class TrueAGIRunner:
    """Main TRUE AGI continuous learning system runner"""
    
    def __init__(self):
        self.running = True
        self.database_config = SystemUtils.load_database_config(PROJECT_ROOT)
        
        # Core components
        self.world_simulator = None
        self.agi_agent = None
        self.knowledge_graph = None
        
        # System components
        self.gpu_processor = None
        self.database_manager = None
        self.agi_monitor = None
        self.gpu_worker = None
        
        # Setup signal handling
        self.signal_handler = SignalHandler(self._shutdown_gracefully)
        
        flush_print("[INIT] ✅ TRUE AGI Runner initialized")
    
    def initialize_components(self):
        """Initialize all system components"""
        flush_print("[INIT] 🔧 Initializing system components...")
        
        # Initialize GPU processor
        self.gpu_processor = GPUProcessor()
        
        if self.gpu_processor.use_gpu:
            flush_print("✅ [GPU] GPU acceleration models initialized")
        else:
            flush_print("⚠️ [GPU] GPU acceleration models failed to initialize")
        
        # Initialize knowledge graph
        self.knowledge_graph = EnhancedKnowledgeGraph(
            backend='neo4j',
            config=self.database_config,
            auto_connect=True
        )
        
        if self.knowledge_graph.connect():
            flush_print("[OK] ✅ Knowledge graph connection established")
        else:
            flush_print("[ERROR] ❌ Knowledge graph connection failed")
            return False
        
        # Initialize database manager
        self.database_manager = DatabaseManager(self.knowledge_graph)
        
        # Initialize world simulator and AGI agent
        self.world_simulator = WorldSimulator()
        self.agi_agent = AGIAgent(self.world_simulator, self.knowledge_graph)
        
        # Configure for continuous learning
        self.world_simulator.set_simulation_speed(0.1)
        self.world_simulator.set_auto_generate_events(True, 0.2)
        self.agi_agent.set_exploration_rate(0.3)
        self.agi_agent.set_novelty_threshold(0.6)
        
        # Initialize monitoring components
        self.agi_monitor = AGIMonitor(
            self.world_simulator, 
            self.agi_agent, 
            self.gpu_processor, 
            self.database_manager
        )
        
        self.gpu_worker = GPUWorker(
            self.gpu_processor, 
            self.agi_agent, 
            self.database_manager
        )
        
        flush_print("[INIT] ✅ All components initialized successfully")
        return True
    
    def start_learning(self):
        """Start the TRUE AGI learning process"""
        flush_print("[AGI] 🚀 Starting TRUE AGI Learning Process...")
        
        # Start the AGI learning system
        self.agi_agent.start_learning()
        
        # Start monitoring
        self.agi_monitor.start_monitoring()
        
        # Start GPU worker
        self.gpu_worker.start_worker()
        
        flush_print("[AGI] ✅ TRUE AGI Learning Process started")
        return True
    
    def run(self):
        """Main run loop"""
        flush_print("🚀 STARTING TRUE AGI CONTINUOUS LEARNING SYSTEM")
        flush_print("=" * 60)
        
        # Initialize components first
        if not self.initialize_components():
            flush_print("[ERROR] ❌ Failed to initialize components")
            return
        
        # Show GPU status AFTER initialization
        if self.gpu_processor and self.gpu_processor.use_gpu:
            flush_print("✅ [GPU] GPU Acceleration: ENABLED - 6GB utilization optimized")
        else:
            flush_print("⚠️ [GPU] GPU Acceleration: DISABLED (using CPU)")
        
        flush_print("=" * 60)
        
        # Start learning
        if not self.start_learning():
            flush_print("[ERROR] ❌ Failed to start learning")
            return
        
        flush_print("[SUCCESS] ✅ TRUE AGI Continuous Learning System running!")
        flush_print("[INFO] 🎯 System learning autonomously through environmental interaction")
        flush_print("[INFO] 🛑 Press Ctrl+C to stop gracefully")
        flush_print("=" * 60)
        
        # Main loop
        self._main_loop()
    
    def _main_loop(self):
        """Main system loop"""
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                
                # System health check
                if self.world_simulator and self.agi_agent:
                    world_stats = self.world_simulator.get_learning_statistics()
                    sim_stats = world_stats.get('simulation', {})
                    
                    if sim_stats.get('steps', 0) > 0:
                        status_msg = f"[STATUS] ✅ TRUE AGI Learning Active - {sim_stats.get('steps', 0)} steps"
                        flush_print(status_msg)
                    else:
                        flush_print("[STATUS] ⚠️ TRUE AGI Learning appears inactive")
                
        except KeyboardInterrupt:
            flush_print("\n[STOP] 🛑 Interrupted by user (Ctrl+C)")
        except Exception as e:
            flush_print(f"[ERROR] ❌ Unexpected error: {e}")
        finally:
            self._shutdown_gracefully()
    
    def _shutdown_gracefully(self):
        """Gracefully shutdown the system"""
        flush_print("\n[SHUTDOWN] 🛑 Shutting down TRUE AGI system...")
        
        self.running = False
        
        # Stop components
        if self.agi_monitor:
            self.agi_monitor.stop_monitoring()
        
        if self.gpu_worker:
            self.gpu_worker.stop_worker()
        
        if self.agi_agent:
            self.agi_agent.stop_learning()
            flush_print("[SHUTDOWN] ✅ AGI Agent stopped")
        
        if self.world_simulator:
            self.world_simulator.stop()
            flush_print("[SHUTDOWN] ✅ World Simulator stopped")
        
        if self.gpu_processor:
            self.gpu_processor.clear_gpu_memory()
            flush_print("[SHUTDOWN] ✅ GPU memory cleared")
        
        flush_print("[SHUTDOWN] ✅ TRUE AGI system shutdown complete")


def main():
    """Main entry point"""
    runner = TrueAGIRunner()
    runner.run()


if __name__ == "__main__":
    main()
