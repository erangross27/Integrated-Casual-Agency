#!/usr/bin/env python3
"""
Continuous Learning Runner
Keeps the ICA Framework learning running indefinitely using TRUE AGI system
"""

# Set encoding for Windows console
import sys
import os
if sys.platform == "win32":
    # Force UTF-8 encoding for Windows console
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    # Set console code page to UTF-8
    os.system('chcp 65001 > nul')
    
    # Set environment variable for unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'

import time
import signal
import json
import logging
import subprocess
import threading
import traceback
import random
import ctypes
import psutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from scripts folder
sys.path.insert(0, str(PROJECT_ROOT))

# DISABLE VERBOSE ICA FRAMEWORK LOGGING IMMEDIATELY
logging.getLogger('ica_framework').setLevel(logging.WARNING)
logging.getLogger('ica_framework.utils.logger').setLevel(logging.WARNING)
logging.getLogger('ica_framework.core').setLevel(logging.WARNING)
logging.getLogger('ica_framework.components').setLevel(logging.WARNING)

# Import TRUE AGI SYSTEM ONLY
from ica_framework.sandbox import WorldSimulator, AGIAgent
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
from ica_framework.utils.logger import ica_logger

def flush_print(*args, **kwargs):
    """Print with immediate flush to ensure output appears"""
    print(*args, **kwargs)
    sys.stdout.flush()


class TrueAGIContinuousRunner:
    """Enhanced continuous runner with TRUE AGI learning capabilities"""

    def __init__(self):
        self.running = True
        self.learning = None
        self.kg = None
        self.database_config = self._load_database_config()
        self.spawned_processes = []  # Track all spawned processes
        self.main_pid = os.getpid()  # Remember main process ID
        
        # TRUE AGI SYSTEM COMPONENTS
        self.world_simulator = None
        self.agi_agent = None
        self.true_agi_active = False
        
        # Learning statistics
        self.learning_sessions = []
        self.performance_history = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Windows 11 specific: Register for console control handler
        if sys.platform == "win32":
            try:
                from ctypes import wintypes
                
                def windows_console_handler(dwCtrlType):
                    """Windows console control handler for Ctrl+C, Ctrl+Break, etc."""
                    if dwCtrlType in (0, 1, 2):  # CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT
                        flush_print(f"\n[STOP] 🛑 Windows console event {dwCtrlType}, IMMEDIATE SHUTDOWN...")
                        self.running = False
                        self.true_agi_active = False
                        self._cleanup_all_processes()
                        os._exit(0)
                    return True
                
                # Register the console control handler
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleCtrlHandler(
                    ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(windows_console_handler),
                    True
                )
                flush_print("[INIT] ✅ Windows 11 console control handler registered")
            except Exception as e:
                flush_print(f"[INIT] Windows 11 console handler setup failed: {e}")

    def _load_database_config(self):
        """Load Neo4j configuration from config file"""
        config_file = PROJECT_ROOT / "config/database/neo4j.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    db_config = config_data['config']
                    flush_print(f"[OK] Loaded Neo4j config from {config_file}")
                    flush_print(f"   URI: {db_config['uri']}")
                    flush_print(f"   Database: {db_config['database']}")
                    flush_print(f"   User: {db_config['username']}")
                    flush_print()
                    return db_config
                
            except Exception as e:
                flush_print(f"[ERROR] Failed to load config file: {e}")
                
        flush_print("[WARNING] Using default Neo4j configuration")
        return {
            'uri': 'neo4j://127.0.0.1:7687',
            'username': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully and kill all Python processes - Windows 11 optimized"""
        flush_print(f"\n[STOP] 🛑 Received signal {signum}, IMMEDIATE WINDOWS 11 SHUTDOWN...")
        self.running = False
        self.true_agi_active = False
        
        # Stop TRUE AGI system
        if self.agi_agent:
            self.agi_agent.stop_learning()
        if self.world_simulator:
            self.world_simulator.stop()
        
        # Immediate nuclear cleanup
        flush_print("[STOP] 🔥 SIGNAL HANDLER - Windows 11 Emergency cleanup...")
        self._cleanup_all_processes()
        
        # Force exit immediately
        flush_print("[EXIT] 💀 Windows 11 Force exiting NOW...")
        os._exit(0)

    def _cleanup_all_processes(self):
        """Windows 11 optimized comprehensive cleanup of all processes and resources"""
        flush_print("[CLEANUP] Starting Windows 11 comprehensive cleanup...")
        
        # Stop continuous learning workers first
        if self.learning and hasattr(self.learning, 'continuous_manager'):
            try:
                flush_print("[CLEANUP] Stopping continuous learning workers...")
                if self.learning.continuous_manager:
                    self.learning.continuous_manager.stop_workers()
                flush_print("[CLEANUP] ✅ Workers stopped")
            except Exception as e:
                flush_print(f"[CLEANUP] Warning: Error stopping workers: {e}")
        
        # Stop TRUE AGI system
        if self.agi_agent:
            try:
                self.agi_agent.stop_learning()
                flush_print("[CLEANUP] ✅ AGI Agent stopped")
            except Exception as e:
                flush_print(f"[CLEANUP] Warning: Error stopping AGI agent: {e}")
        
        if self.world_simulator:
            try:
                self.world_simulator.stop()
                flush_print("[CLEANUP] ✅ World Simulator stopped")
            except Exception as e:
                flush_print(f"[CLEANUP] Warning: Error stopping world simulator: {e}")
        
        # WINDOWS 11 NUCLEAR OPTION: Multi-stage aggressive cleanup
        flush_print("[CLEANUP] 🔥 WINDOWS 11 NUCLEAR CLEANUP - Killing ALL Python processes...")
        current_pid = os.getpid()
        
        # Stage 1: Use taskkill with /T flag to kill process tree
        flush_print("[CLEANUP] Stage 1: Process tree termination with taskkill...")
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "python.exe", "/T"], 
                capture_output=True, 
                text=True,
                check=False,
                timeout=15
            )
            flush_print(f"[CLEANUP] Stage 1 result: {result.returncode}")
        except Exception as e:
            flush_print(f"[CLEANUP] Stage 1 error: {e}")
        
        flush_print("[CLEANUP] ✅ Windows 11 nuclear cleanup completed")

    def initialize_knowledge_graph(self):
        """Initialize the knowledge graph with database connection"""
        flush_print("[INIT] 📊 Initializing Enhanced Knowledge Graph...")
        
        try:
            # Initialize Enhanced Knowledge Graph with correct parameters
            self.kg = EnhancedKnowledgeGraph(
                backend='neo4j',
                config=self.database_config,
                auto_connect=True
            )
            
            # Test connection
            if self.kg.connect():
                flush_print("[OK] ✅ Knowledge graph connection established")
                return True
            else:
                flush_print("[ERROR] ❌ Knowledge graph connection failed")
                return False
                
        except Exception as e:
            flush_print(f"[ERROR] ❌ Knowledge graph initialization failed: {e}")
            return False

    def initialize_true_agi_system(self):
        """Initialize the TRUE AGI learning system with Neo4j integration"""
        flush_print("[INIT] 🧠 Initializing TRUE AGI Learning System...")
        
        try:
            # Create world simulator
            self.world_simulator = WorldSimulator()
            
            # Initialize Enhanced Knowledge Graph with Neo4j
            db_config = self._load_database_config()
            flush_print("[INIT] 🔗 Initializing Enhanced Knowledge Graph with Neo4j...")
            
            knowledge_graph = EnhancedKnowledgeGraph(
                backend='neo4j',
                config=db_config
            )
            
            # Test connection
            if knowledge_graph.connect():
                flush_print("[OK] ✅ Neo4j knowledge graph connected successfully")
            else:
                flush_print("[WARNING] ⚠️ Neo4j connection failed, falling back to memory")
                knowledge_graph = EnhancedKnowledgeGraph(backend='memory')
            
            # Create AGI agent with knowledge graph
            self.agi_agent = AGIAgent(self.world_simulator, knowledge_graph)
            
            # Configure for continuous learning
            self.world_simulator.set_simulation_speed(0.1)  # Fast simulation
            self.world_simulator.set_auto_generate_events(True, 0.2)  # Regular events
            self.agi_agent.set_exploration_rate(0.3)  # Moderate exploration
            self.agi_agent.set_novelty_threshold(0.6)  # Moderate novelty threshold
            
            # Display learning progress summary
            progress = self.agi_agent.learning_progress
            flush_print(f"[INIT] 📊 Learning Progress Summary:")
            flush_print(f"   • Concepts Learned: {progress['concepts_learned']}")
            flush_print(f"   • Hypotheses Formed: {progress['hypotheses_formed']}")
            flush_print(f"   • Hypotheses Confirmed: {progress['hypotheses_confirmed']}")
            flush_print(f"   • Causal Relationships: {progress['causal_relationships_discovered']}")
            flush_print(f"   • Patterns Recognized: {progress['patterns_recognized']}")
            flush_print(f"   • Curiosity Level: {self.agi_agent.curiosity_level:.2f}")
            
            flush_print("[OK] ✅ TRUE AGI System initialized successfully with Neo4j integration")
            return True
            
        except Exception as e:
            flush_print(f"[ERROR] ❌ TRUE AGI System initialization failed: {e}")
            traceback.print_exc()
            return False

    def start_true_agi_learning(self):
        """Start the TRUE AGI learning process"""
        if not self.world_simulator or not self.agi_agent:
            flush_print("[ERROR] ❌ TRUE AGI System not initialized")
            return False
        
        try:
            flush_print("[AGI] 🚀 Starting TRUE AGI Learning Process...")
            
            # Start the AGI learning system
            self.agi_agent.start_learning()
            self.true_agi_active = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._agi_monitor_loop)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            flush_print("[AGI] ✅ TRUE AGI Learning Process started")
            return True
            
        except Exception as e:
            flush_print(f"[ERROR] ❌ Failed to start TRUE AGI learning: {e}")
            return False

    def _agi_monitor_loop(self):
        """Monitor TRUE AGI learning progress"""
        flush_print("[AGI] 👁️ AGI monitoring thread started")
        
        cycle_count = 0
        
        while self.running and self.true_agi_active:
            try:
                cycle_count += 1
                
                # Get learning statistics
                world_stats = self.world_simulator.get_learning_statistics()
                agent_summary = self.agi_agent.get_learning_summary()
                
                # Display progress every 10 cycles
                if cycle_count % 10 == 0:
                    self._display_agi_progress(world_stats, agent_summary, cycle_count)
                
                # Inject learning challenges periodically
                if cycle_count % 25 == 0:
                    self._inject_learning_challenges()
                
                # Adaptive sleep based on learning activity
                learning_activity = world_stats.get('learning', {}).get('learning_events', 0)
                if learning_activity > 50:
                    sleep_time = 5  # Fast learning
                elif learning_activity > 20:
                    sleep_time = 8  # Moderate learning
                else:
                    sleep_time = 12  # Slow learning
                
                time.sleep(sleep_time)
                
            except Exception as e:
                flush_print(f"[AGI] ⚠️ Error in AGI monitoring: {e}")
                time.sleep(30)  # Longer sleep on error

    def _display_agi_progress(self, world_stats, agent_summary, cycle_count):
        """Display AGI learning progress"""
        
        # World simulation stats
        sim_stats = world_stats.get('simulation', {})
        learning_stats = world_stats.get('learning', {})
        
        # Agent learning progress
        progress = agent_summary.get('learning_progress', {})
        
        flush_print(f"\n[AGI] 📊 TRUE AGI Learning Progress (Cycle {cycle_count})")
        flush_print(f"[AGI] 🌍 Simulation: {sim_stats.get('steps', 0)} steps, {sim_stats.get('steps_per_second', 0):.1f} steps/sec")
        flush_print(f"[AGI] 🎯 Learning Events: {learning_stats.get('learning_events', 0)}")
        flush_print(f"[AGI] 🔍 Discoveries: {learning_stats.get('discovery_events', 0)}")
        flush_print(f"[AGI] ❓ Curiosity Events: {learning_stats.get('curiosity_events', 0)}")
        flush_print(f"[AGI] 🧠 Concepts Learned: {progress.get('concepts_learned', 0)}")
        flush_print(f"[AGI] 🧪 Hypotheses: {progress.get('hypotheses_formed', 0)} formed, {progress.get('hypotheses_confirmed', 0)} confirmed")
        flush_print(f"[AGI] 🔗 Causal Relationships: {progress.get('causal_relationships_discovered', 0)}")
        flush_print(f"[AGI] 🎨 Patterns: {progress.get('patterns_recognized', 0)}")
        flush_print(f"[AGI] 🤔 Curiosity Level: {agent_summary.get('curiosity_level', 0):.2f}")
        
        # Memory usage
        memory = agent_summary.get('memory_usage', {})
        flush_print(f"[AGI] 💾 Memory: ST={memory.get('short_term', 0)}, LT={memory.get('long_term', 0)}")
        
        # Knowledge base size
        kb_size = agent_summary.get('knowledge_base_size', 0)
        causal_models = agent_summary.get('causal_models', 0)
        flush_print(f"[AGI] 📚 Knowledge Base: {kb_size} concepts, {causal_models} causal models")

    def _inject_learning_challenges(self):
        """Inject learning challenges to stimulate AGI growth"""
        
        challenges = [
            ('mass_experiment', {}),
            ('gravity_change', {'gravity': random.uniform(5, 15)}),
            ('pendulum', {}),
        ]
        
        challenge_type, kwargs = random.choice(challenges)
        
        try:
            self.world_simulator.inject_learning_challenge(challenge_type, **kwargs)
            flush_print(f"[AGI] 🎯 Injected learning challenge: {challenge_type}")
            
            # Boost curiosity after challenge
            self.agi_agent.inject_curiosity(0.2)
            
        except Exception as e:
            flush_print(f"[AGI] ⚠️ Error injecting challenge: {e}")

    def run(self):
        """Main run loop for TRUE AGI continuous learning"""
        flush_print("🚀 STARTING TRUE AGI CONTINUOUS LEARNING SYSTEM")
        flush_print("=" * 60)
        
        # Initialize components
        if not self.initialize_knowledge_graph():
            flush_print("[ERROR] ❌ Failed to initialize knowledge graph")
            return
        
        if not self.initialize_true_agi_system():
            flush_print("[ERROR] ❌ Failed to initialize TRUE AGI system")
            return
        
        # Start TRUE AGI learning
        if not self.start_true_agi_learning():
            flush_print("[ERROR] ❌ Failed to start TRUE AGI learning")
            return
        
        flush_print("[SUCCESS] ✅ TRUE AGI Continuous Learning System running!")
        flush_print("[INFO] 🎯 The system is now learning autonomously through environmental interaction")
        flush_print("[INFO] 🧠 Press Ctrl+C to stop and see learning summary")
        flush_print("=" * 60)
        
        # Main loop
        try:
            while self.running:
                time.sleep(60)  # Check every minute
                
                # Periodic system health check
                if self.world_simulator and self.agi_agent:
                    # Check if systems are still active
                    if not self.world_simulator.running:
                        flush_print("[WARNING] ⚠️ World simulator stopped, restarting...")
                        self.world_simulator.start()
                    
                    # Get brief status
                    world_stats = self.world_simulator.get_learning_statistics()
                    sim_stats = world_stats.get('simulation', {})
                    
                    if sim_stats.get('steps', 0) > 0:
                        flush_print(f"[STATUS] ✅ TRUE AGI Learning Active - {sim_stats.get('steps', 0)} simulation steps")
                    else:
                        flush_print("[STATUS] ⚠️ TRUE AGI Learning appears inactive")
                
        except KeyboardInterrupt:
            flush_print("\n[STOP] 🛑 Interrupted by user")
        except Exception as e:
            flush_print(f"[ERROR] ❌ Unexpected error in main loop: {e}")
        finally:
            self._shutdown_gracefully()

    def _shutdown_gracefully(self):
        """Gracefully shutdown the TRUE AGI system"""
        flush_print("\n[SHUTDOWN] 🛑 Shutting down TRUE AGI system...")
        
        self.running = False
        self.true_agi_active = False
        
        # Stop TRUE AGI components
        if self.agi_agent:
            try:
                # Save final progress before stopping
                flush_print("[SHUTDOWN] 💾 Saving final learning progress...")
                self.agi_agent.save_learning_progress()
                
                self.agi_agent.stop_learning()
                flush_print("[SHUTDOWN] ✅ AGI Agent stopped")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error stopping AGI agent: {e}")
        
        if self.world_simulator:
            try:
                self.world_simulator.stop()
                flush_print("[SHUTDOWN] ✅ World Simulator stopped")
            except Exception as e:
                flush_print(f"[SHUTDOWN] ⚠️ Error stopping world simulator: {e}")
        
        # Display final learning summary
        self._display_final_learning_summary()
        
        flush_print("[SHUTDOWN] ✅ TRUE AGI system shutdown complete")

    def _display_final_learning_summary(self):
        """Display final learning summary"""
        if not self.agi_agent or not self.world_simulator:
            return
        
        try:
            flush_print("\n" + "="*60)
            flush_print("🎓 FINAL TRUE AGI LEARNING SUMMARY")
            flush_print("="*60)
            
            # Get comprehensive statistics
            world_stats = self.world_simulator.get_learning_statistics()
            agent_summary = self.agi_agent.get_learning_summary()
            knowledge_base = self.agi_agent.get_knowledge_base()
            causal_models = self.agi_agent.get_causal_models()
            insights = self.agi_agent.get_learning_insights()
            
            # Display final statistics
            flush_print(f"📚 Total Knowledge Base: {len(knowledge_base)} concepts")
            flush_print(f"🔗 Causal Models: {len(causal_models)} relationships")
            flush_print(f"🧪 Active Hypotheses: {len(self.agi_agent.get_active_hypotheses())}")
            flush_print(f"✅ Confirmed Hypotheses: {len(self.agi_agent.get_confirmed_hypotheses())}")
            
            # Learning progress
            progress = agent_summary.get('learning_progress', {})
            flush_print(f"🎯 Learning Progress:")
            flush_print(f"  • Concepts Learned: {progress.get('concepts_learned', 0)}")
            flush_print(f"  • Hypotheses Formed: {progress.get('hypotheses_formed', 0)}")
            flush_print(f"  • Hypotheses Confirmed: {progress.get('hypotheses_confirmed', 0)}")
            flush_print(f"  • Causal Relationships: {progress.get('causal_relationships_discovered', 0)}")
            flush_print(f"  • Patterns Recognized: {progress.get('patterns_recognized', 0)}")
            
            # Performance metrics
            flush_print(f"📈 Performance Metrics:")
            flush_print(f"  • Learning Velocity: {insights['learning_velocity']:.2f}")
            flush_print(f"  • Discovery Rate: {insights['discovery_rate']:.2f}")
            flush_print(f"  • Final Curiosity Level: {agent_summary.get('curiosity_level', 0):.2f}")
            
            # World simulation statistics
            sim_stats = world_stats.get('simulation', {})
            learning_stats = world_stats.get('learning', {})
            
            flush_print(f"🌍 World Simulation:")
            flush_print(f"  • Total Steps: {sim_stats.get('steps', 0)}")
            flush_print(f"  • Runtime: {sim_stats.get('runtime_seconds', 0):.1f} seconds")
            flush_print(f"  • Learning Events: {learning_stats.get('learning_events', 0)}")
            flush_print(f"  • Discovery Events: {learning_stats.get('discovery_events', 0)}")
            flush_print(f"  • Curiosity Events: {learning_stats.get('curiosity_events', 0)}")
            
            flush_print("="*60)
            flush_print("🎯 TRUE AGI CONTINUOUS LEARNING COMPLETE")
            flush_print("="*60)
            
        except Exception as e:
            flush_print(f"[ERROR] ❌ Error generating final summary: {e}")


def main():
    """Main entry point"""
    runner = TrueAGIContinuousRunner()
    runner.run()


if __name__ == "__main__":
    main()