#!/usr/bin/env python3
"""
Component Initializer Module
Handles initialization of all system components
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import component modules
from ..system import SystemUtils
from ..gpu import GPUProcessor
from ..database import DatabaseManager
from ..database.modern_database_manager import create_modern_database_manager
from ..monitoring import AGIMonitor
from ..gpu import GPUWorker

# Import TRUE AGI Framework
from ica_framework.sandbox import WorldSimulator, AGIAgent

# Setup utilities
flush_print = SystemUtils.flush_print


class ComponentInitializer:
    """Handles initialization of all system components"""
    
    def __init__(self, database_config):
        self.database_config = database_config
        
        # Core components
        self.world_simulator = None
        self.agi_agent = None
        self.knowledge_graph = None
        
        # System components
        self.gpu_processor = None
        self.database_manager = None
        self.agi_monitor = None
        self.gpu_worker = None
    
    def initialize_gpu_processor(self):
        """Initialize GPU processor with full initialization"""
        flush_print("[INIT] 🔧 Initializing GPU processor...")
        
        self.gpu_processor = GPUProcessor()
        
        if self.gpu_processor.use_gpu:
            flush_print("✅ [GPU] GPU acceleration models initialized")
            
            # Ensure GPU models are fully initialized before attempting restoration
            if (hasattr(self.gpu_processor, 'pattern_recognizer') and 
                hasattr(self.gpu_processor, 'hypothesis_generator') and
                self.gpu_processor.pattern_recognizer is not None and
                self.gpu_processor.hypothesis_generator is not None):
                flush_print("✅ [GPU] Neural network models ready for weight restoration")
                return True
            else:
                flush_print("⚠️ [GPU] Neural network models not fully initialized - weights may not restore properly")
                return True
        else:
            flush_print("⚠️ [GPU] GPU acceleration models failed to initialize")
            return True  # Still continue with CPU
    
    def initialize_knowledge_graph(self):
        """Initialize PostgreSQL database (knowledge stored in neural networks)"""
        flush_print("[INIT] 🔧 Initializing PostgreSQL database...")
        
        # For PostgreSQL-only architecture, we don't need a separate knowledge graph
        # The neural networks ARE the knowledge storage
        try:
            # Test PostgreSQL connection
            import psycopg2
            conn = psycopg2.connect(
                host=self.database_config['host'],
                port=self.database_config['port'],
                database=self.database_config['database'],
                user=self.database_config['user'],
                password=self.database_config['password']
            )
            conn.close()
            flush_print("[OK] ✅ PostgreSQL connection established")
            flush_print("[OK] 🧠 Neural networks will store the knowledge")
            
            # Create a simple placeholder for compatibility
            self.knowledge_graph = None  # No separate knowledge graph needed
            return True
            
        except Exception as e:
            flush_print(f"[ERROR] ❌ PostgreSQL connection failed: {e}")
            return False
    
    def initialize_database_manager(self):
        """Initialize modern file-based database manager"""
        flush_print("[INIT] 🔧 Initializing modern database manager...")
        
        # Use modern file-based neural storage + PostgreSQL events
        session_id = f"agi_session_{int(__import__('time').time())}"
        self.database_manager = create_modern_database_manager(session_id)
        
        flush_print("[OK] ✅ Modern database manager initialized")
        flush_print("[OK] 🧠 Neural networks: File storage (PyTorch + HDF5)")
        flush_print("[OK] 📊 Events: PostgreSQL (if available)")
        return True
    
    def initialize_simulators(self):
        """Initialize world simulator and AGI agent"""
        flush_print("[INIT] 🔧 Initializing simulators...")
        
        # For PostgreSQL-only architecture, we don't need a knowledge graph
        # The neural networks and database manager handle knowledge storage
        
        # Initialize world simulator and AGI agent
        self.world_simulator = WorldSimulator()
        self.agi_agent = AGIAgent(self.world_simulator, self.database_manager)
        
        # Configure for continuous learning
        self.world_simulator.set_simulation_speed(0.1)
        self.world_simulator.set_auto_generate_events(True, 0.2)
        self.agi_agent.set_exploration_rate(0.3)
        self.agi_agent.set_novelty_threshold(0.6)
        
        flush_print("[OK] ✅ Simulators initialized")
        flush_print("[OK] 🧠 AGI agent will learn from environment")
        return True
    
    def initialize_monitoring_components(self):
        """Initialize monitoring components"""
        flush_print("[INIT] 🔧 Initializing monitoring components...")
        
        if not all([self.world_simulator, self.agi_agent, self.gpu_processor, self.database_manager]):
            flush_print("[ERROR] ❌ Cannot initialize monitoring - missing core components")
            return False
        
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
        
        flush_print("[OK] ✅ Monitoring components initialized")
        return True
    
    def initialize_all_components(self):
        """Initialize all system components in the correct order"""
        flush_print("[INIT] 🔧 Initializing system components...")
        
        # Initialize in dependency order
        if not self.initialize_gpu_processor():
            return False
        
        if not self.initialize_knowledge_graph():
            return False
        
        if not self.initialize_database_manager():
            return False
        
        if not self.initialize_simulators():
            return False
        
        if not self.initialize_monitoring_components():
            return False
        
        flush_print("[INIT] ✅ All components initialized successfully")
        return True
    
    def get_components(self):
        """Get all initialized components"""
        return {
            'world_simulator': self.world_simulator,
            'agi_agent': self.agi_agent,
            'knowledge_graph': self.knowledge_graph,
            'gpu_processor': self.gpu_processor,
            'database_manager': self.database_manager,
            'agi_monitor': self.agi_monitor,
            'gpu_worker': self.gpu_worker
        }
