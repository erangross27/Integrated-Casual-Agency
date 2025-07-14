#!/usr/bin/env python3
"""
Continuous Learning Runner
Keeps the ICA Framework learning running indefinitely using the modular system
"""

import time
import sys
import signal
import json
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# DISABLE VERBOSE ICA FRAMEWORK LOGGING IMMEDIATELY
logging.getLogger('ica_framework').setLevel(logging.WARNING)
logging.getLogger('ica_framework.utils.logger').setLevel(logging.WARNING)
logging.getLogger('ica_framework.core').setLevel(logging.WARNING)
logging.getLogger('ica_framework.components').setLevel(logging.WARNING)

from ica_framework.learning import ContinuousLearning


class ContinuousRunner:
    """Simple runner that keeps learning active"""

    def __init__(self):
        self.running = True
        self.learning = None
        self.database_config = self._load_database_config()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_database_config(self):
        """Load Neo4j configuration from config file"""
        config_file = Path("config/database/neo4j.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                    db_config = config_data['config']
                    print(f"✅ Loaded Neo4j config from {config_file}")
                    print(f"   URI: {db_config['uri']}")
                    print(f"   Database: {db_config['database']}")
                    print(f"   User: {db_config['username']}")
                    print()
                    return db_config
                
            except Exception as e:
                print(f"❌ Failed to load config file: {e}")
                
        print("⚠️ Using default Neo4j configuration")
        return {
            'uri': 'neo4j://127.0.0.1:7687',
            'username': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        
        # Explicitly stop learning if it's running
        if self.learning and hasattr(self.learning, 'continuous_manager'):
            try:
                print("🛑 Stopping workers...")
                if self.learning.continuous_manager:
                    self.learning.continuous_manager.stop_workers()
            except Exception as e:
                print(f"⚠️ Error stopping workers: {e}")
        
        # Force exit if needed
        import os
        os._exit(0)
    
    def run_forever(self):
        """Run learning continuously - let it handle its own infinite loop"""
        
        # Suppress verbose logging before starting
        logging.disable(logging.INFO)  # Disable INFO level and below
        logging.getLogger().setLevel(logging.WARNING)  # Only show warnings and errors
        
        print("🔄 Starting continuous learning (spam-free mode)...")
        print("   Press Ctrl+C to stop gracefully")
        print()
        
        try:
            # Create learning instance with proper config (only once)
            self.learning = ContinuousLearning(
                database_backend="neo4j",
                database_config=self.database_config,
                num_workers=15,
                batch_size=20,
                enable_parallel=True,   # ENABLE PARALLEL FOR FULL SPEED
                continuous_mode=True    # Use TRUE continuous mode
            )
            
            print("🚀 Starting infinite learning session...")
            print("📈 Progress will be shown every 30 seconds (no spam)")
            print("📊 Scenario count should increase constantly...")
            print("🛑 Press Ctrl+C to stop (will force-kill all workers)")
            print()
            
            # Let continuous learning run its own infinite loop
            self.learning.run_continuous_learning()
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user - cleaning up workers...")
            if self.learning and hasattr(self.learning, 'continuous_manager'):
                try:
                    if self.learning.continuous_manager:
                        self.learning.continuous_manager.stop_workers()
                except:
                    pass
            
            # Force kill any remaining Python processes
            try:
                import subprocess
                subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                             capture_output=True, check=False)
                print("🧹 Forced cleanup of any remaining Python processes")
            except:
                pass
            return
        except Exception as e:
            print(f"❌ Critical error: {e}")
            import traceback
            traceback.print_exc()
            return
        finally:
            print("🏁 Continuous learning stopped")

if __name__ == "__main__":
    runner = ContinuousRunner()
    runner.run_forever()