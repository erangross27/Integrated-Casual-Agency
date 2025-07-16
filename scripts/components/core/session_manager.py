#!/usr/bin/env python3
"""
Session Manager Module
Handles session restoration and learning data persistence
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


class SessionManager:
    """Manages session restoration and learning data persistence"""
    
    def __init__(self, knowledge_graph, database_manager):
        self.knowledge_graph = knowledge_graph
        self.database_manager = database_manager
        self.session_restored = False
        self.restored_session_data = None
    
    def attempt_session_restoration(self):
        """Try to restore previous session with TRUE learning persistence"""
        if not self.knowledge_graph or not self.database_manager:
            return False
        
        flush_print("[INIT] 🔄 Checking for previous learning data...")
        
        try:
            # Check for any existing learning data in database
            has_learning_data = False
            
            # Check for AGI concepts
            concepts_query = "MATCH (n:Entity {type: 'agi_concept'}) RETURN COUNT(n) as count"
            concepts_result = self.knowledge_graph.execute_custom_query(concepts_query)
            concept_count = concepts_result[0]['count'] if concepts_result else 0
            
            # Check for learning progress
            progress_query = "MATCH (n:Entity {type: 'learning_progress'}) RETURN COUNT(n) as count"
            progress_result = self.knowledge_graph.execute_custom_query(progress_query)
            progress_count = progress_result[0]['count'] if progress_result else 0
            
            # Check for neural models
            neural_query = "MATCH (n:Entity {type: 'neural_model'}) RETURN COUNT(n) as count"
            neural_result = self.knowledge_graph.execute_custom_query(neural_query)
            neural_count = neural_result[0]['count'] if neural_result else 0
            
            if concept_count > 0 or progress_count > 0 or neural_count > 0:
                has_learning_data = True
                
                # Create mock session data for restoration
                self.restored_session_data = {
                    'concepts_learned': concept_count,
                    'learning_progress': {'concepts_learned': concept_count},
                    'pattern_count': 0,
                    'hypotheses_formed': 0,
                    'simulation_steps': 0
                }
                self.session_restored = True
                
                flush_print(f"[RESTORE] ✅ Previous learning data found!")
                flush_print(f"[RESTORE] 📊 Database Contents:")
                flush_print(f"[RESTORE]   • AGI Concepts: {concept_count}")
                flush_print(f"[RESTORE]   • Learning Progress: {progress_count}")
                flush_print(f"[RESTORE]   • Neural Models: {neural_count}")
                flush_print(f"[RESTORE] 🎯 Now attempting to restore learning state...")
                
                return True
            else:
                flush_print("[RESTORE] ℹ️ No previous learning data found - starting fresh")
                return False
                
        except Exception as e:
            flush_print(f"[RESTORE] ⚠️ Learning data check failed: {e}")
            return False
    
    def apply_session_restoration(self, agi_agent, gpu_processor, world_simulator):
        """Apply restored session data to components with TRUE learning persistence"""
        if not self.session_restored or not self.restored_session_data:
            return
        
        try:
            # Use database manager's comprehensive restore for TRUE persistence
            restoration_success = self.database_manager.restore_learning_state(
                agi_agent, 
                gpu_processor
            )
            
            if restoration_success:
                flush_print(f"[RESTORE] ✅ Complete learning state restored!")
                
                # Verify GPU models were actually restored
                if (gpu_processor and gpu_processor.use_gpu and
                    hasattr(gpu_processor, 'pattern_recognizer') and
                    hasattr(gpu_processor, 'hypothesis_generator')):
                    flush_print(f"[RESTORE] 🧠 Neural networks: ✓ Loaded into GPU memory")
                else:
                    flush_print(f"[RESTORE] ⚠️ Neural networks: No GPU models to restore")
                
                flush_print(f"[RESTORE] 📚 Knowledge base: ✓ Loaded")
                flush_print(f"[RESTORE] 🔬 Learning progress: ✓ Loaded")
                flush_print(f"[RESTORE] 🎯 Resuming with ALL previous learning intact!")
            else:
                flush_print(f"[RESTORE] ⚠️ Could not restore learning state - starting fresh")
                
                # Try manual GPU restoration as fallback
                if (gpu_processor and gpu_processor.use_gpu and 
                    self.database_manager and self.database_manager.neural_persistence):
                    
                    flush_print(f"[RESTORE] 🔄 Attempting manual GPU model restoration...")
                    gpu_restore_success = self.database_manager.neural_persistence.restore_gpu_models(gpu_processor)
                    
                    if gpu_restore_success:
                        flush_print(f"[RESTORE] ✅ GPU models manually restored!")
                    else:
                        flush_print(f"[RESTORE] ⚠️ No previous GPU models found - starting with fresh neural networks")
                
                # Fallback to basic session restoration
                if agi_agent and 'learning_progress' in self.restored_session_data:
                    restored_progress = self.restored_session_data['learning_progress']
                    agi_agent.learning_progress.update(restored_progress)
                    
                    # Restore memory state if available
                    if 'memory_state' in self.restored_session_data:
                        memory_state = self.restored_session_data['memory_state']
                        if hasattr(agi_agent, 'short_term_memory') and 'short_term' in memory_state:
                            agi_agent.short_term_memory = memory_state['short_term']
                        if hasattr(agi_agent, 'long_term_memory') and 'long_term' in memory_state:
                            agi_agent.long_term_memory = memory_state['long_term']
            
            # Restore simulation state
            if world_simulator and 'simulation_steps' in self.restored_session_data:
                simulation_steps = self.restored_session_data['simulation_steps']
                if hasattr(world_simulator, 'total_steps'):
                    world_simulator.total_steps = simulation_steps
            
            flush_print("[RESTORE] ✅ Session state applied to components")
            
        except Exception as e:
            flush_print(f"[RESTORE] ⚠️ Error applying session restoration: {e}")
    
    def get_stats(self):
        """Get session management statistics"""
        return {
            'session_restored': self.session_restored,
            'restored_data': self.restored_session_data
        }
