#!/usr/bin/env python3
"""
Session Manager Module
Handles session restoration and learning data persistence
"""

import os
import sys
import torch
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
        """Temporarily disabled - always start fresh to test learning fixes"""
        flush_print("[INIT] 🔄 Starting fresh session (restoration temporarily disabled for testing)")
        return False
    def apply_session_restoration(self, agi_agent, gpu_processor, world_simulator):
        """Apply restored session data to components with file-based persistence"""
        if not self.session_restored or not self.restored_session_data:
            return
        
        try:
            # Use database manager's restore method for file-based restoration
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
                
                flush_print(f"[RESTORE] 📚 Knowledge base: ✓ File-based storage active")
                flush_print(f"[RESTORE] 🔬 Learning progress: ✓ Session continuity enabled")
                flush_print(f"[RESTORE] 🎯 Resuming with ALL previous learning intact!")
            else:
                flush_print(f"[RESTORE] ⚠️ Could not restore learning state - starting fresh")
                
                # Try manual restoration from checkpoint files
                if (gpu_processor and gpu_processor.use_gpu and 
                    self.database_manager and self.database_manager.neural_persistence and
                    hasattr(self, 'latest_session_path')):
                    
                    flush_print(f"[RESTORE] 🔄 Attempting manual checkpoint restoration...")
                    
                    # Try to manually load from latest checkpoint
                    try:
                        models_dir = self.latest_session_path / "models"
                        
                        # Load pattern recognizer if exists
                        pattern_file = models_dir / "pattern_recognizer_latest.pth"
                        if pattern_file.exists() and hasattr(gpu_processor, 'pattern_recognizer'):
                            state_dict = torch.load(pattern_file, map_location=gpu_processor.device)
                            gpu_processor.pattern_recognizer.load_state_dict(state_dict)
                            flush_print(f"[RESTORE] ✅ Pattern recognizer restored from {pattern_file.name}")
                        
                        # Load hypothesis generator if exists
                        hypothesis_file = models_dir / "hypothesis_generator_latest.pth"
                        if hypothesis_file.exists() and hasattr(gpu_processor, 'hypothesis_generator'):
                            state_dict = torch.load(hypothesis_file, map_location=gpu_processor.device)
                            gpu_processor.hypothesis_generator.load_state_dict(state_dict)
                            flush_print(f"[RESTORE] ✅ Hypothesis generator restored from {hypothesis_file.name}")
                        
                        flush_print(f"[RESTORE] ✅ Manual checkpoint restoration completed!")
                        
                    except Exception as e:
                        flush_print(f"[RESTORE] ⚠️ Manual restoration failed: {e}")
                        flush_print(f"[RESTORE] 🔄 Starting with fresh neural networks")
            
            # Log restoration event
            if agi_agent and self.database_manager:
                try:
                    self.database_manager.log_learning_event(
                        'session_restore',
                        getattr(agi_agent, 'environment_state', {}),
                        getattr(agi_agent, 'current_action', {}),
                        outcome=f"File-based restoration: {restoration_success}"
                    )
                except Exception as e:
                    flush_print(f"[RESTORE] ⚠️ Could not log restoration event: {e}")
            
            flush_print("[RESTORE] ✅ Session state applied to components")
            
        except Exception as e:
            flush_print(f"[RESTORE] ⚠️ Error applying session restoration: {e}")
            import traceback
            traceback.print_exc()
    
    def get_stats(self):
        """Get session management statistics"""
        return {
            'session_restored': self.session_restored,
            'restored_data': self.restored_session_data
        }
    
    def _migrate_session_data(self, old_session_path, new_session_id):
        """Migrate data from old session to new session"""
        try:
            import shutil
            from pathlib import Path
            
            # Get new session path
            new_session_path = Path("agi_checkpoints") / new_session_id
            
            # Create new session directory structure
            new_session_path.mkdir(parents=True, exist_ok=True)
            (new_session_path / "models").mkdir(exist_ok=True)
            (new_session_path / "metadata").mkdir(exist_ok=True)
            
            # Copy models from old to new
            old_models = old_session_path / "models"
            new_models = new_session_path / "models"
            
            if old_models.exists():
                for model_file in old_models.glob("*.pth"):
                    shutil.copy2(model_file, new_models / model_file.name)
                    flush_print(f"[MIGRATE] ✅ Copied model: {model_file.name}")
            
            # Copy metadata from old to new
            old_metadata = old_session_path / "metadata"
            new_metadata = new_session_path / "metadata"
            
            if old_metadata.exists():
                for metadata_file in old_metadata.glob("*.json"):
                    shutil.copy2(metadata_file, new_metadata / metadata_file.name)
                    flush_print(f"[MIGRATE] ✅ Copied metadata: {metadata_file.name}")
            
            # Update database manager to point to new session
            if hasattr(self.database_manager, 'neural_persistence'):
                # Update the neural persistence to use new session path
                self.database_manager.neural_persistence.session_path = new_session_path
                self.database_manager.neural_persistence.session_id = new_session_id
                flush_print(f"[MIGRATE] ✅ Updated database manager to new session")
            
            return True
            
        except Exception as e:
            flush_print(f"[MIGRATE] ❌ Migration failed: {e}")
            return False
    
    def _cleanup_old_sessions(self, old_session_dirs):
        """Clean up old session directories"""
        try:
            import shutil
            
            deleted_count = 0
            for session_dir in old_session_dirs:
                try:
                    shutil.rmtree(session_dir)
                    deleted_count += 1
                    flush_print(f"[CLEANUP] 🗑️ Deleted old session: {session_dir.name}")
                except Exception as e:
                    flush_print(f"[CLEANUP] ⚠️ Failed to delete {session_dir.name}: {e}")
            
            if deleted_count > 0:
                flush_print(f"[CLEANUP] ✅ Cleaned up {deleted_count} old session directories")
                
        except Exception as e:
            flush_print(f"[CLEANUP] ⚠️ Cleanup failed: {e}")
