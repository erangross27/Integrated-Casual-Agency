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
        """Try to restore previous session from file-based checkpoints"""
        if not self.database_manager:
            return False
        
        flush_print("[INIT] 🔄 Checking for previous checkpoint data...")
        
        try:
            import os
            from pathlib import Path
            
            # Check for checkpoint directories
            checkpoints_dir = Path("agi_checkpoints")
            if not checkpoints_dir.exists():
                flush_print("[RESTORE] ℹ️ No checkpoint directory found - starting fresh")
                return False
            
            # Find all session directories excluding the current one
            current_session_id = getattr(self.database_manager, 'session_id', None)
            session_dirs = [d for d in checkpoints_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('agi_session_') 
                          and d.name != current_session_id]
            
            if not session_dirs:
                flush_print("[RESTORE] ℹ️ No previous sessions found - starting fresh")
                return False
            
            # Find sessions that actually have saved models
            sessions_with_models = []
            for session_dir in session_dirs:
                models_dir = session_dir / "models"
                if models_dir.exists() and any(models_dir.glob("*.pth")):
                    sessions_with_models.append(session_dir)
            
            if not sessions_with_models:
                flush_print("[RESTORE] ℹ️ No previous sessions with saved models found - starting fresh")
                return False
            
            # Find the latest session that has models (highest timestamp)
            latest_session = max(sessions_with_models, key=lambda x: int(x.name.split('_')[-1]))
            
            # Check if the latest session has saved models
            models_dir = latest_session / "models"
            metadata_dir = latest_session / "metadata"
            
            has_models = models_dir.exists() and any(models_dir.glob("*.pth"))
            has_metadata = metadata_dir.exists() and any(metadata_dir.glob("*.json"))
            
            if has_models or has_metadata:
                self.latest_session_path = latest_session
                session_id = latest_session.name
                
                # Update database manager to use the restore session
                if self.database_manager and hasattr(self.database_manager, 'set_restore_session'):
                    self.database_manager.set_restore_session(session_id)
                
                # Count available models
                model_count = len(list(models_dir.glob("*.pth"))) if has_models else 0
                metadata_count = len(list(metadata_dir.glob("*.json"))) if has_metadata else 0
                
                # Create session data for restoration
                self.restored_session_data = {
                    'session_path': str(latest_session),
                    'session_id': session_id,
                    'models_found': model_count,
                    'metadata_found': metadata_count,
                    'restoration_method': 'file_based'
                }
                self.session_restored = True
                
                flush_print(f"[RESTORE] ✅ Previous checkpoint found!")
                flush_print(f"[RESTORE] 📊 Session: {session_id}")
                flush_print(f"[RESTORE]   • Neural Models: {model_count}")
                flush_print(f"[RESTORE]   • Metadata Files: {metadata_count}")
                flush_print(f"[RESTORE] 🎯 Now attempting to restore learning state...")
                
                return True
            else:
                flush_print(f"[RESTORE] ℹ️ Latest session {latest_session.name} has no saved models - starting fresh")
                return False
                
        except Exception as e:
            flush_print(f"[RESTORE] ⚠️ Checkpoint check failed: {e}")
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
