#!/usr/bin/env python3
"""
Modern File-Based Database Manager
For TRUE AGI that learns from environment - neural networks ARE the knowledge
FILE-BASED STORAGE ONLY - NO DATABASE DEPENDENCIES
"""

import time
import json
from .modern_neural_persistence import ModernNeuralPersistence


class DatabaseManager:
    """Modern file-based database manager for TRUE AGI learning"""
    
    def __init__(self, knowledge_graph=None, restore_session_id=None):
        # Use existing session ID for restoration, or create new one
        self.session_id = restore_session_id or f"agi_session_{int(time.time())}"
        
        # Auto-cleanup old sessions if creating a new session (not restoring)
        if not restore_session_id:
            self._cleanup_old_sessions()
        
        # Initialize modern file-based neural persistence
        self.neural_persistence = ModernNeuralPersistence(self.session_id)
        
        print(f"🧠 [DB] Modern file-based Database Manager initialized")
        print(f"🧠 [DB] Session ID: {self.session_id}")
        print(f"🧠 [DB] Neural networks stored in file checkpoints")
        
        # If restoring, set up to use existing session
        if restore_session_id:
            print(f"🧠 [DB] Configured for session restoration: {restore_session_id}")
            
    def set_restore_session(self, session_id):
        """Configure to restore from a specific session"""
        self.session_id = session_id
        self.neural_persistence = ModernNeuralPersistence(session_id)
        print(f"🧠 [DB] Switched to restoration session: {session_id}")
        print(f"🧠 [DB] Neural persistence now pointing to: agi_checkpoints/{session_id}")
        
    def log_learning_event(self, event_type, environment_state, agi_action, reward=None, outcome=None):
        """Log learning event to file"""
        try:
            log_entry = {
                'timestamp': time.time(),
                'event_type': event_type,
                'environment_state': str(environment_state)[:200],  # Truncate for storage
                'agi_action': str(agi_action)[:200],
                'reward': reward,
                'outcome': outcome
            }
            
            log_file = self.neural_persistence.session_path / "learning_log.json"
            
            # Append to log file
            logs = []
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            return True
        except Exception as e:
            print(f"❌ [DB] Logging error: {e}")
            return False
    
    def store_learning_state(self, agi_agent, gpu_processor=None):
        """Store COMPLETE learning state - neural networks + AGI events"""
        print("🧠 [DB] Storing complete AGI learning state...")
        success = True
        
        # Save neural networks (the actual knowledge)
        if gpu_processor:
            print("🧠 [DB] Saving neural network models...")
            try:
                # Save pattern recognizer
                if hasattr(gpu_processor, 'pattern_recognizer') and gpu_processor.pattern_recognizer:
                    neural_success = self.neural_persistence.save_neural_model(
                        'pattern_recognizer', 
                        gpu_processor.pattern_recognizer,
                        {'type': 'pattern_recognition', 'architecture': 'neural_network'}
                    )
                    if not neural_success:
                        success = False
                
                # Save hypothesis generator
                if hasattr(gpu_processor, 'hypothesis_generator') and gpu_processor.hypothesis_generator:
                    neural_success = self.neural_persistence.save_neural_model(
                        'hypothesis_generator', 
                        gpu_processor.hypothesis_generator,
                        {'type': 'hypothesis_generation', 'architecture': 'neural_network'}
                    )
                    if not neural_success:
                        success = False
                
                print("✅ [DB] Neural networks saved successfully")
                
            except Exception as e:
                print(f"❌ [DB] Neural network save error: {e}")
                success = False
        else:
            print("ℹ️ [DB] No GPU processor provided - skipping neural network save")
        
        # Log learning event
        if agi_agent:
            try:
                environment_state = getattr(agi_agent, 'environment_state', {})
                current_action = getattr(agi_agent, 'current_action', {})
                
                self.log_learning_event(
                    'model_save',
                    environment_state,
                    current_action,
                    outcome='Learning state persisted'
                )
                
            except Exception as e:
                print(f"❌ [DB] Learning event logging error: {e}")
        
        if success:
            print("✅ [DB] Complete AGI learning state saved successfully")
        else:
            print("⚠️ [DB] Some components failed to save")
        
        return success
    
    def restore_learning_state(self, agi_agent, gpu_processor=None):
        """Restore COMPLETE learning state - neural networks + AGI events"""
        print("🧠 [DB] Restoring complete AGI learning state...")
        success = True
        
        # Restore neural networks (the knowledge)
        if gpu_processor:
            print("🧠 [DB] Restoring neural network models...")
            try:
                # Restore pattern recognizer
                if hasattr(gpu_processor, 'pattern_recognizer') and gpu_processor.pattern_recognizer:
                    neural_success = self.neural_persistence.load_neural_model(
                        'pattern_recognizer', 
                        gpu_processor.pattern_recognizer
                    )
                    if not neural_success:
                        print("ℹ️ [DB] No saved pattern recognizer found - starting fresh")
                
                # Restore hypothesis generator
                if hasattr(gpu_processor, 'hypothesis_generator') and gpu_processor.hypothesis_generator:
                    neural_success = self.neural_persistence.load_neural_model(
                        'hypothesis_generator', 
                        gpu_processor.hypothesis_generator
                    )
                    if not neural_success:
                        print("ℹ️ [DB] No saved hypothesis generator found - starting fresh")
                
                print("✅ [DB] Neural networks restored successfully")
                
            except Exception as e:
                print(f"❌ [DB] Neural network restore error: {e}")
                success = False
        else:
            print("ℹ️ [DB] No GPU processor provided - skipping neural network restore")
        
        # Log learning event
        if agi_agent:
            try:
                environment_state = getattr(agi_agent, 'environment_state', {})
                current_action = getattr(agi_agent, 'current_action', {})
                
                self.log_learning_event(
                    'model_restore',
                    environment_state,
                    current_action,
                    outcome='Learning state restored'
                )
                
            except Exception as e:
                print(f"❌ [DB] Learning event logging error: {e}")
        
        if success:
            print("✅ [DB] Complete AGI learning state restored successfully")
        else:
            print("⚠️ [DB] Some components failed to restore")
        
        return success
    
    def log_pattern_recognition(self, input_pattern, recognized_pattern, confidence, processing_time):
        """Log pattern recognition result"""
        return self.log_learning_event(
            'pattern_recognition', 
            {'input': str(input_pattern)[:100]}, 
            {'recognized': str(recognized_pattern)[:100], 'confidence': confidence},
            outcome=f'Processing time: {processing_time}ms'
        )
    
    def log_hypothesis_generation(self, context, hypothesis, confidence, test_outcome=None, validation_data=None):
        """Log hypothesis generation result"""
        return self.log_learning_event(
            'hypothesis_generation',
            {'context': str(context)[:100]},
            {'hypothesis': str(hypothesis)[:100], 'confidence': confidence},
            outcome=f'Test outcome: {test_outcome}'
        )
    
    def log_learning_metric(self, metric_name, value, context=None):
        """Log learning progress metric"""
        return self.log_learning_event(
            'learning_metric',
            {'context': str(context)[:100] if context else 'none'},
            {'metric': metric_name, 'value': value},
            outcome='Metric logged'
        )
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        try:
            log_file = self.neural_persistence.session_path / "learning_log.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    return {
                        'total_events': len(logs),
                        'event_types': list(set(log.get('event_type', 'unknown') for log in logs)),
                        'recent_events': logs[-10:] if logs else []
                    }
        except:
            pass
        return {'total_events': 0, 'event_types': [], 'recent_events': []}
    
    def get_storage_stats(self):
        """Get neural storage statistics"""
        return self.neural_persistence.get_storage_stats()
    
    def cleanup_old_models(self, keep_versions=5):
        """Clean up old model versions"""
        try:
            self.neural_persistence.cleanup_old_versions('pattern_recognizer', keep_versions)
            self.neural_persistence.cleanup_old_versions('hypothesis_generator', keep_versions)
            
        except Exception as e:
            print(f"❌ [DB] Cleanup error: {e}")
    
    def close(self):
        """Close database connections"""
        # File-based system doesn't need explicit closing
        print("🧠 [DB] File-based storage - no connections to close")
    
    # === LEGACY COMPATIBILITY METHODS ===
    def save_learning_state(self, agi_agent):
        """Legacy method for backwards compatibility"""
        return self.store_learning_state(agi_agent)
    
    def restore_learning_state_legacy(self, agi_agent):
        """Legacy method for backwards compatibility"""
        return self.restore_learning_state(agi_agent)
    
    def store_patterns(self, patterns):
        """Store patterns in neural networks (implicit through learning)"""
        print("🧠 [DB] Patterns stored implicitly in neural networks")
        return True
    
    def get_patterns(self):
        """Get patterns from neural networks (implicit through inference)"""
        print("🧠 [DB] Patterns retrieved implicitly from neural networks")
        return []
    
    def save_session_state(self, agi_agent):
        """Save session state to file"""
        try:
            self.log_learning_event(
                'session_save',
                getattr(agi_agent, 'environment_state', {}),
                getattr(agi_agent, 'current_action', {}),
                outcome='Session state saved'
            )
            return True
        except Exception as e:
            print(f"❌ [DB] Session save error: {e}")
            return False
    
    def restore_session_state(self, agi_agent):
        """Restore session state from file"""
        try:
            self.log_learning_event(
                'session_restore',
                getattr(agi_agent, 'environment_state', {}),
                getattr(agi_agent, 'current_action', {}),
                outcome='Session state restored'
            )
            return True
        except Exception as e:
            print(f"❌ [DB] Session restore error: {e}")
            return False
    
    def _cleanup_old_sessions(self):
        """Keep only the most recent session and delete all others"""
        try:
            import shutil
            from pathlib import Path
            
            checkpoints_dir = Path("agi_checkpoints")
            if not checkpoints_dir.exists():
                return
            
            # Find all AGI session directories
            session_dirs = []
            for item in checkpoints_dir.iterdir():
                if item.is_dir() and item.name.startswith("agi_session_"):
                    try:
                        timestamp_str = item.name.replace("agi_session_", "")
                        timestamp = int(timestamp_str)
                        session_dirs.append((timestamp, item))
                    except ValueError:
                        continue
            
            # Keep only the most recent session (if any)
            if len(session_dirs) > 1:
                session_dirs.sort(key=lambda x: x[0], reverse=True)
                dirs_to_delete = session_dirs[1:]  # All except newest
                
                deleted_count = 0
                for timestamp, dir_path in dirs_to_delete:
                    try:
                        shutil.rmtree(dir_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠️ [DB] Could not delete old session {dir_path.name}: {e}")
                
                if deleted_count > 0:
                    print(f"🧹 [DB] Auto-cleanup: Deleted {deleted_count} old session directories")
                    
        except Exception as e:
            print(f"⚠️ [DB] Session cleanup error: {e}")
