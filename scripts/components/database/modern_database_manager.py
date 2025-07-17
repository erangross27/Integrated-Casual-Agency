#!/usr/bin/env python3
"""
Modern Database Manager for TRUE AGI
Uses file-based storage for neural networks + PostgreSQL for events
"""

import time
from .modern_neural_persistence import ModernNeuralPersistence
from .postgresql_agi_persistence import PostgreSQLAGIPersistence


class ModernDatabaseManager:
    """Modern database manager using file-based neural storage + PostgreSQL events"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        
        # Modern file-based neural network storage
        self.neural_persistence = ModernNeuralPersistence(session_id)
        
        # PostgreSQL for learning events and metadata
        try:
            self.agi_persistence = PostgreSQLAGIPersistence(session_id)
            self.postgres_available = True
            print(f"✅ [Modern] PostgreSQL available for event logging")
        except Exception as e:
            print(f"⚠️ [Modern] PostgreSQL unavailable: {e}")
            self.postgres_available = False
        
        print(f"🧠 [Modern] Database manager initialized")
        print(f"🗂️ [Modern] Neural networks: File-based storage (PyTorch + HDF5)")
        print(f"📊 [Modern] Events & metadata: {'PostgreSQL' if self.postgres_available else 'Local files'}")
    
    def store_learning_state(self, agi_agent, gpu_processor=None):
        """Store COMPLETE learning state using modern approach"""
        print("🧠 [Modern] Storing complete AGI learning state...")
        success = True
        
        # Save neural networks to files (much more efficient)
        if gpu_processor:
            print("🧠 [Modern] Saving neural network models to files...")
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
                
                print("✅ [Modern] Neural networks saved to files successfully")
                
            except Exception as e:
                print(f"❌ [Modern] Neural network save error: {e}")
                success = False
        else:
            print("ℹ️ [Modern] No GPU processor provided - skipping neural network save")
        
        # Log learning event to PostgreSQL (if available)
        if agi_agent and self.postgres_available:
            try:
                environment_state = getattr(agi_agent, 'environment_state', {})
                current_action = getattr(agi_agent, 'current_action', {})
                
                self.agi_persistence.log_learning_event(
                    'model_save',
                    environment_state,
                    current_action,
                    outcome='Learning state persisted to files'
                )
                
            except Exception as e:
                print(f"⚠️ [Modern] Event logging failed: {e}")
                # Don't fail the whole save for event logging issues
        
        if success:
            print("✅ [Modern] Complete AGI learning state saved successfully")
        else:
            print("⚠️ [Modern] Some components failed to save")
        
        return success
    
    def restore_learning_state(self, agi_agent, gpu_processor=None):
        """Restore COMPLETE learning state using modern approach"""
        print("🧠 [Modern] Restoring complete AGI learning state...")
        success = True
        
        # Restore neural networks from files
        if gpu_processor:
            print("🧠 [Modern] Restoring neural network models from files...")
            try:
                # Restore pattern recognizer
                if hasattr(gpu_processor, 'pattern_recognizer') and gpu_processor.pattern_recognizer:
                    neural_success = self.neural_persistence.load_neural_model(
                        'pattern_recognizer', 
                        gpu_processor.pattern_recognizer
                    )
                    if not neural_success:
                        print("ℹ️ [Modern] No saved pattern recognizer found - starting fresh")
                
                # Restore hypothesis generator
                if hasattr(gpu_processor, 'hypothesis_generator') and gpu_processor.hypothesis_generator:
                    neural_success = self.neural_persistence.load_neural_model(
                        'hypothesis_generator', 
                        gpu_processor.hypothesis_generator
                    )
                    if not neural_success:
                        print("ℹ️ [Modern] No saved hypothesis generator found - starting fresh")
                
                print("✅ [Modern] Neural networks restored from files successfully")
                
            except Exception as e:
                print(f"❌ [Modern] Neural network restore error: {e}")
                success = False
        else:
            print("ℹ️ [Modern] No GPU processor provided - skipping neural network restore")
        
        # Log learning event to PostgreSQL (if available)
        if agi_agent and self.postgres_available:
            try:
                environment_state = getattr(agi_agent, 'environment_state', {})
                current_action = getattr(agi_agent, 'current_action', {})
                
                self.agi_persistence.log_learning_event(
                    'model_restore',
                    environment_state,
                    current_action,
                    outcome='Learning state restored from files'
                )
                
            except Exception as e:
                print(f"⚠️ [Modern] Event logging failed: {e}")
        
        if success:
            print("✅ [Modern] Complete AGI learning state restored successfully")
        else:
            print("⚠️ [Modern] Some components failed to restore")
        
        return success
    
    def get_storage_info(self):
        """Get information about storage usage"""
        print("📊 [Modern] Storage Information:")
        
        # Neural network storage
        try:
            neural_size, model_count = self.neural_persistence.get_total_storage_usage()
            print(f"🧠 Neural Networks: {model_count} models")
        except Exception as e:
            print(f"⚠️ Neural storage info error: {e}")
        
        # List saved models
        try:
            self.neural_persistence.list_saved_models()
        except Exception as e:
            print(f"⚠️ Model list error: {e}")
    
    def cleanup_old_data(self, keep_latest=3):
        """Cleanup old data"""
        print(f"🧹 [Modern] Cleaning up old data (keeping latest {keep_latest})...")
        
        try:
            self.neural_persistence.cleanup_old_checkpoints(keep_latest)
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    # Legacy compatibility methods
    def store_learning_state_legacy(self, agi_agent):
        """Legacy compatibility - store without GPU processor"""
        return self.store_learning_state(agi_agent)
    
    def restore_learning_state_legacy(self, agi_agent):
        """Legacy compatibility - restore without GPU processor"""
        return self.restore_learning_state(agi_agent)


# Factory function for creating modern database manager
def create_modern_database_manager(session_id=None):
    """Create a modern database manager instance"""
    if session_id is None:
        session_id = f"agi_session_{int(time.time())}"
    
    return ModernDatabaseManager(session_id)
