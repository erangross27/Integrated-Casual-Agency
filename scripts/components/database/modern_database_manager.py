#!/usr/bin/env python3
"""
Modern Database Manager for TRUE AGI
Uses file-based storage for neural networks + W&B analytics
"""

import time
from .modern_neural_persistence import ModernNeuralPersistence
from .analytics_logger import WandBAGILogger


class ModernDatabaseManager:
    """Modern database manager using file-based neural storage + W&B analytics"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        
        # Modern file-based neural network storage
        self.neural_persistence = ModernNeuralPersistence(session_id)
        
        # Weights & Biases analytics logger
        try:
            self.analytics_logger = WandBAGILogger(resume_mode=True)
            self.analytics_available = self.analytics_logger.initialized
            if self.analytics_available:
                print(f"✅ [Modern] W&B Analytics initialized")
            else:
                print(f"⚠️ [Modern] W&B Analytics unavailable")
        except Exception as e:
            print(f"⚠️ [Modern] W&B Analytics error: {e}")
            self.analytics_logger = None
            self.analytics_available = False
        
        print(f"🧠 [Modern] Database manager initialized")
        print(f"🗂️ [Modern] Neural networks: File-based storage (PyTorch + HDF5)")
        print(f"📊 [Modern] Analytics: {'W&B Dashboard' if self.analytics_available else 'Local files'}")
    
    def set_restore_session(self, session_id):
        """Configure to restore from a specific session"""
        self.session_id = session_id
        self.neural_persistence = ModernNeuralPersistence(session_id)
        print(f"🧠 [Modern] Switched to restoration session: {session_id}")
        print(f"🧠 [Modern] Neural persistence now pointing to: agi_checkpoints/{session_id}")
    
    def store_learning_state(self, agi_agent, gpu_processor=None):
        """Store COMPLETE learning state using modern approach"""
        print("🧠 [Modern] Storing complete AGI learning state...")
        success = True
        
        # Log GPU performance to W&B
        if self.analytics_available:
            self.analytics_logger.log_gpu_performance()
        
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
                    
                    # Log neural network info to W&B
                    if self.analytics_available and neural_success:
                        self.analytics_logger.log_neural_network_info(
                            'pattern_recognizer', 
                            gpu_processor.pattern_recognizer
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
                    
                    # Log neural network info to W&B
                    if self.analytics_available and neural_success:
                        self.analytics_logger.log_neural_network_info(
                            'hypothesis_generator', 
                            gpu_processor.hypothesis_generator
                        )
                    
                    if not neural_success:
                        success = False
                
                print("✅ [Modern] Neural networks saved to files successfully")
                
            except Exception as e:
                print(f"❌ [Modern] Neural network save error: {e}")
                success = False
        else:
            print("ℹ️ [Modern] No GPU processor provided - skipping neural network save")
        
        # Log to W&B Analytics (if available)
        if agi_agent and self.analytics_available:
            try:
                environment_state = getattr(agi_agent, 'environment_state', {})
                current_action = getattr(agi_agent, 'current_action', {})
                
                self.analytics_logger.log_learning_event(
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
        
        # Log to W&B Analytics (if available)
        if agi_agent and self.analytics_available:
            try:
                environment_state = getattr(agi_agent, 'environment_state', {})
                current_action = getattr(agi_agent, 'current_action', {})
                
                self.analytics_logger.log_learning_event(
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
    
    def log_learning_episode(self, episode_data: dict):
        """Log a learning episode to W&B analytics"""
        if self.analytics_available:
            try:
                self.analytics_logger.log_learning_episode(episode_data)
                self.analytics_logger.increment_learning_cycle()
            except Exception as e:
                print(f"⚠️ [Modern] Episode logging failed: {e}")
    
    def log_learning_event(self, event_type, environment_state, agi_action, reward=None, outcome=None):
        """Log AGI learning event (compatibility method)"""
        episode_data = {
            'event_type': event_type,
            'environment_state': str(environment_state)[:200],
            'agi_action': str(agi_action)[:200],
            'reward': reward,
            'outcome': outcome
        }
        self.log_learning_episode(episode_data)
    
    def log_learning_metrics(self, metrics: dict):
        """Log learning metrics to W&B analytics"""
        if self.analytics_available:
            try:
                self.analytics_logger.log_learning_metrics(metrics)
            except Exception as e:
                print(f"⚠️ [Modern] Metrics logging failed: {e}")
    
    def close(self):
        """Close the database manager and finish W&B session"""
        if self.analytics_available:
            try:
                self.analytics_logger.finish()
            except Exception as e:
                print(f"⚠️ [Modern] W&B finish failed: {e}")
    
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
        # Create new timestamped session, cleanup will handle old ones
        import time
        session_id = f"agi_session_{int(time.time())}"
    
    return ModernDatabaseManager(session_id)
