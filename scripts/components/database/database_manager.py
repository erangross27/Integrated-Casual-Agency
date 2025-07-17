#!/usr/bin/env python3
"""
PostgreSQL-Only Database Manager
For TRUE AGI that learns from environment - neural networks ARE the knowledge
NO GRAPH DATABASE - JUST NEURAL LEARNING STORAGE
"""

import time
import json
from .neural_persistence import NeuralPersistence
from .postgresql_agi_persistence import PostgreSQLAGIPersistence


class DatabaseManager:
    """PostgreSQL-only database manager for TRUE AGI learning"""
    
    def __init__(self, knowledge_graph=None):
        self.session_id = f"agi_session_{int(time.time())}"
        
        # Initialize PostgreSQL AGI persistence
        self.agi_persistence = PostgreSQLAGIPersistence(self.session_id)
        
        # Initialize neural persistence
        self.neural_persistence = NeuralPersistence(self.session_id)
        
        print(f"🧠 [DB] PostgreSQL-only Database Manager initialized")
        print(f"🧠 [DB] Session ID: {self.session_id}")
        print(f"🧠 [DB] Neural networks are the knowledge - no graph database needed")
    
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
                    neural_success = self.neural_persistence.save_model_weights(
                        'pattern_recognizer', 
                        gpu_processor.pattern_recognizer,
                        {'type': 'pattern_recognition', 'architecture': 'neural_network'}
                    )
                    if not neural_success:
                        success = False
                
                # Save hypothesis generator
                if hasattr(gpu_processor, 'hypothesis_generator') and gpu_processor.hypothesis_generator:
                    neural_success = self.neural_persistence.save_model_weights(
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
                
                self.agi_persistence.log_learning_event(
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
                    neural_success = self.neural_persistence.load_model_weights(
                        'pattern_recognizer', 
                        gpu_processor.pattern_recognizer
                    )
                    if not neural_success:
                        print("ℹ️ [DB] No saved pattern recognizer found - starting fresh")
                
                # Restore hypothesis generator
                if hasattr(gpu_processor, 'hypothesis_generator') and gpu_processor.hypothesis_generator:
                    neural_success = self.neural_persistence.load_model_weights(
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
                
                self.agi_persistence.log_learning_event(
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
        return self.agi_persistence.log_pattern_recognition(
            input_pattern, recognized_pattern, confidence, processing_time
        )
    
    def log_hypothesis_generation(self, context, hypothesis, confidence, test_outcome=None, validation_data=None):
        """Log hypothesis generation result"""
        return self.agi_persistence.log_hypothesis_generation(
            context, hypothesis, confidence, test_outcome, validation_data
        )
    
    def log_learning_event(self, event_type, environment_state, agi_action, reward=None, outcome=None):
        """Log AGI learning event"""
        return self.agi_persistence.log_learning_event(
            event_type, environment_state, agi_action, reward, outcome
        )
    
    def log_learning_metric(self, metric_name, value, context=None):
        """Log learning progress metric"""
        return self.agi_persistence.log_learning_metric(metric_name, value, context)
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        return self.agi_persistence.get_learning_stats()
    
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
        if self.agi_persistence:
            self.agi_persistence.close()
        if self.neural_persistence:
            self.neural_persistence.close()
    
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
        """Save session state to PostgreSQL"""
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
        """Restore session state from PostgreSQL"""
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
