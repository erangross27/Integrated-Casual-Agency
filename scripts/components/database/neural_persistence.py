#!/usr/bin/env python3
"""
PostgreSQL-Only Neural Persistence
For TRUE AGI that learns from environment - neural networks ARE the knowledge
"""

import torch
import pickle
import gzip
import json
import time
import hashlib
from datetime import datetime
import logging
from .postgresql_agi_persistence import PostgreSQLAGIPersistence


class NeuralPersistence:
    """PostgreSQL-only neural network persistence for TRUE AGI"""
    
    def __init__(self, session_id, database_manager=None):
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize PostgreSQL AGI persistence
        self.agi_persistence = PostgreSQLAGIPersistence(session_id)
        
        # Storage configuration
        self.use_compression = True
        self.compression_level = 6
        
        print(f"🧠 [Neural] Initialized PostgreSQL-only persistence for session {session_id}")
        print(f"🧠 [Neural] Neural networks are the knowledge storage - no graph database needed")
        
    def save_model_weights(self, model_name, model, metadata=None):
        """Save neural network model weights to PostgreSQL"""
        return self.agi_persistence.save_neural_model(model_name, model, metadata)
    
    def load_model_weights(self, model_name, model):
        """Load neural network model weights from PostgreSQL"""
        return self.agi_persistence.load_neural_model(model_name, model)
    
    def log_learning_event(self, event_type, environment_state, agi_action, reward=None, outcome=None):
        """Log AGI learning event"""
        return self.agi_persistence.log_learning_event(event_type, environment_state, agi_action, reward, outcome)
    
    def log_pattern_recognition(self, input_pattern, recognized_pattern, confidence, processing_time):
        """Log pattern recognition result"""
        return self.agi_persistence.log_pattern_recognition(input_pattern, recognized_pattern, confidence, processing_time)
    
    def log_hypothesis_generation(self, context, hypothesis, confidence, test_outcome=None, validation_data=None):
        """Log hypothesis generation result"""
        return self.agi_persistence.log_hypothesis_generation(context, hypothesis, confidence, test_outcome, validation_data)
    
    def log_learning_metric(self, metric_name, value, context=None):
        """Log learning progress metric"""
        return self.agi_persistence.log_learning_metric(metric_name, value, context)
    
    def get_model_versions(self, model_name):
        """Get all versions of a model"""
        versions = []
        
        try:
            if self.agi_persistence.connected:
                with self.agi_persistence.connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT version, created_at, parameter_count, model_size_bytes, is_current
                        FROM neural_models
                        WHERE session_id = %s AND model_name = %s
                        ORDER BY version DESC
                    """, (self.session_id, model_name))
                    
                    for row in cursor.fetchall():
                        versions.append({
                            'version': row[0],
                            'created_at': row[1],
                            'parameter_count': row[2],
                            'model_size_bytes': row[3],
                            'is_current': row[4]
                        })
                        
        except Exception as e:
            print(f"❌ [Neural] Error getting versions: {e}")
        
        return versions
    
    def cleanup_old_versions(self, model_name, keep_versions=5):
        """Clean up old model versions"""
        print(f"🧠 [Neural] Cleaning up old versions of {model_name} (keeping {keep_versions})")
        
        try:
            if self.agi_persistence.connected:
                with self.agi_persistence.connection.cursor() as cursor:
                    cursor.execute("""
                        DELETE FROM neural_models 
                        WHERE session_id = %s AND model_name = %s 
                        AND version NOT IN (
                            SELECT version FROM neural_models 
                            WHERE session_id = %s AND model_name = %s
                            ORDER BY version DESC 
                            LIMIT %s
                        )
                    """, (self.session_id, model_name, self.session_id, model_name, keep_versions))
                    
                    deleted = cursor.rowcount
                    self.agi_persistence.connection.commit()
                    
                    if deleted > 0:
                        print(f"✅ [Neural] Cleaned up {deleted} old versions")
                        
        except Exception as e:
            print(f"❌ [Neural] Cleanup error: {e}")
    
    def get_storage_stats(self):
        """Get neural storage statistics"""
        stats = {
            'total_models': 0,
            'total_size_bytes': 0,
            'models_by_name': {}
        }
        
        try:
            if self.agi_persistence.connected:
                with self.agi_persistence.connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT model_name, COUNT(*) as versions, SUM(model_size_bytes) as total_size
                        FROM neural_models
                        WHERE session_id = %s
                        GROUP BY model_name
                    """, (self.session_id,))
                    
                    for row in cursor.fetchall():
                        model_name = row[0]
                        versions = row[1]
                        total_size = row[2] or 0
                        
                        stats['models_by_name'][model_name] = {
                            'versions': versions,
                            'total_size_bytes': total_size
                        }
                        
                        stats['total_models'] += versions
                        stats['total_size_bytes'] += total_size
                        
        except Exception as e:
            print(f"❌ [Neural] Error getting stats: {e}")
        
        return stats
    
    def get_learning_stats(self):
        """Get AGI learning statistics"""
        return self.agi_persistence.get_learning_stats()
    
    def close(self):
        """Close database connection"""
        if self.agi_persistence:
            self.agi_persistence.close()
