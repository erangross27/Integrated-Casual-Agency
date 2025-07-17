#!/usr/bin/env python3
"""
PostgreSQL-Only Neural Persistence for TRUE AGI
All knowledge stored in neural network weights - no graph database needed
"""

import torch
import pickle
import gzip
import json
import time
import hashlib
from datetime import datetime
import logging


class PostgreSQLAGIPersistence:
    """PostgreSQL-only persistence for TRUE AGI learning"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.connection = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        self.setup_database()
    
    def setup_database(self):
        """Setup PostgreSQL database connection"""
        print(f"🧠 [AGI] Setting up PostgreSQL neural storage...")
        
        try:
            import psycopg2
            self.connection = psycopg2.connect(
                host='localhost',
                port=5432,
                database='ica_neural',
                user='ica_user',
                password='ica_password'
            )
            self.connection.autocommit = False
            self.connected = True
            print(f"✅ [AGI] Connected to PostgreSQL neural database")
            
            self.create_tables()
            return True
            
        except Exception as e:
            print(f"❌ [AGI] Failed to connect to PostgreSQL: {e}")
            self.connected = False
            return False
    
    def create_tables(self):
        """Create tables optimized for AGI learning"""
        create_sql = """
        -- AGI Learning Sessions
        CREATE TABLE IF NOT EXISTS agi_sessions (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            environment_type VARCHAR(100),
            learning_objectives TEXT,
            session_metadata JSONB
        );
        
        -- Neural Network Models (the AGI's brain)
        CREATE TABLE IF NOT EXISTS neural_models (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(255) REFERENCES agi_sessions(session_id),
            model_name VARCHAR(255) NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_size_bytes BIGINT,
            compression_ratio FLOAT,
            parameter_count BIGINT,
            architecture_info JSONB,
            is_current BOOLEAN DEFAULT TRUE,
            UNIQUE(session_id, model_name, version)
        );
        
        -- Neural Weights Storage (compressed binary)
        CREATE TABLE IF NOT EXISTS neural_weights (
            id BIGSERIAL PRIMARY KEY,
            model_id BIGINT REFERENCES neural_models(id) ON DELETE CASCADE,
            weights_data BYTEA NOT NULL,
            checksum VARCHAR(64),
            compressed BOOLEAN DEFAULT TRUE
        );
        
        -- Environmental Learning Events
        CREATE TABLE IF NOT EXISTS learning_events (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(255) REFERENCES agi_sessions(session_id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type VARCHAR(100),
            environment_state JSONB,
            agi_action JSONB,
            reward_signal FLOAT,
            learning_outcome TEXT
        );
        
        -- Pattern Recognition Results  
        CREATE TABLE IF NOT EXISTS pattern_recognitions (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(255) REFERENCES agi_sessions(session_id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_pattern JSONB,
            recognized_pattern VARCHAR(255),
            confidence_score FLOAT,
            processing_time_ms FLOAT
        );
        
        -- Hypothesis Generation Results
        CREATE TABLE IF NOT EXISTS hypothesis_generations (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(255) REFERENCES agi_sessions(session_id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_context JSONB,
            generated_hypothesis TEXT,
            confidence_score FLOAT,
            test_outcome VARCHAR(50),
            validation_data JSONB
        );
        
        -- Learning Progress Metrics
        CREATE TABLE IF NOT EXISTS learning_metrics (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(255) REFERENCES agi_sessions(session_id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metric_name VARCHAR(100),
            metric_value FLOAT,
            context_data JSONB
        );
        
        -- Performance Indexes
        CREATE INDEX IF NOT EXISTS idx_neural_models_session ON neural_models(session_id);
        CREATE INDEX IF NOT EXISTS idx_neural_models_current ON neural_models(session_id, model_name, is_current);
        CREATE INDEX IF NOT EXISTS idx_learning_events_session ON learning_events(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_pattern_recognitions_session ON pattern_recognitions(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_hypothesis_generations_session ON hypothesis_generations(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_learning_metrics_session ON learning_metrics(session_id, timestamp);
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_sql)
                self.connection.commit()
                print(f"✅ [AGI] Neural learning tables created successfully")
                
                # Create the session
                self.create_agi_session()
                return True
                
        except Exception as e:
            print(f"❌ [AGI] Failed to create tables: {e}")
            self.connection.rollback()
            return False
    
    def create_agi_session(self):
        """Create a new AGI learning session"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO agi_sessions (session_id, environment_type, learning_objectives)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        environment_type = EXCLUDED.environment_type,
                        learning_objectives = EXCLUDED.learning_objectives
                """, (
                    self.session_id,
                    'environmental_learning',
                    'Learn from surroundings through neural pattern recognition and hypothesis generation'
                ))
                self.connection.commit()
                print(f"✅ [AGI] Session {self.session_id} created")
                
        except Exception as e:
            print(f"❌ [AGI] Failed to create session: {e}")
            self.connection.rollback()
    
    def save_neural_model(self, model_name, model, metadata=None):
        """Save neural network model (the AGI's learned knowledge)"""
        print(f"🧠 [AGI] Saving {model_name} neural knowledge...")
        
        try:
            # Get model state dict and serialize
            state_dict = model.state_dict()
            model_bytes = pickle.dumps(state_dict)
            original_size = len(model_bytes)
            
            # Compress the neural data
            compressed_data = gzip.compress(model_bytes, compresslevel=6)
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Calculate checksum for integrity
            checksum = hashlib.sha256(compressed_data).hexdigest()
            
            # Count parameters
            parameter_count = sum(p.numel() for p in model.parameters())
            
            print(f"🧠 [AGI] Compressed {original_size:,} bytes → {compressed_size:,} bytes ({compression_ratio:.2f}x)")
            print(f"🧠 [AGI] Model has {parameter_count:,} learnable parameters")
            
            with self.connection.cursor() as cursor:
                # Mark previous versions as not current
                cursor.execute("""
                    UPDATE neural_models 
                    SET is_current = FALSE 
                    WHERE session_id = %s AND model_name = %s
                """, (self.session_id, model_name))
                
                # Insert model metadata
                cursor.execute("""
                    INSERT INTO neural_models 
                    (session_id, model_name, model_size_bytes, compression_ratio, 
                     parameter_count, architecture_info)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    self.session_id, model_name, original_size, compression_ratio,
                    parameter_count, json.dumps(metadata or {})
                ))
                
                model_id = cursor.fetchone()[0]
                
                # Insert compressed weights
                cursor.execute("""
                    INSERT INTO neural_weights 
                    (model_id, weights_data, checksum, compressed)
                    VALUES (%s, %s, %s, %s)
                """, (model_id, compressed_data, checksum, True))
                
                self.connection.commit()
                print(f"✅ [AGI] Saved {model_name} knowledge (ID: {model_id})")
                return True
                
        except Exception as e:
            print(f"❌ [AGI] Failed to save {model_name}: {e}")
            self.connection.rollback()
            return False
    
    def load_neural_model(self, model_name, model):
        """Load neural network model (restore AGI's learned knowledge)"""
        print(f"🧠 [AGI] Loading {model_name} neural knowledge...")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT nm.*, nw.weights_data, nw.checksum, nw.compressed
                    FROM neural_models nm
                    JOIN neural_weights nw ON nm.id = nw.model_id
                    WHERE nm.session_id = %s AND nm.model_name = %s AND nm.is_current = TRUE
                    ORDER BY nm.created_at DESC
                    LIMIT 1
                """, (self.session_id, model_name))
                
                result = cursor.fetchone()
                if not result:
                    print(f"ℹ️ [AGI] No saved knowledge found for {model_name} - starting fresh")
                    return False
                
                # Verify checksum
                weights_data = result[8]  # weights_data column
                expected_checksum = result[9]  # checksum column
                actual_checksum = hashlib.sha256(weights_data).hexdigest()
                
                if expected_checksum != actual_checksum:
                    print(f"❌ [AGI] Neural knowledge corrupted for {model_name}")
                    return False
                
                # Decompress and deserialize
                if result[10]:  # compressed column
                    weights_data = gzip.decompress(weights_data)
                
                state_dict = pickle.loads(weights_data)
                model.load_state_dict(state_dict)
                
                print(f"✅ [AGI] Restored {model_name} knowledge (Version: {result[3]})")
                print(f"🧠 [AGI] Loaded {result[6]:,} parameters from neural memory")
                return True
                
        except Exception as e:
            print(f"❌ [AGI] Failed to load {model_name}: {e}")
            return False
    
    def log_learning_event(self, event_type, environment_state, agi_action, reward=None, outcome=None):
        """Log AGI learning event from environmental interaction"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO learning_events 
                    (session_id, event_type, environment_state, agi_action, reward_signal, learning_outcome)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    self.session_id, event_type, 
                    json.dumps(environment_state), 
                    json.dumps(agi_action),
                    reward, outcome
                ))
                self.connection.commit()
                
        except Exception as e:
            print(f"❌ [AGI] Failed to log learning event: {e}")
            self.connection.rollback()
    
    def log_pattern_recognition(self, input_pattern, recognized_pattern, confidence, processing_time):
        """Log pattern recognition result"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO pattern_recognitions 
                    (session_id, input_pattern, recognized_pattern, confidence_score, processing_time_ms)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    self.session_id,
                    json.dumps(input_pattern),
                    recognized_pattern,
                    confidence,
                    processing_time
                ))
                self.connection.commit()
                
        except Exception as e:
            print(f"❌ [AGI] Failed to log pattern recognition: {e}")
            self.connection.rollback()
    
    def log_hypothesis_generation(self, context, hypothesis, confidence, test_outcome=None, validation_data=None):
        """Log hypothesis generation result"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO hypothesis_generations 
                    (session_id, input_context, generated_hypothesis, confidence_score, test_outcome, validation_data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    self.session_id,
                    json.dumps(context),
                    hypothesis,
                    confidence,
                    test_outcome,
                    json.dumps(validation_data) if validation_data else None
                ))
                self.connection.commit()
                
        except Exception as e:
            print(f"❌ [AGI] Failed to log hypothesis generation: {e}")
            self.connection.rollback()
    
    def log_learning_metric(self, metric_name, value, context=None):
        """Log learning progress metric"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO learning_metrics 
                    (session_id, metric_name, metric_value, context_data)
                    VALUES (%s, %s, %s, %s)
                """, (
                    self.session_id,
                    metric_name,
                    value,
                    json.dumps(context) if context else None
                ))
                self.connection.commit()
                
        except Exception as e:
            print(f"❌ [AGI] Failed to log learning metric: {e}")
            self.connection.rollback()
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        try:
            with self.connection.cursor() as cursor:
                # Get learning event counts
                cursor.execute("""
                    SELECT event_type, COUNT(*) 
                    FROM learning_events 
                    WHERE session_id = %s 
                    GROUP BY event_type
                """, (self.session_id,))
                
                event_stats = dict(cursor.fetchall())
                
                # Get latest metrics
                cursor.execute("""
                    SELECT metric_name, metric_value, timestamp
                    FROM learning_metrics 
                    WHERE session_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 20
                """, (self.session_id,))
                
                latest_metrics = cursor.fetchall()
                
                return {
                    'session_id': self.session_id,
                    'event_stats': event_stats,
                    'latest_metrics': latest_metrics,
                    'total_learning_events': sum(event_stats.values())
                }
                
        except Exception as e:
            print(f"❌ [AGI] Failed to get learning stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print(f"🔌 [AGI] Neural database connection closed")
