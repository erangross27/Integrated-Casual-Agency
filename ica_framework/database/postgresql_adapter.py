#!/usr/bin/env python3
"""
PostgreSQL Database Adapter for Neural Network Storage
Handles large binary data storage for PyTorch models
"""

import psycopg2
import psycopg2.extras
import pickle
import json
import time
import logging
from typing import Dict, Any, Optional, List


class PostgreSQLAdapter:
    """PostgreSQL adapter for neural network and training data storage"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PostgreSQL connection"""
        self.config = config
        self.connection = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                database=self.config.get('database', 'ica_neural'),
                user=self.config.get('user', 'ica_user'),
                password=self.config.get('password', 'ica_password')
            )
            self.connection.autocommit = True
            self.connected = True
            self.logger.info("✅ Connected to PostgreSQL")
            self._initialize_tables()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ PostgreSQL connection failed: {e}")
            self.connected = False
            return False
    
    def _initialize_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            cursor = self.connection.cursor()
            
            # Neural models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS neural_models (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(255) UNIQUE NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    model_weights BYTEA NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_current BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Training sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) UNIQUE NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'active',
                    metadata JSONB
                )
            """)
            
            # Training logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    log_type VARCHAR(50) NOT NULL,
                    log_data JSONB NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_models_session ON neural_models(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_models_name ON neural_models(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_logs_session ON training_logs(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_logs_timestamp ON training_logs(timestamp)")
            
            cursor.close()
            self.logger.info("✅ PostgreSQL tables initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize PostgreSQL tables: {e}")
            raise
    
    def save_neural_model(self, model_id: str, model_name: str, session_id: str, 
                         model_weights: bytes, metadata: Dict[str, Any] = None) -> bool:
        """Save neural network model to PostgreSQL"""
        if not self.connected:
            self.logger.error("❌ Not connected to PostgreSQL")
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # Mark previous models as not current
            cursor.execute("""
                UPDATE neural_models 
                SET is_current = FALSE 
                WHERE model_name = %s AND session_id = %s
            """, (model_name, session_id))
            
            # Insert new model
            cursor.execute("""
                INSERT INTO neural_models (model_id, model_name, session_id, model_weights, metadata, is_current)
                VALUES (%s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (model_id) DO UPDATE SET
                    model_weights = EXCLUDED.model_weights,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP,
                    is_current = TRUE
            """, (model_id, model_name, session_id, model_weights, json.dumps(metadata) if metadata else None))
            
            cursor.close()
            self.logger.info(f"✅ Saved neural model {model_name} ({len(model_weights)} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save neural model {model_name}: {e}")
            return False
    
    def load_neural_model(self, model_name: str, session_id: str = None) -> Optional[Dict[str, Any]]:
        """Load neural network model from PostgreSQL"""
        if not self.connected:
            self.logger.error("❌ Not connected to PostgreSQL")
            return None
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if session_id:
                cursor.execute("""
                    SELECT model_id, model_name, session_id, model_weights, metadata, created_at
                    FROM neural_models 
                    WHERE model_name = %s AND session_id = %s AND is_current = TRUE
                    ORDER BY created_at DESC LIMIT 1
                """, (model_name, session_id))
            else:
                cursor.execute("""
                    SELECT model_id, model_name, session_id, model_weights, metadata, created_at
                    FROM neural_models 
                    WHERE model_name = %s AND is_current = TRUE
                    ORDER BY created_at DESC LIMIT 1
                """, (model_name,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'model_id': result['model_id'],
                    'model_name': result['model_name'],
                    'session_id': result['session_id'],
                    'model_weights': result['model_weights'],
                    'metadata': json.loads(result['metadata']) if result['metadata'] else {},
                    'created_at': result['created_at']
                }
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load neural model {model_name}: {e}")
            return None
    
    def create_training_session(self, session_id: str, metadata: Dict[str, Any] = None) -> bool:
        """Create a new training session"""
        if not self.connected:
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO training_sessions (session_id, metadata)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    metadata = EXCLUDED.metadata
            """, (session_id, json.dumps(metadata) if metadata else None))
            cursor.close()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create training session: {e}")
            return False
    
    def log_training_event(self, session_id: str, log_type: str, log_data: Dict[str, Any]) -> bool:
        """Log training event"""
        if not self.connected:
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO training_logs (session_id, log_type, log_data)
                VALUES (%s, %s, %s)
            """, (session_id, log_type, json.dumps(log_data)))
            cursor.close()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log training event: {e}")
            return False
    
    def get_training_history(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training history for a session"""
        if not self.connected:
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT log_type, log_data, timestamp
                FROM training_logs 
                WHERE session_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (session_id, limit))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get training history: {e}")
            return []
    
    def close(self):
        """Close PostgreSQL connection"""
        if self.connection:
            self.connection.close()
            self.connected = False
            self.logger.info("🔌 PostgreSQL connection closed")
