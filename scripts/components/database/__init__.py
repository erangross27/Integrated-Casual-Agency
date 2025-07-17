#!/usr/bin/env python3
"""
Database Module - PostgreSQL-Only TRUE AGI Storage
Neural networks ARE the knowledge - no graph database needed
"""

from .database_manager import DatabaseManager
from .neural_persistence import NeuralPersistence
from .postgresql_agi_persistence import PostgreSQLAGIPersistence

__all__ = [
    'DatabaseManager',
    'NeuralPersistence',
    'PostgreSQLAGIPersistence'
]
