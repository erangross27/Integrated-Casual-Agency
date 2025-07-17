"""
Database adapters for ICA Framework
Supports multiple graph database backends for scalable knowledge storage
"""

from .graph_database import GraphDatabase
from .memory_adapter import MemoryAdapter

__all__ = ['GraphDatabase', 'MemoryAdapter']
