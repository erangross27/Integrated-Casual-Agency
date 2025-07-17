"""
ICA Framework - TRU    "MemoryAdapter",AGI System
A framework for genuine Artificial General Intelligence through autonomous environmental learning.
"""

__version__ = "0.2.0"
__author__ = "ICA Development Team"
__email__ = "ica@example.com"

# TRUE AGI SYSTEM - Core components for genuine autonomous learning
from .sandbox import WorldSimulator, AGIAgent, PhysicsEngine, LearningEnvironment
from .enhanced_knowledge_graph import EnhancedKnowledgeGraph
from .utils import Config, ica_logger as Logger
from .database import GraphDatabase, MemoryAdapter

__all__ = [
    "WorldSimulator",
    "AGIAgent", 
    "PhysicsEngine",
    "LearningEnvironment",
    "EnhancedKnowledgeGraph",
    "Config",
    "Logger",
    "GraphDatabase",
    "Neo4jAdapter",
    "MemoryAdapter"
]
