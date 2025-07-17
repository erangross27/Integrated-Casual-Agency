"""
AGI Modules Package
Modular components for the TRUE AGI Agent system
"""

from .memory_system import MemorySystem
from .hypothesis_manager import HypothesisManager
from .causal_reasoning import CausalReasoning
from .attention_system import AttentionSystem
from .curiosity_engine import CuriosityEngine
from .pattern_learner import PatternLearner
from .physics_learner import PhysicsLearner
from .exploration_controller import ExplorationController
from .learning_progress import LearningProgress
from .sensory_processor import SensoryProcessor

__all__ = [
    'MemorySystem',
    'HypothesisManager', 
    'CausalReasoning',
    'AttentionSystem',
    'CuriosityEngine',
    'PatternLearner',
    'PhysicsLearner',
    'ExplorationController',
    'LearningProgress',
    'SensoryProcessor'
]
