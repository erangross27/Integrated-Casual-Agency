"""
ICA Framework Learning Module
Continuous learning components for AGI training
"""

from .continuous_learning import ContinuousLearning
from .parallel_manager import ParallelLearningManager, ContinuousParallelManager
from .scenario_generators import PhysicsSimulation, ProceduralScenarioGenerator
from .session_manager import SessionManager
from .comprehensive_scenarios import ComprehensiveScenarioLibrary

__all__ = [
    "ContinuousLearning",
    "ParallelLearningManager", 
    "ContinuousParallelManager",
    "PhysicsSimulation",
    "ProceduralScenarioGenerator",
    "SessionManager",
    "ComprehensiveScenarioLibrary"
]
