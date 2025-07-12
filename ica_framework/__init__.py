"""
ICA Framework - Integrated Causal Agency
A framework for Artificial General Intelligence driven by intrinsic curiosity
and active causal model construction.
"""

__version__ = "0.1.0"
__author__ = "ICA Development Team"
__email__ = "ica@example.com"

from .core import ICAAgent
from .components import (
    CausalKnowledgeGraph,
    WorldModel,
    CuriosityModule,
    ActionPlanner,
    HierarchicalAbstraction
)
from .utils import Config, Logger

__all__ = [
    "ICAAgent",
    "CausalKnowledgeGraph", 
    "WorldModel",
    "CuriosityModule",
    "ActionPlanner",
    "HierarchicalAbstraction",
    "Config",
    "Logger"
]
