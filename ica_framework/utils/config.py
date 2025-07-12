"""
Configuration management for ICA Framework
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path


class GraphConfig(BaseModel):
    """Configuration for Causal Knowledge Graph"""
    initial_nodes: int = Field(default=500, description="Initial number of nodes")
    initial_edges: int = Field(default=1000, description="Initial number of edges")
    max_nodes: int = Field(default=50000, description="Maximum number of nodes")
    confidence_threshold: float = Field(default=0.1, description="Minimum confidence threshold")
    beta_prior_alpha: float = Field(default=1.0, description="Beta distribution prior alpha")
    beta_prior_beta: float = Field(default=1.0, description="Beta distribution prior beta")


class WorldModelConfig(BaseModel):
    """Configuration for World Model (Bayesian GNN)"""
    embedding_dim: int = Field(default=128, description="Embedding dimension")
    hidden_dim: int = Field(default=256, description="Hidden layer dimension")
    num_layers: int = Field(default=3, description="Number of GNN layers")
    num_heads: int = Field(default=8, description="Number of attention heads")
    dropout: float = Field(default=0.1, description="Dropout rate")
    learning_rate: float = Field(default=0.001, description="Learning rate")


class CuriosityConfig(BaseModel):
    """Configuration for Curiosity Module"""
    complexity_weight: float = Field(default=0.1, description="Weight for complexity normalization")
    adversarial_weight: float = Field(default=0.5, description="Weight for adversarial probing")
    reward_threshold: float = Field(default=0.01, description="Minimum reward threshold")
    exploration_bonus: float = Field(default=0.1, description="Exploration bonus multiplier")


class PlannerConfig(BaseModel):
    """Configuration for Action Planner (SAC)"""
    lr_actor: float = Field(default=3e-4, description="Actor learning rate")
    lr_critic: float = Field(default=3e-4, description="Critic learning rate")
    alpha: float = Field(default=0.2, description="Entropy coefficient")
    gamma: float = Field(default=0.99, description="Discount factor")
    tau: float = Field(default=0.005, description="Soft update coefficient")
    buffer_size: int = Field(default=1000000, description="Replay buffer size")
    batch_size: int = Field(default=256, description="Batch size")


class AbstractionConfig(BaseModel):
    """Configuration for Hierarchical Abstraction"""
    motif_min_size: int = Field(default=3, description="Minimum motif size")
    motif_max_size: int = Field(default=10, description="Maximum motif size")
    clustering_algorithm: str = Field(default="kmeans", description="Clustering algorithm")
    num_clusters: int = Field(default=50, description="Number of clusters")
    utility_decay: float = Field(default=0.01, description="Utility score decay factor (γ)")
    significance_threshold: float = Field(default=0.05, description="Statistical significance threshold")


class SandboxConfig(BaseModel):
    """Configuration for Sandbox Environment"""
    dataset_size: int = Field(default=500, description="Dataset size for testing")
    test_ratio: float = Field(default=0.2, description="Test set ratio")
    validation_ratio: float = Field(default=0.1, description="Validation set ratio")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


class Config(BaseModel):
    """Main configuration class for ICA Framework"""
    graph: GraphConfig = Field(default_factory=GraphConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    curiosity: CuriosityConfig = Field(default_factory=CuriosityConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    abstraction: AbstractionConfig = Field(default_factory=AbstractionConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    
    # Global settings
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")
    log_level: str = Field(default="INFO", description="Logging level")
    wandb_project: Optional[str] = Field(default=None, description="Weights & Biases project name")
    
    class Config:
        extra = "forbid"
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def save(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)
    
    def get_device(self) -> str:
        """Get the appropriate device"""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
