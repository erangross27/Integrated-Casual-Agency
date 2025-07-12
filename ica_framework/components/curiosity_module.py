"""
Curiosity Module implementation for ICA Framework
Implements complexity-normalized intrinsic reward and adversarial probing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ..utils.config import CuriosityConfig
from ..utils.logger import ica_logger


class ComplexityEstimator(nn.Module):
    """Estimates model complexity for MDL-based normalization"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.complexity_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive complexity
        )
    
    def forward(self, model_representation: torch.Tensor) -> torch.Tensor:
        """Estimate complexity of model state"""
        return self.complexity_net(model_representation)


class AdversarialProber(nn.Module):
    """Adversarial agent that finds points of confusion for the world model"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.probe_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Action space normalization
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate probing action and value estimate"""
        action = self.probe_net(state)
        value = self.value_net(state)
        return action, value


class CuriosityModule:
    """
    Curiosity Module for intrinsic motivation in ICA Framework
    
    Implements:
    1. Complexity-normalized intrinsic rewards
    2. Adversarial probing for robust uncertainty estimation
    3. Information-theoretic exploration bonuses
    """
    
    def __init__(self, config: CuriosityConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = ica_logger
        
        # Initialize components
        self.complexity_estimator = ComplexityEstimator()
        self.adversarial_prober = AdversarialProber(state_dim, action_dim)
        
        # Optimizers
        self.complexity_optimizer = torch.optim.Adam(
            self.complexity_estimator.parameters(), lr=0.001
        )
        self.prober_optimizer = torch.optim.Adam(
            self.adversarial_prober.parameters(), lr=0.001
        )
        
        # History tracking
        self.prediction_errors = []
        self.complexity_costs = []
        self.intrinsic_rewards = []
        self.prober_success_rate = []
        
        self.logger.info("Initialized Curiosity Module")
    
    def calculate_intrinsic_reward(self, 
                                 prediction_error: torch.Tensor,
                                 model_before: torch.Tensor,
                                 model_after: torch.Tensor,
                                 normalize: bool = True) -> torch.Tensor:
        """
        Calculate complexity-normalized intrinsic reward
        
        Args:
            prediction_error: Error before model update
            model_before: Model representation before update
            model_after: Model representation after update
            normalize: Whether to apply complexity normalization
            
        Returns:
            Intrinsic reward value
        """
        
        # Base reward from prediction error reduction
        base_reward = prediction_error.mean()
        
        if not normalize:
            reward = base_reward
        else:
            # Calculate complexity cost
            complexity_before = self.complexity_estimator(model_before)
            complexity_after = self.complexity_estimator(model_after)
            complexity_cost = torch.abs(complexity_after - complexity_before)
            
            # Normalize by complexity (MDL principle)
            normalized_reward = base_reward / (1.0 + self.config.complexity_weight * complexity_cost)
            reward = normalized_reward.squeeze()
        
        # Add exploration bonus
        exploration_bonus = self._calculate_exploration_bonus(prediction_error)
        final_reward = reward + self.config.exploration_bonus * exploration_bonus
        
        # Track metrics
        self.prediction_errors.append(prediction_error.mean().item())
        if normalize:
            self.complexity_costs.append(complexity_cost.item())
        self.intrinsic_rewards.append(final_reward.item())
        
        return final_reward
    
    def _calculate_exploration_bonus(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """Calculate exploration bonus based on prediction uncertainty"""
        
        # Use prediction error variance as exploration signal
        error_variance = torch.var(prediction_error)
        
        # Information-theoretic bonus (simplified)
        bonus = torch.log(1.0 + error_variance)
        
        return bonus
    
    def train_adversarial_prober(self, 
                               states: torch.Tensor,
                               world_model: nn.Module,
                               num_epochs: int = 10) -> float:
        """
        Train adversarial prober to find confusion points
        
        Args:
            states: Batch of states to probe
            world_model: World model to probe against
            num_epochs: Number of training epochs
            
        Returns:
            Average probing success rate
        """
        
        world_model.eval()  # Keep world model frozen
        
        total_success = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            self.prober_optimizer.zero_grad()
            
            # Generate probing actions
            probe_actions, probe_values = self.adversarial_prober(states)
            
            # Evaluate world model uncertainty on these actions
            with torch.no_grad():
                # Simulate world model prediction with probing actions
                # This is simplified - in practice you'd need proper graph data
                combined_input = torch.cat([states, probe_actions], dim=-1)
                
                # Mock uncertainty calculation (replace with actual world model call)
                uncertainty = torch.randn(states.size(0), 1).abs()  # Placeholder
            
            # Prober objective: maximize world model uncertainty
            prober_loss = -uncertainty.mean()  # Negative because we want to maximize
            
            prober_loss.backward()
            self.prober_optimizer.step()
            
            # Calculate success rate (high uncertainty = success)
            success_rate = (uncertainty > uncertainty.median()).float().mean()
            total_success += success_rate.item()
            num_batches += 1
        
        avg_success = total_success / num_batches if num_batches > 0 else 0.0
        self.prober_success_rate.append(avg_success)
        
        self.logger.debug(f"Adversarial prober success rate: {avg_success:.3f}")
        return avg_success
    
    def calculate_robust_uncertainty(self, 
                                   state: torch.Tensor,
                                   world_model: nn.Module,
                                   num_probe_samples: int = 10) -> torch.Tensor:
        """
        Calculate robust uncertainty using adversarial probing
        
        Args:
            state: Current state
            world_model: World model to evaluate
            num_probe_samples: Number of probing samples
            
        Returns:
            Robust uncertainty estimate
        """
        
        uncertainties = []
        
        for _ in range(num_probe_samples):
            # Generate adversarial probe
            probe_action, _ = self.adversarial_prober(state)
            
            # Evaluate world model uncertainty
            with torch.no_grad():
                # This is simplified - replace with actual world model uncertainty calculation
                uncertainty = torch.randn(1).abs()  # Placeholder
                uncertainties.append(uncertainty)
        
        # Robust statistic (e.g., median or high percentile)
        uncertainties_tensor = torch.stack(uncertainties)
        robust_uncertainty = torch.quantile(uncertainties_tensor, 0.8)  # 80th percentile
        
        return robust_uncertainty
    
    def update_complexity_estimator(self, 
                                  model_representations: torch.Tensor,
                                  true_complexities: torch.Tensor) -> float:
        """
        Update complexity estimator based on true complexity measurements
        
        Args:
            model_representations: Model state representations
            true_complexities: Ground truth complexity measures
            
        Returns:
            Training loss
        """
        
        self.complexity_optimizer.zero_grad()
        
        predicted_complexities = self.complexity_estimator(model_representations)
        loss = F.mse_loss(predicted_complexities.squeeze(), true_complexities)
        
        loss.backward()
        self.complexity_optimizer.step()
        
        return loss.item()
    
    def get_curiosity_metrics(self) -> Dict[str, float]:
        """Get comprehensive curiosity metrics"""
        
        metrics = {}
        
        if self.prediction_errors:
            metrics.update({
                "avg_prediction_error": np.mean(self.prediction_errors),
                "prediction_error_trend": self._calculate_trend(self.prediction_errors)
            })
        
        if self.complexity_costs:
            metrics.update({
                "avg_complexity_cost": np.mean(self.complexity_costs),
                "complexity_cost_trend": self._calculate_trend(self.complexity_costs)
            })
        
        if self.intrinsic_rewards:
            metrics.update({
                "avg_intrinsic_reward": np.mean(self.intrinsic_rewards),
                "reward_variance": np.var(self.intrinsic_rewards),
                "reward_trend": self._calculate_trend(self.intrinsic_rewards)
            })
        
        if self.prober_success_rate:
            metrics.update({
                "prober_success_rate": np.mean(self.prober_success_rate),
                "prober_improvement": self._calculate_trend(self.prober_success_rate)
            })
        
        return metrics
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> float:
        """Calculate trend direction in recent values"""
        
        if len(values) < window:
            return 0.0
        
        recent = values[-window:]
        early = values[-window*2:-window] if len(values) >= window*2 else values[:-window]
        
        if not early:
            return 0.0
        
        recent_avg = np.mean(recent)
        early_avg = np.mean(early)
        
        return (recent_avg - early_avg) / early_avg if early_avg != 0 else 0.0
    
    def reset_history(self):
        """Reset tracking history"""
        self.prediction_errors.clear()
        self.complexity_costs.clear()
        self.intrinsic_rewards.clear()
        self.prober_success_rate.clear()
        
        self.logger.debug("Reset curiosity module history")
    
    def save_state(self, filepath: str):
        """Save curiosity module state"""
        
        state = {
            'complexity_estimator': self.complexity_estimator.state_dict(),
            'adversarial_prober': self.adversarial_prober.state_dict(),
            'complexity_optimizer': self.complexity_optimizer.state_dict(),
            'prober_optimizer': self.prober_optimizer.state_dict(),
            'config': self.config.dict(),
            'metrics': {
                'prediction_errors': self.prediction_errors,
                'complexity_costs': self.complexity_costs,
                'intrinsic_rewards': self.intrinsic_rewards,
                'prober_success_rate': self.prober_success_rate
            }
        }
        
        torch.save(state, filepath)
        self.logger.info(f"Saved curiosity module to {filepath}")
    
    def load_state(self, filepath: str):
        """Load curiosity module state"""
        
        state = torch.load(filepath, map_location='cpu')
        
        self.complexity_estimator.load_state_dict(state['complexity_estimator'])
        self.adversarial_prober.load_state_dict(state['adversarial_prober'])
        self.complexity_optimizer.load_state_dict(state['complexity_optimizer'])
        self.prober_optimizer.load_state_dict(state['prober_optimizer'])
        
        # Restore metrics
        metrics = state['metrics']
        self.prediction_errors = metrics['prediction_errors']
        self.complexity_costs = metrics['complexity_costs']
        self.intrinsic_rewards = metrics['intrinsic_rewards']
        self.prober_success_rate = metrics['prober_success_rate']
        
        self.logger.info(f"Loaded curiosity module from {filepath}")
