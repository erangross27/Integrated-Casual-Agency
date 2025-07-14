"""
Action Planner implementation for ICA Framework
Soft Actor-Critic (SAC) with dynamic entropy adjustment and HER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random
from ..utils.config import PlannerConfig
from ..utils.logger import ica_logger


class ReplayBuffer:
    """Experience replay buffer with Hindsight Experience Replay (HER) support"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool, goal: Optional[np.ndarray] = None):
        """Add experience to buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e['state'] for e in batch])
        actions = torch.FloatTensor([e['action'] for e in batch])
        rewards = torch.FloatTensor([e['reward'] for e in batch])
        next_states = torch.FloatTensor([e['next_state'] for e in batch])
        dones = torch.BoolTensor([e['done'] for e in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def add_her_experiences(self, episode_buffer: List[Dict], 
                          her_ratio: float = 0.8) -> None:
        """Add Hindsight Experience Replay experiences"""
        
        if len(episode_buffer) < 2:
            return
        
        # Sample goals from future states in the episode
        for i, experience in enumerate(episode_buffer):
            if random.random() < her_ratio:
                # Sample a future state as the goal
                future_idx = random.randint(i + 1, len(episode_buffer) - 1)
                new_goal = episode_buffer[future_idx]['state']
                
                # Create modified experience with new goal
                modified_exp = experience.copy()
                modified_exp['goal'] = new_goal
                
                # Recalculate reward based on new goal
                modified_exp['reward'] = self._calculate_goal_reward(
                    experience['next_state'], new_goal
                )
                
                self.buffer.append(modified_exp)
    
    def _calculate_goal_reward(self, achieved_state: np.ndarray, 
                              desired_goal: np.ndarray) -> float:
        """Calculate reward based on goal achievement"""
        # Simple L2 distance - customize based on your task
        distance = np.linalg.norm(achieved_state - desired_goal)
        return -distance  # Negative distance as reward
    
    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for SAC"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Action bounds
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std"""
        
        features = self.network(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std to reasonable range
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization trick"""
        
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh transformation
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Correct for tanh transformation
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for SAC"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Q1 and Q2 values"""
        
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        
        return q1, q2


class ActionPlanner:
    """
    Action Planner using Soft Actor-Critic with dynamic entropy adjustment
    and Hindsight Experience Replay for the ICA Framework
    """
    
    def __init__(self, config: PlannerConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = ica_logger
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # Copy critic parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()  # Convert to float
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_actor)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = {'actor': [], 'critic': [], 'alpha': []}
        
        
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using current policy"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if evaluate:
            # Deterministic action for evaluation
            with torch.no_grad():
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
        else:
            # Stochastic action for exploration
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)
        
        return action.cpu().numpy().flatten()
    
    def update_alpha(self, global_confidence: float):
        """Update entropy coefficient based on global confidence"""
        
        # Dynamic alpha adjustment: lower confidence -> higher exploration
        confidence_factor = 1.0 - global_confidence
        target_alpha = self.config.alpha * (1.0 + confidence_factor)
        
        # Smooth adjustment - ensure alpha remains a float
        current_alpha = self.alpha.item() if hasattr(self.alpha, 'item') else float(self.alpha)
        target_alpha = target_alpha.item() if hasattr(target_alpha, 'item') else float(target_alpha)
        self.alpha = 0.9 * current_alpha + 0.1 * target_alpha
        
        # Convert tensor to float for logging
        confidence_value = global_confidence.item() if hasattr(global_confidence, 'item') else float(global_confidence)
        self.logger.debug(f"Updated alpha to {self.alpha:.3f} based on confidence {confidence_value:.3f}")
    
    def train_step(self, batch_size: int = None) -> Dict[str, float]:
        """Perform one training step"""
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        # Update critic
        critic_loss = self._update_critic(batch)
        
        # Update actor and alpha
        actor_loss, alpha_loss = self._update_actor_and_alpha(batch)
        
        # Update target networks
        self._update_targets()
        
        # Update metrics
        self.training_step += 1
        self.losses['critic'].append(critic_loss)
        self.losses['actor'].append(actor_loss)
        self.losses['alpha'].append(alpha_loss)
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': float(self.alpha) if hasattr(self.alpha, 'item') else self.alpha
        }
    
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update critic networks"""
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Calculate target Q values
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.config.gamma * next_q * (~dones.unsqueeze(1))
        
        # Calculate current Q values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor_and_alpha(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Update actor and alpha"""
        
        states = batch['states']
        
        # Sample new actions
        new_actions, log_probs = self.actor.sample(states)
        
        # Calculate Q values for new actions
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor loss
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()  # Convert to float
        
        return actor_loss.item(), alpha_loss.item()
    
    def _update_targets(self):
        """Soft update target networks"""
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool, goal: Optional[np.ndarray] = None):
        """Add experience to replay buffer"""
        
        self.replay_buffer.push(state, action, reward, next_state, done, goal)
    
    def finish_episode(self, episode_buffer: List[Dict], episode_reward: float):
        """Finish episode and add HER experiences"""
        
        # Add HER experiences
        self.replay_buffer.add_her_experiences(episode_buffer)
        
        # Track episode reward
        self.episode_rewards.append(episode_reward)
        
        self.logger.debug(f"Episode finished with reward {episode_reward:.2f}")
    
    def get_planner_metrics(self) -> Dict[str, float]:
        """Get comprehensive planner metrics"""
        
        metrics = {}
        
        if self.episode_rewards:
            metrics.update({
                'avg_episode_reward': np.mean(self.episode_rewards[-100:]),  # Last 100 episodes
                'episode_reward_std': np.std(self.episode_rewards[-100:]),
                'best_episode_reward': max(self.episode_rewards),
                'recent_reward_trend': self._calculate_trend(self.episode_rewards[-50:])
            })
        
        # Training losses
        for loss_type, losses in self.losses.items():
            if losses:
                metrics[f'avg_{loss_type}_loss'] = np.mean(losses[-100:])
        
        # Buffer statistics
        metrics.update({
            'buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.config.buffer_size,
            'training_steps': self.training_step,
            'current_alpha': float(self.alpha) if hasattr(self.alpha, 'item') else self.alpha
        })
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in recent values"""
        
        if len(values) < 10:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate correlation coefficient as trend measure
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def save_checkpoint(self, filepath: str):
        """Save planner checkpoint"""
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'config': self.config.dict()
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved planner checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load planner checkpoint"""
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()  # Convert to float
        
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        
        self.logger.info(f"Loaded planner checkpoint from {filepath}")
