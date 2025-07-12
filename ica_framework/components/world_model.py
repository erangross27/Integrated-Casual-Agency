"""
World Model implementation for ICA Framework
Bayesian Graph Neural Network (B-GNN) based on R-GCN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Optional torch-geometric imports
try:
    from torch_geometric.nn import RGCNConv, GATConv
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # Create placeholder classes
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class Batch:
        pass

from ..utils.config import WorldModelConfig
from ..utils.logger import ica_logger


class BayesianLayer(nn.Module):
    """Bayesian layer with uncertainty estimation"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Mean parameters
        self.weight_mu = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(output_dim))
        
        # Variance parameters (log-variance for numerical stability)
        self.weight_logvar = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1 - 5)
        self.bias_logvar = nn.Parameter(torch.zeros(output_dim) - 5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick"""
        if self.training:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean parameters for inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return torch.matmul(x, weight) + bias
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from prior (standard normal)"""
        # KL divergence for weights
        weight_kl = -0.5 * torch.sum(1 + self.weight_logvar - self.weight_mu.pow(2) - self.weight_logvar.exp())
        
        # KL divergence for bias
        bias_kl = -0.5 * torch.sum(1 + self.bias_logvar - self.bias_mu.pow(2) - self.bias_logvar.exp())
        
        return weight_kl + bias_kl


class SimpleGraphConv(nn.Module):
    """Simple graph convolution layer as fallback when torch-geometric is not available"""
    
    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        
        # Relation-specific weight matrices
        self.weight_matrices = nn.Parameter(torch.randn(num_relations, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Simple forward pass"""
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        
        # Process each edge type
        for rel_type in range(self.num_relations):
            # Find edges of this type
            mask = edge_type == rel_type
            if mask.sum() == 0:
                continue
            
            # Get edges of this type
            rel_edges = edge_index[:, mask]
            
            if rel_edges.size(1) > 0:
                # Apply relation-specific transformation
                transformed = torch.matmul(x, self.weight_matrices[rel_type])
                
                # Simple message passing (mean aggregation)
                for i in range(x.size(0)):
                    # Find incoming edges
                    incoming_mask = rel_edges[1] == i
                    if incoming_mask.sum() > 0:
                        source_nodes = rel_edges[0, incoming_mask]
                        messages = transformed[source_nodes]
                        out[i] += messages.mean(dim=0)
        
        return out + self.bias


# Use appropriate graph convolution based on availability
if HAS_TORCH_GEOMETRIC:
    GraphConvLayer = RGCNConv
else:
    GraphConvLayer = SimpleGraphConv
    """Bayesian layer with uncertainty quantification"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features))
        
        # Prior parameters
        self.weight_prior_mu = 0.0
        self.weight_prior_logvar = np.log(prior_std ** 2)
        self.bias_prior_mu = 0.0
        self.bias_prior_logvar = np.log(prior_std ** 2)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            # Sample weights and biases
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight_eps = torch.randn_like(weight_std)
            bias_eps = torch.randn_like(bias_std)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean parameters
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence between posterior and prior"""
        weight_kl = self._kl_div(self.weight_mu, self.weight_logvar, 
                                self.weight_prior_mu, self.weight_prior_logvar)
        bias_kl = self._kl_div(self.bias_mu, self.bias_logvar,
                              self.bias_prior_mu, self.bias_prior_logvar)
        return weight_kl + bias_kl
    
    def _kl_div(self, mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
        """KL divergence between two Gaussians"""
        kl = 0.5 * (logvar_p - logvar_q + 
                   torch.exp(logvar_q - logvar_p) + 
                   ((mu_q - mu_p) ** 2) * torch.exp(-logvar_p) - 1)
        return kl.sum()


class AttentionGate(nn.Module):
    """Attention mechanism for edge confidence weighting"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, source_emb: torch.Tensor, target_emb: torch.Tensor, 
                edge_confidence: torch.Tensor) -> torch.Tensor:
        """Calculate attention weights based on node embeddings and edge confidence"""
        combined = torch.cat([source_emb, target_emb], dim=-1)
        attention_weight = self.attention(combined)
        
        # Combine with edge confidence
        final_weight = attention_weight * edge_confidence.unsqueeze(-1)
        return final_weight


class WorldModel(nn.Module):
    """
    Bayesian Graph Neural Network World Model
    Implements R-GCN architecture with attention mechanism and uncertainty quantification
    """
    
    def __init__(self, config: WorldModelConfig, num_relations: int, num_node_features: int):
        super().__init__()
        self.config = config
        self.num_relations = num_relations
        self.num_node_features = num_node_features
        self.logger = ica_logger
        
        # Graph convolutional layers
        self.rgcn_layers = nn.ModuleList()
        
        # First layer
        self.rgcn_layers.append(
            RGCNConv(num_node_features, config.hidden_dim, num_relations)
        )
        
        # Hidden layers
        for _ in range(config.num_layers - 2):
            self.rgcn_layers.append(
                RGCNConv(config.hidden_dim, config.hidden_dim, num_relations)
            )
        
        # Output layer
        self.rgcn_layers.append(
            RGCNConv(config.hidden_dim, config.embedding_dim, num_relations)
        )
        
        # Attention mechanism
        self.attention_gate = AttentionGate(config.embedding_dim)
        
        # Bayesian prediction layers
        self.bayesian_predictor = BayesianLayer(config.embedding_dim, config.embedding_dim)
        
        # Output layers for mean and variance
        self.mean_predictor = nn.Linear(config.embedding_dim, num_node_features)
        self.var_predictor = nn.Sequential(
            nn.Linear(config.embedding_dim, num_node_features),
            nn.Softplus()  # Ensure positive variance
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self.logger.info(f"Initialized World Model with {config.num_layers} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: torch.Tensor, edge_confidence: torch.Tensor,
                sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the world model
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge types [num_edges]
            edge_confidence: Edge confidences [num_edges]
            sample: Whether to sample from Bayesian layers
            
        Returns:
            Tuple of (predicted_mean, predicted_variance)
        """
        
        # Graph convolutions
        h = x
        for i, layer in enumerate(self.rgcn_layers):
            h = layer(h, edge_index, edge_type)
            if i < len(self.rgcn_layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        
        # Apply attention mechanism
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_emb = h[source_nodes]
        target_emb = h[target_nodes]
        
        attention_weights = self.attention_gate(source_emb, target_emb, edge_confidence)
        
        # Aggregate with attention
        # This is a simplified aggregation - in practice you'd want more sophisticated pooling
        weighted_h = h * attention_weights.mean(dim=0, keepdim=True)
        
        # Bayesian prediction
        bayesian_out = self.bayesian_predictor(weighted_h, sample=sample)
        
        # Predict mean and variance
        predicted_mean = self.mean_predictor(bayesian_out)
        predicted_var = self.var_predictor(bayesian_out)
        
        return predicted_mean, predicted_var
    
    def predict_next_state(self, current_state: torch.Tensor, 
                          action: torch.Tensor, graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state given current state and action
        
        Args:
            current_state: Current state features
            action: Action vector
            graph_data: Graph structure data
            
        Returns:
            Tuple of (next_state_mean, next_state_variance)
        """
        
        # Combine state and action
        combined_features = torch.cat([current_state, action.unsqueeze(0).expand(current_state.size(0), -1)], dim=-1)
        
        # Forward pass
        mean, var = self.forward(
            combined_features, 
            graph_data.edge_index, 
            graph_data.edge_type,
            graph_data.edge_confidence
        )
        
        return mean, var
    
    def calculate_uncertainty(self, x: torch.Tensor, edge_index: torch.Tensor,
                            edge_type: torch.Tensor, edge_confidence: torch.Tensor,
                            num_samples: int = 10) -> torch.Tensor:
        """Calculate model uncertainty using Monte Carlo sampling"""
        
        predictions = []
        for _ in range(num_samples):
            mean, var = self.forward(x, edge_index, edge_type, edge_confidence, sample=True)
            predictions.append(mean)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [num_samples, num_nodes, num_features]
        
        # Calculate epistemic uncertainty (variance across samples)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        # Calculate aleatoric uncertainty (average predicted variance)
        _, aleatoric_uncertainty = self.forward(x, edge_index, edge_type, edge_confidence, sample=False)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return total_uncertainty
    
    def get_kl_loss(self) -> torch.Tensor:
        """Get KL divergence loss for Bayesian layers"""
        kl_loss = 0
        for module in self.modules():
            if isinstance(module, BayesianLayer):
                kl_loss += module.kl_divergence()
        return kl_loss
    
    def update_confidence(self, prediction_error: torch.Tensor, 
                         edge_confidences: Dict[Tuple[int, int], float],
                         learning_rate: float = 0.01) -> Dict[Tuple[int, int], float]:
        """Update edge confidences based on prediction error"""
        
        updated_confidences = {}
        avg_error = prediction_error.mean().item()
        
        for edge_id, confidence in edge_confidences.items():
            # Simple update rule - decrease confidence if error is high
            adjustment = -learning_rate * avg_error
            new_confidence = max(0.01, min(1.0, confidence + adjustment))
            updated_confidences[edge_id] = new_confidence
        
        return updated_confidences


class WorldModelTrainer:
    """Training utilities for World Model"""
    
    def __init__(self, model: WorldModel, config: WorldModelConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        self.logger = ica_logger
    
    def train_step(self, batch: Batch, kl_weight: float = 0.01) -> Dict[str, float]:
        """Single training step"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_mean, predicted_var = self.model(
            batch.x, batch.edge_index, batch.edge_type, batch.edge_confidence
        )
        
        # Calculate losses
        # Negative log likelihood loss
        nll_loss = self._negative_log_likelihood(predicted_mean, predicted_var, batch.y)
        
        # KL divergence loss
        kl_loss = self.model.get_kl_loss()
        
        # Total loss
        total_loss = nll_loss + kl_weight * kl_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "nll_loss": nll_loss.item(),
            "kl_loss": kl_loss.item()
        }
    
    def evaluate(self, batch: Batch) -> Dict[str, float]:
        """Evaluate model on batch"""
        
        self.model.eval()
        with torch.no_grad():
            predicted_mean, predicted_var = self.model(
                batch.x, batch.edge_index, batch.edge_type, batch.edge_confidence,
                sample=False
            )
            
            # Calculate metrics
            mse = F.mse_loss(predicted_mean, batch.y)
            nll = self._negative_log_likelihood(predicted_mean, predicted_var, batch.y)
            
            # Calculate uncertainty
            uncertainty = self.model.calculate_uncertainty(
                batch.x, batch.edge_index, batch.edge_type, batch.edge_confidence
            )
            
            return {
                "mse": mse.item(),
                "nll": nll.item(),
                "mean_uncertainty": uncertainty.mean().item(),
                "max_uncertainty": uncertainty.max().item()
            }
    
    def _negative_log_likelihood(self, mean: torch.Tensor, var: torch.Tensor, 
                               target: torch.Tensor) -> torch.Tensor:
        """Calculate negative log likelihood for Gaussian distribution"""
        
        # Add small epsilon to avoid log(0)
        var = var + 1e-6
        
        # NLL = 0.5 * (log(2π) + log(var) + (target - mean)^2 / var)
        nll = 0.5 * (torch.log(2 * np.pi * var) + (target - mean) ** 2 / var)
        
        return nll.mean()
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint
