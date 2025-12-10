"""Core graph neural network layers and components."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder.
    
    Args:
        in_channels: Number of input features
        hidden_channels: List of hidden layer dimensions
        out_channels: Number of output features
        dropout: Dropout rate
        activation: Activation function
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        out_channels: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        prev_channels = in_channels
        for hidden_dim in hidden_channels:
            self.layers.append(GCNConv(prev_channels, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_channels = hidden_dim
        
        # Output layer
        self.layers.append(GCNConv(prev_channels, out_channels))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "none": nn.Identity(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            torch.Tensor: Encoded node features
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x, edge_index)
        if self.batch_norms is not None:
            x = self.batch_norms[-1](x)
        
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder.
    
    Args:
        in_channels: Number of input features
        hidden_channels: List of hidden layer dimensions
        out_channels: Number of output features
        heads: Number of attention heads
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        prev_channels = in_channels
        for hidden_dim in hidden_channels:
            self.layers.append(
                GATConv(prev_channels, hidden_dim, heads=heads, dropout=dropout)
            )
            prev_channels = hidden_dim * heads
        
        # Output layer
        self.layers.append(
            GATConv(prev_channels, out_channels, heads=1, dropout=dropout)
        )
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "none": nn.Identity(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            torch.Tensor: Encoded node features
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x, edge_index)
        
        return x


class GINEncoder(nn.Module):
    """Graph Isomorphism Network encoder.
    
    Args:
        in_channels: Number of input features
        hidden_channels: List of hidden layer dimensions
        out_channels: Number of output features
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        out_channels: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        prev_channels = in_channels
        for hidden_dim in hidden_channels:
            mlp = nn.Sequential(
                nn.Linear(prev_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            self.layers.append(GINConv(mlp))
            prev_channels = hidden_dim
        
        # Output layer
        mlp = nn.Sequential(
            nn.Linear(prev_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.layers.append(GINConv(mlp))
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "none": nn.Identity(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            torch.Tensor: Encoded node features
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x, edge_index)
        
        return x


class VariationalEncoder(nn.Module):
    """Variational encoder for VGAE.
    
    Args:
        encoder: Base encoder (GCN, GAT, or GIN)
        latent_dim: Dimension of latent space
    """
    
    def __init__(self, encoder: nn.Module, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        
        # Get output dimension from encoder
        # This is a simplified approach - in practice, you'd want to be more careful
        self.hidden_dim = getattr(encoder, 'hidden_dim', 64)
        
        # Mean and log-variance layers
        self.mu_layer = nn.Linear(self.hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.hidden_dim, latent_dim)
    
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log-variance
        """
        # Get hidden representation
        h = self.encoder(x, edge_index)
        
        # Compute mean and log-variance
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar


class InnerProductDecoder(nn.Module):
    """Inner product decoder for link prediction.
    
    Args:
        activation: Activation function for output
    """
    
    def __init__(self, activation: str = "sigmoid"):
        super().__init__()
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "none": nn.Identity(),
        }
        return activations.get(activation, nn.Sigmoid())
    
    def forward(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Node embeddings
            edge_index: Edge indices
            
        Returns:
            torch.Tensor: Edge probabilities
        """
        # Compute inner product
        edge_prob = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        
        return self.activation(edge_prob)


class MLPDecoder(nn.Module):
    """Multi-layer perceptron decoder.
    
    Args:
        in_channels: Number of input features
        hidden_channels: List of hidden layer dimensions
        out_channels: Number of output features
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        out_channels: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        for hidden_dim in hidden_channels:
            layers.extend([
                nn.Linear(prev_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_channels = hidden_dim
        
        layers.append(nn.Linear(prev_channels, out_channels))
        
        self.layers = nn.Sequential(*layers)
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "none": nn.Identity(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            torch.Tensor: Decoded features
        """
        x = self.layers(x)
        return self.activation(x)
