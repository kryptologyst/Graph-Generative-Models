"""Basic tests for graph generative models."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import generate_sbm_graph, generate_barabasi_albert_graph
from src.models import create_model
from src.utils import set_seed, get_device


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_sbm_graph_generation(self):
        """Test SBM graph generation."""
        data = generate_sbm_graph(num_nodes=100, num_communities=3)
        
        assert data.num_nodes == 100
        assert data.x.size(1) == 16  # Default feature dimension
        assert data.edge_index.size(0) == 2
        assert data.edge_index.size(1) > 0  # Should have some edges
        assert data.y.size(0) == 100  # Should have labels
    
    def test_barabasi_albert_graph_generation(self):
        """Test Barabási-Albert graph generation."""
        data = generate_barabasi_albert_graph(num_nodes=100)
        
        assert data.num_nodes == 100
        assert data.x.size(1) == 16  # Default feature dimension
        assert data.edge_index.size(0) == 2
        assert data.edge_index.size(1) > 0  # Should have some edges
        assert data.y.size(0) == 100  # Should have labels


class TestModelCreation:
    """Test model creation functions."""
    
    def test_vgae_creation(self):
        """Test VGAE model creation."""
        model = create_model(
            model_type="vgae",
            in_channels=16,
            hidden_channels=[32, 16],
            latent_dim=8,
        )
        
        assert model is not None
        assert hasattr(model, 'encode')
        assert hasattr(model, 'decode')
        assert hasattr(model, 'forward')
    
    def test_graphvae_creation(self):
        """Test GraphVAE model creation."""
        model = create_model(
            model_type="graphvae",
            in_channels=16,
            hidden_channels=[32, 16],
            latent_dim=8,
        )
        
        assert model is not None
        assert hasattr(model, 'encode')
        assert hasattr(model, 'decode')
        assert hasattr(model, 'forward')
    
    def test_graphrnn_creation(self):
        """Test GraphRNN model creation."""
        model = create_model(
            model_type="graphrnn",
            in_channels=16,
            hidden_channels=[32, 16],
            latent_dim=8,
        )
        
        assert model is not None
        assert hasattr(model, 'encode')
        assert hasattr(model, 'decode')
        assert hasattr(model, 'forward')


class TestModelForward:
    """Test model forward passes."""
    
    def setup_method(self):
        """Setup test data."""
        self.data = generate_sbm_graph(num_nodes=50, num_communities=2)
        self.device = get_device()
        self.data = self.data.to(self.device)
    
    def test_vgae_forward(self):
        """Test VGAE forward pass."""
        model = create_model(
            model_type="vgae",
            in_channels=self.data.x.size(1),
            hidden_channels=[16, 8],
            latent_dim=4,
        ).to(self.device)
        
        # Forward pass
        z, edge_prob = model(self.data.x, self.data.edge_index)
        
        assert z.size(0) == self.data.num_nodes
        assert z.size(1) == 4  # latent_dim
        assert edge_prob.size(0) == self.data.edge_index.size(1)
    
    def test_graphvae_forward(self):
        """Test GraphVAE forward pass."""
        model = create_model(
            model_type="graphvae",
            in_channels=self.data.x.size(1),
            hidden_channels=[16, 8],
            latent_dim=4,
        ).to(self.device)
        
        # Forward pass
        z, node_features, adj_matrix, node_probs = model(self.data.x, self.data.edge_index)
        
        assert z.size(0) == 1  # Graph-level representation
        assert z.size(1) == 4  # latent_dim
        assert node_features.size(0) == 100  # max_nodes
        assert adj_matrix.size(0) == 100
        assert node_probs.size(0) == 100
    
    def test_graphrnn_forward(self):
        """Test GraphRNN forward pass."""
        model = create_model(
            model_type="graphrnn",
            in_channels=self.data.x.size(1),
            hidden_channels=[16, 8],
            latent_dim=4,
        ).to(self.device)
        
        # Forward pass
        z, sequence = model(self.data.x, self.data.edge_index)
        
        assert z.size(0) == self.data.num_nodes
        assert z.size(1) == 4  # latent_dim
        assert sequence.size(0) == self.data.num_nodes
        assert sequence.size(1) == 100  # sequence_length


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(10)
        numpy_rand = torch.rand(10).numpy()
        
        # Set seed again
        set_seed(42)
        
        # Should be the same
        torch_rand2 = torch.rand(10)
        numpy_rand2 = torch.rand(10).numpy()
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert torch.allclose(torch.tensor(numpy_rand), torch.tensor(numpy_rand2))
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        
        # Test specific device
        device_cpu = get_device("cpu")
        assert device_cpu.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
