"""Simple example script demonstrating graph generative models."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from src.data import load_dataset, generate_sbm_graph
from src.models import create_model
from src.utils import set_seed, get_device


def main():
    """Simple example of training and evaluating a VGAE model."""
    print("Graph Generative Models - Simple Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load dataset
    print("Loading dataset...")
    data = generate_sbm_graph(num_nodes=500, num_communities=3)
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    # Create model
    print("Creating VGAE model...")
    model = create_model(
        model_type="vgae",
        in_channels=data.x.size(1),
        hidden_channels=[64, 32],
        latent_dim=16,
    )
    
    device = get_device()
    model = model.to(device)
    data = data.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Using device: {device}")
    
    # Simple training loop
    print("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward pass
        z, edge_prob = model(data.x, data.edge_index)
        
        # Simple reconstruction loss
        pos_prob = model.decode(z, data.edge_index)
        recon_loss = -torch.log(pos_prob + 1e-15).mean()
        
        # Backward pass
        recon_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Loss = {recon_loss.item():.4f}")
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        
        # Generate some new edges
        num_new_edges = 100
        new_edge_index = torch.randint(0, data.num_nodes, (2, num_new_edges), device=device)
        new_edge_prob = model.decode(z, new_edge_index)
        
        print(f"Generated {num_new_edges} edge predictions")
        print(f"Average edge probability: {new_edge_prob.mean().item():.4f}")
        print(f"Edge probability std: {new_edge_prob.std().item():.4f}")
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
