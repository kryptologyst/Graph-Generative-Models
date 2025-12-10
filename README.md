# Graph Generative Models

A production-ready implementation of graph generative models using PyTorch Geometric. This project provides implementations of Variational Graph Autoencoders (VGAE), Graph Variational Autoencoders (GraphVAE), and Graph Recurrent Neural Networks (GraphRNN) for graph generation and link prediction tasks.

## Features

- **Multiple Model Architectures**: VGAE, GraphVAE, and GraphRNN implementations
- **Flexible Encoders**: GCN, GAT, and GIN encoder options
- **Comprehensive Evaluation**: Link prediction and graph generation metrics
- **Interactive Demo**: Streamlit-based visualization and exploration
- **Production Ready**: Type hints, configuration management, and proper error handling
- **Device Agnostic**: Automatic CUDA/MPS/CPU device selection
- **Reproducible**: Deterministic seeding and checkpointing

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Apple Silicon with MPS support (optional)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Graph-Generative-Models.git
cd Graph-Generative-Models

# Install dependencies
pip install -r requirements.txt

# Or install with optional dependencies
pip install -e ".[dev,molecules,serving]"
```

### PyTorch Geometric Installation

For the best experience, install PyTorch Geometric with the appropriate PyTorch version:

```bash
# For PyTorch 2.0+
pip install torch-geometric torch-scatter torch-sparse torch-cluster pyg-lib
```

## Quick Start

### 1. Train a Model

```bash
# Train VGAE on Cora dataset
python train.py --model vgae --data cora --epochs 200

# Train GraphVAE on synthetic data
python train.py --model graphvae --data sbm --epochs 100

# Use custom configuration
python train.py --config configs/config.yaml --model graphrnn --data barabasi_albert
```

### 2. Run Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo.py

# Or use Gradio (if installed)
python demo.py --interface gradio
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python train.py --eval-only --model vgae --data cora
```

## Project Structure

```
graph-generative-models/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── layers/            # Neural network layers
│   ├── data/              # Data utilities
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation utilities
│   └── utils/             # General utilities
├── configs/               # Configuration files
│   ├── config.yaml        # Main configuration
│   ├── model/             # Model-specific configs
│   ├── data/              # Dataset configs
│   └── train/             # Training configs
├── data/                  # Data directory
├── outputs/               # Training outputs
├── assets/                # Generated assets
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
├── demo.py                # Interactive demo
├── train.py               # Training script
├── requirements.txt       # Dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Models

### Variational Graph Autoencoder (VGAE)

The VGAE learns probabilistic node embeddings and reconstructs graphs through edge prediction. It's particularly effective for link prediction tasks.

**Key Features:**
- Probabilistic latent space
- KL divergence regularization
- Inner product decoder
- Suitable for transductive learning

### Graph Variational Autoencoder (GraphVAE)

GraphVAE generates entire graphs by learning graph-level representations. It's designed for graph generation tasks.

**Key Features:**
- Graph-level latent space
- Node existence prediction
- Adjacency matrix generation
- Suitable for inductive learning

### Graph Recurrent Neural Network (GraphRNN)

GraphRNN generates graphs sequentially, making it suitable for generating graphs with specific structural properties.

**Key Features:**
- Sequential generation
- RNN-based decoder
- Flexible sequence length
- Good for structured graphs

## Datasets

### Citation Networks
- **Cora**: 2,708 nodes, 5,429 edges
- **Citeseer**: 3,327 nodes, 4,732 edges  
- **PubMed**: 19,717 nodes, 44,338 edges

### Synthetic Graphs
- **SBM**: Stochastic Block Model graphs
- **Barabási-Albert**: Scale-free networks

### Custom Datasets

You can easily add custom datasets by implementing the data loading interface:

```python
from src.data import load_dataset

# Add your dataset
def load_custom_dataset(**kwargs):
    # Your implementation
    pass

# Register in load_dataset function
```

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific settings
- `configs/data/`: Dataset configurations
- `configs/train/`: Training parameters

### Example Configuration

```yaml
# configs/config.yaml
experiment:
  name: "graph_generative_models"
  seed: 42
  device: "auto"

model:
  latent_dim: 64
  hidden_dims: [128, 64]
  dropout: 0.1

train:
  epochs: 200
  learning_rate: 0.01
  batch_size: 1
  patience: 50
```

## Training

### Command Line Interface

```bash
# Basic training
python train.py --model vgae --data cora --epochs 200

# Advanced options
python train.py \
    --model graphvae \
    --data sbm \
    --epochs 500 \
    --lr 0.005 \
    --device cuda \
    --output-dir ./experiments/graphvae_sbm

# Resume training
python train.py --resume ./outputs/best_model.pt
```

### Programmatic Training

```python
from src.data import load_dataset
from src.models import create_model
from src.train import Trainer

# Load data
data, _ = load_dataset("cora")

# Create model
model = create_model("vgae", data.x.size(1), [128, 64])

# Train
trainer = Trainer(model, data, config)
history = trainer.train()
```

## Evaluation

### Link Prediction Metrics

- **AUC**: Area Under the ROC Curve
- **AP**: Average Precision
- **F1**: F1 Score
- **Accuracy**: Binary Classification Accuracy

### Graph Generation Metrics

- **Density**: Graph density distribution
- **Clustering**: Clustering coefficient
- **Degree Distribution**: Node degree statistics
- **Structural Similarity**: Comparison with original graphs

### Evaluation Example

```python
from src.eval import Evaluator

# Create evaluator
evaluator = Evaluator(model, data)

# Evaluate link prediction
lp_results = evaluator.evaluate_link_prediction()
print(f"AUC: {lp_results['auc']:.4f}")

# Evaluate graph generation
gen_results = evaluator.evaluate_graph_generation()
print(f"Density: {gen_results['density_mean']:.4f}")
```

## Interactive Demo

The Streamlit demo provides an interactive interface for:

- **Data Visualization**: Explore dataset statistics and structure
- **Embedding Analysis**: Visualize node embeddings with t-SNE/PCA
- **Graph Generation**: Generate new graphs interactively
- **Model Analysis**: Compare generated vs original graphs

### Running the Demo

```bash
# Streamlit
streamlit run demo.py

# Gradio (alternative)
python demo.py --interface gradio
```

## Advanced Usage

### Custom Models

```python
from src.layers import GCNEncoder, VariationalEncoder
from src.models import VGAE

# Create custom encoder
encoder = GCNEncoder(in_channels=1433, hidden_channels=[256, 128], out_channels=64)
var_encoder = VariationalEncoder(encoder, latent_dim=64)

# Create VGAE
model = VGAE(var_encoder)
```

### Custom Datasets

```python
from torch_geometric.data import Data
from src.data import prepare_data_for_generation

# Create custom data
data = Data(x=features, edge_index=edge_index, y=labels)
data = prepare_data_for_generation(data)
```

### Distributed Training

```python
# Multi-GPU training
python train.py --model vgae --data cora --device cuda --batch-size 4

# DDP training (advanced)
torchrun --nproc_per_node=4 train.py --model graphvae --data sbm
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster training
2. **Batch Processing**: Increase batch size for larger datasets
3. **Memory Optimization**: Use gradient checkpointing for large models
4. **Early Stopping**: Configure patience to prevent overfitting
5. **Learning Rate Scheduling**: Use ReduceLROnPlateau for better convergence

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or model size
   python train.py --batch-size 1 --model vgae
   ```

2. **Import Errors**
   ```bash
   # Reinstall PyTorch Geometric
   pip uninstall torch-geometric
   pip install torch-geometric
   ```

3. **Slow Training**
   ```bash
   # Use GPU acceleration
   python train.py --device cuda
   ```

### Debug Mode

```bash
# Enable debug logging
python train.py --model vgae --data cora --log-level debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
ruff check src/
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{graph_generative_models,
  title={Graph Generative Models},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Graph-Generative-Models}
}
```

## Acknowledgments

- PyTorch Geometric team for the excellent graph learning framework
- Original VGAE paper authors (Kipf & Welling, 2016)
- GraphVAE paper authors (Kipf & Welling, 2016)
- GraphRNN paper authors (You et al., 2018)

## References

1. Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. arXiv preprint arXiv:1611.07308.
2. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. ICLR.
3. You, J., Ying, R., Ren, X., Hamilton, W., & Leskovec, J. (2018). GraphRNN: Generating realistic graphs with deep auto-regressive models. ICML.
4. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. ICLR Workshop.
# Graph-Generative-Models
