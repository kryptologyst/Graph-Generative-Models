#!/usr/bin/env python3
"""Main training script for graph generative models."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import load_dataset, prepare_data_for_generation, get_graph_statistics
from src.models import create_model
from src.train import Trainer
from src.eval import Evaluator
from src.utils import set_seed, get_device, create_directories, load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train graph generative models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["vgae", "graphvae", "graphrnn"],
        default="vgae",
        help="Model type to train",
    )
    
    parser.add_argument(
        "--data",
        type=str,
        choices=["cora", "citeseer", "pubmed", "sbm", "barabasi_albert"],
        default="cora",
        help="Dataset to use",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train",
    )
    
    return parser.parse_args()


def load_and_prepare_data(config: DictConfig) -> torch.Tensor:
    """Load and prepare data for training.
    
    Args:
        config: Configuration object
        
    Returns:
        torch.Tensor: Prepared data
    """
    print("Loading dataset...")
    
    # Load dataset
    data, dataset_name = load_dataset(
        name=config.data.name,
        root=config.data.root,
        **config.data.get("params", {})
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.edge_index.size(1)}")
    print(f"Features: {data.x.size(1)}")
    
    # Prepare data for generation
    data = prepare_data_for_generation(
        data,
        train_ratio=config.data.get("train_ratio", 0.8),
        val_ratio=config.data.get("val_ratio", 0.1),
        test_ratio=config.data.get("test_ratio", 0.1),
        add_negative_edges=config.data.get("add_negative_edges", True),
        num_negative_samples=config.data.get("num_negative_samples", 1),
    )
    
    # Print graph statistics
    stats = get_graph_statistics(data)
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return data


def create_and_train_model(
    config: DictConfig,
    data: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Create and train the model.
    
    Args:
        config: Configuration object
        data: Training data
        args: Command line arguments
        
    Returns:
        Tuple[torch.nn.Module, Dict[str, Any]]: Trained model and training history
    """
    print(f"\nCreating {args.model} model...")
    
    # Create model
    model = create_model(
        model_type=args.model,
        in_channels=data.x.size(1),
        hidden_channels=config.model.get("hidden_dims", [128, 64]),
        latent_dim=config.model.get("latent_dim", 64),
        encoder_type=config.model.get("encoder_type", "gcn"),
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Update config with command line arguments
    config.train.epochs = args.epochs
    config.train.learning_rate = args.lr
    config.train.batch_size = args.batch_size
    config.experiment.device = args.device
    config.experiment.seed = args.seed
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data=data,
        config=config.train,
        device=get_device(args.device),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_model(args.resume)
    
    # Train model
    if not args.eval_only:
        print("\nStarting training...")
        history = trainer.train()
        
        # Save model
        model_path = os.path.join(args.output_dir, "best_model.pt")
        trainer.save_model(model_path)
        print(f"Model saved to: {model_path}")
        
        return model, history
    else:
        print("Skipping training (eval-only mode)")
        return model, {}


def evaluate_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate the trained model.
    
    Args:
        model: Trained model
        data: Test data
        args: Command line arguments
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        data=data,
        device=get_device(args.device),
    )
    
    # Evaluate link prediction
    lp_results = evaluator.evaluate_link_prediction()
    print("\nLink Prediction Results:")
    for metric, value in lp_results.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Evaluate graph generation (if applicable)
    if args.model in ["graphvae", "graphrnn"]:
        gen_results = evaluator.evaluate_graph_generation()
        print("\nGraph Generation Results:")
        for metric, value in gen_results.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Create evaluation report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    report = evaluator.create_evaluation_report(save_path=report_path)
    print(f"\nEvaluation report saved to: {report_path}")
    
    return lp_results


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    create_directories({"output_dir": args.output_dir})
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.model = args.model
    config.data.name = args.data
    
    print("=" * 60)
    print("GRAPH GENERATIVE MODELS TRAINING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Output Directory: {args.output_dir}")
    
    # Load and prepare data
    data = load_and_prepare_data(config)
    
    # Create and train model
    model, history = create_and_train_model(config, data, args)
    
    # Evaluate model
    eval_results = evaluate_model(model, data, args)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.yaml")
    OmegaConf.save(config, config_path)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
