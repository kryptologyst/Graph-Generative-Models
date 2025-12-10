"""Utility functions for graph generative models."""

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: The device to use
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation(activation: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        activation: Name of activation function
        
    Returns:
        nn.Module: Activation function
    """
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=-1),
        "none": nn.Identity(),
    }
    
    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    
    return activations[activation]


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
    """
    from omegaconf import OmegaConf
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    from omegaconf import OmegaConf
    
    OmegaConf.save(config, save_path)


def create_directories(paths: Dict[str, str]) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: Dictionary of path names and their locations
    """
    import os
    
    for name, path in paths.items():
        os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        restore_best_weights: Whether to restore best weights when stopping
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
        """
        self.best_weights = model.state_dict().copy()
