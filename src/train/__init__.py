"""Training utilities for graph generative models."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import EarlyStopping, get_device, format_time
from ..data import prepare_data_for_generation


class Trainer:
    """Trainer class for graph generative models.
    
    Args:
        model: Model to train
        data: Training data
        config: Training configuration
        device: Device to use for training
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: torch.Tensor,
        config: Dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.data = data
        self.config = config
        self.device = device or get_device()
        
        # Move model and data to device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 50),
            min_delta=config.get("min_delta", 0.001),
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
        }
        
        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_model_state = None
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        optimizer_type = self.config.get("optimizer", "adam")
        lr = self.config.get("learning_rate", 0.01)
        weight_decay = self.config.get("weight_decay", 5e-4)
        
        if optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get("scheduler")
        
        if scheduler_type is None:
            return None
        elif scheduler_type == "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=True,
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 200),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.
        
        Returns:
            Tuple[float, float]: Training loss and AUC
        """
        self.model.train()
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if isinstance(self.model, VGAE):
            z, edge_prob = self.model(self.data.x, self.data.train_pos_edge_index)
            
            # Reconstruction loss
            recon_loss = self.model.recon_loss(z, self.data.train_pos_edge_index)
            
            # KL divergence loss
            kl_loss = self.model.kl_loss()
            
            # Total loss
            loss = (
                self.config.get("recon_weight", 1.0) * recon_loss +
                self.config.get("kl_weight", 1.0) * kl_loss
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip_norm"]
                )
            
            self.optimizer.step()
            
            # Compute AUC
            with torch.no_grad():
                auc = self._compute_auc(z, self.data.train_pos_edge_index)
            
            return loss.item(), auc
        
        else:
            # For other model types, implement specific training logic
            raise NotImplementedError(f"Training not implemented for {type(self.model)}")
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model.
        
        Returns:
            Tuple[float, float]: Validation loss and AUC
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(self.model, VGAE):
                z, edge_prob = self.model(self.data.x, self.data.train_pos_edge_index)
                
                # Reconstruction loss
                recon_loss = self.model.recon_loss(z, self.data.train_pos_edge_index)
                kl_loss = self.model.kl_loss()
                
                loss = (
                    self.config.get("recon_weight", 1.0) * recon_loss +
                    self.config.get("kl_weight", 1.0) * kl_loss
                )
                
                # Compute AUC
                auc = self._compute_auc(z, self.data.train_pos_edge_index)
                
                return loss.item(), auc
            
            else:
                raise NotImplementedError(f"Validation not implemented for {type(self.model)}")
    
    def _compute_auc(self, z: torch.Tensor, edge_index: torch.Tensor) -> float:
        """Compute AUC score.
        
        Args:
            z: Node embeddings
            edge_index: Edge indices
            
        Returns:
            float: AUC score
        """
        from sklearn.metrics import roc_auc_score
        
        # Positive edges
        pos_prob = self.model.decode(z, edge_index)
        pos_labels = torch.ones_like(pos_prob)
        
        # Negative edges
        neg_edge_index = torch.randint(
            0, z.size(0), (2, edge_index.size(1)), device=self.device
        )
        neg_prob = self.model.decode(z, neg_edge_index)
        neg_labels = torch.zeros_like(neg_prob)
        
        # Combine
        all_prob = torch.cat([pos_prob, neg_prob])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        return roc_auc_score(all_labels.cpu().numpy(), all_prob.cpu().numpy())
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model.
        
        Returns:
            Dict[str, List[float]]: Training history
        """
        epochs = self.config.get("epochs", 200)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training
            train_loss, train_auc = self.train_epoch()
            
            # Validation
            val_loss, val_auc = self.validate()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # Logging
            if epoch % self.config.get("log_interval", 10) == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
                )
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print("Training completed!")
        return self.history
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]
