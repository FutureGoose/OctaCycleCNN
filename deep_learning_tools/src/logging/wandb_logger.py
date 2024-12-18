import wandb
from typing import Dict, Any, Optional
import torch.nn as nn
import matplotlib.pyplot as plt
from .base import BaseLogger
import os
class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(
        self, 
        run_id: str, 
        project: str,
        entity: Optional[str] = None,
        log_dir: str = "logs",
        **kwargs
    ):
        """Initialize wandb logger.
        
        Args:
            run_id (str): Unique identifier for this run
            project (str): W&B project name
            entity (str, optional): W&B username or team name
            log_dir (str): Local directory for logs
            **kwargs: Additional arguments passed to wandb.init
        """
        # ensure logs are saved in the wandb subdirectory
        wandb_dir = os.path.join(log_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_id,
            dir=wandb_dir,
            reinit=True,
            **kwargs
        )

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B."""
        wandb.config.update(hyperparameters)

    def log_model_summary(self, model: nn.Module) -> None:
        """Log model architecture to W&B."""
        # wandb handles model architecture visualization differently
        self.watch_model(model)

    def watch_model(self, model: nn.Module) -> None:
        """Set up model gradient and parameter tracking."""
        wandb.watch(
            model,
            log="all",  # log gradients and parameters
            log_freq=100  # log every 100 batches
        )

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: Dict[str, float]) -> None:
        """Log epoch metrics to W&B."""
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss
        }
        
        # add all metrics with proper namespacing
        for name, value in metrics.items():
            if name.startswith('train_'):
                log_dict[f"train/{name[6:]}"] = value
            elif name.startswith('val_'):
                log_dict[f"val/{name[4:]}"] = value
            else:
                log_dict[f"metrics/{name}"] = value

        wandb.log(log_dict)

    def log_early_stopping(self) -> None:
        """Log early stopping event."""
        wandb.log({"training/early_stopping": True})

    def save_figure(self, figure: plt.Figure, filename: str) -> None:
        """Save matplotlib figure to W&B."""
        wandb.log({filename: wandb.Image(figure)})

    def close(self) -> None:
        """Finish the wandb run."""
        wandb.finish()