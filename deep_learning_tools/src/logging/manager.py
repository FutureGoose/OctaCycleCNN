from typing import Dict, Any, Optional, Literal
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from .base import BaseLogger
from .factory import create_logger

class LoggerManager:
    """
    Manages logging operations for model training, abstracting away the specific logger implementation.
    
    This class serves as an intermediary between the trainer and the specific logging implementation,
    handling the collection and formatting of data to be logged.

    Attributes:
        logger (BaseLogger): The specific logger implementation to use.

    Example:
        >>> logger_manager = LoggerManager(logger_type="file", log_dir="logs")
        >>> logger_manager.on_training_start(trainer)
        >>> logger_manager.on_epoch_end(trainer, epoch=1)
    """

    def __init__(self, logger_type: Optional[Literal["file", "wandb", "tensorboard"]] = "file", **kwargs):
        """
        Initialize the logger manager.

        Args:
            logger_type (Optional[Literal["file", "wandb", "tensorboard"]]): Type of logger to use.
                If None, logging is disabled.
            **kwargs: Additional arguments passed to the logger constructor.
        """
        self.logger = create_logger(logger_type, **kwargs)

    def close(self) -> None:
        """Close the logger if it supports closing e.g. tensorboard."""
        if hasattr(self.logger, 'close'):
            self.logger.close()

    def collect_hyperparameters(self, trainer) -> Dict[str, Any]:
        """
        Extract and format hyperparameters from trainer state.

        Args:
            trainer: The model trainer instance.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """
        return {
            'batch_size': trainer.batch_size,
            'learning_rate': trainer.optimizer.param_groups[0]['lr'],
            'weight_decay': trainer.optimizer.param_groups[0].get('weight_decay', 0),
            'scheduler_step_size': trainer.scheduler.step_size if trainer.scheduler else None,
            'scheduler_gamma': trainer.scheduler.gamma if trainer.scheduler else None,
            'early_stopping_patience': trainer.early_stopping.patience,
            'early_stopping_delta': trainer.early_stopping.delta,
            'metrics': trainer.metrics_names,
            'optimizer': type(trainer.optimizer).__name__,
            'scheduler': type(trainer.scheduler).__name__ if trainer.scheduler else None
        }
 
    def collect_model_summary(self, model: nn.Module, sample_input: torch.Tensor) -> str:
        """
        Generate model summary string.

        Args:
            model (nn.Module): The PyTorch model.
            sample_input (torch.Tensor): Sample input tensor for the model.

        Returns:
            str: Formatted model summary string.
        """
        try:
            return str(summary(
                model, 
                input_size=tuple(sample_input.size()),
                verbose=0,
                col_width=16,
                col_names=["output_size", "num_params", "kernel_size", "mult_adds", "trainable"],
                row_settings=["var_names"]
            ))
        except Exception as e:
            return f"Failed to generate summary: {e}"

    def on_training_start(self, trainer) -> None:
        """
        Handle start of training events.

        Args:
            trainer: The model trainer instance.
        """
        hyperparams = self.collect_hyperparameters(trainer)
        self.logger.log_hyperparameters(hyperparams)
        
        if trainer.train_loader:
            try:
                sample_data, _ = next(iter(trainer.train_loader))
                summary = self.collect_model_summary(trainer.model, sample_data)
                self.logger.log_model_summary(summary)
            except StopIteration:
                pass  # Handle empty dataset case silently

    def on_epoch_end(self, trainer, epoch: int) -> None:
        """
        Handle end of epoch events.

        Args:
            trainer: The model trainer instance.
            epoch (int): Current epoch number.
        """
        metrics = {
            name: trainer.metrics_history[f'val_{name}'][-1]
            for name in trainer.metrics_names
        }
        self.logger.log_epoch(
            epoch=epoch,
            train_loss=trainer.metrics_history['train_loss'][-1],
            val_loss=trainer.metrics_history['val_loss'][-1],
            metrics=metrics
        )

    def on_early_stopping(self, trainer) -> None:
        """
        Handle early stopping event.

        Args:
            trainer: The model trainer instance.
        """
        self.logger.log_early_stopping()

    def save_figure(self, figure: plt.Figure, filename: str) -> None:
        """
        Save a matplotlib figure.

        Args:
            figure (plt.Figure): The figure to save.
            filename (str): Name of the file to save the figure to.
        """
        self.logger.save_figure(figure, filename)