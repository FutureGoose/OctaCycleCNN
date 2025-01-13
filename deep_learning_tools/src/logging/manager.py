from typing import Dict, Any, Optional, Literal
import torch
from torchinfo import summary
import matplotlib.pyplot as plt
from .factory import create_logger
from .wandb_logger import WandBLogger
from datetime import datetime
import os

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

class LoggerManager:
    def __init__(
        self, 
        logger_type: Optional[Literal["file", "wandb", "tensorboard"]] = "file",
        wandb_project: Optional[str] = None,  # renamed to be explicit
        wandb_entity: Optional[str] = None,   # renamed to be explicit,
        log_dir: str = "logs",
        **kwargs
    ):
        """Initialize the logger manager.
        
        Args:
            logger_type: Type of logger to use ("file", "wandb", or "tensorboard")
            wandb_project: Name of the W&B project to log to (required if using wandb)
            wandb_entity: W&B username or team name (optional)
            log_dir: Base directory for logs (e.g., "logs")
            **kwargs: Additional arguments passed to the logger
        """
        self.logger_type = logger_type
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if logger_type == "wandb":
            if wandb_project is None:
                raise ValueError("wandb_project must be specified when using wandb logger")
            
            # For wandb, we create the logs/wandb directory structure
            # This is the ONLY place where we add the wandb subdirectory
            #wandb_log_dir = os.path.join(log_dir, "wandb")
            os.makedirs(log_dir, exist_ok=True)
            
            self.logger = create_logger(
                logger_type,
                run_id=run_id,
                project=wandb_project,
                entity=wandb_entity,
                log_dir=log_dir,  # WandBLogger will use this directory as-is
                **kwargs
            )
        else:
            # For other loggers, let them handle their own directory structure
            self.logger = create_logger(
                logger_type,
                run_id=run_id,
                log_dir=log_dir,
                **kwargs
            )

    def close(self) -> None:
        """Close the logger if it supports closing e.g. tensorboard."""
        if hasattr(self.logger, 'close'):
            self.logger.close()

    def collect_hyperparameters(self, trainer: "ModelTrainer") -> Dict[str, Any]:
        """
        Extract and format hyperparameters from trainer state.

        Args:
            trainer: The model trainer instance.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """

        # get scheduler info safely
        scheduler_info = {}
        if trainer.scheduler:
            scheduler_type = type(trainer.scheduler).__name__
            scheduler_info = {
            'scheduler_type': scheduler_type,
            # for StepLR
            'step_size': getattr(trainer.scheduler, 'step_size', None),
            'gamma': getattr(trainer.scheduler, 'gamma', None),
            # for LambdaLR
            'lr_lambda': str(getattr(trainer.scheduler, 'lr_lambdas', None))
        }
        return {
            'batch_size': trainer.batch_size,
            'learning_rate': trainer.optimizer.param_groups[0]['lr'],
            'weight_decay': trainer.optimizer.param_groups[0].get('weight_decay', 0),
            'early_stopping_patience': trainer.early_stopping.patience,
            'early_stopping_delta': trainer.early_stopping.delta,
            'metrics': trainer.metrics_names,
            'optimizer': type(trainer.optimizer).__name__,
            'scheduler': type(trainer.scheduler).__name__ if trainer.scheduler else None,
            **scheduler_info  # include scheduler-specific parameters
        }
 
    def collect_model_summary(self, trainer: "ModelTrainer") -> str:
        """Generate model summary string using actual data sample.
        
        Args:
            trainer: The model trainer instance containing model and data loaders
            
        Returns:
            str: Formatted model summary string
        """
        try:
            if trainer.train_loader is None:
                raise ValueError("Training data loader not initialized")
                
            # get sample batch from training data
            sample_data, _ = next(iter(trainer.train_loader))
            if not isinstance(sample_data, torch.Tensor):
                raise TypeError(f"Expected tensor input, got {type(sample_data)}")
                
            return str(summary(
                trainer.model, 
                input_size=tuple(sample_data.size()),
                verbose=0,
                col_width=16,
                col_names=["output_size", "num_params", "kernel_size", "mult_adds", "trainable"],
                row_settings=["var_names"]
            ))
        except Exception as e:
            return f"Failed to generate summary: {str(e)}"

    def on_training_start(self, trainer: "ModelTrainer") -> None:
        """Handle events at the start of training."""
        # collect hyperparameters
        hyperparams = self.collect_hyperparameters(trainer)
        self.logger.log_hyperparameters(hyperparams)
        
        # log model architecture
        if isinstance(self.logger, WandBLogger):
            self.logger.watch_model(trainer.model)
        else:
            model_summary = self.collect_model_summary(trainer)
            self.logger.log_model_summary(model_summary)

    def on_epoch_end(self, trainer: "ModelTrainer", epoch: int) -> None:
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

    def on_early_stopping(self, trainer: "ModelTrainer") -> None:
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