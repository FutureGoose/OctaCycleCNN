from typing import Optional, Literal, Dict, Any
from .base import BaseLogger
from .file_logger import FileLogger
import matplotlib.pyplot as plt

class NullLogger(BaseLogger):
    """
    A no-op logger that implements the BaseLogger interface but does nothing.
    Used when logging is disabled.
    """
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        pass

    def log_model_summary(self, summary_str: str) -> None:
        pass

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  metrics: Optional[Dict[str, float]] = None) -> None:
        pass

    def log_early_stopping(self) -> None:
        pass

    def save_figure(self, figure: plt.Figure, filename: str) -> None:
        pass


def create_logger(
    logger_type: Optional[Literal["file", "wandb"]] = "file",
    **kwargs
) -> BaseLogger:
    """
    Factory function to create the appropriate logger instance.

    Args:
        logger_type (Optional[Literal["file", "wandb"]]): Type of logger to create.
            If None, returns a NullLogger that does nothing.
        **kwargs: Additional arguments passed to the logger constructor.
            For FileLogger:
                - log_dir (str): Directory to save logs (default: "logs")
            For WandBLogger (not implemented yet):
                - project (str): WandB project name
                - entity (str): WandB entity name
                - config (dict): WandB config

    Returns:
        BaseLogger: An instance of the specified logger type.

    Raises:
        ValueError: If an unknown logger_type is specified.

    Example:
        >>> logger = create_logger("file", log_dir="my_logs")
        >>> logger = create_logger(None)  # Creates NullLogger
        >>> logger = create_logger("wandb", project="my_project")  # Not implemented yet
    """
    if logger_type is None:
        return NullLogger()
    
    if logger_type == "file":
        return FileLogger(**kwargs)
    
    if logger_type == "wandb":
        raise NotImplementedError(
            "WandB logger not implemented yet. "
            "Please use 'file' or None for now."
        )
    
    raise ValueError(
        f"Unknown logger_type: {logger_type}. "
        "Valid options are: 'file', 'wandb', or None"
    )