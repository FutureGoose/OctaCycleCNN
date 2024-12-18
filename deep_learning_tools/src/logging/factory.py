from typing import Optional, Literal, Dict, Any
import matplotlib.pyplot as plt
from .base import BaseLogger
from .file_logger import FileLogger
from .tensorboard_logger import TensorBoardLogger

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
            For TensorBoardLogger:
                - log_dir (str): Directory to save logs (default: "logs")
    Returns:
        BaseLogger: An instance of the specified logger type.

    Raises:
        ValueError: If an unknown logger_type is specified.

    Example:
        >>> logger = create_logger("file", log_dir="my_logs")
        >>> logger = create_logger(None)  # Creates NullLogger
        >>> logger = create_logger("wandb", project="my_project")  # Not implemented yet
        >>> logger = create_logger("tensorboard", log_dir="my_logs")
    """
    if logger_type is None:
        return NullLogger()
    
    logger_map = {
        "file": FileLogger,
        "tensorboard": TensorBoardLogger,
        "wandb": None  # not implemented yet
    }
    
    logger_class = logger_map.get(logger_type)
    if logger_class is None:
        raise ValueError(f"Unknown or unimplemented logger type: {logger_type}")
    
    return logger_class(**kwargs)