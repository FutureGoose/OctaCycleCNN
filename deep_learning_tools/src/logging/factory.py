from typing import Optional, Literal, Dict, Any
import matplotlib.pyplot as plt
from .base import BaseLogger
from .file_logger import FileLogger
from .tensorboard_logger import TensorBoardLogger
from .wandb_logger import WandBLogger

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
    logger_type: Optional[Literal["file", "wandb", "tensorboard"]] = None,
    **kwargs
) -> BaseLogger:
    """Create appropriate logger instance.
    
    Args:
        logger_type: Type of logger to use. If None or invalid type, returns NullLogger.
        **kwargs: Additional arguments passed to the logger constructor.
    
    Returns:
        BaseLogger: Appropriate logger instance or NullLogger if logging is disabled.
    """
    if logger_type is None or logger_type not in ["file", "wandb", "tensorboard"]:
        return NullLogger()
    
    logger_map = {
        "file": FileLogger,
        "tensorboard": TensorBoardLogger,
        "wandb": WandBLogger
    }
    
    logger_class = logger_map.get(logger_type)
    return logger_class(**kwargs)