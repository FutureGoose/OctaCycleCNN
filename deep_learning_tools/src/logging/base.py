from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import matplotlib.pyplot as plt
import torch.nn as nn

class BaseLogger(ABC):
    """Abstract base class defining the logging interface."""
    
    @abstractmethod
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_model_summary(self, model_info: Union[str, nn.Module]) -> None:
        """Log model architecture summary.
        
        Args:
            model_info: Either a string containing model summary or the model itself
        """
        pass

    def watch_model(self, model: nn.Module) -> None:  # new method for gradient tracking
        """Set up model gradient and parameter tracking.
        
        Default implementation does nothing. Override in loggers that support gradient tracking.
        """
        pass
    
    @abstractmethod
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  metrics: Dict[str, float] = None) -> None:
        """Log epoch results."""
        pass
    
    @abstractmethod
    def log_early_stopping(self) -> None:
        """Log early stopping event."""
        pass
    
    @abstractmethod
    def save_figure(self, figure: plt.Figure, filename: str) -> None:
        """Save a matplotlib figure."""
        pass