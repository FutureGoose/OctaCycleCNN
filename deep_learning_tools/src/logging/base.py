from abc import ABC, abstractmethod
from typing import Dict, Any
import matplotlib.pyplot as plt

class BaseLogger(ABC):
    """Abstract base class defining the logging interface."""
    
    @abstractmethod
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def log_model_summary(self, summary_str: str) -> None:
        """Log model architecture summary."""
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