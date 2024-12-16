import os
import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
from .base import BaseLogger

class FileLogger(BaseLogger):
    """
    File-based logger implementation that writes training progress to text files.
    
    This logger creates a timestamped directory for each training run and saves:
    - A text log file with hyperparameters, model summary, and training progress
    - Generated figures (e.g., loss plots)
    
    Attributes:
        log_dir (str): Directory where all logs are stored
        log_file (str): Path to the text log file
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the file logger.
        
        Args:
            log_dir (str): Base directory for storing logs.
                A timestamped subdirectory will be created for this run.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "training_log.txt")

    def _write_to_log(self, content: str, header: str = None) -> None:
        """
        Helper method to write content to log file.
        
        Args:
            content (str): Content to write
            header (str, optional): Optional header to add before content
        """
        with open(self.log_file, 'a') as f:
            if header:
                f.write(f"\n{header}\n")
            f.write(content)
            if header:
                f.write("\n" + "=" * len(header) + "\n\n")
            else:
                f.write("\n")

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log hyperparameters to file."""
        content = "\n".join(f"{key}: {value}" for key, value in hyperparameters.items())
        self._write_to_log(content, header="=== Hyperparameters ===")

    def log_model_summary(self, summary_str: str) -> None:
        """Log model architecture summary to file."""
        self._write_to_log(summary_str, header="==== Model Summary ====")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                 metrics: Dict[str, float] = None) -> None:
        """Log epoch results to file."""
        if metrics:
            metric_str = ', '.join([f"{name}: {value:.2f}%" for name, value in metrics.items()])
            log_message = (f"[epoch {epoch:02d}] train loss: {train_loss:.4f} | "
                         f"val loss: {val_loss:.4f} | {metric_str}")
        else:
            log_message = f"[epoch {epoch:02d}] train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        
        self._write_to_log(log_message)

    def log_early_stopping(self) -> None:
        """Log early stopping event to file."""
        self._write_to_log("🚨 Early stopping triggered.")

    def save_figure(self, figure: plt.Figure, filename: str) -> None:
        """Save a matplotlib figure to the log directory."""
        figure.savefig(os.path.join(self.log_dir, filename))