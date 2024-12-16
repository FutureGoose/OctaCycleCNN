import os
import datetime
from typing import Dict, Any

class TrainingLogger:
    """Handles logging functionality for model training."""
    
    def __init__(self, log_dir: str = "logs", enable_logging: bool = True):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory to save logs
            enable_logging (bool): Whether to enable logging
        """
        self.enable_logging = enable_logging
        if enable_logging:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, "training_log.txt")
        else:
            self.log_dir = None
            self.log_file = None

    def log_hyperparameters(self, hyperparameters: Dict[str, Any], verbose: bool = True):
        """
        Log hyperparameters to file and optionally print them.
        
        Args:
            hyperparameters (dict): Dictionary of hyperparameters
            verbose (bool): Whether to print hyperparameters
        """
        log_header = "\n=== Hyperparameters ===\n"
        log_footer = "=======================\n\n"

        if verbose:
            print("\n\033[38;5;180m" + "=== Hyperparameters ===" + "\033[0m")
            for key, value in hyperparameters.items():
                print(f"{key}: {value}")
            print("\033[38;5;180m" + "=======================" + "\033[0m\n")

        if self.enable_logging:
            with open(self.log_file, 'a') as f:
                f.write(log_header)
                for key, value in hyperparameters.items():
                    f.write(f"{key}: {value}\n")
                f.write(log_footer)

    def log_model_summary(self, summary_str: str, verbose: bool = True):
        """
        Log model summary to file and optionally print it.
        
        Args:
            summary_str (str): Model summary string
            verbose (bool): Whether to print the summary
        """
        if verbose:
            print("\033[38;5;180m" + "==== Model Summary ====" + "\033[0m")
            print(summary_str)
            print("\033[38;5;180m" + "=" * 120 + "\033[0m\n")

        if self.enable_logging:
            with open(self.log_file, 'a') as f:
                f.write("==== Model Summary ====\n")
                f.write(summary_str)
                f.write("=" * 120 + "\n\n")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                 metrics: Dict[str, float] = None, verbose: bool = True):
        """
        Log epoch results to file and optionally print them.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss
            val_loss (float): Validation loss
            metrics (dict): Dictionary of metric names and values
            verbose (bool): Whether to print the results
        """
        if metrics:
            metric_str = ', '.join([f"{name}: {value:.2f}%" for name, value in metrics.items()])
            log_message = f"[epoch {epoch:02d}] train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | {metric_str}"
        else:
            log_message = f"[epoch {epoch:02d}] train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"

        if verbose:
            print("\033[38;5;44m" + log_message + "\033[0m")

        if self.enable_logging:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")

    def log_early_stopping(self, verbose: bool = True):
        """Log early stopping message."""
        stop_message = "🚨 early stopping triggered."
        
        if verbose:
            print("\033[38;5;196m" + stop_message + "\033[0m")

        if self.enable_logging:
            with open(self.log_file, 'a') as f:
                f.write(stop_message + "\n")