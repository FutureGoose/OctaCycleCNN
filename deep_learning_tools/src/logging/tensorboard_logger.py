from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from .base import BaseLogger
import torch
import os

class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard.

    This logger writes logs to a specified directory for visualization in TensorBoard.

    To run TensorBoard, use the following command in your terminal:
    ```
    tensorboard --logdir=logs/tensorboard
    ```
    Then, open your web browser and navigate to:
    ```
    http://localhost:6006/
    ```

    Attributes:
        writer (SummaryWriter): The TensorBoard writer instance.
    """

    def __init__(self, run_id: str, log_dir: str = "logs"):
        """Initialize TensorBoard writer."""
        # organize logs under tensorboard/run_id subdirectory
        full_log_dir = os.path.join(log_dir, "tensorboard", run_id)
        self.writer = SummaryWriter(log_dir=full_log_dir)

    def _sanitize_hparams(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert hyperparameters to types that TensorBoard can handle.
        TensorBoard only accepts: int, float, str, bool, or torch.Tensor
        """
        sanitized = {}
        for key, value in hparams.items():
            if value is None:
                sanitized[key] = 'None'
            elif isinstance(value, (list, tuple)):
                sanitized[key] = ','.join(str(v) for v in value)
            elif isinstance(value, (int, float, str, bool, torch.Tensor)):
                sanitized[key] = value
            else:
                # convert any other types to string representation
                sanitized[key] = str(value)
        return sanitized

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        sanitized_hparams = self._sanitize_hparams(hyperparameters)
        self.writer.add_hparams(sanitized_hparams, {})

    def log_model_summary(self, summary_str: str) -> None:
        self.writer.add_text("Model Summary", summary_str)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: Dict[str, float]) -> None:
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/validation", val_loss, epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

    def log_early_stopping(self) -> None:
        self.writer.add_text("Training Status", "Early stopping triggered")

    def save_figure(self, figure: plt.Figure, filename: str) -> None:
        self.writer.add_figure(filename, figure)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()