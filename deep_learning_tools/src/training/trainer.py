import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Callable, Dict, Any, Literal
from collections import defaultdict
import matplotlib.pyplot as plt
from ..visualization import MetricsPlotter
from .early_stopping import EarlyStopping
from ..logging import LoggerManager
from .metrics import accuracy
import signal
import os
import sys
import wandb
from ..utils import initialize_wandb_sweep, log_hyperparameters, handle_wandb_error, handle_sweep_configuration

class ModelTrainer:
    """
    A flexible and intuitive trainer for PyTorch models.

    Attributes:
        model (nn.Module): The neural network model to train.
        device (torch.device): The device to run the training on.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        metrics (list): List of metric functions to evaluate.
        early_stopping (EarlyStopping): Early stopping handler.
        logger_manager (LoggerManager): Manager for logging operations.
        plotter (MetricsPlotter): Plotter for metrics visualization.
        metrics_history (defaultdict): History of training metrics with the following structure:
            - 'train_loss': List[float] - Per-epoch training loss
            - 'val_loss': List[float] - Per-epoch validation loss
            - 'epochs': List[int] - Epoch numbers
            - 'train_batch_losses': List[List[float]] - Per-batch training losses for each epoch
            - 'val_batch_losses': List[List[float]] - Per-batch validation losses for each epoch
            - f'train_{metric_name}': List[float] - Training metrics for custom metrics
            - f'val_{metric_name}': List[float] - Validation metrics for custom metrics
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        sweep: bool = False,           # NEW
        verbose: bool = True,
        save_metrics: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_delta: float = 1e-4,
        metrics: Optional[List[Callable[[torch.Tensor, torch.Tensor], float]]] = None,
        log_dir: str = "logs",
        logger_type: Optional[Literal["file", "wandb", "tensorboard"]] = "file",
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            model (nn.Module): The neural network model to train.
            device (torch.device): The device to run the training on.
            loss_fn (nn.Module, optional): The loss function. Defaults to CrossEntropyLoss.
            optimizer (torch.optim.Optimizer, optional): The optimizer. Defaults to Adam.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            batch_size (int): Batch size for data loaders.
            sweep (bool): If True, uses W&B configurations for training.  # NEW
            verbose (bool): If True, prints training progress.
            save_metrics (bool): If True, saves the metrics visualization.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
            early_stopping_delta (float): Minimum change in monitored quantity to qualify as an improvement.
            metrics (list of callables, optional): List of metric functions to evaluate.
            log_dir (str): Directory to save logs and model checkpoints.
            logger_type (Optional[Literal["file", "wandb", "tensorboard"]]): Type of logger to use.
            wandb_project (Optional[str]): Name of the W&B project to log to.
            wandb_entity (Optional[str]): W&B username or team name.
        """

        if sweep:
            sweep_config = handle_sweep_configuration(
                model=self.model,
                optimizer=self.optimizer,
                batch_size=self.batch_size,
                early_stopping_patience=early_stopping_patience,
                early_stopping_delta=early_stopping_delta
            )

            self.batch_size = sweep_config["batch_size"]
            self.optimizer = sweep_config["optimizer"]
            early_stopping_patience = sweep_config["early_stopping_patience"]
            early_stopping_delta = sweep_config["early_stopping_delta"]

            log_hyperparameters(
                batch_size=self.batch_size,
                optimizer=self.optimizer,
                early_stopping_patience=early_stopping_patience,
                early_stopping_delta=early_stopping_delta
            )

        self.model = model.to(device)
        self.device = device
        self.criterion = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=1e-4)

        self.scheduler = scheduler
        self.batch_size = batch_size

        self.verbose = verbose
        self.save_metrics = save_metrics       
        self.metrics = metrics if metrics else [accuracy]
        self.metrics_names = [metric.__name__ for metric in self.metrics]

        self.logger_manager = LoggerManager(logger_type=logger_type, 
                                            log_dir=log_dir,
                                            wandb_project=wandb_project,
                                            wandb_entity=wandb_entity)
        self.plotter = MetricsPlotter()

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            verbose=verbose,
            delta=early_stopping_delta
        )

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.metrics_history: defaultdict = defaultdict(list)

        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print("\nTraining interrupted. Cleaning up...")
        self.interrupted = True
        self.logger_manager.close()
        if hasattr(self.logger_manager.logger, 'cleanup'):
            self.logger_manager.logger.cleanup()  # call cleanup if available

        if not os.path.exists('checkpoint.pt'):
            torch.save(self.model.state_dict(), 'interrupted_model.pt')
            print("Model state saved as interrupted_model.pt.")
        else:
            print("Model state already saved as checkpoint.pt.")
        raise KeyboardInterrupt("Training interrupted by user.")

    def validate_state(self) -> None:
        """Validates that essential components are initialized."""
        if self.model is None:
            raise ValueError("Model not initialized")
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized")
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized")
        
    def load_best_model(self) -> None:
        """Loads the best model weights saved by EarlyStopping."""
        if self.early_stopping.best_model_path:
            self.model.load_state_dict(torch.load(self.early_stopping.best_model_path, weights_only=False))

    def setup_data_loaders(self, training_set: Dataset, val_set: Dataset) -> None:
        """Sets up the data loaders for training and validation."""
        self.train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

    def train_epoch(self, epoch: int) -> float:
        """Trains the model for one epoch."""
        self.model.train()
        batch_losses = []

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())

            # log metrics for each batch if logger type is wandb
            if self.logger_manager.logger_type == "wandb":
                wandb.log({"batch_loss": loss.item(), "epoch": epoch})

        # epoch total loss / number of batches = epoch average loss
        average_loss = sum(batch_losses) / len(self.train_loader)
        # saving average loss for the epoch
        self.metrics_history['train_loss'].append(average_loss)
        # saving all batch losses for the epoch
        self.metrics_history['train_batch_losses'].append(batch_losses)

        # calculate additional metrics
        outputs_last_batch = outputs
        targets_last_batch = targets
        for metric, name in zip(self.metrics, self.metrics_names):
            metric_value = metric(outputs_last_batch, targets_last_batch)
            self.metrics_history[f'train_{name}'].append(metric_value)

        # log epoch metrics if logger type is wandb
        if self.logger_manager.logger_type == "wandb":
            wandb.log({"epoch_loss": average_loss, "epoch": epoch})

        return average_loss

    def evaluate(self, epoch: int, phase: str = 'val') -> float:
        """Evaluates the model on the validation dataset."""
        if phase == 'val':
            loader = self.val_loader
        else:
            raise ValueError("phase must be 'val'.")

        self.model.eval()
        batch_losses = []
        metrics_results: Dict[str, float] = {name: 0.0 for name in self.metrics_names}
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                batch_losses.append(loss.item())
                for metric in self.metrics:
                    metrics_results[metric.__name__] += metric(outputs, targets)

        average_loss = sum(batch_losses) / len(loader)
        self.metrics_history[f'{phase}_loss'].append(average_loss)
        self.metrics_history[f'{phase}_batch_losses'].append(batch_losses)

        for name in self.metrics_names:
            avg_metric = metrics_results[name] / len(loader)
            self.metrics_history[f'{phase}_{name}'].append(avg_metric)

        if self.verbose:
            metrics_str = ', '.join([f"{name}: {self.metrics_history[f'{phase}_{name}'][-1]:.2f}%" 
                                for name in self.metrics_names])
            progress_msg = f"[epoch {epoch:02d}] train loss: {self.metrics_history['train_loss'][-1]:.4f} | "\
                        f"val loss: {average_loss:.4f} | {metrics_str}"
            print("\033[38;5;44m" + progress_msg + "\033[0m")

        # early stopping
        self.early_stopping(average_loss, self.model)
        if self.early_stopping.early_stop:
            if self.verbose:
                print("\033[38;5;196m" + "🚨 Early stopping triggered." + "\033[0m")
            return average_loss

        return average_loss

    def train(
        self,
        training_set: Dataset,
        val_set: Dataset,
        num_epochs: int = 50,
        scheduler_step: Optional[int] = None,
    ) -> nn.Module:
        """
        Trains the model.

        Args:
            training_set (Dataset): The training dataset.
            val_set (Dataset): The validation dataset.
            num_epochs (int): Number of epochs to train.
            scheduler_step (int, optional): Step size for the scheduler.

        Returns:
            nn.Module: The trained model.
        """
        try:
            self.setup_data_loaders(training_set, val_set)
            self.validate_state()

            self.logger_manager.on_training_start(self)

            for epoch in range(1, num_epochs + 1):
                self.metrics_history['epochs'].append(epoch)
                train_loss = self.train_epoch(epoch)
                val_loss = self.evaluate(epoch, phase='val')

                if self.scheduler:
                    self.scheduler.step()

                self.logger_manager.on_epoch_end(self, epoch)

                if self.early_stopping.early_stop:
                    break

            # plot metrics
            self.plot()

            # load the best model parameters (weights, biases, batchnorm, etc.)
            self.load_best_model()

            # log the existing checkpoint as an artifact if logger type is wandb
            if self.logger_manager.logger_type == "wandb":
                artifact = wandb.Artifact('model_checkpoint', type='model')
                artifact.add_file(self.early_stopping.best_model_path)
                wandb.log_artifact(artifact)

            return self.model
        except KeyboardInterrupt:
            print("Training was manually interrupted.")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            self._handle_interrupt(None, None)
        finally:
            # ensure logger is closed even if exception is raised
            self.logger_manager.close()

    def plot(self) -> None:
        """Plots the training and validation metrics with batch variation bands."""
        epochs = self.metrics_history['epochs']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # prepare batch variation data
        batch_variation = (
            [min(batch_losses) for batch_losses in self.metrics_history['train_batch_losses']],
            [max(batch_losses) for batch_losses in self.metrics_history['train_batch_losses']]
        )

        # use MetricsPlotter for losses
        self.plotter.plot_losses(
            ax=axes[0],
            epochs=epochs,
            train_losses=self.metrics_history['train_loss'],
            test_losses=self.metrics_history['val_loss'],
            batch_variation=batch_variation
        )

        # prepare metrics dictionary for the plotter
        metrics_dict = {}
        for metric in self.metrics_names:
            metrics_dict[f'train_{metric}'] = self.metrics_history[f'train_{metric}']
            metrics_dict[f'val_{metric}'] = self.metrics_history[f'val_{metric}']

        # use MetricsPlotter for other metrics
        self.plotter.plot_metrics(
            ax=axes[1],
            epochs=epochs,
            metrics_dict=metrics_dict
        )

        plt.tight_layout()

        if self.save_metrics:
            self.logger_manager.save_figure(plt.gcf(), "metrics.png")

        plt.show()