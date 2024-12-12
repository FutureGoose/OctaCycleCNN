import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Callable
import matplotlib.pyplot as plt
from early_stopping import EarlyStopping

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
        test_loader (DataLoader): DataLoader for testing data.
        metrics (dict): Dictionary to store training metrics.
        early_stopping (EarlyStopping, optional): Early stopping handler.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        verbose: bool = True,
        early_stopping_patience: int = 5,
        metrics: Optional[List[Callable]] = None,
    ):
        """
        Initializes the ModelTrainer.

        Args:
            model (nn.Module): The neural network model to train.
            device (torch.device): The device to run the training on.
            loss_fn (nn.Module, optional): The loss function. Defaults to CrossEntropyLoss.
            optimizer (torch.optim.Optimizer, optional): The optimizer. Defaults to Adam.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            batch_size (int): Batch size for data loaders.
            verbose (bool): If True, prints training progress.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
            metrics (List[Callable], optional): List of metric functions to evaluate.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.verbose = verbose
        self.metrics = metrics if metrics else [self.accuracy]
        self.metrics_names = [metric.__name__ for metric in self.metrics]

        self.train_loader = None
        self.test_loader = None

        self.metrics_history = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        for metric_name in self.metrics_names:
            self.metrics_history[f'train_{metric_name}'] = []
            self.metrics_history[f'test_{metric_name}'] = []

        self.early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=verbose)

    def setup_data_loaders(self, training_set: Dataset, test_set: Dataset):
        """
        Sets up the data loaders for training and testing.

        Args:
            training_set (Dataset): The training dataset.
            test_set (Dataset): The testing dataset.
        """
        self.train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

    def train_epoch(self, epoch: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(self.train_loader)
        self.metrics_history['train_loss'].append(average_loss)

        # calculate additional metrics
        outputs_last_batch = outputs  # storing the last batch's outputs and targets
        targets_last_batch = targets
        for metric, name in zip(self.metrics, self.metrics_names):
            metric_value = metric(outputs_last_batch, targets_last_batch)
            self.metrics_history[f'train_{name}'].append(metric_value)

        return average_loss

    def evaluate(self, epoch: int, phase: str = 'test') -> float:
        """
        Evaluates the model on the test dataset.

        Args:
            epoch (int): The current epoch number.
            phase (str): Phase of evaluation ('test' or 'validation').

        Returns:
            float: The average loss for the phase.
        """
        if phase == 'test':
            loader = self.test_loader
        else:
            raise ValueError("Phase must be 'test'.")

        self.model.eval()
        running_loss = 0.0
        metrics_results = {name: 0.0 for name in self.metrics_names}

        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                for metric in self.metrics:
                    metrics_results[metric.__name__] += metric(outputs, targets)

        average_loss = running_loss / len(loader)
        self.metrics_history[f'{phase}_loss'].append(average_loss)

        for name in self.metrics_names:
            avg_metric = metrics_results[name] / len(loader)
            self.metrics_history[f'{phase}_{name}'].append(avg_metric)

        if self.verbose:
            if self.metrics_names:
                metrics_str = ', '.join([f"{name}: {self.metrics_history[f'{phase}_{name}'][-1]:.2f}%" for name in self.metrics_names])
                print(f'[epoch {epoch}] train loss: {self.metrics_history["train_loss"][-1]:.4f}, '
                      f'test loss: {average_loss:.4f}, '
                      f'{metrics_str}')
            else:
                print(f'[epoch {epoch}] train loss: {self.metrics_history["train_loss"][-1]:.4f}, '
                      f'test loss: {average_loss:.4f}')

        # early Stopping
        self.early_stopping(average_loss, self.model)
        if self.early_stopping.early_stop:
            if self.verbose:
                print("Early stopping triggered.")
            return average_loss

        return average_loss

    def train(
        self,
        training_set: Dataset,
        test_set: Dataset,
        num_epochs: int = 50,
        scheduler_step: Optional[int] = None,
    ) -> nn.Module:
        """
        Trains the model.

        Args:
            training_set (Dataset): The training dataset.
            test_set (Dataset): The testing dataset.
            num_epochs (int): Number of epochs to train.
            scheduler_step (int, optional): Step size for the scheduler.

        Returns:
            nn.Module: The trained model.
        """
        self.setup_data_loaders(training_set, test_set)

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate(epoch, phase='test')

            if self.scheduler:
                self.scheduler.step()

            self.metrics_history['epochs'].append(epoch)

            if self.early_stopping.early_stop:
                break

        self.plot_metrics()
        self.model.load_state_dict(torch.load(self.early_stopping.best_model_path))
        return self.model

    def plot_metrics(self):
        """
        Plots the training and testing metrics.
        """
        epochs = self.metrics_history['epochs']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # plot Losses
        axes[0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        axes[0].plot(epochs, self.metrics_history['test_loss'], label='Test Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Test Losses')
        axes[0].legend()
        axes[0].grid(True)

        # plot Metrics
        for metric in self.metrics_names:
            axes[1].plot(epochs, self.metrics_history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
            axes[1].plot(epochs, self.metrics_history[f'test_{metric}'], label=f'Test {metric.capitalize()}')

        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Training and Test Metrics')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes the accuracy metric.

        Args:
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The ground truth labels.

        Returns:
            float: Accuracy percentage.
        """
        _, preds = torch.max(outputs, dim=1)
        return (torch.sum(preds == targets).item() / targets.size(0)) * 100