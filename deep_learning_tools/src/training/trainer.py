import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Callable
import matplotlib.pyplot as plt
from ..visualization import MetricsPlotter
from .early_stopping import EarlyStopping
from ..utils import TrainingLogger
from torchinfo import summary
import os

class ModelTrainer:
    """
    a flexible and intuitive trainer for pytorch models.

    attributes:
        model (nn.Module): the neural network model to train.
        device (torch.device): the device to run the training on.
        criterion (nn.Module): the loss function.
        optimizer (torch.optim.Optimizer): the optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): learning rate scheduler.
        train_loader (DataLoader): dataloader for training data.
        val_loader (DataLoader): dataloader for validation data.
        metrics (list): list of metric functions to evaluate.
        early_stopping (EarlyStopping, optional): early stopping handler.
        hyperparameters (dict): dictionary to store hyperparameters.
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
        verbose_details: bool = True,
        enable_logging: bool = True,
        save_metrics: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_delta: float = 1e-4,
        metrics: Optional[List[Callable]] = None,
        log_dir: str = "logs"
    ):
        """
        initializes the modeltrainer.

        args:
            model (nn.Module): the neural network model to train.
            device (torch.device): the device to run the training on.
            loss_fn (nn.Module, optional): the loss function. defaults to crossentropyloss.
            optimizer (torch.optim.Optimizer, optional): the optimizer. defaults to adam.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): learning rate scheduler.
            batch_size (int): batch size for data loaders.
            verbose (bool): if true, prints training progress.
            verbose_details (bool): if true, prints hyperparameters and model summary after training.
            enable_logging (bool): if true, enables logging to a file.
            save_metrics (bool): if true, saves the metrics visualization.
            early_stopping_patience (int): number of epochs with no improvement after which training will be stopped.
            early_stopping_delta (float): minimum change in the monitored quantity to qualify as an improvement.
            metrics (list of callables, optional): list of metric functions to evaluate.
            log_dir (str): directory to save logs and model checkpoints.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.verbose = verbose
        self.verbose_details = verbose_details 
        self.enable_logging = enable_logging   
        self.save_metrics = save_metrics       
        self.metrics = metrics if metrics else [self.accuracy]
        self.metrics_names = [metric.__name__ for metric in self.metrics]

        # initialize logger
        self.logger = TrainingLogger(log_dir=log_dir, enable_logging=enable_logging)

        # initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            verbose=verbose,
            delta=early_stopping_delta
        )

        # initialize plotter
        self.plotter = MetricsPlotter()

        self.train_loader = None
        self.val_loader = None

        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'train_batch_losses': [],
            'val_batch_losses': []
        }
        for metric_name in self.metrics_names:
            self.metrics_history[f'train_{metric_name}'] = []
            self.metrics_history[f'val_{metric_name}'] = []

        # hyperparameters dictionary
        self.hyperparameters = {
            'batch_size': self.batch_size,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0),
            'scheduler_step_size': self.scheduler.step_size if self.scheduler else None,
            'scheduler_gamma': self.scheduler.gamma if self.scheduler else None,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_delta': early_stopping_delta,
            'metrics': self.metrics_names,
            'optimizer': type(self.optimizer).__name__,
            'scheduler': type(self.scheduler).__name__ if self.scheduler else None
        }

    def log_hyperparameters(self):
        """logs the hyperparameters."""
        self.logger.log_hyperparameters(
            hyperparameters=self.hyperparameters,
            verbose=self.verbose_details
        )

    def log_model_summary(self, input_size):
        """logs the model summary."""
        try:
            summary_str = str(summary(
                self.model, 
                input_size=input_size,
                verbose=0,
                col_width=16,
                col_names=["output_size", "num_params", "kernel_size", "mult_adds", "trainable"],
                row_settings=["var_names"]
            ))
            self.logger.log_model_summary(summary_str, verbose=self.verbose_details)
        except Exception as e:
            if self.verbose:
                print("\033[38;5;196m" + f"failed to generate model summary: {e}" + "\033[0m")

    def setup_data_loaders(self, training_set: Dataset, val_set: Dataset):
        """sets up the dataloaders for training and validation."""
        self.train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

    def train_epoch(self, epoch: int) -> float:
        """trains the model for one epoch."""
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

        return average_loss

    def evaluate(self, epoch: int, phase: str = 'val') -> float:
        """evaluates the model on the validation dataset."""
        if phase == 'val':
            loader = self.val_loader
        else:
            raise ValueError("phase must be 'val'.")

        self.model.eval()
        batch_losses = []
        metrics_results = {name: 0.0 for name in self.metrics_names}
        
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

        # log epoch results using TrainingLogger
        metrics_dict = {
            name: self.metrics_history[f'{phase}_{name}'][-1] 
            for name in self.metrics_names
        }
        
        self.logger.log_epoch(
            epoch=epoch,
            train_loss=self.metrics_history['train_loss'][-1],
            val_loss=average_loss,
            metrics=metrics_dict if self.metrics_names else None,
            verbose=self.verbose
        )

        # early stopping
        self.early_stopping(average_loss, self.model)
        if self.early_stopping.early_stop:
            self.logger.log_early_stopping(verbose=self.verbose)
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
        trains the model.

        args:
            training_set (Dataset): the training dataset.
            val_set (Dataset): the validation dataset.
            num_epochs (int): number of epochs to train.
            scheduler_step (int, optional): step size for the scheduler.

        returns:
            nn.Module: the trained model.
        """
        self.setup_data_loaders(training_set, val_set)

        # infer input size from the first batch of training data
        try:
            sample_data, _ = next(iter(self.train_loader))
            input_size = tuple(sample_data.size())
        except StopIteration:
            if self.verbose:
                print("\033[38;5;196mtraining loader is empty. cannot infer input size for model summary.\033[0m")
            input_size = None
        except Exception as e:
            if self.verbose:
                print(f"\033[38;5;196mfailed to infer input size for model summary: {e}\033[0m")
            input_size = None

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch, phase='val')

            if self.scheduler:
                self.scheduler.step()

            self.metrics_history['epochs'].append(epoch)

            if self.early_stopping.early_stop:
                break

        # log hyperparameters and model summary
        if input_size and self.verbose_details:
            self.log_hyperparameters()
            self.log_model_summary(input_size)

        # plot metrics
        self.plot()

        # load the best model weights if logging was enabled
        if self.early_stopping.best_model_path:
            self.model.load_state_dict(torch.load(self.early_stopping.best_model_path))

        return self.model

    def plot(self):
        """plots the training and validation metrics with batch variation bands."""
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

        if self.save_metrics and self.logger.log_dir:
            plt.savefig(os.path.join(self.logger.log_dir, "metrics.png"))

        plt.show()

    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        computes the accuracy metric.

        args:
            outputs (torch.Tensor): the model outputs.
            targets (torch.Tensor): the ground truth labels.

        returns:
            float: accuracy percentage.
        """
        _, preds = torch.max(outputs, dim=1)
        return (torch.sum(preds == targets).item() / targets.size(0)) * 100