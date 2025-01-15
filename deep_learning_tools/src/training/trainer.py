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
import numpy as np
from ..utils.karpathy_verification import KarpathyVerification
import warnings
from ..utils.utils import PrintManager
from torch.functional import F
from tqdm import tqdm

class ModelTrainer:
    """A comprehensive trainer for PyTorch models with advanced features.

    This class provides a robust training framework with support for:
    - Mixed precision training (FP16)
    - Channels last memory format
    - Early stopping with customizable patience
    - Learning rate scheduling
    - Custom metrics tracking
    - Multiple logger backends (file, wandb, tensorboard)
    - Sweep training for hyperparameter optimization
    - Karpathy's training verification tests
    - Graceful interruption handling
    - Reproducible training with seed setting

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
        metrics_history (defaultdict): History of training metrics.
        batch_size (int): Size of training batches.
        use_half_precision (bool): Whether to use FP16 training.
        use_channels_last (bool): Whether to use channels last memory format.
        step_scheduler_batch (bool): Whether to step scheduler per batch.
        run_karpathy_checks (bool): Whether to run verification tests.
        sweep (bool): Whether running in sweep mode.
        seed (Optional[int]): Random seed for reproducibility.
        verbose (bool): Whether to print detailed output.
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
        save_metrics: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_delta: float = 1e-4,
        metrics: Optional[List[Callable[[torch.Tensor, torch.Tensor], float]]] = None,
        log_dir: str = "logs",
        logger_type: Optional[Literal["file", "wandb", "tensorboard"]] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        sweep: bool = False,
        seed: Optional[int] = None,
        strict_reproducibility: bool = False,
        run_karpathy_checks: bool = False,
        use_half_precision: bool = True,
        use_channels_last: bool = True,
        step_scheduler_batch: bool = False,
    ) -> None:
        """Initialize the ModelTrainer with specified configuration.

        Args:
            model (nn.Module): The neural network model to train.
            device (torch.device): The device to run the training on.
            loss_fn (nn.Module, optional): The loss function. Defaults to CrossEntropyLoss.
            optimizer (torch.optim.Optimizer, optional): The optimizer. Defaults to Adam with lr=3e-4.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            batch_size (int): Batch size for data loaders. Defaults to 32.
            verbose (bool): If True, prints training progress. Defaults to True.
            save_metrics (bool): If True, saves the metrics visualization. Defaults to True.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
            early_stopping_delta (float): Minimum change in monitored quantity to qualify as an improvement. Defaults to 1e-4.
            metrics (list of callables, optional): List of metric functions to evaluate.
            log_dir (str): Directory to save logs and model checkpoints. Defaults to "logs".
            logger_type (Optional[Literal["file", "wandb", "tensorboard"]]): Type of logger to use.
            wandb_project (Optional[str]): Name of the W&B project to log to.
            wandb_entity (Optional[str]): W&B username or team name.
            sweep (bool): If True, delegates training to W&B sweep. Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. If None, no seed is set.
            strict_reproducibility (bool): If True, enables strict reproducibility at the cost of performance. Defaults to False.
            run_karpathy_checks (bool): If True, enables Karpathy verification checks. Defaults to False.
            use_half_precision (bool): If True, uses FP16 (half precision) for training. Defaults to True.
            use_channels_last (bool): If True, uses channels last memory format. Defaults to True.
            step_scheduler_batch (bool): If True, steps the scheduler per batch. Defaults to False.

        Notes:
            - If loss_fn is None, defaults to CrossEntropyLoss
            - If optimizer is None, defaults to Adam with lr=3e-4
            - If metrics is None, defaults to accuracy for classification
            - Early stopping monitors validation loss
            - Seed affects both PyTorch and NumPy random states
            - Logger type determines how training progress is recorded
            - W&B parameters only used when logger_type is "wandb"
        """

        # initialize utility managers first
        self.print_manager = PrintManager()
        self.plotter = MetricsPlotter()

        ############# GPU/PERFORMANCE SETTINGS #############
        # create a more robust patched deletion method with warning handling
        def patched_dataloader_del(self):
            """
            Custom deletion method for DataLoader's multiprocessing iterator.

            This method attempts to shut down worker processes safely. 
            If an exception occurs during shutdown, it checks for 
            specific AssertionErrors and issues a warning instead of 
            failing silently, improving error handling and robustness 
            during DataLoader cleanup.
            """
            try:
                self._shutdown_workers()
            except Exception as e:
                if not isinstance(e, AssertionError) or "can only test a child process" not in str(e):
                    warnings.warn(f"Exception in DataLoader shutdown: {str(e)}")

        # apply the patch
        torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__del__ = patched_dataloader_del

        # enable cuDNN benchmarking for better performance
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.model = model
        self.device = device
        self.use_half_precision = use_half_precision and device.type == 'cuda'
        self.use_channels_last = use_channels_last and device.type == 'cuda'
        
        # uses channels last memory format for better performance
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # from float32 to float16 (half precision)
        if self.use_half_precision:
            self.model = self.model.half()
            # keep BatchNorm in float32 for stability
            for mod in self.model.modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.float()
            if verbose:
                self.print_manager.print_message("Using FP16 (half precision) training", msg_type="success")
        
        # move model to device after format conversion
        self.model = self.model.to(device)
        
        ############# TRAINING SETTINGS #############
        self.criterion = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=3e-4)
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.verbose = verbose
        self.save_metrics = save_metrics
        self.seed = seed
        self.strict_reproducibility = strict_reproducibility
        self.current_epoch = 0

        if seed is not None:
            self._set_random_seed(seed)

        self.metrics = metrics if metrics else [accuracy]
        self.metrics_names = [metric.__name__ for metric in self.metrics]
        
        self.logger_manager = LoggerManager(
            logger_type=logger_type, 
            log_dir=log_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity
        )

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            verbose=verbose,
            delta=early_stopping_delta
        )

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.metrics_history: defaultdict = defaultdict(list)

        self.sweep = sweep

        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

        self.run_karpathy_checks = run_karpathy_checks
        self.prediction_history = []
        
        # initialize verification results
        self.verification_results = None

        self.step_scheduler_batch = step_scheduler_batch

    def _handle_interrupt(self, signum, frame):
        self.print_manager.print_message("\nTraining interrupted. Cleaning up...", msg_type="warning")
        self.logger_manager.close()
        if hasattr(self.logger_manager.logger, 'cleanup'):
            self.logger_manager.logger.cleanup()

        if not os.path.exists('checkpoint.pt'):
            torch.save(self.model.state_dict(), 'interrupted_model.pt')
            self.print_manager.print_message("Model state saved as interrupted_model.pt.", msg_type="info")
        else:
            self.print_manager.print_message("Model state already saved as checkpoint.pt.", msg_type="info")
        raise KeyboardInterrupt("Training interrupted by user.")

    def validate_state(self) -> None:
        """Validates that essential components are initialized."""
        if self.model is None:
            raise ValueError("Model not initialized")
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized")
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized")
        
    def load_best_model(self, checkpoint_path=None) -> None:
        """Loads the best model weights saved by EarlyStopping."""
        if checkpoint_path is None:
            checkpoint_path = self.early_stopping.best_model_path
        if checkpoint_path:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )

    def setup_data_loaders(self, training_set: Dataset, val_set: Dataset) -> None:
        """Sets up the data loaders for training and validation."""
        # create a generator for dataloader if seed is set
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        # calculate optimal number of workers (leave one CPU core free)
        num_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 2
        
        self.train_loader = DataLoader(
            training_set, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator if self.seed is not None else None,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed) if self.seed is not None else None, 
            pin_memory=True,            # faster transfer from CPU to GPU
            persistent_workers=True,    # keep workers alive between epochs
            num_workers=num_workers,    # dynamically set based on available cores
            prefetch_factor=1           # reduced from 2 to 1 to prevent OOM
        )
        self.val_loader = DataLoader(
            val_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=num_workers,
            prefetch_factor=1           # reduced from 2 to 1 to prevent OOM
        )

    def train_epoch(self, epoch: int) -> float:
        """Trains the model for one epoch.
        
        Args:
            epoch (int): The current epoch number.
            
        Returns:
            float: The average training loss for the epoch.
            
        Notes:
            - Updates model weights using backpropagation
            - Handles data conversion (channels last, half precision)
            - Updates learning rate if using OneCycleLR or batch-wise scheduling
            - Logs metrics and updates progress bar
            - Calculates additional metrics on the last batch
        """
        self.model.train()
        batch_losses = []

        # update epoch number in dataset if it supports it
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)

        # create progress bar that will be cleared
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}',
            leave=False,  # don't leave the progress bar
            file=sys.stdout,  # use stdout to avoid logging to file
            dynamic_ncols=True  # adapt to terminal width
        )

        for batch_idx, (data, targets) in enumerate(progress_bar):
            # convert to channels last if enabled
            if self.use_channels_last and data.dim() == 4:
                data = data.to(memory_format=torch.channels_last)
            
            # convert to half precision if enabled
            if self.use_half_precision:
                data = data.half()
            
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # step scheduler per batch if configured
            if self.scheduler and self.step_scheduler_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    if self.scheduler._step_count < self.scheduler.total_steps:
                        self.scheduler.step()
                else:
                    self.scheduler.step()

            batch_losses.append(loss.item())

            # update progress bar with current loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'}, refresh=True)

            # log metrics for each batch if using wandb
            if self.logger_manager.logger_type == "wandb":
                wandb.log({"batch_loss": loss.item(), "epoch": epoch})

        progress_bar.close()

        # epoch total loss / number of batches = epoch average loss
        average_loss = sum(batch_losses) / len(self.train_loader)
        # save average loss for the epoch
        self.metrics_history['train_loss'].append(average_loss)
        # save all batch losses for the epoch
        self.metrics_history['train_batch_losses'].append(batch_losses)

        # calculate additional metrics on last batch
        outputs_last_batch = outputs
        targets_last_batch = targets
        for metric, name in zip(self.metrics, self.metrics_names):
            metric_value = metric(outputs_last_batch, targets_last_batch)
            self.metrics_history[f'train_{name}'].append(metric_value)

        # step scheduler per epoch if not stepping per batch
        if self.scheduler and not self.step_scheduler_batch:
            self.scheduler.step()

        return average_loss

    def evaluate(self, epoch: int, phase: str = 'val') -> float:
        """Evaluates the model on the validation dataset.
        
        Args:
            epoch (int): The current epoch number.
            phase (str, optional): The evaluation phase. Currently only supports 'val'. Defaults to 'val'.
            
        Returns:
            float: The average validation loss for the epoch.
            
        Raises:
            ValueError: If phase is not 'val'.
            
        Notes:
            - Sets model to evaluation mode
            - Handles data conversion (channels last, half precision)
            - Calculates loss and additional metrics
            - Updates progress bar with live metrics
            - Checks early stopping conditions
        """
        if phase == 'val':
            loader = self.val_loader
        else:
            raise ValueError("phase must be 'val'.")

        self.model.eval()
        batch_losses = []
        metrics_results: Dict[str, float] = {name: 0.0 for name in self.metrics_names}
        
        progress_bar = tqdm(
            loader,
            desc=f'Validate',
            leave=False,        # don't leave the progress bar
            file=sys.stdout,    # use stdout to avoid logging to file
            dynamic_ncols=True  # adapt to terminal width
        )
        
        with torch.no_grad():
            for data, targets in progress_bar:
                # convert to channels last if enabled
                if self.use_channels_last and data.dim() == 4:
                    data = data.to(memory_format=torch.channels_last)
                
                # convert to half precision if enabled
                if self.use_half_precision:
                    data = data.half()
                
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                batch_losses.append(loss.item())

                # update metrics for progress bar
                current_metrics = {name: metric(outputs, targets) for metric, name in zip(self.metrics, self.metrics_names)}
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', **{k: f'{v:.2f}%' for k, v in current_metrics.items()}}, refresh=True)

                for metric in self.metrics:
                    metrics_results[metric.__name__] += metric(outputs, targets)

        progress_bar.close()

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
            self.print_manager.print_message(progress_msg, msg_type="progress")

        self.early_stopping(average_loss, self.model)
        if self.early_stopping.early_stop:
            if self.verbose:
                self.print_manager.print_message("🚨 Early stopping triggered.", msg_type="error")
            return average_loss

        return average_loss

    def train(
        self,
        training_set: Dataset,
        val_set: Dataset,
        num_epochs: int = 20,
    ) -> nn.Module:
        """Trains the model on the provided datasets.

        Args:
            training_set (Dataset): The training dataset.
            val_set (Dataset): The validation dataset.
            num_epochs (int, optional): Number of epochs to train. Defaults to 20.

        Returns:
            nn.Module: The trained model (best version if early stopping is enabled).

        Notes:
            - Sets up data loaders and validates training state
            - Runs Karpathy verification tests if enabled
            - Handles sweep training if in sweep mode
            - Tracks prediction dynamics if Karpathy checks are enabled
            - Implements early stopping
            - Saves checkpoints and logs metrics
            - Handles training interruptions gracefully
            - Ensures proper cleanup of resources
        """
        try:
            self.setup_data_loaders(training_set, val_set)
            self.validate_state()
            
            # run Karpathy's verification tests if requested
            if self.run_karpathy_checks:
                verifier = KarpathyVerification(
                    model=self.model,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    device=self.device,
                    verbose=self.verbose,
                    use_half_precision=self.use_half_precision,
                    use_channels_last=self.use_channels_last
                )
                verification_results = verifier.run_all_verifications()
                if self.verbose:
                    self.print_manager.print_message("Karpathy verification tests completed.", msg_type="success")
                self.verification_results = verification_results

            if self.sweep:
                self.training_set = training_set
                self.val_set = val_set
                from ..sweeps.sweep import run_sweep
                run_sweep(self)
                return self.model

            self.logger_manager.on_training_start(self)
            
            # store fixed batch for prediction dynamics if Karpathy checks are enabled
            if self.run_karpathy_checks:
                try:
                    self._fixed_batch = next(iter(self.val_loader))
                    self._fixed_data, self._fixed_targets = [x.to(self.device) for x in self._fixed_batch]
                except StopIteration:
                    raise ValueError("Validation loader is empty, cannot run Karpathy checks.")

            # start training
            start_epoch = self.current_epoch + 1
            end_epoch = start_epoch + num_epochs

            for epoch in range(start_epoch, end_epoch):
                self.current_epoch = epoch
                self.metrics_history['epochs'].append(epoch)
                train_loss = self.train_epoch(epoch)
                val_loss = self.evaluate(epoch, phase='val')
                
                # track prediction dynamics if Karpathy checks are enabled
                if self.run_karpathy_checks:
                    pred_info = verifier.track_predictions(self._fixed_data, self._fixed_targets)
                    self.prediction_history.append(pred_info)

                if self.scheduler:
                    self.scheduler.step()

                self.logger_manager.on_epoch_end(self, epoch)

                if self.early_stopping.early_stop:
                    break

            self.plot()
            
            # plot prediction dynamics if Karpathy checks were enabled
            if self.run_karpathy_checks and self.prediction_history:
                verifier.plot_prediction_dynamics(self.prediction_history)

            self.load_best_model()

            # log the existing checkpoint as an artifact if logger type is wandb
            if self.logger_manager.logger_type == "wandb":
                artifact = wandb.Artifact('model_checkpoint', type='model')
                artifact.add_file(self.early_stopping.best_model_path)
                wandb.log_artifact(artifact)

            return self.model
            
        except KeyboardInterrupt:
            self.print_manager.print_message("Training was manually interrupted.", msg_type="warning")
        except Exception as e:
            self.print_manager.print_message(f"Error during training: {str(e)}", msg_type="error")
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
            val_losses=self.metrics_history['val_loss'],
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

    def _set_random_seed(self, seed: int) -> None:
        """
        Sets random seeds for reproducibility.

        Args:
            seed (int): The random seed to use.
        """
        # basic reproducibility settings
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        # for reproducible operations on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # strict reproducibility settings if enabled
        if self.strict_reproducibility:
            if self.verbose:
                self.print_manager.print_message(
                    "Strict reproducibility enabled. This may impact performance.", 
                    msg_type="warning"
                )
            # set environment variable for CUDA reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            # enable deterministic algorithms
            torch.use_deterministic_algorithms(True)
        
        if self.verbose:
            self.print_manager.print_message(
                f"Random seed set to {seed} for reproducibility.", 
                msg_type="success"
            )

    def set_learning_rate(self, new_lr: float) -> None:
        """update learning rate of the optimizer
        
        args:
            new_lr (float): new learning rate to use
        """

        self.early_stopping.counter = 0
        self.early_stopping.early_stop = False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        if self.verbose:
            self.print_manager.print_message(
                f"Learning rate updated to {new_lr}", 
                msg_type="success"
            )

    def evaluate_on_test(self, test_set: Dataset, checkpoint_path=None) -> Dict[str, Any]:
        """Evaluates the best saved model on the test dataset and prints the metrics.
        
        Args:
            test_set (Dataset): The test dataset.
            checkpoint_path (str, optional): Path to the checkpoint file.
        
        Returns:
            Dict containing:
                - predictions: model predictions
                - true_labels: ground truth labels
                - outputs: model outputs (logits)
                - probabilities: softmax probabilities
                - test_loader: dataloader used for evaluation
        """
        self.print_manager.print_message(
            "Loading Best Model for Test Evaluation", 
            msg_type="success", 
            bold=False
        )
        self.load_best_model(checkpoint_path=checkpoint_path)
        
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count() if os.cpu_count() else 2,
            prefetch_factor=2
        )
        
        self.model.eval()
        batch_losses = []
        metrics_results: Dict[str, float] = {name: 0.0 for name in self.metrics_names}
        
        all_outputs = []
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                if self.use_channels_last and data.dim() == 4:
                    data = data.to(memory_format=torch.channels_last)
                
                if self.use_half_precision:
                    data = data.half()
                
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                loss = self.criterion(outputs, targets)
                batch_losses.append(loss.item())
                
                for metric in self.metrics:
                    metric_value = metric(outputs, targets)
                    metrics_results[metric.__name__] += metric_value
                
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        average_loss = sum(batch_losses) / len(test_loader)
        for metric_name in metrics_results:
            metrics_results[metric_name] /= len(test_loader)
        
        metrics_str = ', '.join([f"{name}: {value:.2f}" for name, value in metrics_results.items()])
        self.print_manager.print_message(
            f"Test Loss: {average_loss:.4f} | {metrics_str}", 
            msg_type="info"
        )
        
        return {
            'predictions': np.array(all_preds),
            'true_labels': np.array(all_labels),
            'outputs': np.array(all_outputs),
            'probabilities': np.array(all_probs),
            'test_loader': test_loader
        }