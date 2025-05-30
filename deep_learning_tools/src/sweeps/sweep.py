import wandb
from typing import TYPE_CHECKING, Dict, Any
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import traceback
from contextlib import contextmanager
import torch
from ..training.early_stopping import EarlyStopping
from tqdm import tqdm
import sys

if TYPE_CHECKING:
    from src.training.trainer import ModelTrainer

from .sweep_config import sweep_config, get_count


@contextmanager
def wandb_run(trainer: "ModelTrainer", config: Dict[str, Any]):
    """Context manager for wandb runs."""
    run = wandb.init(
        config=config,
        reinit=True,
        dir=trainer.logger_manager.logger.log_dir  # keep individual runs in logs/wandb
    )
    try:
        yield run
    except KeyboardInterrupt:
        print("Sweep interrupted. Finalizing run...")
        wandb.finish()
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()


def build_optimizer(network: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float):
    """Utility function to build the optimizer."""
    if optimizer_name.lower() == 'adam':
        return Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def run_sweep(trainer: "ModelTrainer"):
    """Initializes and runs the W&B sweep using the provided trainer."""
    if trainer.logger_manager.logger_type != "wandb":
        raise ValueError("LoggerManager must be set to 'wandb' for running sweeps")

    # ensure we're starting with a clean state
    if wandb.run is not None:
        wandb.finish()

    # configure wandb settings globally
    wandb.setup(settings=wandb.Settings(
        _disable_stats=False,  # enable stats for better tracking
        _disable_meta=False,   # enable meta for better reproducibility
        disable_code=False,    # track code for versioning
        disable_git=False,     # track git info for versioning
    ))

    try:
        # create sweep
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=trainer.logger_manager.wandb_project,
            entity=trainer.logger_manager.wandb_entity
        )
        
        print(f"sweep url: https://wandb.ai/{trainer.logger_manager.wandb_entity}/{trainer.logger_manager.wandb_project}/sweeps/{sweep_id}")

        def sweep_train():
            """Function to be executed for each sweep run."""
            try:
                with wandb_run(trainer, wandb.config) as run:
                    if run is None:
                        print("Failed to initialize W&B run.")
                        return

                    # access config after wandb.init()
                    config = wandb.config
                    print(f"\nrun {run.name} - batch_size: {config.batch_size}, lr: {config.learning_rate:.8f}, optimizer: {config.optimizer}, weight_decay: {config.weight_decay:.8f}")

                    # execute the training function for this trial
                    train_function(trainer, config)

            except Exception as e:
                print(f"Error during sweep run: {str(e)}")
                traceback.print_exc()

        print("\nStarting sweep...")

        try:
            count = get_count()
            print(f"Running {count} trials...")
            
            wandb.agent(
                sweep_id, 
                function=sweep_train,
                count=count
            )
            
            print("\nSweep completed successfully!")
            print(f"View results at: https://wandb.ai/{trainer.logger_manager.wandb_entity}/{trainer.logger_manager.wandb_project}/sweeps/{sweep_id}")

        except Exception as e:
            print(f"Error during sweep agent execution: {str(e)}")
            traceback.print_exc()

    except Exception as e:
        print(f"Failed to initialize sweep: {str(e)}")
        traceback.print_exc()


def sweep_train_epoch(trainer: "ModelTrainer", epoch: int) -> float:
    """Custom training function for a single epoch during a sweep."""
    trainer.model.train()
    batch_losses = []

    progress_bar = tqdm(
        trainer.train_loader,
        desc=f'Epoch {epoch}',
        leave=False,
        file=sys.stdout,   # stdout to avoid logging to file
        dynamic_ncols=True # adapt to terminal width
    )

    for batch_idx, (data, targets) in enumerate(progress_bar):
        # move data to device first
        data, targets = data.to(trainer.device), targets.to(trainer.device)
        
        # then convert to channels last if enabled
        if trainer.use_channels_last and data.dim() == 4:
            data = data.to(memory_format=torch.channels_last)
        
        # finally convert to half precision if enabled
        if trainer.use_half_precision:
            data = data.half()

        trainer.optimizer.zero_grad()
        outputs = trainer.model(data)
        loss = trainer.criterion(outputs, targets)
        loss.backward()
        trainer.optimizer.step()

        batch_losses.append(loss.item())
        
        # update progress bar with current loss
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'}, refresh=True)

        # log metrics for each batch if logger type is wandb
        if trainer.logger_manager.logger_type == "wandb":
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

    progress_bar.close()

    average_loss = sum(batch_losses) / len(trainer.train_loader)
    trainer.metrics_history['train_loss'].append(average_loss)

    # calculate additional metrics
    outputs_last_batch = outputs
    targets_last_batch = targets
    for metric, name in zip(trainer.metrics, trainer.metrics_names):
        metric_value = metric(outputs_last_batch, targets_last_batch)
        trainer.metrics_history[f'train_{name}'].append(metric_value)

    # log epoch metrics if logger type is wandb
    if trainer.logger_manager.logger_type == "wandb":
        metrics_dict = {
            "train_loss": average_loss,
            "epoch": epoch
        }
        # add all additional metrics
        for name in trainer.metrics_names:
            metrics_dict[f"train_{name}"] = trainer.metrics_history[f'train_{name}'][-1]
        wandb.log(metrics_dict)

    return average_loss


def sweep_evaluate(trainer: "ModelTrainer", epoch: int, phase: str = 'val') -> float:
    """Custom evaluation function for the validation phase during a sweep."""
    if phase == 'val':
        loader = trainer.val_loader
    else:
        raise ValueError("phase must be 'val'.")

    trainer.model.eval()
    batch_losses = []
    metrics_results: Dict[str, float] = {name: 0.0 for name in trainer.metrics_names}

    progress_bar = tqdm(
        loader,
        desc=f'Validate',
        leave=False,
        file=sys.stdout,     # stdout to avoid logging to file
        dynamic_ncols=True   # adapt to terminal width
    )

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(progress_bar):
            # move data to device first
            data, targets = data.to(trainer.device), targets.to(trainer.device)
            
            # then convert to channels last if enabled
            if trainer.use_channels_last and data.dim() == 4:
                data = data.to(memory_format=torch.channels_last)
            
            # finally convert to half precision if enabled
            if trainer.use_half_precision:
                data = data.half()
            
            outputs = trainer.model(data)
            loss = trainer.criterion(outputs, targets)
            batch_losses.append(loss.item())
            
            # update metrics for progress bar
            current_metrics = {name: metric(outputs, targets) for metric, name in zip(trainer.metrics, trainer.metrics_names)}
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', **{k: f'{v:.2f}%' for k, v in current_metrics.items()}}, refresh=True)
            
            for metric in trainer.metrics:
                metrics_results[metric.__name__] += metric(outputs, targets)

    progress_bar.close()

    average_loss = sum(batch_losses) / len(loader)
    trainer.metrics_history[f'{phase}_loss'].append(average_loss)

    for name in trainer.metrics_names:
        avg_metric = metrics_results[name] / len(loader)
        trainer.metrics_history[f'{phase}_{name}'].append(avg_metric)

    if trainer.logger_manager.logger_type == "wandb":
        # log all metrics together with epoch
        metrics_dict = {
            "val_loss": average_loss,
            "epoch": epoch
        }
        # add all additional metrics
        for name in trainer.metrics_names:
            metrics_dict[f"val_{name}"] = trainer.metrics_history[f'{phase}_{name}'][-1]
        wandb.log(metrics_dict)

    if trainer.verbose:
        metrics_str = ', '.join([f"{name}: {trainer.metrics_history[f'{phase}_{name}'][-1]:.2f}%" 
                            for name in trainer.metrics_names])
        progress_msg = f"[epoch {epoch:02d}] train loss: {trainer.metrics_history['train_loss'][-1]:.4f} | "\
                    f"val loss: {average_loss:.4f} | {metrics_str}"
        print("\033[38;5;44m" + progress_msg + "\033[0m")

    # early stopping
    trainer.early_stopping(average_loss, trainer.model)
    if trainer.early_stopping.early_stop:
        if trainer.verbose:
            print("\033[38;5;196m" + "🚨 Early stopping triggered." + "\033[0m")
        return average_loss

    return average_loss


def train_function(trainer: "ModelTrainer", config: Dict[str, Any]):
    """Training function that will be called by the sweep."""
    try:
        # update trainer with config parameters
        trainer.batch_size = config["batch_size"]
        trainer.optimizer = build_optimizer(
            network=trainer.model,
            optimizer_name=config["optimizer"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        if trainer.scheduler:
            trainer.scheduler = StepLR(trainer.optimizer, step_size=1, gamma=0.75)

        # set up data loaders
        trainer.setup_data_loaders(training_set=trainer.training_set, val_set=trainer.val_set)

        # reset EarlyStopping to ensure independence between runs
        trainer.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 5),
            delta=config.get("early_stopping_delta", 1e-4),
            verbose=trainer.verbose,
            path=trainer.early_stopping.path  # retain the checkpoint path
        )
        # if trainer.verbose:
            # print(f"Initialized EarlyStopping with patience={config.get('early_stopping_patience', 5)} and delta={config.get('early_stopping_delta', 1e-4)}")

        # train for the specified number of epochs
        num_epochs = config.get("epochs", 50)
        for epoch in range(1, num_epochs + 1):
            trainer.metrics_history['epochs'].append(epoch)
            train_loss = sweep_train_epoch(trainer, epoch)
            val_loss = sweep_evaluate(trainer, epoch, phase='val')

            if trainer.scheduler:
                trainer.scheduler.step()

            if trainer.early_stopping.early_stop:
                break

    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        raise  # re-raise the exception to ensure the sweep notices the failure