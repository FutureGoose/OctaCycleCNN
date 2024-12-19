import wandb
from torch.optim import Adam
import torch.nn as nn
from typing import Dict, Any, Optional
from wandb.errors import CommError, Error as WandbError
import torch


def initialize_wandb_sweep(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    batch_size: int,
    early_stopping_patience: int,
    early_stopping_delta: float
) -> Dict[str, Any]:
    """
    Initialize wandb for sweep and retrieve configuration.

    Args:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        batch_size (int): Batch size for data loaders.
        early_stopping_patience (int): Early stopping patience.
        early_stopping_delta (float): Early stopping delta.

    Returns:
        dict: Updated configuration values.

    Raises:
        WandbError: If there's an error with wandb initialization
        RuntimeError: If there's an error with configuration
    """
    try:
        wandb.init()  # initialize W&B
        config = wandb.config  # retrieve sweep config
    except WandbError as e:
        raise WandbError(f"Failed to initialize wandb: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during wandb initialization: {str(e)}")

    try:
        # override hyperparameters with config values if present
        batch_size = config.get("batch_size", batch_size)
        learning_rate = config.get("learning_rate", 1e-4)
        early_stopping_patience = config.get("early_stopping_patience", early_stopping_patience)
        early_stopping_delta = config.get("early_stopping_delta", early_stopping_delta)

        # initialize optimizer with learning rate from config
        if optimizer is None:
            optimizer = Adam(model.parameters(), lr=learning_rate)

        return {
            "batch_size": batch_size,
            "optimizer": optimizer,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_delta": early_stopping_delta
        }
    except AttributeError as e:
        raise RuntimeError(f"Invalid configuration structure: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error processing configuration: {str(e)}")


def log_hyperparameters(
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    early_stopping_patience: int,
    early_stopping_delta: float
) -> None:
    """
    Log hyperparameters to wandb.

    Args:
        batch_size (int): Batch size for data loaders.
        optimizer (torch.optim.Optimizer): The optimizer.
        early_stopping_patience (int): Early stopping patience.
        early_stopping_delta (float): Early stopping delta.

    Raises:
        CommError: If there's an error communicating with wandb servers
        RuntimeError: If there's an error updating the configuration
    """
    try:
        wandb.config.update({
            "batch_size": batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_delta": early_stopping_delta
        })
    except CommError as e:
        raise CommError(f"Failed to communicate with wandb servers: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to update wandb configuration: {str(e)}")


def handle_wandb_error(e: Exception) -> None:
    """
    Handle errors during wandb initialization.

    Args:
        e (Exception): The exception raised during wandb initialization.

    Raises:
        RuntimeError: Always raises with a descriptive message
    """
    if isinstance(e, CommError):
        error_msg = "Failed to communicate with wandb servers"
    elif isinstance(e, WandbError):
        error_msg = "Wandb initialization error"
    else:
        error_msg = "Unexpected error during wandb initialization"
    
    print(f"{error_msg}: {str(e)}")
    raise RuntimeError(f"Exiting due to {error_msg.lower()}")


def handle_sweep_configuration(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    batch_size: int,
    early_stopping_patience: int,
    early_stopping_delta: float
) -> Dict[str, Any]:
    """
    Handle the entire sweep configuration process.

    Args:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        batch_size (int): Batch size for data loaders.
        early_stopping_patience (int): Early stopping patience.
        early_stopping_delta (float): Early stopping delta.

    Returns:
        dict: Updated configuration values.

    Raises:
        RuntimeError: If there's any error during the configuration process
    """
    try:
        sweep_config = initialize_wandb_sweep(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta
        )

        log_hyperparameters(
            batch_size=sweep_config["batch_size"],
            optimizer=sweep_config["optimizer"],
            early_stopping_patience=sweep_config["early_stopping_patience"],
            early_stopping_delta=sweep_config["early_stopping_delta"]
        )

        return sweep_config

    except (CommError, WandbError) as e:
        handle_wandb_error(e)
    except Exception as e:
        print(f"Unexpected error during sweep configuration: {str(e)}")
        raise RuntimeError("Failed to configure sweep") 