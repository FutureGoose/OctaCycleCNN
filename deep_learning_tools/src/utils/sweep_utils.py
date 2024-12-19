import wandb
from torch.optim import Adam
import torch.nn as nn


def initialize_wandb_sweep(model, optimizer, batch_size, early_stopping_patience, early_stopping_delta):
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
    """
    wandb.init()  # initialize W&B
    config = wandb.config  # retrieve sweep config

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


def log_hyperparameters(batch_size, optimizer, early_stopping_patience, early_stopping_delta):
    """
    Log hyperparameters to wandb.

    Args:
        batch_size (int): Batch size for data loaders.
        optimizer (torch.optim.Optimizer): The optimizer.
        early_stopping_patience (int): Early stopping patience.
        early_stopping_delta (float): Early stopping delta.
    """
    wandb.config.update({
        "batch_size": batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_delta": early_stopping_delta
    })


def handle_wandb_error(e):
    """
    Handle errors during wandb initialization.

    Args:
        e (Exception): The exception raised during wandb initialization.
    """
    print(f"Error during wandb initialization: {e}")
    raise RuntimeError("Exiting due to error in wandb initialization.") 


def handle_sweep_configuration(model, optimizer, batch_size, early_stopping_patience, early_stopping_delta):
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

    except Exception as e:
        handle_wandb_error(e) 