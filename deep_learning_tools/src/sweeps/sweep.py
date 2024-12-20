import wandb
from typing import TYPE_CHECKING, Dict, Any, Optional
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import sys
import os
import traceback
from wandb.errors import CommError, Error as WandbError
from contextlib import contextmanager

if TYPE_CHECKING:
    from src.training.trainer import ModelTrainer

# import sweep configuration
from .sweep_config import sweep_config

@contextmanager
def wandb_run(trainer: "ModelTrainer", config: Dict[str, Any]):
    """Context manager for wandb runs."""
    run = wandb.init(
        project=trainer.logger_manager.wandb_project,
        entity=trainer.logger_manager.wandb_entity,
        config=config,
        reinit=True
    )
    try:
        yield run
    finally:
        if wandb.run is not None:
            wandb.finish()

def build_optimizer(network: nn.Module, optimizer_name: str, learning_rate: float):
    """Utility function to build the optimizer."""
    if optimizer_name.lower() == 'adam':
        return Adam(network.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_name.lower() == 'sgd':
        return SGD(network.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_function(trainer: "ModelTrainer", config: Dict[str, Any] = None):
    """Training function that will be called by the sweep."""
    try:
        # print run info
        print(f"\nrun {wandb.run.name} - batch_size: {config.batch_size}, lr: {config.learning_rate:.5f}, optimizer: {config.optimizer}")
        
        # update trainer's hyperparameters based on sweep config
        trainer.batch_size = config.batch_size
        trainer.optimizer = build_optimizer(
            network=trainer.model,
            optimizer_name=config.optimizer,
            learning_rate=config.learning_rate
        )
        if trainer.scheduler:
            trainer.scheduler = StepLR(trainer.optimizer, step_size=1, gamma=0.75)

        # train the model with updated hyperparameters
        metrics = trainer.train(
            training_set=trainer.training_set,
            val_set=trainer.val_set,
            num_epochs=config.epochs,
            scheduler_step=True  # enable scheduler stepping
        )
        
        # log metrics for each epoch
        for epoch in range(len(metrics["train_loss"])):
            wandb.log({
                "epoch": epoch,
                "train_loss": metrics["train_loss"][epoch],
                "val_loss": metrics["val_loss"][epoch],
                "train_acc": metrics["train_acc"][epoch],
                "val_acc": metrics["val_acc"][epoch]
            })
        
        # log best metrics in summary
        wandb.run.summary.update({
            "best_val_loss": min(metrics["val_loss"]),
            "best_val_acc": max(metrics["val_acc"]),
            "best_train_loss": min(metrics["train_loss"]),
            "best_train_acc": max(metrics["train_acc"])
        })

    except Exception as e:
        print(f"error during training: {str(e)}")
        traceback.print_exc()
        raise  # re-raise the exception

def run_sweep(trainer: "ModelTrainer"):
    """Initializes and runs the W&B sweep using the provided trainer."""
    if trainer.logger_manager.wandb_project is None:
        raise ValueError("wandb_project must be specified when using sweep")

    # ensure we're starting with a clean state
    if wandb.run is not None:
        wandb.finish()

    # configure wandb settings globally
    os.environ["WANDB_SILENT"] = "true"
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
                        print("failed to initialize wandb run")
                        return

                    # access config after wandb.init()
                    config = wandb.config
                    print(f"\nrun {run.name} - batch_size: {config.batch_size}, lr: {config.learning_rate:.5f}, optimizer: {config.optimizer}")

                    train_function(trainer, config)

            except (CommError, WandbError) as e:
                print(f"wandb error during sweep run: {str(e)}")
                raise  # re-raise to ensure errors are not silently ignored
            except KeyboardInterrupt:
                print("\nsweep interrupted by user")
                raise  # re-raise to ensure proper cleanup
            except Exception as e:
                print(f"error during sweep run: {str(e)}")
                traceback.print_exc()
                raise  # re-raise to ensure errors are not silently ignored

        print("\nstarting sweep...")
        # run the sweep agent with error handling
        try:
            count = sweep_config.get('count', 20)
            print(f"running {count} trials...")
            
            wandb.agent(
                sweep_id, 
                function=sweep_train,
                count=count
            )
            
            print("\nsweep completed successfully!")
            print(f"view results at: https://wandb.ai/{trainer.logger_manager.wandb_entity}/{trainer.logger_manager.wandb_project}/sweeps/{sweep_id}")
            print("\ntip: in the sweep page, look at the parallel coordinates plot to find the best parameters")
            print("the best parameters will be those that minimize val_loss and maximize val_acc")

        except KeyboardInterrupt:
            print("\nsweep interrupted by user")
            raise  # re-raise to ensure proper cleanup
        except Exception as e:
            print(f"error during sweep agent execution: {str(e)}")
            traceback.print_exc()
            raise  # re-raise to ensure errors are not silently ignored

    except Exception as e:
        print(f"failed to initialize sweep: {str(e)}")
        traceback.print_exc()
        raise