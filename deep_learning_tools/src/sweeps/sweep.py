import wandb
from typing import TYPE_CHECKING, Dict, Any, Optional
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import sys
from wandb.errors import CommError, Error as WandbError

if TYPE_CHECKING:
    from src.training.trainer import ModelTrainer

# import sweep configuration
from .sweep_config import sweep_config

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
            scheduler_step=None
        )
        
        # log final metrics
        wandb.log({
            "final_train_loss": metrics["train_loss"][-1],
            "final_val_loss": metrics["val_loss"][-1],
            "final_train_acc": metrics["train_acc"][-1],
            "final_val_acc": metrics["val_acc"][-1]
        })

    except Exception as e:
        print(f"error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_or_create_sweep(project: str, entity: Optional[str] = None) -> str:
    """Get existing sweep ID or create a new one."""
    api = wandb.Api()
    
    # construct the path to check for existing sweeps
    path = f"{entity}/{project}" if entity else project
    
    try:
        # get list of sweeps for this project
        sweeps = api.sweeps(path=path)
        
        # look for an active sweep with matching configuration
        for sweep in sweeps:
            if (sweep.state == 'running' and 
                sweep.config.get('method') == sweep_config.get('method') and
                sweep.config.get('metric') == sweep_config.get('metric')):
                print(f"\nresuming existing sweep: {sweep.id}")
                return sweep.id
                
    except Exception as e:
        print(f"error checking existing sweeps: {str(e)}")
    
    # if no matching sweep found or error occurred, create new one
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity
    )
    print(f"\ncreated new sweep: {sweep_id}")
    return sweep_id

def run_sweep(trainer: "ModelTrainer"):
    """Initializes and runs the W&B sweep using the provided trainer."""
    if trainer.logger_manager.wandb_project is None:
        raise ValueError("wandb_project must be specified when using sweep")

    # ensure we're starting with a clean state
    if wandb.run is not None:
        wandb.finish()

    try:
        # get existing sweep or create new one
        sweep_id = get_or_create_sweep(
            project=trainer.logger_manager.wandb_project,
            entity=trainer.logger_manager.wandb_entity
        )
        print(f"view sweep at: https://wandb.ai/{trainer.logger_manager.wandb_entity}/{trainer.logger_manager.wandb_project}/sweeps/{sweep_id}")

        def sweep_train():
            """Function to be executed for each sweep run."""
            try:
                # initialize a new wandb run within the sweep
                with wandb.init(reinit=True) as run:
                    if run is None:
                        print("failed to initialize wandb run")
                        return

                    config = wandb.config
                    print(f"\nstarting run with config:")
                    for key, value in dict(config).items():
                        print(f"  {key}: {value}")
                    
                    train_function(trainer, config)

            except (CommError, WandbError) as e:
                print(f"wandb error during sweep run: {str(e)}")
            except KeyboardInterrupt:
                print("\nsweep interrupted by user")
                if wandb.run is not None:
                    wandb.finish()
                return
            except Exception as e:
                print(f"error during sweep run: {str(e)}")
                import traceback
                traceback.print_exc()

        print("\nstarting sweep agent...")
        # run the sweep agent with error handling
        try:
            wandb.agent(
                sweep_id, 
                function=sweep_train,
                count=sweep_config.get('count', 20),
                project=trainer.logger_manager.wandb_project,
                entity=trainer.logger_manager.wandb_entity
            )
        except KeyboardInterrupt:
            print("\nsweep interrupted by user")
            if wandb.run is not None:
                wandb.finish()
            return
        except Exception as e:
            print(f"error during sweep agent execution: {str(e)}")
            import traceback
            traceback.print_exc()
            if wandb.run is not None:
                wandb.finish()
            raise

    except Exception as e:
        print(f"failed to initialize sweep: {str(e)}")
        import traceback
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()
        raise