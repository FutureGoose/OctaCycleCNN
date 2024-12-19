# deep_learning_tools/src/sweeps/sweep.py

import wandb
import subprocess

# define the sweep configuration
sweep_config = {
    "method": "grid",  # options: grid, random, bayesian
    "name": "mnist-sweep",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "batch_size": {
            "values": [32, 64, 128]
        },
        "learning_rate": {
            "values": [0.1, 0.01, 0.001]
        },
        "early_stopping_patience": {
            "values": [3, 5]
        },
        "early_stopping_delta": {
            "values": [0.0001, 0.001]
        }
    }
}

# initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="fashion-mnist", entity="futuregoose")

# function to run the training script
def train():
    # run the training script
    subprocess.run(["python3", "-u", "deep_learning_tools/src/training/train.py"])

# start the sweep agent
wandb.agent(sweep_id, function=train)