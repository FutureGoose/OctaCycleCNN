import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
import wandb
from pathlib import Path
import sys

# append the package root to sys.path
package_root = Path('/home/gustaf/projects/deeplearning/deep_learning_tools')
sys.path.append(str(package_root))

from src.training.trainer import ModelTrainer  # adjust the import based on your package structure
from src.metrics import accuracy  # adjust as necessary

def main():
    # initialize w&b
    wandb.init()

    # define your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the net class
    class Net(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc(x)
            return x

    # initialize model
    model = Net(input_size=28*28, output_size=10)

    # define normalization transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load datasets with the new data root
    data_root = '/home/gustaf/projects/deeplearning'  # updated to one level up
    trainset = datasets.FashionMNIST(root=f'{data_root}/deep_learning_tools/data/', train=True, download=True, transform=transform)
    valset = datasets.FashionMNIST(root=f'{data_root}/deep_learning_tools/data/', train=False, download=True, transform=transform)

    # initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=None,  # let modeltrainer handle optimizer if sweep=True
        scheduler=None,  # optionally handle schedulers via sweep
        batch_size=wandb.config.batch_size,  # fetch from sweep config
        sweep=True,  # enable sweep
        verbose=True,
        save_metrics=False,
        early_stopping_patience=wandb.config.early_stopping_patience,
        early_stopping_delta=wandb.config.early_stopping_delta,
        logger_type="wandb",
        wandb_project="fashion-mnist",
        wandb_entity="futuregoose-org"
        # wandb_entity="futuregoose"
    )

    # train model
    trained_model = trainer.train(
        training_set=trainset,
        val_set=valset,
        num_epochs=10  # can also be a sweep parameter
    )

if __name__ == "__main__":
    main()