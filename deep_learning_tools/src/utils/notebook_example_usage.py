################### Build dataset ###################
import sys
from pathlib import Path
import torch

# package_root = Path('/home/goose/projects/deeplearning/deep_learning_tools')
package_root = Path('/home/gustaf/projects/deeplearning/deep_learning_tools')
sys.path.append(str(package_root))

from src import prepare_datasets

# parameters
dataset_name = 'FashionMNIST'
data_root = '/home/gustaf/projects/deeplearning/data'
download_data = True
normalize_data = False

# prepare datasets
trainset, valset = prepare_datasets(dataset_name, data_root, normalize=normalize_data)


################### Build network ###################
import torch.nn as nn
from torch.optim import Adadelta, Adam, SGD
from torch.optim.lr_scheduler import StepLR
from src import ModelTrainer
from src import accuracy, precision

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
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


#################### Build optimizer and scheduler ####################
# initialize optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)
# initialize scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=0.75)


#################### Build trainer ####################
trainer = ModelTrainer(
    model=model,
    device=device,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=512,  # 128
    verbose=True,             # controls training progress logs
    save_metrics=False,        # controls saving of metrics visualization
    early_stopping_patience=3,
    early_stopping_delta=0.1,
    logger_type="wandb",
    wandb_project="fashion-mnist",
    wandb_entity="futuregoose",
    sweep=False
)


#################### Train model ####################
trained_model = trainer.train(
    training_set=trainset,
    val_set=valset,
    num_epochs=2
)