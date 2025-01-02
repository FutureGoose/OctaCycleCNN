from torchvision import datasets, transforms
import torch
import numpy as np

def calculate_mean_std(dataset):
    """Calculate mean and standard deviation of a dataset."""
    mean = 0.0
    std = 0.0
    total_images = len(dataset)

    for img, _ in dataset:
        mean += img.mean()
        std += img.std()

    mean /= total_images
    std /= total_images

    return mean.item(), std.item()


def get_transforms(normalize=True, mean=None, std=None):
    """Create a composition of transforms with optional normalization."""
    transform_list = [transforms.ToTensor()]
    if normalize and mean is not None and std is not None:
        transform_list.append(transforms.Normalize((mean,), (std,)))
    return transforms.Compose(transform_list)


def load_dataset(name, root, train=True, download=True, transform=None):
    """Load a specified dataset with given parameters."""
    dataset_dict = {
        'FashionMNIST': datasets.FashionMNIST,
        'MNIST': datasets.MNIST,
        'CIFAR10': datasets.CIFAR10,
        # add datasets here
    }
    if name not in dataset_dict:
        raise ValueError(f"dataset {name} is not supported.")
    return dataset_dict[name](root=root, train=train, download=download, transform=transform)


def verify_normalization(dataset):
    """Verify that the normalization worked by calculating mean and std."""
    norm_mean, norm_std = calculate_mean_std(dataset)
    print(f'after normalization: mean: {norm_mean}, std: {norm_std}')


def prepare_datasets(dataset_name, data_root, normalize=True, precalculated_stats=None, transform_train=None, transform_test=None):
    """
    Prepare training and validation datasets with optional normalization or custom transforms.
    
    Args:
        dataset_name (str): Name of the dataset to load
        data_root (str): Root directory for dataset
        normalize (bool): Whether to normalize the data (ignored if custom transforms are provided)
        precalculated_stats (tuple, optional): Tuple of (mean, std) if already calculated
        transform_train (transforms.Compose, optional): Custom transform for training data
        transform_test (transforms.Compose, optional): Custom transform for test/validation data
    """
    # If custom transforms are provided, use them directly
    if transform_train is not None and transform_test is not None:
        trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=transform_train)
        valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=transform_test)
        return trainset, valset
    
    # Load datasets first
    trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=None)
    valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=None)
    
    # Handle different dataset formats
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        # CIFAR datasets: uint8 NHWC format
        if hasattr(trainset, 'data') and isinstance(trainset.data, (torch.Tensor, np.ndarray)):
            trainset.data = torch.tensor(trainset.data, dtype=torch.float16) / 255
            trainset.data = trainset.data.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
            valset.data = torch.tensor(valset.data, dtype=torch.float16) / 255
            valset.data = valset.data.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
    elif dataset_name in ['MNIST', 'FashionMNIST']:
        # MNIST-like datasets: uint8 NHW format
        if hasattr(trainset, 'data') and isinstance(trainset.data, (torch.Tensor, np.ndarray)):
            trainset.data = torch.tensor(trainset.data, dtype=torch.float16) / 255
            trainset.data = trainset.data.unsqueeze(1).to(memory_format=torch.channels_last)  # Add channel dim
            valset.data = torch.tensor(valset.data, dtype=torch.float16) / 255
            valset.data = valset.data.unsqueeze(1).to(memory_format=torch.channels_last)  # Add channel dim
    else:
        # For other datasets, we'll rely on transforms
        print(f"Dataset {dataset_name} will use standard transform pipeline")
    
    # Apply normalization if requested
    if normalize and precalculated_stats:
        mean, std = precalculated_stats
        print(f'using precalculated stats - mean: {mean}, std: {std}')
        normalized_transform = get_transforms(normalize=True, mean=mean, std=std)
        trainset.transform = normalized_transform
        valset.transform = normalized_transform

        # verify normalization with precalculated stats
        verify_normalization(trainset)
        return trainset, valset
    
    # Calculate stats if needed
    if normalize:
        mean, std = calculate_mean_std(trainset)
        print(f'calculated stats - mean: {mean}, std: {std}')
        print(f'for future runs, use: precalculated_stats=({mean}, {std})')
        
        # update transforms with normalization
        normalized_transform = get_transforms(normalize=True, mean=mean, std=std)
        trainset.transform = normalized_transform
        valset.transform = normalized_transform

        # verify normalization after calculating mean and std
        verify_normalization(trainset)
    
    return trainset, valset