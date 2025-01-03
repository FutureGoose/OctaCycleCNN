from torchvision import datasets, transforms
import torch

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
    dataset = dataset_dict[name](root=root, train=train, download=download, transform=transform)
    
    # optimize memory layout for GPU if using CIFAR10
    if name == 'CIFAR10' and transform is not None:
        # convert first batch to check if it's a tensor output
        sample_batch = transform(dataset[0][0])
        if isinstance(sample_batch, torch.Tensor) and len(sample_batch.shape) == 3:
            old_transform = transform
            def optimized_transform(x):
                return old_transform(x).to(memory_format=torch.channels_last)
            dataset.transform = optimized_transform
    
    return dataset


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
    
    # Otherwise, use the original normalization logic
    if normalize and precalculated_stats:
        mean, std = precalculated_stats
        print(f'using precalculated stats - mean: {mean}, std: {std}')
        normalized_transform = get_transforms(normalize=True, mean=mean, std=std)
        trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=normalized_transform)
        valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=normalized_transform)

        # verify normalization with precalculated stats
        verify_normalization(trainset)
        return trainset, valset
    
    # initial transform without normalization to calculate mean and std
    initial_transform = get_transforms(normalize=False)
    
    trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=initial_transform)
    valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=initial_transform)

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