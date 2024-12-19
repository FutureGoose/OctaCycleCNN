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
    """create a composition of transforms with optional normalization."""
    transform_list = [transforms.ToTensor()]
    if normalize and mean is not None and std is not None:
        transform_list.append(transforms.Normalize((mean,), (std,)))
    return transforms.Compose(transform_list)


def load_dataset(name, root, train=True, download=True, transform=None):
    """load a specified dataset with given parameters."""
    dataset_dict = {
        'FashionMNIST': datasets.FashionMNIST,
        'MNIST': datasets.MNIST,  # added support for MNIST
        'CIFAR10': datasets.CIFAR10,  # added support for CIFAR10
        # add other datasets here if needed
    }
    if name not in dataset_dict:
        raise ValueError(f"dataset {name} is not supported.")
    return dataset_dict[name](root=root, train=train, download=download, transform=transform)


def prepare_datasets(dataset_name, data_root, normalize=True):
    """prepare training and validation datasets with optional normalization."""
    # initial transform without normalization to calculate mean and std
    initial_transform = get_transforms(normalize=False)
    
    trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=initial_transform)
    valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=initial_transform)

    if normalize:
        mean, std = calculate_mean_std(trainset)
        print(f'calculated mean: {mean}, std: {std}')
        # update transforms with normalization
        normalized_transform = get_transforms(normalize=True, mean=mean, std=std)
        trainset.transform = normalized_transform
        valset.transform = normalized_transform

        # recalculate mean and std after normalization
        norm_mean, norm_std = calculate_mean_std(trainset)
        print(f'after normalization: mean: {norm_mean}, std: {norm_std}')
    
    return trainset, valset