from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split, Dataset

class SubsetWithTransform(Dataset):
    """A dataset wrapper that applies a specific transform to a subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)


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
        transform_list.append(transforms.Normalize(mean, std))
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
        raise ValueError(f"Dataset {name} is not supported.")
    return dataset_dict[name](root=root, train=train, download=download, transform=transform)


def verify_normalization(dataset):
    """Verify that the normalization worked by calculating mean and std."""
    norm_mean, norm_std = calculate_mean_std(dataset)
    print(f'After normalization: mean: {norm_mean}, std: {norm_std}')


def prepare_datasets(dataset_name, 
                     data_root, normalize=True, 
                     precalculated_stats=None, 
                     transform_train=None, 
                     transform_val=None, 
                     use_validation_split=False, 
                     validation_split=0.1, 
                     random_seed=42):
    """
    Prepare training, validation, and test datasets with optional normalization or custom transforms.
    
    Args:
        dataset_name (str): Name of the dataset to load
        data_root (str): Root directory for dataset
        normalize (bool): Whether to normalize the data (ignored if custom transforms are provided)
        precalculated_stats (tuple, optional): Tuple of (mean, std) if already calculated
        transform_train (transforms.Compose, optional): Custom transform for training data
        transform_val (transforms.Compose, optional): Custom transform for validation/test data
        use_validation_split (bool): Whether to split the training set into training and validation sets
        validation_split (float): Fraction of the training set to use as validation
        random_seed (int): Seed for random number generator to ensure reproducibility
    """
    if transform_train is not None and transform_val is not None and use_validation_split:
        # 1. Load the entire training dataset without any transforms
        full_trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=None)

        # 2. Calculate validation size
        val_size = int(len(full_trainset) * validation_split)
        train_size = len(full_trainset) - val_size

        # 3. Split the dataset into training and validation subsets
        train_subset, val_subset = random_split(full_trainset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

        # 4. Wrap subsets with respective transforms
        trainset = SubsetWithTransform(train_subset, transform=transform_train)
        valset = SubsetWithTransform(val_subset, transform=transform_val)

        # 5. Load the test set with transform_val
        testset = load_dataset(name=dataset_name, root=data_root, train=False, transform=transform_val)

        return trainset, valset, testset

    elif transform_train is not None and transform_val is not None:
        # If no validation split is used, apply transforms directly
        trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=transform_train)
        valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=transform_val)
        return trainset, valset

    else:
        # If custom transforms are not provided, proceed with normalization logic

        if normalize and precalculated_stats:
            mean, std = precalculated_stats
            print(f'Using precalculated stats - mean: {mean}, std: {std}')
            normalized_transform = get_transforms(normalize=True, mean=mean, std=std)
            trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=normalized_transform)
            valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=normalized_transform)

            if use_validation_split:
                testset = valset
                val_size = int(len(trainset) * validation_split)
                train_size = len(trainset) - val_size
                train_subset, val_subset = random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))
                # Wrap subsets without additional transforms since normalization is already applied
                trainset, valset = train_subset, val_subset
                verify_normalization(trainset)
                return trainset, valset, testset

            verify_normalization(trainset)
            return trainset, valset

        # Initial transform without normalization to calculate mean and std
        initial_transform = get_transforms(normalize=False)

        trainset = load_dataset(name=dataset_name, root=data_root, train=True, transform=initial_transform)
        valset = load_dataset(name=dataset_name, root=data_root, train=False, transform=initial_transform)

        if use_validation_split:
            testset = valset
            val_size = int(len(trainset) * validation_split)
            train_size = len(trainset) - val_size
            train_subset, val_subset = random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

            if normalize:
                mean, std = calculate_mean_std(train_subset)
                print(f'Calculated stats - mean: {mean}, std: {std}')
                print(f'For future runs, use: precalculated_stats=({mean}, {std})')
                
                # Update transforms with normalization
                normalized_transform = get_transforms(normalize=True, mean=mean, std=std)
                trainset = SubsetWithTransform(train_subset, transform=transform_train if transform_train else get_transforms(normalize=True, mean=mean, std=std))
                valset = SubsetWithTransform(val_subset, transform=transform_val if transform_val else get_transforms(normalize=True, mean=mean, std=std))
                testset = SubsetWithTransform(testset, transform=normalized_transform)

                # Verify normalization
                verify_normalization(train_subset)
            return trainset, valset, testset

        if normalize:
            mean, std = calculate_mean_std(trainset)
            print(f'Calculated stats - mean: {mean}, std: {std}')
            print(f'For future runs, use: precalculated_stats=({mean}, {std})')
            
            # Update transforms with normalization
            normalized_transform = get_transforms(normalize=True, mean=mean, std=std)
            trainset.transform = normalized_transform
            valset.transform = normalized_transform

            # Verify normalization
            verify_normalization(trainset)
        
        return trainset, valset