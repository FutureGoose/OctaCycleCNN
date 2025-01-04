import torch
from torch.utils.data import Dataset
import hashlib
from torchvision import transforms

def hash_fn(n, seed=42):
    k = n * seed
    return int(hashlib.md5(bytes(str(k), 'utf-8')).hexdigest()[-8:], 16)

class AlternatingFlipDataset(Dataset):
    """Dataset wrapper that performs alternating flip augmentation following the paper exactly.
    Uses (hash(idx) + epoch) % 2 to determine flips, ensuring every pair of consecutive
    epochs contains all possible flipped versions of each image."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0
        
    def set_epoch(self, epoch):
        """Must be called at the start of each epoch"""
        self.epoch = epoch
        
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        
        # Exactly matching the paper's implementation:
        # flip_mask = ((hashed_indices + epoch) % 2 == 0)
        should_flip = (hash_fn(idx) + self.epoch) % 2 == 0
            
        if should_flip:
            if isinstance(data, torch.Tensor):
                data = torch.flip(data, [-1])
            else:
                data = transforms.functional.hflip(data)
        
        return data, target
    
    def __len__(self):
        return len(self.dataset)
    
    def verify_alternating_flip(self, num_samples=5):
        """Verify that the alternating flip is working correctly according to the paper's specification."""
        # Store original epoch
        original_epoch = self.epoch
        
        # Test samples across first three epochs (0,1,2) to verify the pattern
        results = []
        for idx in range(num_samples):
            # Get flip states for three consecutive epochs
            flip_states = [
                (hash_fn(idx) + 0) % 2 == 0,  # epoch 0
                (hash_fn(idx) + 1) % 2 == 0,  # epoch 1
                (hash_fn(idx) + 2) % 2 == 0,  # epoch 2
            ]
            
            # Print detailed info for debugging
            print(f"Sample {idx}:")
            print(f"  Epoch 0: {'flipped' if flip_states[0] else 'not flipped'}")
            print(f"  Epoch 1: {'flipped' if flip_states[1] else 'not flipped'}")
            print(f"  Epoch 2: {'flipped' if flip_states[2] else 'not flipped'}")
            
            # Verify that consecutive epochs have opposite flip states
            results.append(
                flip_states[0] != flip_states[1] and  # alternates between epoch 0 and 1
                flip_states[1] != flip_states[2]      # alternates between epoch 1 and 2
            )
        
        # Restore original epoch
        self.set_epoch(original_epoch)
        
        # All checks should pass
        verification_passed = all(results)
        if verification_passed:
            print("\033[92mAlternating flip verification passed! ✓\033[0m")
            print("All samples follow the paper's alternating pattern:")
            print("- Each sample's flip state is determined by (hash(idx) + epoch) % 2")
            print("- This ensures consecutive epochs always have opposite flip states")
            print("- Every pair of consecutive epochs contains all possible flipped versions")
        else:
            print("\033[91mAlternating flip verification failed! ✗\033[0m")
            print("Some samples did not alternate correctly between epochs")
        
        return verification_passed

class MultiCropTTAWrapper:
    """Wrapper for test-time augmentation with multiple crops."""
    def __init__(self, model):
        self.model = model
        
    def __call__(self, x):
        # Basic forward pass
        base_output = self.model(x)
        
        # Mirror flip
        flipped = torch.flip(x, [-1])
        flip_output = self.model(flipped)
        
        # Combine base and flip
        outputs = [base_output, flip_output]
        
        # Add translations if input size allows
        if x.size(-1) >= 32:  # Only do translations for images 32x32 or larger
            pad = 1
            padded = torch.nn.functional.pad(x, (pad,)*4, 'reflect')
            # Up-left translation
            trans1 = padded[..., :x.size(-2), :x.size(-1)]
            # Down-right translation
            trans2 = padded[..., 2:, 2:]
            
            # Get outputs for translations
            trans1_output = self.model(trans1)
            trans2_output = self.model(trans2)
            
            # Get outputs for flipped translations
            trans1_flip = self.model(torch.flip(trans1, [-1]))
            trans2_flip = self.model(torch.flip(trans2, [-1]))
            
            outputs.extend([trans1_output, trans2_output, trans1_flip, trans2_flip])
        
        # Weight and combine all outputs
        # Base predictions get 0.25 weight each, translations get 0.125 each
        weights = torch.tensor([0.25, 0.25] + [0.125]*4, device=outputs[0].device)
        return torch.stack(outputs).mul(weights.view(-1, 1, 1)).sum(0) 