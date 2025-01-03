import torch
from torch.utils.data import Dataset
import hashlib
from torchvision import transforms

def hash_fn(n, seed=42):
    k = n * seed
    return int(hashlib.md5(bytes(str(k), 'utf-8')).hexdigest()[-8:], 16)

class AlternatingFlipDataset(Dataset):
    """Dataset wrapper that performs alternating flip augmentation."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0
        
    def set_epoch(self, epoch):
        """Must be called at the start of each epoch"""
        self.epoch = epoch
        
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Determine if this sample should be flipped in this epoch
        should_flip = (hash_fn(idx) + self.epoch) % 2 == 0
        if should_flip:
            if isinstance(data, torch.Tensor):
                data = torch.flip(data, [-1])  # Flip the last dimension (width)
            else:
                # If data is a PIL image
                data = transforms.functional.hflip(data)
        return data, target
    
    def __len__(self):
        return len(self.dataset)

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