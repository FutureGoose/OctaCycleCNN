import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

def conv(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding='same', bias=False)

class Airbench94InspiredNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Airbench94InspiredNet, self).__init__()
        
        act = lambda: nn.GELU()
        bn = lambda ch: nn.BatchNorm2d(ch)  # Note: momentum=0.6 in original, but PyTorch uses 1-momentum  momentum=0.4, eps=1e-12
        
        self.net = nn.Sequential(
            # Whitening layer (will be initialized separately)
            nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
                act(),
            ),
            # First block
            nn.Sequential(
                conv(24, 64),
                nn.MaxPool2d(2),
                bn(64), act(),
                conv(64, 64),
                bn(64), act(),
            ),
            # Second block
            nn.Sequential(
                conv(64, 256),
                nn.MaxPool2d(2),
                bn(256), act(),
                conv(256, 256),
                bn(256), act(),
            ),
            # Third block
            nn.Sequential(
                conv(256, 256),
                nn.MaxPool2d(2),
                bn(256), act(),
                conv(256, 256),
                bn(256), act(),
            ),
            nn.MaxPool2d(3),
            Flatten(),
            nn.Linear(256, num_classes, bias=False),
            Mul(1/9)  # Scaling factor
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Freeze whitening layer weights
        self.net[0][0].weight.requires_grad = False

    def forward(self, x):
        return self.net(x)
    
    def _initialize_weights(self):
        # Initialize all convolutions after whitening as partial identity transforms
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.net[0][0]:
                w = m.weight.data
                nn.init.dirac_(w[:w.size(1)])  # Initialize first M filters as identity
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Disable affine scale parameters but keep biases
                m.weight.requires_grad = False
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                
    def init_whitening(self, train_images, eps=5e-4):
        """Initialize the whitening layer using training images statistics.
        This should be called before training with a batch of training images."""
        def get_patches(x, patch_shape):
            c, (h, w) = x.shape[1], patch_shape
            return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
            
        def get_whitening_parameters(patches):
            n,c,h,w = patches.shape
            patches_flat = patches.view(n, -1)
            est_patch_covariance = (patches_flat.T @ patches_flat) / n
            eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
            return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)
            
        patches = get_patches(train_images, patch_shape=self.net[0][0].weight.data.shape[2:])
        eigenvalues, eigenvectors = get_whitening_parameters(patches)
        eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
        self.net[0][0].weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))