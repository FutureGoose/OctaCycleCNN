import torch
import torch.nn as nn
import torch.nn.functional as F

class Airbench94InspiredNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Airbench94InspiredNet, self).__init__()
        
        # Whitening layer (will be initialized separately)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whitening = nn.Sequential(
            nn.Conv2d(3, whiten_width, kernel_size=whiten_kernel_size, padding=0, bias=True),
            nn.GELU()
        )
        
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(whiten_width, 64, kernel_size=3, padding='same', bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64, momentum=0.4, eps=1e-12),  # Note: momentum=0.6 in original, but PyTorch uses 1-momentum
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(64, momentum=0.4, eps=1e-12),
            nn.GELU()
        )
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding='same', bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256, momentum=0.4, eps=1e-12),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256, momentum=0.4, eps=1e-12),
            nn.GELU()
        )
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256, momentum=0.4, eps=1e-12),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256, momentum=0.4, eps=1e-12),
            nn.GELU()
        )
        
        # Final layers
        self.pool = nn.MaxPool2d(3)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes, bias=False)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Freeze whitening layer weights
        self.whitening[0].weight.requires_grad = False
        
        # Scaling factor
        self.scaling_factor = 1/9

    def forward(self, x):
        x = self.whitening(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x * self.scaling_factor
    
    def _initialize_weights(self):
        # Initialize all convolutions after whitening as partial identity transforms
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.whitening[0]:
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
            
        patches = get_patches(train_images, patch_shape=self.whitening[0].weight.data.shape[2:])
        eigenvalues, eigenvectors = get_whitening_parameters(patches)
        eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
        self.whitening[0].weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))