import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """LayerNorm that supports BCHW format by internally converting to BHWC"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if x.dim() == 4:
            # Convert from BCHW to BHWC
            x = x.permute(0, 2, 3, 1)
            
            # Normalize in channels dimension
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            # Apply weight and bias
            x = self.weight * x + self.bias
            
            # Convert back to BCHW
            x = x.permute(0, 3, 1, 2)
            return x
        else:
            # For linear layers (BC format)
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
            return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)  # [B, C, H, W]
        
        # Channel norm, so transpose to [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        # MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # Residual
        x = input + self.drop_path(self.gamma * x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ModernConvNeXt(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Configuration for CIFAR-10
        dims = [64, 128, 256, 512]  # Channel dimensions
        depths = [2, 2, 4, 2]  # Number of blocks at each stage
        
        # Stem for 32x32 input
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm(dims[0])
        )
        
        # Downsampling layers
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            ))
        
        # ConvNeXt stages
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, 0.1, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j]) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Final norm and head
        self.norm = LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Weight init
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.stem(x)
        print(f"After stem: {x.shape}")
        
        for i in range(4):
            x = self.stages[i](x)
            print(f"After stage {i}: {x.shape}")
            if i < 3:
                x = self.downsample_layers[i](x)
                print(f"After downsample {i}: {x.shape}")
        
        # Global average pooling and final norm
        x = x.mean([-2, -1])
        print(f"After pooling: {x.shape}")
        x = self.norm(x)
        print(f"After norm: {x.shape}")
        x = self.head(x)
        print(f"After head: {x.shape}")
        
        # Ensure logits are in correct shape [B, num_classes]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            print(f"After reshape: {x.shape}")
        
        print(f"Final output shape: {x.shape}")
        return x
