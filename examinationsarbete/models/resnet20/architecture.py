import torch
import torch.nn as nn
from torchvision import transforms

class BasicBlock(nn.Module):
    """basic building block for resnet-20"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # first conv: 3x3, maintain spatial size with padding
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # second conv: 3x3, maintain spatial size with padding
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNet20(nn.Module):
    """resnet-20 implementation for cifar-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # initial conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 3 stages, each with 6 BasicBlocks (3 * 6 + 2 = 20 layers)
        self.stage1 = self._make_stage(16, 16, 6, stride=1)  # 32x32
        self.stage2 = self._make_stage(16, 32, 6, stride=2)  # 16x16
        self.stage3 = self._make_stage(32, 64, 6, stride=2)  # 8x8
        
        # global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # weight initialization
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # first block might have stride > 1 to reduce spatial dimensions
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # remaining blocks maintain spatial dimensions
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 3 stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # global average pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # he initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # initialize gamma to 1, beta to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # initialize weights with normal distribution
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# data augmentation transforms for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# transforms for testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]) 