"""
The EightLayerConvNet is a streamlined, VGG-inspired CNN tailored for the CIFAR-10 dataset. Key design choices include:

- Orthogonal Weight Initialization: Promotes stable and efficient training.
- Excluding Bias Terms in Convolutional Layers: Eliminates redundancy by leveraging Batch Normalization parameters.
- Small Kernel Sizes and Progressive Pooling: Efficiently extract and downsample features from small images.

These architectural decisions collectively contribute to the model's performance and training efficiency on CIFAR-10.
"""


import torch.nn as nn
from torchvision import  transforms

class EightLayerConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(EightLayerConvNet, self).__init__()
        # input size: [batch, 3, 32, 32]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),      # size: [batch, 64, 32, 32]
            nn.BatchNorm2d(64, momentum=0.6),                            # size maintained
            nn.ReLU(inplace=True),                                       # size maintained
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),     # size: [batch, 64, 32, 32]
            nn.BatchNorm2d(64, momentum=0.6),                            # size maintained
            nn.ReLU(inplace=True),                                       # size maintained
            nn.MaxPool2d(kernel_size=2, stride=2),                       # size: [batch, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),    # size: [batch, 128, 16, 16]
            nn.BatchNorm2d(128, momentum=0.6),                           # size maintained
            nn.ReLU(inplace=True),                                       # size maintained
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),   # size: [batch, 128, 16, 16]
            nn.BatchNorm2d(128, momentum=0.6),                           # size maintained
            nn.ReLU(inplace=True),                                       # size maintained
            nn.MaxPool2d(kernel_size=2, stride=2),                       # size: [batch, 128, 8, 8]

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),   # size: [batch, 256, 8, 8]
            nn.BatchNorm2d(256, momentum=0.6),                           # size maintained
            nn.ReLU(inplace=True),                                       # size maintained
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),   # size: [batch, 256, 8, 8]
            nn.BatchNorm2d(256, momentum=0.6),                           # size maintained
            nn.ReLU(inplace=True),                                       # size maintained
            nn.AdaptiveAvgPool2d((1, 1))                                 # size: [batch, 256, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                                # size: [batch, 256]
            nn.Linear(256, 512),                                         # size: [batch, 512]
            nn.BatchNorm1d(512, momentum=0.6),                           # size maintained
            nn.ReLU(inplace=True),                                       # size maintained
            nn.Dropout(p=0.2),                                           # size maintained
            nn.Linear(512, num_classes, bias=False)                      # size: [batch, num_classes]
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# data augmentation transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])