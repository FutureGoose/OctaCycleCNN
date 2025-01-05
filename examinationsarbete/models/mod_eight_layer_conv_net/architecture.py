import torch.nn as nn
from torchvision import  transforms
import math

def conv(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding='same', bias=False)

class ModEightLayerConvNet(nn.Module):
    def __init__(self):
        super(ModEightLayerConvNet, self).__init__()
        
        act  = lambda: nn.ReLU(inplace=True)
        bn = lambda ch: nn.BatchNorm2d(ch, momentum=0.6)

        self.features = nn.Sequential(
            conv(3, 64),
            bn(64), act(),
            conv(64, 64),
            bn(64), act(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv(64, 128),
            bn(128), act(),
            conv(128, 128),
            bn(128), act(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv(128, 256),
            bn(256), act(),
            conv(256, 256),
            bn(256), act(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            bn(256), act(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10)
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
                nn.init.constant_(m.bias, 0)