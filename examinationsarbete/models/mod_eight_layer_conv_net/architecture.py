import torch.nn as nn
import math

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

def conv(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding='same', bias=False)

class ModEightLayerConvNet(nn.Module):
    def __init__(self):
        super(ModEightLayerConvNet, self).__init__()
        
        act  = lambda: nn.ReLU(inplace=True)
        bn = lambda ch: nn.BatchNorm2d(ch, momentum=0.6)
        pool = lambda: nn.MaxPool2d(kernel_size=2, stride=2)

        # input size: [batch, 3, 32, 32]
        self.features = nn.Sequential( 
            conv(3, 64),                  # size: [batch, 64, 32, 32]
            bn(64), act(),                # size maintained
            conv(64, 64),                 # size: [batch, 64, 32, 32]
            bn(64), act(),                # size maintained
            pool(),                       # size: [batch, 64, 16, 16]

            conv(64, 128),                # size: [batch, 128, 16, 16]
            bn(128), act(),               # size maintained
            conv(128, 128),               # size: [batch, 128, 16, 16]
            bn(128), act(),               # size maintained
            pool(),                       # size: [batch, 128, 8, 8]

            conv(128, 256),               # size: [batch, 256, 8, 8]
            bn(256), act(),               # size maintained
            conv(256, 256),               # size: [batch, 256, 8, 8]
            bn(256), act(),               # size maintained
            nn.AdaptiveAvgPool2d((1, 1))  # size: [batch, 256, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # size: [batch, 256]
            nn.Linear(256, 512),          # size: [batch, 512]
            nn.BatchNorm1d(512), act(),   # size maintained
            nn.Dropout(p=0.2),            # size maintained
            nn.Linear(512, 10, bias=False),           # size: [batch, 10]
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)