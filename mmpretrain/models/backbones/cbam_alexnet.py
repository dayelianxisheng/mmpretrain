import torch.nn as nn
import torch
from mmpretrain.registry import MODELS
from mmpretrain.models.utils.cbam import cbam
from mmpretrain.models.backbones.base_backbone import BaseBackbone


@MODELS.register_module()
class CBAMAlexNet(BaseBackbone):

    def __init__(self, in_channels=3):
        super(CBAMAlexNet, self).__init__()

        self.features = nn.Sequential(
            # Conv1 -> 64
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # cbam(64, ratio=16),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2 -> 192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # cbam(192, ratio=16),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3 -> 384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            cbam(384, ratio=16),

            # Conv4 -> 256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # cbam(256,ratio=16),

            # Conv5 -> 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # cbam(256,ratio=16),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


    def forward(self, x):
        x = self.features(x)
        return (x, )
