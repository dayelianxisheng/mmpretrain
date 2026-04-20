# Copyright (c) OpenMMLab. All rights reserved.
"""AlexNet with SE Block."""

import torch.nn as nn

from mmpretrain.registry import MODELS
from ..utils.se_layer import SELayer
from .base_backbone import BaseBackbone


@MODELS.register_module()
class SEAlexNet(BaseBackbone):
    """AlexNet with Squeeze-and-Excitation block.

    The input for AlexNet is a 224x224 RGB image.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
        se_ratio (int): SE reduction ratio. Default: 16.
    """

    def __init__(self, se_ratio=16):
        super(SEAlexNet, self).__init__()

        # Original AlexNet architecture
        # Conv1: 64 channels, conv2: 192 channels, conv3: 384 channels
        # conv4: 256 channels, conv5: 256 channels
        self.features = nn.Sequential(
            # Conv1 -> 64
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            SELayer(64, ratio=se_ratio),  # SE before pooling
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2 -> 192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            SELayer(192, ratio=se_ratio),  # SE before pooling
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3 -> 384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SELayer(384, ratio=se_ratio),

            # Conv4 -> 256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SELayer(256, ratio=se_ratio),

            # Conv5 -> 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SELayer(256, ratio=se_ratio),  # SE before pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 移除内置分类器，使用 neck + head 进行分类

    def forward(self, x):
        x = self.features(x)
        # 总是返回特征，不包含分类器
        # neck 会做全局平均池化
        return (x, )
