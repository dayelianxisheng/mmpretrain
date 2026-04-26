# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn


class ECALayer(nn.Module):
    """Efficient Channel Attention layer.

    Args:
        channels (int): Number of channels of the input feature map.
        k_size (int): Kernel size for 1D convolution. Default: 3.
    """

    def __init__(self, channels, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
