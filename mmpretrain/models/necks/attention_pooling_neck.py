# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


class SEAttention(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Args:
        in_channels (int): Input channels.
        reduction (int): Reduction ratio. Default: 4.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention (CBAM style).

    Args:
        in_channels (int): Input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x)


@MODELS.register_module()
class AttentionPoolingNeck(BaseModule):
    """Neck with attention-weighted pooling.

    Applies attention to spatial features, then pools to (B, C).
    Works between backbone (featmap) and head (GAP pooled).

    Args:
        in_channels (int): Input channels from backbone.
        attn_mode (str): 'se', 'spatial', 'cbam', or 'none'. Default: 'cbam'.
        reduction (int): SE reduction ratio. Default: 4.
        init_cfg (dict): Init config.
    """

    def __init__(self,
                 in_channels: int,
                 attn_mode: str = 'cbam',
                 reduction: int = 4,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.attn_mode = attn_mode

        if attn_mode == 'se':
            self.attention = SEAttention(in_channels, reduction)
        elif attn_mode == 'spatial':
            self.attention = SpatialAttention(in_channels)
        elif attn_mode == 'cbam':
            self.channel_attn = SEAttention(in_channels, reduction)
            self.spatial_attn = SpatialAttention(in_channels)
        elif attn_mode == 'none':
            self.attention = None
        else:
            raise ValueError(f'Unknown attn_mode: {attn_mode}')

    def forward(self, feats: tuple) -> tuple:
        x = feats[-1]  # (B, C, H, W)

        if self.attn_mode == 'cbam':
            x = self.channel_attn(x)
            x = self.spatial_attn(x)
        elif self.attention is not None:
            x = self.attention(x)

        # Global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return (x, )
