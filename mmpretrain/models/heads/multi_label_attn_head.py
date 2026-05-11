# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample, label_to_onehot


class SEAttention(BaseModule):
    """Squeeze-and-Excitation attention for channel recalibration.

    Args:
        in_channels (int): Input channels.
        reduction (int): Reduction ratio. Default: 16.
    """

    def __init__(self, in_channels: int, reduction: int = 4, init_cfg=None):
        super().__init__(init_cfg)
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


class SpatialAttention(BaseModule):
    """Spatial attention for spatial recalibration.

    Args:
        in_channels (int): Input channels.
    """

    def __init__(self, in_channels: int, init_cfg=None):
        super().__init__(init_cfg)
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
class MultiLabelAttnClsHead(BaseModule):
    """Multi-label classification head with attention mechanism.

    Supports three attention modes:
    - 'se': Channel attention (Squeeze-and-Excitation)
    - 'spatial': Spatial attention
    - 'cbam': Sequential channel + spatial attention (CBAM style)
    - 'none': No attention, just FC (baseline)

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Input channels from backbone.
        attn_mode (str): Attention mode. Default: 'se'.
        reduction (int): SE reduction ratio. Default: 4.
        init_cfg (dict): Init config. Default: Normal Linear.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 attn_mode: str = 'se',
                 reduction: int = 4,
                 loss: Dict = dict(type='CrossEntropyLoss', use_sigmoid=True),
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01)):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.attn_mode = attn_mode

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

        if thr is None and topk is None:
            thr = 0.5
        self.thr = thr
        self.topk = topk

        # Attention modules
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

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward: apply attention then FC."""
        x = feats[-1]  # (B, C, H, W) or (B, C) after GAP

        # If feature is already pooled (2D), just FC
        if x.dim() == 2:
            return self.fc(x)

        # Apply attention
        if self.attn_mode == 'cbam':
            x = self.channel_attn(x)
            x = self.spatial_attn(x)
        elif self.attention is not None:
            x = self.attention(x)

        # Global average pooling -> FC
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        return feats[-1]

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        cls_score = self(feats)
        num_classes = cls_score.size(-1)

        if 'gt_score' in data_samples[0]:
            target = torch.stack([i.gt_score.float() for i in data_samples])
        else:
            target = torch.stack([
                label_to_onehot(i.gt_label, num_classes) for i in data_samples
            ]).float()

        losses = dict()
        losses['loss'] = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        return losses

    def predict(self,
                feats: Tuple[torch.Tensor],
                data_samples: List[DataSample] = None) -> List[DataSample]:
        cls_score = self(feats)
        pred_scores = torch.sigmoid(cls_score)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(cls_score.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            if self.thr is not None:
                label = torch.where(score >= self.thr)[0]
            else:
                _, label = score.topk(self.topk)
            data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples
