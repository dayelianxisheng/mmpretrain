# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpretrain.models.utils import SELayer
from mmpretrain.registry import MODELS
from .mobilenet_v2 import MobileNetV2


class SEHSwishInvertedResidual(BaseModule):
    """InvertedResidual block with SE + optional H-Swish for MobileNetV2.

    Structure: Expand(act) → DWConv(act) → SE(mid_channels) → Project

    Args:
        act_cfg (dict): Activation config for expand and depthwise conv.
        se_ratio (int): Channel reduction ratio for SE. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 se_ratio=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.stride = stride
        self.with_cp = with_cp
        self.use_res_connect = stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))

        # Expand: 1x1 conv
        self.expand_conv = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # DWConv: 3x3 depthwise conv
        self.depthwise_conv = ConvModule(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # SE: 作用在 mid_channels（DWConv 输出）上
        self.se = SELayer(
            channels=hidden_dim,
            ratio=se_ratio
        )

        # Project: 1x1 conv, no activation
        self.linear_conv = ConvModule(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        def _inner_forward(x):
            out = self.expand_conv(x)
            out = self.depthwise_conv(out)
            out = self.se(out)
            out = self.linear_conv(out)
            if self.use_res_connect:
                return x + out
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


@MODELS.register_module()
class SEHSwishMobileNetV2(MobileNetV2):
    """MobileNetV2 backbone with SE + partial H-Swish.

    Matches MobileNetV3 activation pattern:
    - Layer 1-4 (shallow): ReLU6
    - Layer 5-7 (deep): H-Swish

    SE is always present, applied after DWConv on mid_channels.

    Args:
        widen_factor (float): Width multiplier. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages. Default: (7, ).
        frozen_stages (int): Stages to be frozen. Default: -1.
        act_start_layer (int): Blocks with global idx >= this use HSwish; before use ReLU6.
            MobileNetV3 starts HSwish at block index 6 (0-based). Default: 6.
        se_ratio (int): Channel reduction ratio for SE. Default: 4.
        conv_cfg (dict, optional): Config for convolution. Default: None.
        norm_cfg (dict): Config for normalization. Default: dict(type='BN').
        norm_eval (bool): Whether to set norm layers to eval mode. Default: False.
        with_cp (bool): Use checkpoint. Default: False.
    """

    def __init__(self,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 act_start_layer=6,
                 se_ratio=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        self.se_ratio = se_ratio
        # act_start_layer: blocks with global idx >= this use HSwish; before use ReLU6
        # V3 HSwish starts at block index 6 (layer7+), so default=6
        self.act_start_layer = act_start_layer
        self.layer_idx = 0
        super().__init__(
            widen_factor=widen_factor,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6'),
            norm_eval=norm_eval,
            with_cp=with_cp,
            init_cfg=init_cfg
        )

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            # Layer index >= act_start_layer uses HSwish, before uses ReLU6
            if self.layer_idx >= self.act_start_layer:
                act_cfg = dict(type='HSwish')
            else:
                act_cfg = dict(type='ReLU6')
            layers.append(
                SEHSwishInvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                    se_ratio=self.se_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels
            self.layer_idx += 1
        return nn.Sequential(*layers)
