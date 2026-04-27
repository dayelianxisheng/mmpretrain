# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.utils import make_divisible, CoordAtt
from mmpretrain.registry import MODELS
from .mobilenet_v2 import MobileNetV2, InvertedResidual


class CAInvertedResidual(InvertedResidual):
    """InvertedResidual block with CoordAttention for MobileNetV2.

    Inherits from the original InvertedResidual and adds CoordAttention module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 reduction=32,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 init_cfg=None):
        # Initialize the parent InvertedResidual (don't pass ca params to parent)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp,
            init_cfg=init_cfg
        )

        # Add CoordAttention module after the convolution
        self.ca = CoordAtt(
            inp=out_channels,
            oup=out_channels,
            reduction=reduction
        )

    def forward(self, x):
        """Forward function with CoordAttention.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map with CoordAttention.
        """
        def _inner_forward(x):
            # Get the output from convolution (inherited from parent)
            out = self.conv(x)

            # Apply CoordAttention
            out = self.ca(out)

            # Residual connection
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
class CAMobileNetV2(MobileNetV2):
    """MobileNetV2 backbone with CoordAttention.

    This backbone inherits from MobileNetV2 and replaces all InvertedResidual
    blocks with CAInvertedResidual blocks.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        reduction (int): Reduction ratio for CoordAttention. Default: 32.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 reduction=32,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None):
        # Store CA parameter BEFORE calling parent __init__
        # because parent __init__ will call self.make_layer()
        self.reduction = reduction

        # Initialize parent MobileNetV2 (don't pass ca params)
        super().__init__(
            widen_factor=widen_factor,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            with_cp=with_cp,
            init_cfg=init_cfg
        )

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack CAInvertedResidual blocks to build a layer for CA-MobileNetV2.

        This overrides the parent's make_layer to use CAInvertedResidual
        instead of InvertedResidual.

        Args:
            out_channels (int): Output channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.

        Returns:
            nn.Sequential: A layer composed of multiple CAInvertedResidual blocks.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                CAInvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                    reduction=self.reduction,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
