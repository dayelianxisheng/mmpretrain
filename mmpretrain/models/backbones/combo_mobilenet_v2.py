# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.utils import (make_divisible, SELayer, cbam,
                                     CoordAtt)
from mmpretrain.registry import MODELS
from .mobilenet_v2 import MobileNetV2, InvertedResidual


class ComboInvertedResidual(InvertedResidual):
    """InvertedResidual block with SE + CBAM + CA attention for MobileNetV2.

    Inherits from the original InvertedResidual and adds triple attention:
    SE -> CBAM -> CA
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 se_ratio=16,
                 cbam_ratio=16,
                 cbam_kernel=3,
                 ca_reduction=32,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 init_cfg=None):
        # Initialize the parent InvertedResidual
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

        # Add triple attention modules after the convolution
        self.se = SELayer(
            channels=out_channels,
            ratio=se_ratio
        )
        self.cbam = cbam(
            in_planes=out_channels,
            ratio=cbam_ratio,
            kernel_size=cbam_kernel
        )
        self.ca = CoordAtt(
            inp=out_channels,
            oup=out_channels,
            reduction=ca_reduction
        )

    def forward(self, x):
        """Forward function with triple attention.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map with SE -> CBAM -> CA attention.
        """
        def _inner_forward(x):
            # Get the output from convolution (inherited from parent)
            out = self.conv(x)

            # Apply triple attention: SE -> CBAM -> CA
            out = self.se(out)
            out = self.cbam(out)
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
class ComboMobileNetV2(MobileNetV2):
    """MobileNetV2 backbone with SE + CBAM + CA triple attention.

    This backbone inherits from MobileNetV2 and replaces all InvertedResidual
    blocks with ComboInvertedResidual blocks (with SE, CBAM, and CA).

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        se_ratio (int): Reduction ratio for SE module. Default: 16.
        cbam_ratio (int): Reduction ratio for CBAM channel attention.
            Default: 16.
        cbam_kernel (int): Kernel size for CBAM spatial attention.
            Default: 3.
        ca_reduction (int): Reduction ratio for CoordAttention. Default: 32.
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
                 se_ratio=16,
                 cbam_ratio=16,
                 cbam_kernel=3,
                 ca_reduction=32,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None):
        # Store triple attention parameters BEFORE calling parent __init__
        # because parent __init__ will call self.make_layer()
        self.se_ratio = se_ratio
        self.cbam_ratio = cbam_ratio
        self.cbam_kernel = cbam_kernel
        self.ca_reduction = ca_reduction

        # Initialize parent MobileNetV2
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
        """Stack ComboInvertedResidual blocks to build a layer.

        This overrides the parent's make_layer to use ComboInvertedResidual
        instead of InvertedResidual.

        Args:
            out_channels (int): Output channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.

        Returns:
            nn.Sequential: A layer composed of multiple ComboInvertedResidual blocks.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                ComboInvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                    se_ratio=self.se_ratio,
                    cbam_ratio=self.cbam_ratio,
                    cbam_kernel=self.cbam_kernel,
                    ca_reduction=self.ca_reduction,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
