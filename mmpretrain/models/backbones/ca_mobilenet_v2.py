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
    """InvertedResidual block with CoordAttention applied between
    depthwise and pointwise convolution.

    Structure:
    PW(expand) -> DW(3x3) -> [CA] -> PW(linear projection)
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
        # 先初始化父类，构建完整的 self.conv
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

        # 关键：self.conv 是一个 nn.Sequential，我们需要知道它的内部结构
        # 默认结构: [ConvModule(1x1 expand), ConvModule(3x3 DW), ConvModule(1x1 project)]
        # 我们需要在索引 1（DW）之后，索引 2（project）之前插入 CA

        # 注意：CA 的输入/输出通道数应该是 expand 后的通道数
        # expand 后的通道数 = in_channels * expand_ratio
        expanded_channels = in_channels * expand_ratio

        # 使 expanded_channels 能被 8 整除（MobileNetV2 的 make_divisible 逻辑）
        # 这里简化处理，实际可能需要调用 make_divisible
        self.ca = CoordAtt(
            inp=expanded_channels,
            oup=expanded_channels,
            reduction=reduction
        )

    def forward(self, x):
        """Forward with CA placed between DW and PW."""

        def _inner_forward(x):
            # 手动执行 self.conv 的子模块，而不是直接调用 self.conv
            # self.conv 是 nn.Sequential，包含：
            #   [0]: PW expand (1x1)
            #   [1]: DW (3x3)
            #   [2]: PW project (1x1)

            if self.expand_ratio != 1:
                # 有 expand 层
                # Step 1: Pointwise expand
                hidden = self.conv[0](x)
                # Step 2: Depthwise + CA
                hidden = self.conv[1](hidden)
                hidden = self.ca(hidden)  # ★ CA 插入在这里
                # Step 3: Pointwise project
                out = self.conv[2](hidden)
            else:
                # expand_ratio=1 时 self.conv 只有两个子模块：[DW, PW]
                hidden = self.conv[0](x)
                hidden = self.ca(hidden)  # ★ CA 插入在这里
                out = self.conv[1](hidden)

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
