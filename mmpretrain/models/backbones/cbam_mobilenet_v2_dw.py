import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule

from mmpretrain.models.utils import make_divisible, cbam
from mmpretrain.registry import MODELS
from .mobilenet_v2 import MobileNetV2, InvertedResidual


class CBAMAfterDWConvInvertedResidual(InvertedResidual):
    """InvertedResidual block with CBAM attention placed after DW Conv.

    CBAM is placed after depthwise convolution for better feature calibration
    before the final linear projection (1x1 conv).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 cbam_ratio=16,
                 kernel_size=3,
                 cbam_in_planes=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 init_cfg=None):
        # 保存 expand_ratio（父类不保存）
        self.expand_ratio = expand_ratio

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
        # CBAM after depthwise convolution (before final 1x1 conv)
        # cbam_in_planes 是 DW Conv 输出的通道数
        in_planes = cbam_in_planes if cbam_in_planes else out_channels
        self.cbam = cbam(
            in_planes=in_planes,
            ratio=cbam_ratio,
            kernel_size=kernel_size
        )

    def forward(self, x):
        def _inner_forward(x):
            # InvertedResidual structure:
            # conv: [expand_conv, dw_conv, project_conv]

            if self.expand_ratio != 1:
                # Has expansion layer
                out = self.conv[0](x)  # expand_conv
                out = self.conv[1](out)  # dw_conv
            else:
                # No expansion, start from dw_conv
                out = self.conv[0](x)  # dw_conv

            # Apply CBAM after DW Conv (before final 1x1 projection)
            out = self.cbam(out)

            # Final 1x1 projection
            if self.expand_ratio != 1:
                out = self.conv[2](out)  # project_conv
            else:
                out = self.conv[1](out)  # project_conv

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
class CBAMMobileNetV2DWConv(MobileNetV2):
    """MobileNetV2 backbone with CBAM attention after DW Conv.

    CBAM is placed after depthwise convolution, before the final linear
    projection. This helps calibrate features before they're projected back
    to high-dimensional space.
    """

    def __init__(self,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 cbam_ratio=16,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None):
        self.cbam_ratio = cbam_ratio
        self.kernel_size = kernel_size
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
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            # 计算 DW Conv 后的通道数（用于 CBAM）
            hidden_dim = int(round(self.in_channels * expand_ratio))
            layers.append(
                CBAMAfterDWConvInvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                    cbam_ratio=self.cbam_ratio,
                    kernel_size=self.kernel_size,
                    cbam_in_planes=hidden_dim,  # DW Conv 输出通道数
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
