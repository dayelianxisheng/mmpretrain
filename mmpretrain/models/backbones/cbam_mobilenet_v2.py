import torch.nn as nn
import torch.utils.checkpoint as cp

from mmpretrain.models.utils import make_divisible, cbam
from mmpretrain.registry import MODELS
from .mobilenet_v2 import MobileNetV2, InvertedResidual


class CBAMInvertedResidual(InvertedResidual):
    """InvertedResidual block with CBAM applied between
    depthwise and pointwise convolution.

    Structure:
    PW(expand) -> DW(3x3) -> [CBAM] -> PW(linear projection)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 cbam_ratio=16,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 init_cfg=None):
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

        # 从 DW 层取真实的 expand 后通道数（确保 make_divisible 对齐）
        dw_layer = self.conv[-2]
        expanded_channels = dw_layer.conv.in_channels

        # CBAM 作用在 expand 后的通道数上（不再是 out_channels）
        self.cbam = cbam(
            in_planes=expanded_channels,
            ratio=cbam_ratio,
            kernel_size=kernel_size
        )

    def forward(self, x):
        """Forward with CBAM placed between DW and PW."""

        def _inner_forward(x):
            # 手动执行 self.conv 的子模块
            # self.conv 结构：
            #   expand_ratio != 1: [PW expand, DW 3x3, PW project]
            #   expand_ratio == 1: [DW 3x3, PW project]

            if self.expand_ratio != 1:
                # 有 expand 层
                assert len(self.conv) == 3, \
                    f"Expected 3 sub-modules, but got {len(self.conv)}"
                # Step 1: Pointwise expand (1x1)
                hidden = self.conv[0](x)
                # Step 2: Depthwise (3x3) + CBAM
                hidden = self.conv[1](hidden)
                hidden = self.cbam(hidden)  # ★ CBAM 插入在这里
                # Step 3: Pointwise project (1x1)
                out = self.conv[2](hidden)
            else:
                # expand_ratio=1 时只有两个子模块
                assert len(self.conv) == 2, \
                    f"Expected 2 sub-modules, but got {len(self.conv)}"
                # Step 1: Depthwise (3x3) + CBAM
                hidden = self.conv[0](x)
                hidden = self.cbam(hidden)  # ★ CBAM 插入在这里
                # Step 2: Pointwise project (1x1)
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
class CBAMMobileNetV2(MobileNetV2):
    """MobileNetV2 backbone with CBAM.

    This backbone inherits from MobileNetV2 and replaces all InvertedResidual
    blocks with CBAMInvertedResidual blocks.

    Args:
        widen_factor (float): Width multiplier. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen. Default: -1.
        cbam_ratio (int): Reduction ratio for CBAM channel attention.
            Default: 16.
        kernel_size (int): Kernel size for CBAM spatial attention.
            Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        norm_eval (bool): Whether to set norm layers to eval mode.
        with_cp (bool): Use checkpoint or not.
    """

    def __init__(self,
                 widen_factor=1.,
                 out_indices=(7,),
                 frozen_stages=-1,
                 cbam_ratio=16,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None):
        # 存储 CBAM 参数，父类初始化时会调用 self.make_layer
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
        """Stack CBAMInvertedResidual blocks to build a layer.

        Overrides parent's make_layer to use CBAMInvertedResidual.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                CBAMInvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                    cbam_ratio=self.cbam_ratio,
                    kernel_size=self.kernel_size,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)