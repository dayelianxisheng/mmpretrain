# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.models.utils import make_divisible
from mmpretrain.registry import MODELS
from .mobilenet_v2 import InvertedResidual
from .vision_transformer import TransformerEncoderLayer


class MobileVitBlock(BaseModule):
    """MobileViT block.

    A lightweight transformer block that provides global modeling capability
    by unfolding feature maps into patches, applying transformer encoder,
    and folding back.

    Args:
        in_channels (int): Number of input channels.
        transformer_dim (int): Dimension for transformer tokens.
        ffn_dim (int): FFN hidden dimension.
        out_channels (int): Number of output channels.
        conv_ksize (int): Kernel size for local representation. Default: 3.
        conv_cfg (dict, optional): Conv config. Default: None.
        norm_cfg (dict): Norm config. Default: dict(type='BN').
        act_cfg (dict): Act config. Default: dict(type='HSwish').
        num_transformer_blocks (int): Number of transformer blocks. Default: 1.
        patch_size (int): Patch size for unfold/fold. Default: 2.
        num_heads (int): Number of attention heads. Default: 4.
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout. Default: 0.
        drop_path_rate (float): Drop path rate. Default: 0.
        no_fusion (bool): Skip fusion layer. Default: False.
        transformer_norm_cfg (dict): Norm config for transformer. Default: dict(type='LN').
        with_cp (bool): Use checkpoint. Default: False.
    """

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            ffn_dim: int,
            out_channels: int,
            conv_ksize: int = 3,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish'),
            num_transformer_blocks: int = 1,
            patch_size: int = 2,
            num_heads: int = 4,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            no_fusion: bool = False,
            transformer_norm_cfg=dict(type='LN'),
            with_cp: bool = False,
            init_cfg=None,
    ):
        super(MobileVitBlock, self).__init__(init_cfg)
        self.with_cp = with_cp

        # Local representation: 3x3 conv + 1x1 conv
        self.local_rep = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                padding=int((conv_ksize - 1) / 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=transformer_dim,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None),
        )

        # Global representation via transformer
        global_rep = [
            TransformerEncoderLayer(
                embed_dims=transformer_dim,
                num_heads=num_heads,
                feedforward_channels=ffn_dim,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=True,
                act_cfg=dict(type='GELU'),
                norm_cfg=transformer_norm_cfg)
            for _ in range(num_transformer_blocks)
        ]
        global_rep.append(
            build_norm_layer(transformer_norm_cfg, transformer_dim)[1])
        self.global_rep = nn.Sequential(*global_rep)

        # Projection back to conv features
        self.conv_proj = ConvModule(
            in_channels=transformer_dim,
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Fusion: shortcut + projected features
        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = ConvModule(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                kernel_size=conv_ksize,
                padding=int((conv_ksize - 1) / 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.patch_size = (patch_size, patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.local_rep(x)

        # Unfold: [B, C, H, W] -> patches
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(
            W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w
        num_patches = num_patch_h * num_patch_w
        interpolate = False
        if new_h != H or new_w != W:
            x = nn.functional.interpolate(
                x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True

        # [B, C, H, W] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w,
                      patch_w).transpose(1, 2)
        # -> [BP, N, C] where P = p_h*p_w, N = n_h*n_w
        x = x.reshape(B, C, num_patches,
                      self.patch_area).transpose(1, 3).reshape(
                          B * self.patch_area, num_patches, -1)

        # Global representations
        def _inner_forward(x):
            return self.global_rep(x)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        # Fold: patches -> [B, C, H, W]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w,
                                      patch_h, patch_w)
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h,
                                      num_patch_w * patch_w)
        if interpolate:
            x = nn.functional.interpolate(
                x, size=(H, W), mode='bilinear', align_corners=False)

        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


@MODELS.register_module()
class MobileVitMobileNetV2(BaseModule):
    """MobileNetV2 with MobileViT blocks for global modeling.

    Combines efficient MobileNetV2 local feature extraction with lightweight
    Transformer blocks for global context, targeting improved multi-label
    classification on VOC.

    Args:
        widen_factor (float): Width multiplier. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages. Default: (5, ).
        frozen_stages (int): Freeze stages. Default: -1.
        conv_cfg (dict, optional): Conv config. Default: None.
        norm_cfg (dict): Norm config. Default: dict(type='BN').
        act_cfg (dict): Act config. Default: dict(type='HSwish').
        norm_eval (bool): Set BN to eval mode. Default: False.
        with_cp (bool): Use checkpoint. Default: False.
        mobilevit_layers (list): Which stages (0-based, after stem) use
            MobileVitBlock instead of InvertedResidual.
            E.g. [3, 4] replaces stage-3 and stage-4 with MobileVitBlocks.
            Default: [4] (last stage only).
        transformer_dim (int): Transformer embedding dimension. Default: 96.
        ffn_dim (int): FFN hidden dim. Default: 192.
        num_transformer_blocks (int): Transformer blocks per MobileVitBlock.
            Default: 1.
        init_cfg (dict): Init config.
    """

    # arch: [expand_ratio, out_channels, num_blocks, stride]
    arch_settings = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(self,
                 widen_factor=1.,
                 out_indices=(5, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 norm_eval=False,
                 with_cp=False,
                 mobilevit_layers=(4, ),
                 transformer_dim=96,
                 ffn_dim=192,
                 num_transformer_blocks=1,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1,
                          layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MobileVitMobileNetV2, self).__init__(init_cfg)
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.mobilevit_layers = set(mobilevit_layers)
        self.transformer_dim = transformer_dim
        self.ffn_dim = ffn_dim
        self.num_transformer_blocks = num_transformer_blocks

        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError(f'out_indices must in range(0, 7), got {index}')

        self.in_channels = make_divisible(32 * widen_factor, 8)

        # Stem: 3x3 conv, stride=2
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.layers = []
        self.layer_idx = 0

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)

            layer_name = f'layer{i + 1}'
            layer = self._make_layer(
                in_channels=self.in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio,
                mobilevit_stage=(i in self.mobilevit_layers),
            )
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)
            self.in_channels = out_channels
            self.layer_idx += 1

        # Final 1x1 expansion conv
        if widen_factor > 1.0:
            self.out_channel = int(1280 * widen_factor)
        else:
            self.out_channel = 1280

        self.conv2 = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers.append('conv2')

    def _make_layer(self, in_channels, out_channels, num_blocks, stride,
                    expand_ratio, mobilevit_stage):
        """Build a stage: either standard InvertedResiduals or MobileVitBlock."""
        layers = []
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1

            if mobilevit_stage and i == num_blocks - 1:
                # Replace the last block of this stage with MobileVitBlock
                layers.append(
                    MobileVitBlock(
                        in_channels=in_channels,
                        transformer_dim=self.transformer_dim,
                        ffn_dim=self.ffn_dim,
                        out_channels=out_channels,
                        num_transformer_blocks=self.num_transformer_blocks,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp,
                    ))
            else:
                act = dict(type='HSwish') if self.layer_idx >= 5 else dict(
                    type='ReLU6')
                layers.append(
                    InvertedResidual(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=act,
                        with_cp=self.with_cp))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileVitMobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if hasattr(m, '_ BatchNorm'):
                    m.eval()

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
