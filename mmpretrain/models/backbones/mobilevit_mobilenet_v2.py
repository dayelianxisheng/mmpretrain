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


class WeightedMobileVitBlock(BaseModule):
    """InvertedResidual + MobileVitBlock in parallel, weighted fusion.

    Keeps the original InvertedResidual (pretrained weight compatible) and adds
    MobileVitBlock in parallel, fusing outputs: alpha*IR_out + beta*MobileVit_out.
    This preserves pretrained weights while injecting global modeling ability.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the first 3x3 conv. Default: 1.
        expand_ratio (int): Expand ratio for InvertedResidual. Default: 6.
        transformer_dim (int): Transformer embedding dimension. Default: 96.
        ffn_dim (int): FFN hidden dim. Default: 192.
        num_transformer_blocks (int): Transformer blocks. Default: 1.
        conv_cfg (dict, optional): Conv config. Default: None.
        norm_cfg (dict): Norm config. Default: dict(type='BN').
        act_cfg (dict): Act config. Default: dict(type='HSwish').
        mobilevit_weight (float): Weight for MobileVitBlock output. Default: 0.5.
            Final: out = (1 - mobilevit_weight) * IR_out + mobilevit_weight * MobileVit_out
        with_cp (bool): Use checkpoint. Default: False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 expand_ratio: int = 6,
                 transformer_dim: int = 96,
                 ffn_dim: int = 192,
                 num_transformer_blocks: int = 1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 mobilevit_weight: float = 0.5,
                 freeze_ir: bool = True,
                 with_cp: bool = False,
                 init_cfg=None):
        super(WeightedMobileVitBlock, self).__init__(init_cfg)
        self.mobilevit_weight = mobilevit_weight
        self.res_weight = 1.0 - mobilevit_weight
        self.with_cp = with_cp

        # Original InvertedResidual (pretrained compatible)
        self.ir = InvertedResidual(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp)

        # MobileVitBlock in parallel
        self.mobilevit = MobileVitBlock(
            in_channels=in_channels,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            out_channels=out_channels,
            num_transformer_blocks=num_transformer_blocks,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp)

        if freeze_ir:
            for p in self.ir.parameters():
                p.requires_grad = False
            self.ir.eval()

    def forward(self, x):
        ir_out = self.ir(x)
        ir_out = ir_out.detach()

        mobilevit_out = self.mobilevit(x)

        if mobilevit_out.shape[2:] != ir_out.shape[2:]:
            mobilevit_out = nn.functional.interpolate(
                mobilevit_out, size=ir_out.shape[2:],
                mode='bilinear', align_corners=False)

        return self.res_weight * ir_out + self.mobilevit_weight * mobilevit_out


@MODELS.register_module()
class MobileVitMobileNetV2(BaseModule):
    """MobileNetV2 with optional MobileViT blocks for global modeling.

    Supports two modes per layer:
    - 'mobilevit': Replace last block with MobileVitBlock (no pretrained load)
    - 'weighted': Keep InvertedResidual, add MobileVitBlock in parallel with
      weighted fusion (pretrained compatible)

    Args:
        widen_factor (float): Width multiplier. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages. Default: (6, ).
        frozen_stages (int): Freeze stages. Default: -1.
        conv_cfg (dict, optional): Conv config. Default: None.
        norm_cfg (dict): Norm config. Default: dict(type='BN').
        act_cfg (dict): Act config. Default: dict(type='HSwish').
        norm_eval (bool): Set BN to eval mode. Default: False.
        with_cp (bool): Use checkpoint. Default: False.
        mobilevit_layers (list): Stages where MobileVitBlock replaces IR.
            Default: ().
        weighted_layers (dict): Stages where WeightedMobileVitBlock is used.
            Format: {stage_idx: mobilevit_weight (0.0~1.0)}.
            E.g. {4: 0.5} uses weighted fusion at stage-4 with 0.5 weight.
            Default: {4: 0.5}.
        freeze_ir_in_weighted (bool): Freeze InvertedResidual in WeightedMobileVitBlock.
            Keeps pretrained knowledge intact, only trains MobileViT branch.
            Default: True.
        transformer_dim (int): Transformer embedding dim. Default: 96.
        ffn_dim (int): FFN hidden dim. Default: 192.
        num_transformer_blocks (int): Transformer blocks per MobileVitBlock.
            Default: 1.
        init_cfg (dict): Init config.
    """

    # arch: [expand_ratio, out_channels, num_blocks, stride]
    # Must match pretrained MobileNetV2 exactly (layer0-7), with layer8 as WeightedMobileVitBlock
    arch_settings = [
        [1, 16, 1, 1],    # layer0: 16ch, 1 block
        [6, 24, 2, 2],    # layer1: 24ch, 2 blocks
        [6, 32, 3, 2],    # layer2: 32ch, 3 blocks
        [6, 64, 4, 2],    # layer3: 64ch, 4 blocks
        [6, 96, 3, 1],    # layer4: 96ch, 3 blocks
        [6, 160, 4, 2],   # layer5: 160ch, 4 blocks
        [6, 320, 3, 2],   # layer6: 320ch, 3 blocks
        [6, 320, 1, 1],   # layer7: 320ch, 1 block
        [6, 320, 1, 1],   # layer8: 320ch, 1 block -> replaced with WeightedMobileVitBlock
    ]

    def __init__(self,
                 widen_factor=1.,
                 out_indices=(6, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 norm_eval=False,
                 with_cp=False,
                 mobilevit_layers=(),
                 weighted_layers=None,
                 freeze_ir_in_weighted=True,
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
        self.weighted_layers = weighted_layers or {}
        self.transformer_dim = transformer_dim
        self.ffn_dim = ffn_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.freeze_ir_in_weighted = freeze_ir_in_weighted

        for index in out_indices:
            if index not in range(0, 9):
                raise ValueError(f'out_indices must in range(0, 9), got {index}')

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
            mobilevit_stage = i in self.mobilevit_layers
            weighted_stage_info = self.weighted_layers.get(i, None)

            layer = self._make_layer(
                in_channels=self.in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio,
                mobilevit_stage=mobilevit_stage,
                weighted_stage=weighted_stage_info is not None,
                mobilevit_weight=weighted_stage_info if isinstance(weighted_stage_info, float) else 0.5,
                freeze_ir=self.freeze_ir_in_weighted,
            )
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)
            self.in_channels = out_channels

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

    def _make_layer(self, in_channels, out_channels, num_blocks, stride,
                    expand_ratio, mobilevit_stage, weighted_stage,
                    mobilevit_weight=0.5, freeze_ir=True):
        """Build a stage with support for replacement and weighted fusion."""
        layers = []
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1

            if mobilevit_stage and i == num_blocks - 1:
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
            elif weighted_stage and i == num_blocks - 1:
                layers.append(
                    WeightedMobileVitBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        transformer_dim=self.transformer_dim,
                        ffn_dim=self.ffn_dim,
                        num_transformer_blocks=self.num_transformer_blocks,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        mobilevit_weight=mobilevit_weight,
                        freeze_ir=freeze_ir,
                        with_cp=self.with_cp,
                    ))
            else:
                layers.append(
                    InvertedResidual(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
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

    def init_weights(self):
        """Custom init: remap pretrained MobileNetV2 keys before loading."""
        import re
        if self.init_cfg:
            for cfg in (self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]):
                if cfg.get('type') == 'Pretrained' or cfg.get('type') is None:
                    if 'checkpoint' in cfg:
                        import torch
                        state_dict = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
                        if 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        prefix = cfg.get('prefix', '')
                        mapped = {}
                        for k, v in state_dict.items():
                            # Strip prefix (e.g. 'backbone.' -> 'xxx')
                            if prefix and k.startswith(prefix + '.'):
                                k_stripped = k[len(prefix) + 1:]
                            elif prefix and k == prefix:
                                k_stripped = ''
                            else:
                                k_stripped = k
                            k_mapped = re.sub(r'(?<!\.)layer7\.3\.', 'layer8.0.ir.', k_stripped)
                            mapped[k_mapped] = v
                        self.load_state_dict(mapped, strict=False)
                        return
        super().init_weights()

    def train(self, mode=True):
        super(MobileVitMobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if hasattr(m, '_ BatchNorm'):
                    m.eval()

    def _load_from_pretrained_checkpoint(self, checkpoint, prefix='', mapping=None, strict=False, logger=None):
        """Remap pretrained MobileNetV2 keys to match weighted model structure.

        Pretrained: layer4.3 (last IR block) -> New model: layer4.0.ir.
        """
        import re
        mapped_state_dict = {}
        state_dict = checkpoint['state_dict']
        for k, v in state_dict.items():
            # Strip prefix first (e.g. 'backbone.')
            k_stripped = k[len(prefix):] if prefix and k.startswith(prefix) else k
            # Remap layer4.3 -> layer4.0.ir
            k_mapped = re.sub(r'\blayer4\.3\.', 'layer4.0.ir.', k_stripped)
            mapped_state_dict[k_mapped] = v

        self.load_state_dict(mapped_state_dict, strict=strict, logger=logger)

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            # After the last backbone stage, apply 1x1 expansion conv
            if i == len(self.layers) - 1:
                x = self.conv2(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
