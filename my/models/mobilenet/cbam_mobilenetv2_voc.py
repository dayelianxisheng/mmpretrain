# CBAM-MobileNetV2 VOC2012 训练配置
# 改进：在 MobileNetV2 的每个 InvertedResidual 后添加 CBAM 注意力模块（通道 + 空间注意力）
# 训练策略：冻结解冻 + EMA

_base_ = [
    '../../../configs/_base_/models/mobilenet_v2_1x.py',
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adam_bs64.py'
]

# CBAM-MobileNetV2 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='CBAMMobileNetV2',
        widen_factor=1.0,
        cbam_ratio=16,          # CBAM 通道注意力的压缩比
        kernel_size=3,           # CBAM 空间注意力的卷积核大小
        out_indices=(7, ),       # 输出最后一层
        frozen_stages=-1,       # 不在 backbone 内部冻结，由 FreezeLayersHook 控制
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone'
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    data_preprocessor=dict(
        num_classes=20,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        to_onehot=True,
        batch_augments=dict(
            augments=[
                dict(type='Mixup', alpha=0.2),
                dict(type='CutMix', alpha=1.0),
            ],
        ),
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,  # MobileNetV2 输出通道
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

# 冻结解冻策略：前5个epoch冻结backbone只训练head和CBAM，解冻后用不同学习率
custom_hooks = [
    dict(
        type='FreezeLayersHook',
        freeze_layers=['backbone'],     # 冻结整个 backbone
        freeze_epochs=5,               # 5个epoch后解冻
        unfreeze_backbone=True,
        priority='ABOVE_NORMAL'
    ),
    dict(
        type='EMAHook',
        momentum=0.0002,
        priority='ABOVE_NORMAL'
    ),
]

# 学习率调度：冻结期小lr，解冻后cosine恢复
param_scheduler = [
    # 前5个epoch冻结期用小学习率
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=5),
    # 解冻后用常规学习率+余弦退火
    dict(type='CosineAnnealingLR', T_max=95, eta_min=5e-6, by_epoch=True, begin=5, end=100),
]

# 优化器配置：参数分组，backbone用低学习率，head和CBAM用正常学习率
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.0001,
    ),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            # backbone 用 0.1x 学习率
            'backbone': dict(lr_mult=0.1),
            # head 用正常学习率
            'head': dict(lr_mult=1.0),
        }
    )
)

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=15,
        monitor='multi-label/mAP'
    )
)
