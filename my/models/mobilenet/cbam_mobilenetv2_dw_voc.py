# CBAM-MobileNetV2-DWConv VOC2012 训练配置
# 改进点：
# 1. CBAM 放在 DW Conv 之后（特征校准后再投影回高维）
# 2. 给 CBAM 模块单独的高学习率 (.cbam: lr_mult=1.0)

_base_ = [
    '../../../configs/_base_/models/mobilenet_v2_1x.py',
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adam_bs64.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='CBAMMobileNetV2DWConv',  # 新 backbone：CBAM 在 DW Conv 之后
        widen_factor=1.0,
        cbam_ratio=16,
        kernel_size=3,
        out_indices=(7, ),
        frozen_stages=-1,
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
        in_channels=1280,
        topk=1,
        loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        )
    )
)

# 冻结解冻 + EMA
custom_hooks = [
    dict(
        type='FreezeLayersHook',
        freeze_layers=['backbone'],
        freeze_epochs=5,
        unfreeze_backbone=True,
        priority='ABOVE_NORMAL'
    ),
    dict(
        type='EMAHook',
        momentum=0.0002,
        priority='ABOVE_NORMAL'
    ),
]

# 学习率调度
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=95, eta_min=5e-6, by_epoch=True, begin=5, end=100),
]

# 优化器配置：CBAM 模块单独高学习率
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
            # backbone 主干用低学习率
            'backbone': dict(lr_mult=0.1),
            # CBAM 模块用正常学习率（改进2）
            '.cbam': dict(lr_mult=1.0),
            # head 用正常学习率
            'head': dict(lr_mult=1.0),
        }
    )
)

# 保存 checkpoint 配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='multi-label/mAP',
        rule='greater'
    ),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=15,
        monitor='multi-label/mAP',
        rule='greater'
    )
)

# 关闭确定性算法
randomness = dict(seed=42, deterministic=False)
