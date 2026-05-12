# EMA-MobileNetV2 VOC2012 训练配置
# 轻量化改进：仅添加 EMA 权重平均，零参数增量
# 训练策略：冻结解冻 + EMA

_base_ = [
    '../../../configs/_base_/models/mobilenet_v2_1x.py',
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adam_bs64.py',
]

# MobileNetV2 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
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
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,
        topk=1,
        loss=dict(
            type='AsymmetricLoss',
            gamma_pos=0,
            gamma_neg=4,
            loss_weight=1.0
        )
    )
)
custom_hooks = [
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

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.0001,
    ),
    clip_grad=dict(max_norm=1.0),
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
