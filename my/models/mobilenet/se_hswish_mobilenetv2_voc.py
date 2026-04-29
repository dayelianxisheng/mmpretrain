# MobileNetV2 with SE + H-Swish (消融实验)
# 改进：融合 MobileNetV3 的两大核心改进

_base_ = [
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEHSwishMobileNetV2',
        widen_factor=1.0,
        se_ratio=4,              # SE 压缩比，与 MobileNetV3 一致
        out_indices=(7, ),
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone'
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

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
