# MobileNetV3-Large VOC2012 训练配置
# MobileNetV3: 融合 SE 注意力 + H-Swish 激活函数，轻量化高性能

_base_ = [
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py',
]

# MobileNetV3-Large 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileNetV3',
        arch='large',
        out_indices=(16, ),
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/mobilenet_v2/mobilenet_v3_large-3ea3c186.pth',
            prefix='backbone'
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=960,
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
