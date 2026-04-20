# EfficientNet-B0 VOC2012 训练配置

_base_ = [
    '../../../configs/_base_/models/efficientnet_b0.py',
    '../../datasets/voc/voc_bs32.py',
    '../../../configs/_base_/schedules/imagenet_bs256.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py'
]

# 多标签分类模型配置
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/efficientnet/efficientnet_b0_3rdparty.pth',
            prefix='backbone'
        )
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,  # EfficientNet-B0 输出通道
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=10,
        monitor='multi-label/mAP'
    )
)
