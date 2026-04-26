# SE-MobileNetV2 VOC2012 训练配置
# 改进：在 MobileNetV2 的每个 InvertedResidual 后添加 SE 注意力模块

_base_ = [
    '../../../configs/_base_/models/mobilenet_v2_1x.py',
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adam_bs64.py'
]

# SE-MobileNetV2 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEMobileNetV2',
        widen_factor=1.0,
        se_ratio=16,           # SE 模块的通道压缩比
        out_indices=(7, ),    # 输出最后一层
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

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=10,
        monitor='multi-label/mAP'
    )
)
