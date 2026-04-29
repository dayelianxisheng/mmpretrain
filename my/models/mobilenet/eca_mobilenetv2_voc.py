# ECA-MobileNetV2 VOC2012 训练配置
# ECA: Efficient Channel Attention，无降维的通道注意力，适合轻量化网络

_base_ = [
    '../../../configs/_base_/models/mobilenet_v2_1x.py',
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adam_bs64.py'
]

# ECA-MobileNetV2 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ECAMobileNetV2',
        widen_factor=1.0,
        eca_k_size=3,           # ECA 1D 卷积核大小
        out_indices=(7, ),       # 输出最后一层
        frozen_stages=-1,       # 不冻结，由 FreezeLayersHook 控制
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
        # 多标签分类不使用 Mixup/CutMix
    ),
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
