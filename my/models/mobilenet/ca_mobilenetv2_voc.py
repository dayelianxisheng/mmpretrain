_base_ = [
    '../../../configs/_base_/models/mobilenet_v2_1x.py',
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py'
]

# CA-MobileNetV2 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='CAMobileNetV2',
        widen_factor=1.0,
        reduction=32,           # CoordAttention 的通道压缩比
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
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,  # MobileNetV2 输出通道
        topk=3,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

# # 冻结解冻策略：前5个epoch冻结backbone只训练head和CA，解冻后用不同学习率
# custom_hooks = [
#     dict(
#         type='FreezeLayersHook',
#         freeze_layers=['backbone'],     # 冻结整个 backbone
#         freeze_epochs=5,               # 5个epoch后解冻
#         unfreeze_backbone=True,
#         priority='ABOVE_NORMAL'
#     ),
#     # dict(
#     #     type='EMAHook',
#     #     momentum=0.0002,
#     #     priority='ABOVE_NORMAL'
#     # ),
# ]


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

# 关闭确定性算法，避免 CuBLAS 非确定性警告
randomness = dict(seed=42, deterministic=False)
