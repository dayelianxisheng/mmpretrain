# CBAM-MobileNetV2 VOC2012 训练配置
# 改进：在 MobileNetV2 的每个 InvertedResidual 后添加 CBAM 注意力模块（通道 + 空间注意力）
# 训练策略：冻结解冻 + EMA

_base_ = [
    '../../../configs/_base_/models/mobilenet_v2_1x.py',
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py'
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
        to_onehot=False,
        # 多标签分类不使用 Mixup/CutMix，会导致标签混淆和loss震荡
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,  # MobileNetV2 输出通道
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

# 保存 checkpoint 配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='multi-label/mAP',  # 保存最佳 mAP 模型
        rule='greater'  # mAP 越高越好
    ),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=20,
        monitor='multi-label/mAP',
        rule='greater'
    )
)
