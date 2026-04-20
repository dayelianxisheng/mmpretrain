# AlexNet Flower100 训练配置

_base_ = [
    '../../datasets/flower100/flower100_bs32.py',
    '../../../../configs/_base_/schedules/imagenet_bs256.py',
    '../../../../configs/_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(type='AlexNet'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)

# 训练参数配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, begin=5, end=100, convert_to_iter_based=True)
]

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2)
)
