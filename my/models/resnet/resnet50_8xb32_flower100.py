# Flower100 数据集 ResNet-50 训练配置

_base_ = [
    '../../../../configs/_base_/models/resnet50.py',
    '../../datasets/flower100/flower100_bs32.py',
    '../../../../configs/_base_/schedules/imagenet_bs256.py',
    '../../../../configs/_base_/default_runtime.py'
]

# 训练参数配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, begin=5, end=100, convert_to_iter_based=True)
]

# 优化器配置（单GPU，batch_size=32时 lr=0.1）
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
)

# 每10个epoch保存一次权重，最后一个epoch总是保存
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2)
)


