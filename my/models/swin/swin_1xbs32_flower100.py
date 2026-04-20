_base_ = [
    '../../../configs/_base_/models/swin_transformer/base_384.py',
    '../../datasets/flower100/flower100_bs32.py',
    '../../../configs/_base_/schedules/imagenet_bs256.py',
    '../../../configs/_base_/default_runtime.py'
]

# 训练参数配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, begin=5, end=100, convert_to_iter_based=True)
]

# 优化器配置（SGD）
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
)

# 每10个epoch保存一次权重，最后一个epoch总是保存
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2)
)

# 修改类别数为100，并设置 pad_small_map=True 避免 window size 问题
model = dict(
    head=dict(num_classes=100),
    backbone=dict(stage_cfgs=dict(block_cfgs=dict(pad_small_map=True)))
)
data_preprocessor = dict(num_classes=100)

# 使用预训练模型（替换为你的权重文件路径）
load_from = "D:\download\swin_base_patch4_window12_384-02c598a4.pth"

# 测试时使用混淆矩阵
test_evaluator = dict(type='ConfusionMatrix')