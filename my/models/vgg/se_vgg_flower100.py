_base_ = [
    '../../../configs/_base_/models/se_vgg.py',  # 改用带BN的VGG16
    'flower100_bs24_224.py',
    '../../../configs/_base_/schedules/adamw_bs224.py',
    '../../../configs/_base_/default_runtime.py'
]

# 训练参数配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, begin=5, end=100, convert_to_iter_based=True)
]

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0005,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8
    ),
    clip_grad=dict(max_norm=1.0),  # 添加梯度裁剪
    paramwise_cfg=dict(norm_decay_mult=0.0)  # 不对BN层进行权重衰减
)

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2)
)

# 修改类别数为100
model = dict(backbone=dict(num_classes=100))
data_preprocessor = dict(num_classes=100)

auto_scale_lr = dict(base_batch_size=224)