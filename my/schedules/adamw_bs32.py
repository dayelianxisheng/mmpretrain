
# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=1.0),
)

# 学习率调度
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=10, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, begin=10, end=100, convert_to_iter_based=True),
]

# 训练配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

val_cfg = dict()
test_cfg = dict()

# 自动学习率缩放
auto_scale_lr = dict(base_batch_size=32)
