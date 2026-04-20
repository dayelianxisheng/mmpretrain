# AdamW优化器配置 - 适合CNN网络
# 适用于中小规模数据集和传统CNN模型

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8
    ),
    # 可选：梯度裁剪防止梯度爆炸
    clip_grad=dict(max_norm=1.0)
)

# learning policy - 适合100 epoch训练
warmup_epochs = 3  # 适中的warmup长度
param_scheduler = [
    # 简短的warmup
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        convert_to_iter_based=True
    ),
    # 主训练阶段：余弦退火
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=True,
        begin=warmup_epochs,
        end=100  # 假设��练100个epoch
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# 对于小数据集，通常不需要自动缩放学习率
auto_scale_lr = dict(base_batch_size=224)
