# SGD 优化器配置 - batch_size=64
# 适合：最终模型、比赛刷榜、追求最佳泛化

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.0001,           # 降低初始学习率，避免震荡
        momentum=0.9,
        weight_decay=0.0001,
    ),
    clip_grad=dict(max_norm=5.0),
)

param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=10),
    dict(type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1),
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=64)
