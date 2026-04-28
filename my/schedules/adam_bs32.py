optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.0001,
    ),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # 主干 0.1 倍小学习率
            'head': dict(lr_mult=1.0),  # 分类头正常学习率
            'cbam': dict(lr_mult=1.0),  # CBAM 注意力正常学习率
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=5e-6,
        by_epoch=True,
        begin=5,
        end=100
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=32)
