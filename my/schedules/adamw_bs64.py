optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'head': dict(lr_mult=1.0),
            'cbam': dict(lr_mult=1.0),
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=190,
        eta_min=1e-6,
        by_epoch=True,
        begin=10,
        end=200,
        convert_to_iter_based=True
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(base_batch_size=64)