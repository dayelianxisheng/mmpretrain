# optimizer (AdamW)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
)

# learning policy (CosineAnnealing + warmup)
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=85, begin=5, end=90, convert_to_iter_based=True),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=90, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=32)
