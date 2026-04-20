_base_ = [
    '../../../configs/_base_/models/vit-base-p16.py',
    'flower100_bs24_224.py',
    '../../../configs/_base_/schedules/adamw_bs224.py',
    '../../../configs/_base_/default_runtime.py'
]

# 迁移学习：加载预训练权重
model = dict(
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        arch='base',
        img_size=384,
        patch_size=16,
        out_indices=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./my/vit/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth',
            map_location='cpu',
            prefix='backbone'
        )
    ),
    head=dict(num_classes=100)
)
data_preprocessor = dict(num_classes=100)

# 训练参数配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, begin=5, end=100, convert_to_iter_based=True)
]

# 优化器配置 - 参数分组，不同层不同学习率
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0005,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8
    ),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        # 预训练层用低学习率
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # 预训练层学习率降低10倍
        }
    )
)

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2)
)

auto_scale_lr = dict(base_batch_size=224)
