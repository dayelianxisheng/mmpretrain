# Swin Transformer Base VOC2012 训练配置

_base_ = [
    '../../../configs/_base_/models/swin_transformer/base_224.py',
    '../../datasets/voc/voc_bs16.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs16.py'
]

# 多标签分类模型配置
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/swin/swin_base_patch4_window7_224-4670dd19.pth',
            prefix='backbone'
        )
    ),
    head=dict(
        _delete_=True,
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1024,  # Swin-Base 输出通道
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    )
)

# Swin Transformer 的优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=15,  # Swin 收敛较慢
        monitor='multi-label/mAP'
    )
)
