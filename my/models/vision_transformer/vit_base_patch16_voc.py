# Vision Transformer (ViT-Base/P16) VOC2012 训练配置

_base_ = [
    '../../../configs/_base_/models/vit-base-p16.py',
    '../../datasets/voc/voc_bs32.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py'
]

# 多标签分类模型配置
model = dict(
    backbone=dict(
        init_cfg=[dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/vit/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth',
            prefix='backbone'
        )]
    ),
    head=dict(
        _delete_=True,
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=768,  # ViT-Base 输出通道
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    )
)

# ViT 需要更大的学习率和更长的训练时间
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0003,  # ViT 通常用稍大的学习率
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=1.0),
)

# 每10个epoch保存一次权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=15,  # ViT 收敛较慢，增加耐心值
        monitor='multi-label/mAP'
    )
)
