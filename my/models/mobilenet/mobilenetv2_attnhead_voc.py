_base_ = [
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        out_indices=(7, ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone'
        )
    ),
    neck=dict(
        type='AttentionPoolingNeck',
        in_channels=1280,
        attn_mode='ema',    # 'se' | 'spatial' | 'cbam' | 'none'
        reduction=4,
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,
        topk=1,
        loss=dict(
            type='AsymmetricLoss',
            gamma_pos=0,
            gamma_neg=4,
            loss_weight=1.0
        )
    )
)

custom_hooks = [
    dict(
        type='EMAHook',
        momentum=0.0002,
        priority='ABOVE_NORMAL'
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='multi-label/mAP',
        rule='greater'
    ),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=15,
        monitor='multi-label/mAP',
        rule='greater'
    )
)
randomness = dict(seed=42, deterministic=False)