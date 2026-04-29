# MobileNetV2 + MobileViT Block (消融实验)
# 在 MobileNetV2 的最后 stage 替换为 MobileVitBlock，引入全局建模能力

_base_ = [
    '../../datasets/voc/voc_bs64.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs64.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileVitMobileNetV2',
        widen_factor=1.0,
        out_indices=(5, ),          # 5 = output after layer5 (160ch -> 1280ch conv)
        frozen_stages=-1,
        mobilevit_layers=(4, ),     # stage-4 (最后一个) 替换为 MobileVitBlock
        transformer_dim=96,         # 保持轻量
        ffn_dim=192,
        num_transformer_blocks=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone'
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=1280,
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

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
