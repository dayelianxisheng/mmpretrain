# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEAlexNet',
        se_ratio=16
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
