# VOC2012 数据集 ResNet-50 训练配置

_base_ = [
    '../../../configs/_base_/models/resnet50.py',
    '../../datasets/voc/voc_bs32.py',
    '../../../configs/_base_/default_runtime.py',
    '../../schedules/adamw_bs96.py'
]

# 多标签分类模型配置
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='my/checkpoints/backbone/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone'
        )
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=20,
        in_channels=2048,
        topk=1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

# 训练参数配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# 学习率调度器
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, begin=5, end=100, convert_to_iter_based=True)
]

# 每10个epoch保存一次权重，最后一个epoch总是保存
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2),
    # 早停：如果验证集 mAP 连续 10 个 epoch 没有提升则停止训练
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=10,
        monitor='multi-label/mAP'  # 监控 mAP 指标
    )
)
