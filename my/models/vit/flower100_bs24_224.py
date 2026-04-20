# 数据集类型
dataset_type = 'CustomDataset'

# 数据预处理配置
data_preprocessor = dict(
    num_classes=100,  # 100 类
    # RGB 格式归一化参数
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # 图像从 BGR 转换为 RGB
    to_rgb=True,
)

# 训练数据增强 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

# 测试 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=512, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# 训练数据加载器
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=r"E:\dataset\flower100_mmp_r50_tv80",
        ann_file=r"meta\train.txt",
        data_prefix='train',
        with_label=True,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=24,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=r"E:\dataset\flower100_mmp_r50_tv80",
        ann_file=r'meta\val.txt',
        data_prefix='val',
        with_label=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# 评估器
val_evaluator = dict(type='Accuracy', topk=(1,5))

# 测试数据加载器（与验证集相同）
test_dataloader = val_dataloader
test_evaluator = val_evaluator

