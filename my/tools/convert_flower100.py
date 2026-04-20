# Copyright (c) OpenMMLab. All rights reserved.
"""Convert Flower100 dataset to MMPretrain format."""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random
import shutil


def convert_flower100(sample_ratio=1.0, train_val_ratio=0.9):
    """
    将 Flower100 数据集转换为 MMPretrain 格式

    Args:
        sample_ratio: 数据采样比例，1.0 表示使用全部数据，0.1 表示使用 10% 数据
        train_val_ratio: 训练集划分比例，0.9 表示 90% 训练，10% 验证
    """
    # 路径配置
    dataset_root = Path('E:/dataset/flower100')

    # 根据参数设置输出目录（包含比例信息）
    ratio_str = f"r{int(sample_ratio*100)}_tv{int(train_val_ratio*100)}"
    output_root = Path(f'E:/dataset/flower100_mmp_{ratio_str}')

    print(f"采样比例: {sample_ratio*100:.0f}%")
    print(f"训练/验证比例: {train_val_ratio:.1f}/{1-train_val_ratio:.1f}")
    print(f"输出目录: {output_root}")

    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'train').mkdir(exist_ok=True)
    (output_root / 'val').mkdir(exist_ok=True)
    (output_root / 'meta').mkdir(exist_ok=True)

    # 读取标注文件
    labels_df = pd.read_csv(dataset_root / 'train_labels.csv')

    # 获取唯一的category_id并排序，建立映射
    unique_ids = sorted(labels_df['category_id'].unique())
    print(f"原始类别数: {len(unique_ids)}")
    print(f"类别ID范围: {unique_ids[0]} - {unique_ids[-1]}")

    # 建立旧ID到新ID(0-99)的映射
    id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

    # 打印类别映射信息
    print("\n类别映射示例 (前10个):")
    for old_id in unique_ids[:10]:
        chinese_name = labels_df[labels_df['category_id'] == old_id]['chinese_name'].iloc[0]
        print(f"  {old_id} -> {id_to_new_id[old_id]}: {chinese_name}")

    # 按类别分组
    class_samples = defaultdict(list)
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        old_id = row['category_id']
        new_id = id_to_new_id[old_id]
        class_samples[new_id].append((filename, new_id, idx))

    # 划分训练集和验证集（标注文件只写文件名，不包含目录）
    train_lines = []
    val_lines = []

    for class_id, samples in class_samples.items():
        for i, (filename, label, original_idx) in enumerate(samples):
            # 按 train_val_ratio 划分
            if i % int(1 / (1 - train_val_ratio)) == 0:
                val_lines.append(f"{filename} {label}\n")
            else:
                train_lines.append(f"{filename} {label}\n")

    # 按比例采样训练集
    if sample_ratio < 1.0:
        random.seed(42)
        sample_size = int(len(train_lines) * sample_ratio)
        train_lines = random.sample(train_lines, sample_size)
        print(f"\n采样后训练集: {len(train_lines)} 张图片")

    # 写入标注文件
    with open(output_root / 'meta' / 'train.txt', 'w') as f:
        f.writelines(train_lines)

    with open(output_root / 'meta' / 'val.txt', 'w') as f:
        f.writelines(val_lines)

    # 创建类别名称文件
    class_names = []
    for old_id in unique_ids:
        chinese_name = labels_df[labels_df['category_id'] == old_id]['chinese_name'].iloc[0]
        class_names.append(chinese_name)

    with open(output_root / 'meta' / 'classes.txt', 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # 复制图片到输出目录
    print("\n复制图片...")
    train_src = dataset_root / 'train'

    # 复制训练图片
    for i, line in enumerate(train_lines):
        filename = line.strip().split()[0]  # 直接获取文件名
        src = train_src / filename
        dst = output_root / 'train' / filename
        if not dst.exists():
            shutil.copy(src, dst)

        if (i + 1) % 2000 == 0:
            print(f"  已处理 {i+1}/{len(train_lines)} 训练图片")

    # 复制验证图片
    for i, line in enumerate(val_lines):
        filename = line.strip().split()[0]  # 直接获取文件名
        src = train_src / filename
        dst = output_root / 'val' / filename
        if not dst.exists():
            shutil.copy(src, dst)

    print(f"\n转换完成!")
    print(f"  训练集: {len(train_lines)} 张图片")
    print(f"  验证集: {len(val_lines)} 张图片")
    print(f"  类别数: {len(unique_ids)}")
    print(f"\n输出目录: {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Flower100 dataset to MMPretrain format')
    parser.add_argument(
        '--ratio', type=float, default=0.5,
        help='数据采样比例，如 0.5 表示使用 50%% 数据')
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='训练集划分比例，如 0.8 表示 80%% 训练，20%% 验证')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    convert_flower100(sample_ratio=args.ratio, train_val_ratio=args.train_ratio)
