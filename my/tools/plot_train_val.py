"""
绘制训练过程中的 train loss 和 val mAP 曲线
用法: python my/tools/plot_train_val.py <vis_data/scalars.json路径> [--out 输出图片路径]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scalars(path):
    """加载 scalars.json 文件，按 step 分组"""
    train_data = {}  # step -> {loss, lr, ...}
    val_data = {}    # step -> {mAP, precision, ...}

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            step = obj.get('step')

            # 训练数据：有 epoch 和 iter 字段，有 loss
            if 'epoch' in obj and 'iter' in obj and 'loss' in obj:
                if step not in train_data or train_data[step]['loss'] > obj['loss']:
                    train_data[step] = {
                        'loss': obj['loss'],
                        'lr': obj.get('lr', 0),
                        'epoch': obj.get('epoch'),
                        'iter': obj.get('iter'),
                        'grad_norm': obj.get('grad_norm', None),
                    }

            # 验证数据：有 multi-label 前缀
            elif 'multi-label/mAP' in obj:
                val_data[step] = {
                    'mAP': obj.get('multi-label/mAP'),
                    'precision_micro': obj.get('multi-label/precision_micro'),
                    'recall_micro': obj.get('multi-label/recall_micro'),
                }

    return train_data, val_data


def smooth(data, window=5):
    """移动平均平滑"""
    if len(data) < window:
        return data
    result = np.convolve(data, np.ones(window)/window, mode='valid')
    # 前后各补一些值使长度一致
    pad = window - 1
    result = np.concatenate([
        np.full(pad, result[0]),
        result
    ])
    return result


def plot_loss_and_map(scalars_path, out_path=None):
    """绘制 train loss 和 val mAP 曲线"""
    train_data, val_data = load_scalars(scalars_path)

    if not train_data:
        print("未找到训练数据 (train loss)")
        return
    if not val_data:
        print("未找到验证数据 (val mAP)")
        return

    # 提取数据并排序
    train_steps = sorted(train_data.keys())
    val_steps = sorted(val_data.keys())

    train_loss = [train_data[s]['loss'] for s in train_steps]
    train_epochs = [train_data[s]['epoch'] for s in train_steps]

    val_epochs = [s for s in val_steps]
    val_mAP = [val_data[s]['mAP'] for s in val_steps]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 上图: Train Loss
    color1 = '#e74c3c'
    ax1.plot(train_steps, train_loss, color=color1, alpha=0.4, linewidth=0.8, label='Train Loss')
    if len(train_loss) > 10:
        smooth_loss = smooth(np.array(train_loss), window=10)
        ax1.plot(train_steps, smooth_loss, color=color1, linewidth=2, label='Train Loss (smooth)')
    ax1.set_xlabel('Step (iterations)')
    ax1.set_ylabel('Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title(f'Train Loss\n(last: {train_loss[-1]:.4f})', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # 添加 epoch 标注（每90个step一个epoch）
    epoch_ticks = [i for i, e in enumerate(train_epochs) if e % 10 == 0]
    if epoch_ticks:
        ax1_xticks = [train_steps[i] for i in epoch_ticks]
        ax1_xticklabels = [train_epochs[i] for i in epoch_ticks]
        ax1.set_xticks(ax1_xticks)
        ax1.set_xticklabels([f'Ep{int(e)}' for e in ax1_xticklabels])

    # 下图: Val mAP
    color2 = '#2ecc71'
    ax2.plot(val_steps, val_mAP, color=color2, linewidth=2, marker='o', markersize=3, label='Val mAP')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    best_step = val_steps[np.argmax(val_mAP)]
    best_mAP = max(val_mAP)
    ax2.set_title(f'Val mAP\n(best: {best_mAP:.2f}% at epoch {best_step})', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'图片已保存到: {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制 train loss 和 val mAP 曲线')
    parser.add_argument('scalars_path', help='vis_data/scalars.json 文件路径')
    parser.add_argument('--out', '-o', help='输出图片路径 (默认显示)')
    args = parser.parse_args()

    if not os.path.exists(args.scalars_path):
        print(f'文件不存在: {args.scalars_path}')
        sys.exit(1)

    plot_loss_and_map(args.scalars_path, args.out)