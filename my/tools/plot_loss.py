"""
绘制训练过程中的 train loss 和 val loss 曲线
用法: python my/tools/plot_loss.py <log文件路径> [--out 输出图片路径]
"""
import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """解析 .log 文件，提取 train loss 和 val loss"""
    train_losses = []  # [(epoch, step, loss), ...]
    val_losses = []    # [(epoch, step, loss), ...]

    train_pattern = re.compile(
        r'Epoch\(train\)\s*\[(\d+)\]\[(\d+)/\d+\].*?loss:\s*([\d.]+)'
    )
    val_pattern = re.compile(
        r'Epoch\(val\)\s*\[(\d+)\]\[(\d+)/\d+\].*?multi-label/mAP:\s*([\d.]+)'
    )

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = train_pattern.search(line)
            if m:
                epoch = int(m.group(1))
                step = int(m.group(2))
                loss = float(m.group(3))
                train_losses.append((epoch, step, loss))

            m = val_pattern.search(line)
            if m:
                epoch = int(m.group(1))
                step = int(m.group(2))
                loss = float(m.group(3))
                val_losses.append((epoch, step, loss))

    return train_losses, val_losses


def smooth(data, window=10):
    if len(data) < window:
        return data
    result = np.convolve(data, np.ones(window)/window, mode='valid')
    pad = window - 1
    return np.concatenate([np.full(pad, result[0]), result])


def plot(train_losses, val_losses, out_path=None):
    if not train_losses:
        print("未找到训练 loss 数据")
        return
    if not val_losses:
        print("未找到验证 loss 数据（仅绘制训练 loss）")
        return

    train_epochs = [t[0] for t in train_losses]
    train_steps = [t[1] for t in train_losses]
    train_loss = [t[2] for t in train_losses]

    val_epochs = [v[0] for v in val_losses]
    val_steps = [v[1] for v in val_losses]
    val_loss = [v[2] for v in val_losses]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Train loss
    ax.plot(train_steps, train_loss, color='#e74c3c', alpha=0.3, linewidth=0.8, label='Train Loss')
    if len(train_loss) > 15:
        ax.plot(train_steps, smooth(np.array(train_loss), 15), color='#e74c3c', linewidth=1.5, label='Train Loss (smooth)')

    # Val loss
    ax.plot(val_steps, val_loss, color='#3498db', linewidth=1.5, marker='o', markersize=3, label='Val Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Train vs Val Loss\n'
                 f'Train last: {train_loss[-1]:.4f}  |  Val last: {val_loss[-1]:.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # x轴每10个epoch标注
    unique_epochs = sorted(set(train_epochs + val_epochs))
    tick_epochs = [e for e in unique_epochs if e % 10 == 0]
    ax.set_xticks(tick_epochs)
    ax.set_xticklabels([f'{e}' for e in tick_epochs])

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'图片已保存到: {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制 train loss 和 val loss 曲线')
    parser.add_argument('log_path', help='.log 日志文件路径')
    parser.add_argument('--out', '-o', help='输出图片路径 (默认显示)')
    args = parser.parse_args()

    import os
    if not os.path.exists(args.log_path):
        print(f'文件不存在: {args.log_path}')
        sys.exit(1)

    train_losses, val_losses = parse_log(args.log_path)
    plot(train_losses, val_losses, args.out)